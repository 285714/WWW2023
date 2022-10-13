import itertools
from data import *
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.vq import kmeans2, ClusterError
import time
# import cvxpy as cp
USE_CVXPY = False
CVXPY_SOLVER = "Gurobi"


def svd_learn_new(sample, n, L=None, compress=True, verbose=None, sample_dist=1, stats={}, em_refine_max_iter=0, n_repeat=5):

    class PartialR:
        def __init__(self, R, states, i, j):
            self.R = R
            self.Rinv = np.linalg.pinv(R)
            self.states = set(states)
            self.i = i
            self.j = j

        def reconstruct(self):
            Ys = self.R @ Ys_
            Ps = Ys @ Ps_
            S = np.real(np.diagonal(self.R @ Zs_ @ np.transpose(Ys_, axes=(0,2,1)) @ self.R.T, axis1=1, axis2=2).T)
            Ms = np.real(np.transpose(Ps / S, axes=(1,2,0)))
            return Mixture(S, Ms)

    def reconstruct_at(i, j):
        E = ProdsInv[i] @ Prods[j]
        eigs, w = np.linalg.eig(E)
        if L is not None:
            mask = np.argpartition(eigs, -L)[-L:]
        else:
            mask = eigs > 1e-5
        R_ = w[:, mask]
        d, _, _, _ = np.linalg.lstsq((R_.T @ Ys_[i] @ Ps_[i]).T, Os[i] @ np.ones(n), rcond=None)
        R = np.diag(d) @ R_.T
        return R

    def combine(parts):
        collectedRinv = np.real(np.vstack(list(p.Rinv.T for p in parts)))
        collectedRinvOrigin = [ i for i, p in enumerate(parts) for _ in range(len(p.Rinv.T)) ]

        dists = np.zeros(len(collectedRinv) * (len(collectedRinv) - 1) // 2)
        for k, ((p1, rinv1), (p2, rinv2)) in enumerate(itertools.combinations(zip(collectedRinvOrigin, collectedRinv), 2)):
            dists[k] = np.linalg.norm(rinv1 - rinv2)**2 / (np.linalg.norm(rinv1) * np.linalg.norm(rinv2))
            if p1 == p2:
                dists[k] = 10
            elif parts[p1].i == parts[p2].j or parts[p1].j == parts[p2].i:
                dists[k] /= 10

        # fcluster seems buggy, so here's a quick fix
        dist_mtrx = squareform(dists + 1e-10 * np.random.rand(*dists.shape))
        double_dists = [0 if i//2 == j//2 else dist_mtrx[i//2, j//2]
                        for i, j in itertools.combinations(range(2 * len(dist_mtrx)), 2)]
        lnk = linkage(double_dists, method="complete")
        double_groups = fcluster(lnk, r, criterion="maxclust") - 1
        groups = np.array([g for i, g in enumerate(double_groups) if i%2])
        assert(max(groups)+1 == r)

        combinedRinv = np.zeros((r, r))
        for l in range(r):
            cluster = collectedRinv[groups==l]
            intra_dists = np.sum(squareform(dists)[groups==l][:,groups==l], axis=0)
            center = cluster[np.argmin(intra_dists)]
            combinedRinv[l] = center

        assert(len(combinedRinv) == r)
        R = np.linalg.pinv(combinedRinv.T)

        def asgn_mtrx_cvx(mass):
            A = cp.Variable((L, r), boolean=True)
            objective = cp.Minimize(cp.sum(cp.max(A @ mass, axis=0)))
            constraint = cp.sum(A, axis=0) == 1
            prob = cp.Problem(objective, [constraint])
            try:
                prob.solve(verbose=False, solver=CVXPY_SOLVER, maximumSeconds=5)
                assert(A.value.shape == (L, r))
                return A.value
            except Exception as e:
                print("solver exception:", e)
                return np.tile(np.eye(L), r // L + 1)[:,:r]

        def asgn_mtrx_bf(mass):
            asgn = np.zeros(r ,dtype=int)
            asgn_mass = np.zeros((L, n))
            asgn_cost = 0
            min_asgn_cost = np.inf
            min_asgn = np.zeros_like(asgn)

            def assign(i, j):
                nonlocal asgn_cost
                asgn[i] = j
                cost = np.linalg.norm(np.minimum(asgn_mass[j], mass[i])) ** 2
                asgn_cost += cost
                asgn_mass[j] += mass[i]
                return (i, j, cost)

            def unassign(i, j, cost):
                nonlocal asgn_cost
                asgn_cost -= cost
                asgn_mass[j] -= mass[i]

            # greedy initialization
            xs = []
            for i in range(len(mass)):
                j = np.argmin(np.linalg.norm(asgn_mass, axis=1))
                xs.append(assign(i, j))
            min_asgn[:] = asgn
            min_asgn_cost = asgn_cost
            for x in xs:
                unassign(*x)

            solve_time = time.time()
            def comp_rec(i):
                nonlocal asgn_cost, min_asgn_cost
                if asgn_cost > min_asgn_cost:
                    return False
                if i >= len(mass):
                    min_asgn[:] = asgn
                    min_asgn_cost = asgn_cost
                    return time.time() > solve_time + 1
                js = list(range(L))
                np.random.shuffle(js)
                for j in js:
                    x = assign(i, j)
                    if comp_rec(i + 1): return True
                    unassign(*x)

            if comp_rec(0) and verbose: print("timeout for asgnm")
            mtrx = np.zeros((L, r), dtype=int)
            mtrx[min_asgn, range(r)] = 1
            return mtrx

        Ys = R @ Ys_
        asgn_mtrx = asgn_mtrx_cvx if USE_CVXPY else asgn_mtrx_bf
        comp = asgn_mtrx(np.linalg.norm(Ys, axis=2, ord=1).T) if compress else np.eye(r)
        if verbose: print(comp)
        compressedR = comp @ R
        if verbose:
            print(f"compressedR.shape = {compressedR.shape} (ideal is {L},{r})")

        Ys = compressedR @ Ys_
        Zs = compressedR @ Zs_
        Ps = Ys @ Ps_
        S = np.real(np.diagonal(Zs @ np.transpose(Ys, axes=(0,2,1)), axis1=1, axis2=2).T)
        Ms = np.real(np.transpose(Ps / S, axes=(1,2,0)))
        return S, Ms

    def em_refine(m, states=range(n), em_refine_max_iter=2):
        states = list(states)
        d = sample #.restrict_to(states)
        # m = mixture.restrict_to(states)
        return em_learn(d, len(states), L, max_iter=em_refine_max_iter, init_mixture=m)

    def find_representative(X):
        X = np.array(X)
        Y = np.empty(X.shape)
        Y[:] = np.nan
        ixs = (X > -0.5) & (X < 1.5)
        Y[ixs] = X[ixs]
        return np.nanmedian(Y, axis=0)

    Os = np.moveaxis(sample.all_trail_probs(), 1, 0)
    us, ss, vhs = np.linalg.svd(Os)
    if L is None:
        ss_norm = np.linalg.norm(ss, axis=0)
        for i, s_norm in enumerate(ss_norm):
            stats[f"sval-{i}"] = s_norm
        ratios = ss_norm[:-1] / ss_norm[1:]
        for i, ratio in enumerate(ratios):
            stats[f"sval-ratio-{i}"] = ratio
        L = 1 + np.argmax(ratios * (ss_norm[:-1] > 1e-6))
        stats["guessedL"] = L
        if mixture is not None:
            sigma_min = min(np.min(np.linalg.svd(X, compute_uv=False)) for i in range(n) for X in [mixture.Ms[:,i,:], mixture.Ms[:,:,i]])
            stats["sigma_min"] = sigma_min
    Ps_ = np.moveaxis(us, 1, 2)[:,:L]
    Qs_ = (ss[:,:L].T * vhs[:,:L].T).T

    A = np.zeros((2 * n * L, n ** 2))
    for j in range(n):
        A[L * j:L * (j + 1), n * j:n * (j + 1)] = Ps_[j]
        A[L * (n + j):L * (n + j + 1), j + n * (np.arange(n))] = -Qs_[j]

    _, s, vh = np.linalg.svd(A.T, full_matrices=True)
    s_inc = s[::-1]
    s_inc_c = s_inc[:n-1]
    ratios = s_inc_c[1:] / s_inc_c[:-1]
    r = max(L, 1 + np.argmax(ratios * (s_inc_c[1:] < 0.1)))
    if verbose:
        print(s_inc[:4])
        with np.printoptions(precision=5, suppress=True, linewidth=np.inf):
            print("singular values of A:", " ".join( f"{x:.5f}" for x in s_inc))
            print("ratios:                      ", " ".join( f"{x:.5f}" for x in ratios))
            print(f"suggest r={r} (vs L={L})")
    B = vh[-r:] # np.random.rand(r,r) @
    Bre = np.moveaxis(B.reshape((r, L, 2 * n), order="F"), -1, 0)
    Ys_ = Bre[0:n]
    Zs_ = Bre[n:2*n]

    Prods = Zs_ @ np.transpose(Ys_, axes=(0,2,1))
    ProdsInv = np.linalg.pinv(Prods)

    dists = []
    dists2 = []
    for i, j in itertools.combinations(range(n), 2):
        E = ProdsInv[i] @ Prods[j]
        Einv = ProdsInv[j] @ Prods[i]
        # randomized pseudoinverse test
        x = E @ np.random.rand(r, 100)
        y = Einv @ E @ x
        dists.append(np.linalg.norm(x - y)**2 / (np.linalg.norm(x) * np.linalg.norm(y)))
        dists2.append(np.linalg.norm(x - y))

    lnk = linkage(np.array(dists), method="complete")
    groups = fcluster(lnk, sample_dist, criterion="distance")

    if verbose:
        for t in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]:
            groups_ = fcluster(lnk, t, criterion="distance")
            print(groups_, t)

    dist_mtrx = squareform(dists)
    np.fill_diagonal(dist_mtrx, np.inf)
    parts = []

    best_learned_mixture = None
    best_dist = np.inf
    for _ in range(n_repeat):
        try:
            for g in range(max(groups)):
                states = np.where(groups == g+1)[0]
                if len(states) > 1:
                    i, j = np.random.choice(states, 2, replace=False)
                else:
                    [i] = states
                    j = np.argmin(dist_mtrx[i])
                if verbose: print(f"=== group {g}: {states} i={i} j={j} {'='*50}")
                R = reconstruct_at(i, j)
                parts.append(PartialR(R, states, i, j))

            if len(parts) > 1:
                S, Ms = combine(parts)
            else:
                m = parts[0].reconstruct()
                S, Ms = m.S, m.Ms

            S, Ms = np.abs(S), np.abs(Ms)
            learned_mixture = Mixture(S, Ms)
            if em_refine_max_iter > 0:
                learned_mixture = em_refine(learned_mixture, em_refine_max_iter=em_refine_max_iter)
            learned_mixture.normalize()

            d = Distribution.from_mixture(learned_mixture, 3).dist(sample) if n_repeat > 1 else 0
            if d < best_dist:
                best_dist = d
                best_learned_mixture = learned_mixture
        except Exception as e:
            import traceback
            print(traceback.format_exc())

    return best_learned_mixture


def svd_learn(sample, n, L=None, verbose=None, stats={}):
    Os = np.moveaxis(sample.all_trail_probs(), 1, 0)

    svds = [ np.linalg.svd(Os[j], full_matrices=True) for j in range(n) ]

    if verbose:
        for i, (_, s, _) in enumerate(svds):
            print(f"{i}: {s[:L+1]} ...")

    Ps_ = np.zeros((n, L, n))
    Qs_ = np.zeros((n, L, n))
    for j, (u, s, vh) in enumerate(svds):
        Ps_[j, 0:min(n,L), :] = u[:, 0:L].T
        Qs_[j, 0:min(n,L), :] = (np.diag(s) @ (vh))[0:L, :]

    A = np.zeros((2 * n * L, n**2))
    for j in range(n):
        A[L*j:L*(j+1), n*j:n*(j+1)] = Ps_[j]
        A[L*(n+j):L*(n+j+1), j+n*(np.arange(n))] = -Qs_[j]

    _, s, vh = np.linalg.svd(A.T, full_matrices=True)
    small = list(s < 1e-5)
    if True in small:
        fst = small.index(True)
        if verbose: print(2*L*n - fst, L, s[[fst-1, fst]])
    B = vh[-L:]
    Bre = np.moveaxis(B.reshape((L, L, 2*n), order="F"), -1, 0)
    Ys_ = Bre[0:n]
    Zs_ = Bre[n:2*n]

    Xs = [ np.linalg.pinv(Zs_[j] @ Ys_[j].T) @ (Zs_[j+1] @ Ys_[j+1].T) for j in range(n-1) ]
    X = np.sum(Xs, axis=0)
    _, R_ = np.linalg.eig(X)
    d, _, _, _ = np.linalg.lstsq((R_.T @ Ys_[0] @ Ps_[0]).T, Os[0] @ np.ones(n), rcond=None)

    R = np.diag(d) @ R_.T
    Ys = R @ Ys_

    Ps = np.array([ Y @ P_ for Y, P_ in zip(Ys, Ps_) ])
    Ss = np.array([ R @ Z_ @ Y_.T @ R.T for Z_, Y_ in zip(Zs_, Ys_) ])

    S_ = np.zeros((L, n))
    Ms_ = np.zeros((L, n, n))
    for l in range(L):
        for i in range(n):
            S_[l,i] = Ss[i,l,l]
            for j in range(n):
                Ms_[l,i,j] = Ps[j,l,i] / S_[l,i]

    S_ = np.abs(S_)
    Ms_ = np.abs(Ms_)
    learned_mixture = Mixture(S_, Ms_)
    learned_mixture.normalize()
    return learned_mixture


def em_learn(sample, n, L, max_iter=10000, ll_stop=1e-4, verbose=None, init_mixture=None, stats={}, mixture=None,
             write_stats=True):
    flat_mixture = (init_mixture or Mixture.random(n, L)).flat()
    flat_trails, trail_probs = sample.flat_trails()

    prev_lls = 0
    for n_iter in range(max_iter):
        lls = flat_trails @ np.log(flat_mixture + 1e-20).transpose()
        if ll_stop is not None and np.max(np.abs(prev_lls - lls)) < ll_stop: break
        prev_lls = lls

        raw_probs = np.exp(lls)
        cond_probs = raw_probs / np.sum(raw_probs, axis=1)[:, np.newaxis]
        cond_probs[np.any(np.isnan(cond_probs), axis=1)] = 1 / L

        flat_mixture = cond_probs.transpose() @ (flat_trails * trail_probs[:, np.newaxis])
        # normalize:
        flat_mixture[:, :n] /= np.sum(flat_mixture[:, :n])
        for i in range(n):
            rows = flat_mixture[:, (i+1)*n:(i+2)*n]
            rows[:] = rows / rows.sum(axis=1)[:, np.newaxis]
            rows[np.any(np.isnan(rows), axis=1)] = 1 / n

        if verbose is not None:
            learned_mixture = Mixture.from_flat(flat_mixture, n)
            learned_distribution = Distribution.from_mixture(learned_mixture, sample.t_len)
            print("Iteration {}: recovery_error={} tv_dist={}".format(
                n_iter + 1, verbose.recovery_error(learned_mixture) if isinstance(verbose, Mixture) else np.inf,
                learned_distribution.dist(sample)))

    if write_stats: stats["n_iter"] = n_iter
    return Mixture.from_flat(flat_mixture, n)
