"""A compact Python implementation of a proximal interior-point QP solver,
following the algorithm in

    R. Schwan, Y. Jiang, D. Kuhn, C. N. Jones,
    "PIQP: A Proximal Interior-Point Quadratic Programming Solver",
    IEEE CDC 2023. arXiv:2304.00290

It solves

    minimize    1/2 x'Px + c'x
    subject to  Ax  = b
                Gx <= h

using a Mehrotra predictor-corrector interior-point method combined with a
proximal method of multipliers (regularization parameters rho, delta).
Linear systems are solved with a dense LU factorization, so the solver is
intended for small/medium problems and for learning purposes.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve


@dataclass
class Result:
    x: np.ndarray       # primal solution
    y: np.ndarray       # multipliers of Ax = b
    z: np.ndarray       # multipliers of Gx <= h (z >= 0)
    s: np.ndarray       # slacks, Gx + s = h (s >= 0)
    status: str         # "solved" or "max_iter_reached"
    iterations: int
    pres: float         # final primal residual (inf-norm)
    dres: float         # final dual residual (inf-norm)
    gap: float          # final duality gap


def norm_inf(v):
    return np.max(np.abs(v)) if np.size(v) else 0.


def kkt_matrix(P, A, G, w, rho, delta):
    n, p, m = P.shape[0], A.shape[0], G.shape[0]
    return np.block([[P + rho * np.eye(n), A.T,                 G.T],
                     [A,                  -delta * np.eye(p),   np.zeros((p, m))],
                     [G,                   np.zeros((m, p)),   -np.diag(w + delta)]])


def initialization(qp, rho, delta):
    """Mehrotra-style starting point: solve the regularized KKT system with
    W = I, then shift s and z into the strictly positive orthant."""
    P, c, A, b, G, h = qp
    n, p, m = c.size, A.shape[0], G.shape[0]

    K = kkt_matrix(P, A, G, np.ones(m), rho, delta)
    sol = np.linalg.solve(K, np.concatenate([-c, b, h]))
    x, y, nu = sol[:n], sol[n:n + p], sol[n + p:]
    if m == 0:
        return x, y, np.zeros(0), np.zeros(0)

    s = -nu
    ds, dnu = max(0., -1.5 * s.min()), max(0., -1.5 * nu.min())
    sp, nup = s + ds, nu + dnu
    gram = sp @ nup
    eps = 1e-12  # guards against division by zero when nu == 0
    s = sp + 0.5 * gram / max(nup.sum(), eps)
    z = nup + 0.5 * gram / max(sp.sum(), eps)
    # make sure the starting point is strictly interior
    s, z = np.maximum(s, 1e-8), np.maximum(z, 1e-8)
    return x, y, z, s


def termination(qp, x, y, z, s, eps_abs, eps_rel):
    P, c, A, b, G, h = qp
    Px = P @ x
    pres = norm_inf(np.concatenate([A @ x - b, G @ x - h + s]))
    dres = norm_inf(Px + A.T @ y + G.T @ z + c)
    gap = abs(x @ Px + c @ x + b @ y + h @ z)
    ok = (pres <= eps_abs + eps_rel * max(norm_inf(A @ x), norm_inf(b), norm_inf(G @ x),
                                          norm_inf(h), norm_inf(s))
          and dres <= eps_abs + eps_rel * max(norm_inf(Px), norm_inf(A.T @ y),
                                              norm_inf(G.T @ z), norm_inf(c))
          and gap <= eps_abs + eps_rel * max(abs(x @ Px), abs(c @ x), abs(b @ y), abs(h @ z)))
    return ok, pres, dres, gap


def max_step(v, dv, tau=0.995):
    """Largest alpha in (0, 1] such that v + alpha*dv >= (1-tau)*v."""
    neg = dv < 0
    return 1. if not np.any(neg) else float(min(1., np.min(-tau * v[neg] / dv[neg])))


def piqp(qp, eps_abs=1e-8, eps_rel=1e-8, max_iter=100, rho=1e-6, delta=1e-4,
         rho_min=1e-10, delta_min=1e-10, verbose=False):
    """Solve the QP given as the tuple qp = (P, c, A, b, G, h).

    Use A = np.zeros((0, n)), b = np.zeros(0) for no equality constraints and
    G = np.zeros((0, n)), h = np.zeros(0) for no inequality constraints.

    Returns a Result. Note: there is no infeasibility detection. On an
    infeasible or unbounded problem the solver stops after max_iter iterations
    with status "max_iter_reached".
    """
    P, c, A, b, G, h = (np.asarray(v, dtype=float) for v in qp)
    qp = (P, c, A, b, G, h)
    n, p, m = c.size, A.shape[0], G.shape[0]
    x, y, z, s = initialization(qp, rho, delta)
    xi, lam, nu = x.copy(), y.copy(), z.copy()   # proximal centres

    pk = norm_inf(np.concatenate([A @ x - b, G @ x - h + s]))
    dk = norm_inf(P @ x + c + A.T @ y + G.T @ z)
    status = "max_iter_reached"

    for k in range(max_iter):
        ok, pres, dres, gap = termination(qp, x, y, z, s, eps_abs, eps_rel)
        mu = s @ z / m if m else 0.
        if verbose:
            print(f"k={k:3d}: pres={pres:.1e} dres={dres:.1e} gap={gap:.1e} mu={mu:.1e}")
        if ok:
            status = "solved"
            break

        # residuals of the proximal (regularized) KKT system
        rx = -(P @ x + c + rho * (x - xi) + A.T @ y + G.T @ z)
        ry = -(A @ x + delta * (lam - y) - b)
        rz = -(G @ x + delta * (nu - z) - h + s)
        lu = lu_factor(kkt_matrix(P, A, G, s / z, rho, delta))

        def direction(rs):
            d = lu_solve(lu, np.concatenate([rx, ry, rz - rs / z]))
            dz = d[n + p:]
            return d[:n], d[n:n + p], dz, (rs - s * dz) / z

        if m:
            # predictor (affine-scaling) step
            dxa, dya, dza, dsa = direction(-s * z)
            ap, ad = max_step(s, dsa), max_step(z, dza)
            sigma = max(0., min(1., ((s + ap * dsa) @ (z + ad * dza)) / (s @ z))) ** 3
            # corrector step
            dx, dy, dz, ds = direction(-s * z - dsa * dza + sigma * mu)
            ap, ad = max_step(s, ds), max_step(z, dz)
        else:
            dx, dy, dz, ds = direction(np.zeros(0))
            ap = ad = 1.

        sz_old = s @ z
        x = x + ap * dx
        s = s + ap * ds
        y = y + ad * dy
        z = z + ad * dz

        # proximal-parameter update: r in [0, 1] measures progress in s'z
        r = min(1., abs(sz_old - s @ z) / sz_old) if m and sz_old > 0 else 0.5
        pk1 = norm_inf(np.concatenate([A @ x - b, G @ x - h + s]))
        dk1 = norm_inf(P @ x + c + A.T @ y + G.T @ z)
        if pk1 <= 0.95 * pk:
            lam, nu = y.copy(), z.copy()
            delta = (1 - r) * delta
        else:
            delta = (1 - r / 3) * delta
        if dk1 <= 0.95 * dk:
            xi = x.copy()
            rho = (1 - r) * rho
        else:
            rho = (1 - r / 3) * rho
        delta, rho = max(delta, delta_min), max(rho, rho_min)
        pk, dk = pk1, dk1
    else:
        ok, pres, dres, gap = termination(qp, x, y, z, s, eps_abs, eps_rel)
        k = max_iter
        if ok:
            status = "solved"

    return Result(x, y, z, s, status, k, pres, dres, gap)
