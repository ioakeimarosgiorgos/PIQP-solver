"""
It solves

    minimize    1/2 x'Px + c'x
    subject to  Ax  = b
                Gx <= h

using a Mehrotra predictor-corrector interior-point method combined with a
proximal method of multipliers (regularization parameters rho, delta).
Linear systems are solved with a dense LU factorization, so the solver is
intended for small/medium problems and for learning purposes.
"""
import numpy as np
from scipy.linalg import lu_factor, lu_solve

def norm_inf(v):
    return np.max(np.abs(v)) if np.size(v) else 0.

def kkt_matrix(P, A, G, w, rho, delta):
    n, p, m = P.shape[0], A.shape[0], G.shape[0]
    return np.block([[P + rho * np.eye(n), A.T,                 G.T],
                     [A,                  -delta * np.eye(p),   np.zeros((p, m))],
                     [G,                   np.zeros((m, p)),   -np.diag(w + delta)]])

def initialization(qp, rho, delta):
    P, c, A, b, G, h = qp
    n, p, m = c.size, A.shape[0], G.shape[0]

    K = np.block([[P+rho*np.eye(n), A.T, G.T],
                  [A, -delta*np.eye(p), np.zeros((p,m))],
                  [G, np.zeros((m,p)), -(1+delta)*np.eye(m)]])
    solution = np.linalg.solve(K, np.concatenate([-c, b, h]))
    x, y, nu = solution[:n], solution[n:n+p], solution[n+p:]
    if m == 0: # no inequalities -> no s, z
        return x, y, np.zeros(0), np.zeros(0)
    s = -nu
    ds, dnu = max(0, -1.5*s.min()), max(0, -1.5*nu.min())
    sp, nup = s + ds, nu + dnu
    gram = sp @ nup
    s = s + ds + 0.5*gram/max(nup.sum(), 1e-12) # max() to avoid dividing by 0 if nu = 0
    z = nu + dnu + 0.5 * gram / max(sp.sum(), 1e-12)
    s, z = np.maximum(s, 1e-8), np.maximum(z, 1e-8)

    return x, y, z, s

def termination(qp, x, y, z, s, eps_abs, eps_rel):
    P, c, A, b, G, h = qp
    pres = norm_inf(np.concatenate([A @ x -b, G @ x -h +s]))
    dres = norm_inf(P@x + A.T@y + G.T@z + c)
    gap = abs(x.T @ P @ x + c.T @ x + b.T @ y + h.T @ z)
    ok = ( pres <= eps_abs + eps_rel*max(norm_inf(A@x), norm_inf(b), norm_inf(G@x), norm_inf(h), norm_inf(s)) and
              dres <= eps_abs + eps_rel*max(norm_inf(P@x), norm_inf(A.T@y), norm_inf(G.T@z), norm_inf(c)) and
              gap <= eps_abs + eps_rel*max(abs(x.T@P@x), abs(c.T@x), abs(b.T@y), abs(h.T@z)))
    return ok, pres, dres, gap

def max_step(v, dv, tau=0.995):
    neg = dv < 0
    return 1. if not np.any(neg) else float(min(1., np.min(-tau * v[neg] / dv[neg])))

def piqp(qp, eps_abs=1e-8, eps_rel=1e-8, max_iter=100, rho=1e-6, delta=1e-4, rho_min=1e-10, delta_min=1e-10, verbose=False):
    P, c, A, b, G, h = qp
    n, p, m = c.size, A.shape[0], G.shape[0]
    x, y, z, s = initialization(qp, rho, delta)
    xi, lam, nu = x.copy(), y.copy(), z.copy()

    pk = norm_inf(np.concatenate([A@x-b, G@x - h + s]))
    dk = norm_inf(P@x + c + A.T@y + G.T@z)

    for k in range(max_iter):
        ok, pres, dres, gap = termination(qp, x, y, z, s, eps_abs, eps_rel)
        mu = s@z/m if m > 0 else 0.
        if verbose: print(f"k={k}: pres={pres:.1e} dres={dres:.1e} gap={gap:.1e} mu={mu:.1e}")
        if ok: return x, y, z, s, "solved"

        rx = -(P@x + c + rho * (x-xi) +A.T@y + G.T@z)
        ry = -(A@x +delta*(lam - y) - b)
        rz = -(G@x + delta*(nu - z) - h + s)
        lu = lu_factor(kkt_matrix(P, A, G, s/z, rho, delta))
        def direction(rs):
            d = lu_solve(lu, np.concatenate([rx,ry,rz - rs/ z]))
            return d[:n], d[n:n+p], d[n+p:], (rs - s*d[n+p:])/z

        if m > 0:
            #prediction step
            dxa, dya, dza, dsa = direction(-s*z)
            ap = max_step(s, dsa)
            ad = max_step(z, dza)
            sigma = max(0., min(1., (((s+ap*dsa)@(z+ad*dza))/(s@z))))**3
            #correction step
            dx, dy, dz, ds = direction(-s*z - dsa*dza + sigma*mu)
            ap = max_step(s, ds)
            ad = max_step(z, dz)
        else: # only equalities -> plain newton step
            dx, dy, dz, ds = direction(np.zeros(0))
            ap, ad = 1., 1.

        sz_old = s@z
        s = s + ap*ds
        x = x + ap*dx
        y = y + ad*dy
        z = z + ad*dz

        r = min(1., abs(sz_old - s@z)/sz_old) if sz_old > 0 else 0.5 # r>1 would make delta, rho negative
        pk1 = norm_inf(np.concatenate([A@x-b, G@x - h + s]))
        dk1 = norm_inf(P@x + c + A.T@y + G.T@z)
        if pk1 <= 0.95*pk:
            lam, nu = y.copy(), z.copy()
            delta = (1-r)*delta
        else: delta = (1-r/3)*delta
        if dk1 <= 0.95*dk:
            xi = x.copy()
            rho = (1-r)*rho
        else: rho = (1-r/3)*rho
        delta, rho = max(delta, delta_min), max(rho, rho_min)
        pk, dk = pk1, dk1

    ok, pres, dres, gap = termination(qp, x, y, z, s, eps_abs, eps_rel)
    if ok: return x, y, z, s, "solved"
    print("warning: max_iter reached (problem may be infeasible)")
    return x, y, z, s, "max_iter_reached"
