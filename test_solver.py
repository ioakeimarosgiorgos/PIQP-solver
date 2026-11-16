"""Checks the solver against SciPy's SLSQP on random feasible QPs and on a few
special cases. Run with `python test_solver.py` or `pytest`."""
import numpy as np
from scipy.optimize import minimize

from piqp_solver import piqp


def random_qp(n, p, m, seed):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    P = M @ M.T + 1e-2 * np.eye(n)
    c = rng.standard_normal(n)
    A = rng.standard_normal((p, n))
    G = rng.standard_normal((m, n))
    x0 = rng.standard_normal(n)                  # strictly feasible point
    return P, c, A, A @ x0, G, G @ x0 + rng.uniform(0.1, 1, m)


def reference(P, c, A, b, G, h):
    cons = [{"type": "ineq", "fun": lambda x: h - G @ x, "jac": lambda x: -G}]
    if A.shape[0]:
        cons.append({"type": "eq", "fun": lambda x: A @ x - b, "jac": lambda x: A})
    return minimize(lambda x: 0.5 * x @ P @ x + c @ x, np.zeros(c.size),
                    jac=lambda x: P @ x + c, constraints=cons, method="SLSQP",
                    options={"ftol": 1e-12, "maxiter": 500}).fun


def test_random_qps():
    for seed in range(30):
        n, p, m = 10 + seed % 7, seed % 4, 15 + seed % 5
        qp = random_qp(n, p, m, seed)
        P, c, *_ = qp
        res = piqp(qp)
        assert res.status == "solved", seed
        f = 0.5 * res.x @ P @ res.x + c @ res.x
        f_ref = reference(*qp)
        assert abs(f - f_ref) <= 1e-6 * max(1., abs(f_ref)), (seed, f, f_ref)


def test_lp():
    # min -x1 - x2  s.t.  x1 + x2 <= 1, x >= 0   -> optimal value -1
    res = piqp((np.zeros((2, 2)), -np.ones(2), np.zeros((0, 2)), np.zeros(0),
                np.array([[1., 1], [-1, 0], [0, -1]]), np.array([1., 0, 0])))
    assert res.status == "solved"
    assert abs(res.x.sum() - 1) < 1e-6


def test_equality_only():
    # min 1/2 |x|^2  s.t.  x1 + x2 = 1   -> x = (0.5, 0.5)
    res = piqp((np.eye(2), np.zeros(2), np.array([[1., 1]]), np.array([1.]),
                np.zeros((0, 2)), np.zeros(0)))
    assert res.status == "solved"
    assert np.allclose(res.x, [0.5, 0.5], atol=1e-6)


def test_infeasible_not_reported_as_solved():
    # x <= -1 and x >= 1
    res = piqp((np.eye(1), np.zeros(1), np.zeros((0, 1)), np.zeros(0),
                np.array([[1.], [-1]]), np.array([-1., -1])), max_iter=30)
    assert res.status != "solved"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: passed")
