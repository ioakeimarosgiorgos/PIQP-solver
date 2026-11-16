# QP solver based on the PIQP paper

A compact, readable Python/NumPy implementation of a **proximal interior-point
method** for convex quadratic programs, following the algorithm of

> R. Schwan, Y. Jiang, D. Kuhn, C. N. Jones,
> *PIQP: A Proximal Interior-Point Quadratic Programming Solver*, IEEE CDC 2023.
> [arXiv:2304.00290](https://arxiv.org/abs/2304.00290)

It solves

```
minimize    1/2 x'Px + c'x
subject to  Ax  = b
            Gx <= h
```

with `P` symmetric positive semidefinite. LPs (`P = 0`) work too.

This is an educational implementation written for the *Applied Optimal Control*
course. It is not the official PIQP solver: for real applications, use the
[official C++ implementation with Python bindings](https://github.com/PREDICT-EPFL/piqp).

## How it works

- **Interior-point method**: Mehrotra predictor-corrector steps on the KKT
  conditions with slacks `s >= 0` (`Gx + s = h`) and multipliers `z >= 0`.
  Primal and dual step lengths are chosen separately, with the
  fraction-to-boundary rule (`tau = 0.995`).
- **Proximal method of multipliers**: the proximal terms `rho/2 |x - xi|^2`
  and `delta`-regularization of the multipliers make the KKT matrix
  quasi-definite. The solver therefore also handles a singular `P` and
  redundant equality constraints. The proximal centres `xi, lambda, nu` are
  updated only when the primal or dual residual has decreased enough. `rho`
  and `delta` shrink as the complementarity `s'z` goes down.
- **Linear algebra**: each iteration builds the regularized KKT matrix

  ```
  [ P + rho I    A'         G'           ]
  [ A           -delta I    0            ]
  [ G            0         -(S/Z + delta I) ]
  ```

  and factorizes it once with a dense LU. The predictor and corrector steps
  reuse that factorization.
- **Initialization**: Mehrotra-style starting point, from one regularized KKT
  solve followed by a shift of `s, z` into the positive orthant.
- **Termination**: primal residual, dual residual and duality gap, each below
  `eps_abs + eps_rel * scale`.

## Usage

```python
import numpy as np
from piqp_solver import piqp

# min (x1-2)^2 + (x2-2)^2  s.t.  0 <= x <= 1.5,  x1 + 2 x2 <= 2.5
P = 2 * np.eye(2)
c = np.array([-4., -4.])
A, b = np.zeros((0, 2)), np.zeros(0)          # no equality constraints
G = np.array([[1., 0], [0, 1], [-1, 0], [0, -1], [1, 2]])
h = np.array([1.5, 1.5, 0, 0, 2.5])

res = piqp((P, c, A, b, G, h), verbose=True)
print(res.status, res.x)                      # solved [1.3 0.6]
```

`piqp` returns a `Result` with fields `x, y, z, s, status, iterations, pres,
dres, gap`. `status` is `"solved"` or `"max_iter_reached"`.

Optional arguments: `eps_abs`, `eps_rel` (default `1e-8`), `max_iter` (100),
the initial and minimum regularization `rho`, `delta`, `rho_min`, `delta_min`,
and `verbose`.

To leave out a constraint type, pass empty arrays of the right shape, e.g.
`A = np.zeros((0, n)), b = np.zeros(0)`.

## Files

| File | Contents |
|------|----------|
| `piqp_solver.py` | the solver |
| `demo.ipynb` | 2-D example with a plot, plus a random QP checked against SciPy |
| `test_solver.py` | tests: 30 random QPs vs. SciPy SLSQP, an LP, an equality-only QP, an infeasible QP |

```bash
pip install -r requirements.txt
python test_solver.py      # or: pytest
```

## Limitations

- **No infeasibility or unboundedness detection.** On such problems the solver
  runs until `max_iter` and returns `status = "max_iter_reached"`.
- Dense linear algebra (`O((n+p+m)^3)` per iteration), so it only suits
  small and medium problems. The real PIQP uses a sparse LDLᵀ factorization
  and a specialized variant for multistage problems such as MPC.
- No iterative refinement, problem scaling or box-constraint specialization.

## License

MIT. See [LICENSE](LICENSE).
