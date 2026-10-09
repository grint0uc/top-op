"""Method of Moving Asymptotes: numpy port of Svanberg's `mmasub` + `subsolv` (2007 MATLAB code).

Solves  min f0(x) + a0 z + sum(c y + d y^2 / 2)  s.t.  f_i(x) - a_i z - y_i <= 0,  xmin <= x <= xmax
by a sequence of convex separable subproblems (primal-dual interior point in `_subsolv`).
With a0 = 1, a = 0, d = 0 and large c this is the plain problem  min f0  s.t.  f_i <= 0.
"""

from __future__ import annotations

import numpy as np


class MMA:
    """Stateful MMA optimizer: keeps xold1, xold2 and the asymptotes low, upp between updates."""

    asyinit = 0.5
    asyincr = 1.2
    asydecr = 0.7
    albefa = 0.1
    raa0 = 1e-5
    epsimin = 1e-7

    def __init__(
        self,
        n: int,
        m: int,
        xmin: float | np.ndarray,
        xmax: float | np.ndarray,
        move: float,
        *,
        a0: float = 1.0,
        a: np.ndarray | None = None,
        c: np.ndarray | None = None,
        d: np.ndarray | None = None,
    ):
        self.n, self.m = int(n), int(m)
        self.xmin = np.broadcast_to(np.asarray(xmin, dtype=np.float64), (self.n,)).copy()
        self.xmax = np.broadcast_to(np.asarray(xmax, dtype=np.float64), (self.n,)).copy()
        if np.any(self.xmax <= self.xmin):
            raise ValueError("xmax must exceed xmin")
        self.move = float(move)
        self.a0 = float(a0)
        self.a = np.zeros(self.m) if a is None else np.asarray(a, dtype=np.float64)
        self.c = np.full(self.m, 1000.0) if c is None else np.asarray(c, dtype=np.float64)
        self.d = np.zeros(self.m) if d is None else np.asarray(d, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        """Forget the history: the next two updates re-initialize the asymptotes, as at iter 1, 2."""
        self.xold1: np.ndarray | None = None
        self.xold2: np.ndarray | None = None
        self.low: np.ndarray | None = None
        self.upp: np.ndarray | None = None
        self.lam = np.zeros(self.m)  # multipliers of the last subproblem
        self.y = np.zeros(self.m)  # artificial variables (> 0: the subproblem was infeasible)

    def update(
        self,
        iter: int,
        x: np.ndarray,
        f0: float,
        df0dx: np.ndarray,
        fval: np.ndarray,
        dfdx: np.ndarray,
    ) -> np.ndarray:
        """One MMA step from `x`; `iter` is 1-based (iter <= 2 re-initializes the asymptotes, as
        do the first two updates after `reset()` whatever `iter` is).

        fval (m,) are the constraint values f_i(x) (<= 0 feasible), dfdx (m, n) their gradients.
        """
        n, m = self.n, self.m
        x = np.asarray(x, dtype=np.float64).reshape(n)
        df0dx = np.asarray(df0dx, dtype=np.float64).reshape(n)
        fval = np.asarray(fval, dtype=np.float64).reshape(m)
        dfdx = np.asarray(dfdx, dtype=np.float64).reshape(m, n)
        xmin, xmax = self.xmin, self.xmax
        span = xmax - xmin

        if iter <= 2 or self.xold2 is None or self.low is None:
            low = x - self.asyinit * span
            upp = x + self.asyinit * span
        else:
            zzz = (x - self.xold1) * (self.xold1 - self.xold2)
            factor = np.ones(n)
            factor[zzz > 0] = self.asyincr
            factor[zzz < 0] = self.asydecr
            low = x - factor * (self.xold1 - self.low)
            upp = x + factor * (self.upp - self.xold1)
            low = np.clip(low, x - 10 * span, x - 0.01 * span)
            upp = np.clip(upp, x + 0.01 * span, x + 10 * span)

        alfa = np.maximum(np.maximum(low + self.albefa * (x - low), x - self.move * span), xmin)
        beta = np.minimum(np.minimum(upp - self.albefa * (upp - x), x + self.move * span), xmax)

        xmamiinv = 1.0 / np.maximum(span, 1e-5)
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        p0 = np.maximum(df0dx, 0.0)
        q0 = np.maximum(-df0dx, 0.0)
        pq0 = 0.001 * (p0 + q0) + self.raa0 * xmamiinv
        p0 = (p0 + pq0) * ux2
        q0 = (q0 + pq0) * xl2
        P = np.maximum(dfdx, 0.0)
        Q = np.maximum(-dfdx, 0.0)
        PQ = 0.001 * (P + Q) + self.raa0 * xmamiinv[None, :]
        P = (P + PQ) * ux2[None, :]
        Q = (Q + PQ) * xl2[None, :]
        b = _mv(P, 1.0 / ux1) + _mv(Q, 1.0 / xl1) - fval

        sub = (low, upp, alfa, beta, p0, q0, P, Q, b)
        xnew, y, _z, lam = _subsolv(*sub, self.a0, self.a, self.c, self.d, self.epsimin)
        self.xold2 = self.xold1
        self.xold1 = x.copy()
        self.low, self.upp = low, upp
        self.lam, self.y = lam, y
        return xnew


# m x n products without BLAS: OpenBLAS wakes its thread pool even for these small operands,
# which costs milliseconds per call while the solver's threads are around (docs/PERF.md)
def _mv(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("ij,j->i", A, v)


def _vm(v: np.ndarray, A: np.ndarray) -> np.ndarray:
    return np.einsum("i,ij->j", v, A)


def _subsolv(low, upp, alfa, beta, p0, q0, P, Q, b, a0, a, c, d, epsimin):
    """Primal-dual Newton method for the MMA subproblem; returns (x, y, z, lam)."""
    m, n = P.shape
    epsi = 1.0
    x = 0.5 * (alfa + beta)
    y = np.ones(m)
    z = 1.0
    lam = np.ones(m)
    xsi = np.maximum(1.0 / (x - alfa), 1.0)
    eta = np.maximum(1.0 / (beta - x), 1.0)
    mu = np.maximum(np.ones(m), 0.5 * c)
    zet = 1.0
    s = np.ones(m)

    def residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi) -> tuple[float, float]:
        ux1 = upp - x
        xl1 = x - low
        plam = p0 + _vm(lam, P)
        qlam = q0 + _vm(lam, Q)
        gvec = _mv(P, 1.0 / ux1) + _mv(Q, 1.0 / xl1)
        r = np.concatenate(
            [
                plam / (ux1 * ux1) - qlam / (xl1 * xl1) - xsi + eta,  # rex
                c + d * y - mu - lam,  # rey
                [a0 - zet - a @ lam],  # rez
                gvec - a * z - y + s - b,  # relam
                xsi * (x - alfa) - epsi,
                eta * (beta - x) - epsi,
                mu * y - epsi,
                [zet * z - epsi],
                lam * s - epsi,
            ]
        )
        return float(np.sqrt(np.einsum("i,i->", r, r))), float(np.abs(r).max())

    while epsi > epsimin:
        resnorm, resmax = residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi)
        ittt = 0
        while resmax > 0.9 * epsi and ittt < 200:
            ittt += 1
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            uxinv2 = 1.0 / ux2
            xlinv2 = 1.0 / xl2
            plam = p0 + _vm(lam, P)
            qlam = q0 + _vm(lam, Q)
            gvec = _mv(P, 1.0 / ux1) + _mv(Q, 1.0 / xl1)
            GG = P * uxinv2[None, :] - Q * xlinv2[None, :]
            dpsidx = plam * uxinv2 - qlam * xlinv2
            delx = dpsidx - epsi / (x - alfa) + epsi / (beta - x)
            dely = c + d * y - lam - epsi / y
            delz = a0 - a @ lam - epsi / z
            dellam = gvec - a * z - y - b + epsi / lam
            diagx = 2 * (plam / (ux2 * ux1) + qlam / (xl2 * xl1)) + xsi / (x - alfa)
            diagx += eta / (beta - x)
            diagxinv = 1.0 / diagx
            diagy = d + mu / y
            diagyinv = 1.0 / diagy
            diaglamyi = s / lam + diagyinv
            if m < n:
                blam = dellam + dely / diagy - _mv(GG, delx * diagxinv)
                AA = np.empty((m + 1, m + 1))
                AA[:m, :m] = np.einsum("ij,kj->ik", GG * diagxinv[None, :], GG)
                AA[:m, :m] += np.diag(diaglamyi)
                AA[:m, m] = a
                AA[m, :m] = a
                AA[m, m] = -zet / z
                sol = np.linalg.solve(AA, np.concatenate([blam, [delz]]))
                dlam = sol[:m]
                dz = float(sol[m])
                dx = -delx * diagxinv - _vm(dlam, GG) * diagxinv
            else:
                diaglamyiinv = 1.0 / diaglamyi
                dellamyi = dellam + dely / diagy
                AA = np.empty((n + 1, n + 1))
                AA[:n, :n] = np.diag(diagx) + (GG.T * diaglamyiinv[None, :]) @ GG
                axz = -GG.T @ (a * diaglamyiinv)
                AA[:n, n] = axz
                AA[n, :n] = axz
                AA[n, n] = zet / z + a @ (a * diaglamyiinv)
                bx = delx + GG.T @ (dellamyi * diaglamyiinv)
                bz = delz - a @ (dellamyi * diaglamyiinv)
                sol = np.linalg.solve(AA, -np.concatenate([bx, [bz]]))
                dx = sol[:n]
                dz = float(sol[n])
                dlam = (GG @ dx) * diaglamyiinv - dz * (a * diaglamyiinv) + dellamyi * diaglamyiinv
            dy = -dely * diagyinv + dlam * diagyinv
            dxsi = -xsi + epsi / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsi / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsi / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsi / lam - (s * dlam) / lam

            # largest step (times 1/1.01) keeping every slack and x strictly inside its bounds
            xx = np.concatenate([y, [z], lam, xsi, eta, mu, [zet], s])
            dxx = np.concatenate([dy, [dz], dlam, dxsi, deta, dmu, [dzet], ds])
            stm = max(
                float(np.max(-1.01 * dxx / xx)),
                float(np.max(-1.01 * dx / (x - alfa))),
                float(np.max(1.01 * dx / (beta - x))),
                1.0,
            )
            steg = 1.0 / stm
            old = (x, y, z, lam, xsi, eta, mu, zet, s)
            itto = 0
            resinew = 2 * resnorm
            while resinew > resnorm and itto < 50:
                itto += 1
                x = old[0] + steg * dx
                y = old[1] + steg * dy
                z = old[2] + steg * dz
                lam = old[3] + steg * dlam
                xsi = old[4] + steg * dxsi
                eta = old[5] + steg * deta
                mu = old[6] + steg * dmu
                zet = old[7] + steg * dzet
                s = old[8] + steg * ds
                resinew, resmax = residual(x, y, z, lam, xsi, eta, mu, zet, s, epsi)
                steg /= 2
            resnorm = resinew
        epsi *= 0.1
    return x, y, z, lam
