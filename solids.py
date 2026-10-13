"""
solids.py -- best-fit primitives (box, cylinder, cone, sphere) from picked surface points,
plus mesh builders. Output is plain polygon data (V, F) with outward-facing windings, so the
normal Extrude / Move / Subdivide tools work on every vertex and face of the result.
numpy + scipy only.
"""
import numpy as np
from scipy.optimize import least_squares

from mesh_core import _newell

SHAPES = ["box", "cylinder", "cone", "sphere"]
SHAPE_NAMES = {"box": "Box", "cylinder": "Cylinder", "cone": "Cone", "sphere": "Sphere"}
MIN_POINTS = {"box": 3, "cylinder": 5, "cone": 6, "sphere": 4}


# ------------------------------------------------------------------ helpers
def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-15 else v


def _basis(a):
    """Two unit vectors perpendicular to a."""
    a = _unit(a)
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = _unit(np.cross(a, t))
    return u, np.cross(a, u)


def _fib_axes(n=120):
    """Roughly uniform unit axes over a hemisphere (axis sign does not matter)."""
    i = np.arange(n) + 0.5
    z = i / n
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z)
    return np.c_[r * np.cos(phi), r * np.sin(phi), z]


def _circle2d(Q):
    """Algebraic (Kasa) circle fit -> centre, radius."""
    x = np.linalg.lstsq(np.c_[2 * Q, np.ones(len(Q))], (Q ** 2).sum(1), rcond=None)[0]
    return x[:2], np.sqrt(max(x[2] + x[:2] @ x[:2], 1e-18))


def _orient(V, F, interior):
    """Flip any face whose normal points toward `interior` (valid for convex solids)."""
    out = []
    for f in F:
        P = V[f]
        out.append(f[::-1] if _newell(P) @ (P.mean(0) - interior) < 0 else list(f))
    return out


# --------------------------------------------------------------------- fits
def fit_sphere(P):
    c = P.mean(0)
    x = np.linalg.lstsq(np.c_[2 * (P - c), np.ones(len(P))], ((P - c) ** 2).sum(1), rcond=None)[0]
    c0 = x[:3]
    r0 = np.sqrt(max(x[3] + c0 @ c0, 1e-18))
    res = lambda q: np.linalg.norm(P - c - q[:3], axis=1) - q[3]
    sol = least_squares(res, np.r_[c0, r0])
    return c + sol.x[:3], abs(sol.x[3]), float(np.sqrt(np.mean(sol.fun ** 2)))


def fit_cylinder(P):
    cen = P.mean(0)
    X = P - cen
    scored = []
    for a0 in _fib_axes():
        u, w = _basis(a0)
        Q = np.c_[X @ u, X @ w]
        c, r = _circle2d(Q)
        scored.append((np.sqrt(np.mean((np.linalg.norm(Q - c, axis=1) - r) ** 2)), a0, u, w, c, r))
    scored.sort(key=lambda s: s[0])
    best = None
    for _, a0, u, w, c, r in scored[:4]:
        def res(q, a0=a0, u=u, w=w):
            a = _unit(a0 + q[0] * u + q[1] * w)
            d = X - (q[2] * u + q[3] * w)
            return np.linalg.norm(d - np.outer(d @ a, a), axis=1) - q[4]
        sol = least_squares(res, [0, 0, c[0], c[1], r])
        rms = float(np.sqrt(np.mean(sol.fun ** 2)))
        if best is None or rms < best[0]:
            best = (rms, a0, u, w, sol.x)
    rms, a0, u, w, q = best
    a = _unit(a0 + q[0] * u + q[1] * w)
    c0 = cen + q[2] * u + q[3] * w
    r = abs(q[4])
    t = (P - c0) @ a
    t0, t1 = float(t.min()), float(t.max())
    h = t1 - t0
    if h < 0.1 * r:                          # rim-only picks -> give it some body
        h = r
    mid = (t0 + t1) / 2
    return c0 + a * (mid - h / 2), a, r, h, rms


def fit_cone(P):
    cen = P.mean(0)
    X = P - cen
    scale = float(np.ptp(P, axis=0).max()) + 1e-12
    scored = []
    for a0 in _fib_axes():
        u, w = _basis(a0)
        Q = np.c_[X @ u, X @ w]
        c, _ = _circle2d(Q)
        perp = np.linalg.norm(Q - c, axis=1)
        A = np.c_[X @ a0, np.ones(len(X))]
        (m, b), *_ = np.linalg.lstsq(A, perp, rcond=None)
        scored.append((np.sqrt(np.mean((A @ [m, b] - perp) ** 2)), a0, u, w, c, m, b))
    scored.sort(key=lambda s: s[0])
    best = None
    for _, a0, u, w, c, m, b in scored[:4]:
        m = m if abs(m) > 1e-4 else 1e-4
        apex0 = c[0] * u + c[1] * w + (-b / m) * a0
        def res(q, a0=a0, u=u, w=w):
            a = _unit(a0 + q[0] * u + q[1] * w)
            d = X - q[2:5]
            al = d @ a
            return np.linalg.norm(d - np.outer(al, a), axis=1) * np.cos(q[5]) - al * np.sin(q[5])
        sol = least_squares(res, np.r_[0, 0, apex0, np.arctan(m)])
        rms = float(np.sqrt(np.mean(sol.fun ** 2)))
        if best is None or rms < best[0]:
            best = (rms, a0, u, w, sol.x)
    rms, a0, u, w, q = best
    a = _unit(a0 + q[0] * u + q[1] * w)
    apex, alpha = cen + q[2:5], q[5]
    al = (P - apex) @ a
    if np.median(al) < 0:                    # make the axis point from apex into the cone
        a, alpha, al = -a, -alpha, -al
    beta = float(np.clip(abs(alpha), np.radians(1), np.radians(85)))
    h = float(al.max())
    if h <= 1e-9 or h > 50 * scale:
        raise ValueError("points look cylindrical or flat - try Cylinder / Box")
    return apex, a, beta, h, rms


def fit_box(P):
    cen = P.mean(0)
    X = P - cen
    _, v = np.linalg.eigh(np.cov(X.T) + 1e-18 * np.eye(3))
    R_pca = v[:, ::-1].copy()
    if np.linalg.det(R_pca) < 0:
        R_pca[:, 2] *= -1
    best_yaw = min(np.radians(np.arange(0, 90.0)),
                   key=lambda t: np.ptp(X[:, 0] * np.cos(t) + X[:, 1] * np.sin(t)) *
                                 np.ptp(-X[:, 0] * np.sin(t) + X[:, 1] * np.cos(t)))
    c, s = np.cos(best_yaw), np.sin(best_yaw)
    R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])   # upright box, best yaw
    best = None
    for R in (R_pca, R_z):
        Q = X @ R
        lo, hi = Q.min(0), Q.max(0)
        d = hi - lo
        vol = np.prod(np.maximum(d, 0.05 * max(d.max(), 1e-12)))
        if best is None or vol < best[0]:
            best = (vol, R, lo, hi)
    _, R, lo, hi = best
    dims = hi - lo
    mx = max(float(dims.max()), 1e-9)
    dims = np.where(dims < 0.05 * mx, 0.25 * mx, dims)      # flat picks -> give it thickness
    return cen + R @ ((lo + hi) / 2), R, dims


# ------------------------------------------------------------------ builders
def build_box(center, R, dims):
    h = dims / 2.0
    V = np.array([center + R @ (h * [sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    F = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return V, _orient(V, F, center)


def build_cylinder(base, a, r, h, seg):
    u, w = _basis(a)
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = r * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), w))
    V = np.vstack([base + ring, base + a * h + ring])
    F = [[k, (k + 1) % seg, seg + (k + 1) % seg, seg + k] for k in range(seg)]
    F += [list(range(seg)), list(range(seg, 2 * seg))]
    return V, _orient(V, F, base + a * h / 2)


def build_cone(apex, a, beta, h, seg):
    u, w = _basis(a)
    ang = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = apex + a * h + h * np.tan(beta) * (np.outer(np.cos(ang), u) + np.outer(np.sin(ang), w))
    V = np.vstack([apex, ring])
    F = [[0, 1 + k, 1 + (k + 1) % seg] for k in range(seg)] + [list(range(1, seg + 1))]
    return V, _orient(V, F, apex + a * 0.75 * h)


def build_sphere(c, r, seg):
    """Quad sphere (cube projected onto a sphere): no poles, easy to edit."""
    n = max(2, seg // 4)
    ids, V, F = {}, [], []

    def vid(p):
        k = tuple(np.round(p, 7))
        if k not in ids:
            ids[k] = len(V)
            V.append(p)
        return ids[k]

    g = lambda i: np.tan(np.pi / 4 * (-1 + 2 * i / n))
    for ax in range(3):
        for sg in (-1, 1):
            ua, va = (ax + 1) % 3, (ax + 2) % 3
            for i in range(n):
                for j in range(n):
                    quad = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        p = np.zeros(3)
                        p[ax], p[ua], p[va] = sg, g(i + di), g(j + dj)
                        quad.append(vid(p / np.linalg.norm(p)))
                    F.append(quad)
    V = c + r * np.array(V)
    return V, _orient(V, F, c)


# ---------------------------------------------------------------------- API
def fit_solid(shape, P, seg=24):
    """Best-fit `shape` to points P (N,3). Returns (V, F, info string)."""
    P = np.asarray(P, float)
    if shape == "sphere":
        c, r, rms = fit_sphere(P)
        V, F = build_sphere(c, r, seg)
        info = "r=%.3g RMS=%.2g" % (r, rms)
        if r > 20 * (np.ptp(P, axis=0).max() + 1e-12):
            info += " (poorly constrained - pick points spread around it)"
    elif shape == "cylinder":
        base, a, r, h, rms = fit_cylinder(P)
        V, F = build_cylinder(base, a, r, h, seg)
        info = "r=%.3g h=%.3g RMS=%.2g" % (r, h, rms)
    elif shape == "cone":
        apex, a, beta, h, rms = fit_cone(P)
        V, F = build_cone(apex, a, beta, h, seg)
        info = "half-angle %.1f deg, h=%.3g RMS=%.2g" % (np.degrees(beta), h, rms)
    elif shape == "box":
        c, R, dims = fit_box(P)
        V, F = build_box(c, R, dims)
        info = "%.3g x %.3g x %.3g" % tuple(dims)
        if dims.min() < 0.15 * dims.max():
            info += " (thin - pick a second side too, or extrude)"
    else:
        raise ValueError("unknown shape %r" % shape)
    return V, F, info
