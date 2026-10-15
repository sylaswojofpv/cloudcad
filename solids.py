"""
solids.py -- best-fit primitives (box, cylinder, cone, sphere) from picked surface points,
plus mesh builders. Output is plain polygon data (V, F) with outward-facing windings, so the
normal Extrude / Move / Subdivide tools work on every vertex and face of the result.
numpy + scipy only.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from mesh_core import _newell

SHAPES = ["box", "cylinder", "cone", "sphere", "plane", "loft", "extrude"]
SHAPE_NAMES = {"box": "Box", "cylinder": "Cylinder", "cone": "Cone", "sphere": "Sphere",
               "plane": "Plane", "loft": "Loft", "extrude": "Extrusion"}
MIN_POINTS = {"box": 3, "cylinder": 5, "cone": 6, "sphere": 4, "plane": 3, "loft": 8, "extrude": 6}
SWEEP_SHAPES = ("loft", "extrude")       # these two have a linear / fitted option
SWEEP_MODES = ("linear", "fitted")


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


def fit_plane(P):
    """Least-squares plane through P. Returns (centre, R, size, rms): R's columns are the
    in-plane axes (u, v) and the normal; size = (width, height) of the tightest rectangle."""
    c = P.mean(0)
    X = P - c
    _, sv, vt = np.linalg.svd(X, full_matrices=False)
    if len(sv) < 2 or sv[1] < 1e-9 * max(sv[0], 1e-30):
        raise ValueError("points are in a straight line - pick points spread over a surface")
    n = np.cross(vt[0], vt[1])
    u, v = vt[0], vt[1]
    rms = float(np.sqrt(np.mean((X @ n) ** 2)))
    Q = np.c_[X @ u, X @ v]
    best = None                                   # smallest-area rectangle (rotating search)
    for t in np.radians(np.arange(0, 90.0, 0.5)):
        ct, st = np.cos(t), np.sin(t)
        a, b = Q @ [ct, st], Q @ [-st, ct]
        area = np.ptp(a) * np.ptp(b)
        if best is None or area < best[0]:
            best = (area, t, a, b)
    _, t, a, b = best
    ct, st = np.cos(t), np.sin(t)
    ur, vr = u * ct + v * st, -u * st + v * ct
    centre = c + ur * (a.min() + a.max()) / 2 + vr * (b.min() + b.max()) / 2
    size = np.array([np.ptp(a), np.ptp(b)]) * 1.1          # 10% margin so picks sit inside
    return centre, np.c_[ur, vr, np.cross(ur, vr)], size, rms


# ------------------------------------------------------------------ builders
def build_plane(center, R, size):
    h = size / 2.0
    V = np.array([center + R[:, 0] * sx * h[0] + R[:, 1] * sy * h[1]
                  for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    return V, [[0, 1, 2, 3]]
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
def fit_solid(shape, P, seg=24, mode="linear", lseg=0):
    """Best-fit `shape` to points P (N,3). Returns (V, F, info string).
    mode / lseg (segments along the length, 0 = automatic) only matter for loft and extrude."""
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
    elif shape == "plane":
        c, R, size, rms = fit_plane(P)
        V, F = build_plane(c, R, size)
        info = "%.3g x %.3g RMS=%.2g" % (size[0], size[1], rms)
    elif shape == "box":
        c, R, dims = fit_box(P)
        V, F = build_box(c, R, dims)
        info = "%.3g x %.3g x %.3g" % tuple(dims)
        if dims.min() < 0.15 * dims.max():
            info += " (thin - pick a second side too, or extrude)"
    elif shape == "extrude":
        V, F, info = fit_extrusion(P, mode, lseg, seg)
    elif shape == "loft":
        V, F, info = fit_loft(P, mode, lseg, seg)
    else:
        raise ValueError("unknown shape %r" % shape)
    return V, F, info



# =============================================================== extrusion / loft
# Both are fitted to ALL picked points at once.
#   Extrusion: find the direction along which the points collapse onto a thin closed curve
#              (that curve is the 2D profile), then sweep the profile along it.
#   Loft:      find the axis, then fit a smooth radius field r(angle, position along axis);
#              every cross-section of the result is a closed shape around the axis.
# Linear = straight axis, straight blends.  Fitted = the centre line bends and the blends are
# cubic, so the solid follows the points as well as it can.
def _outline_score(Q):
    """How close 2D points are to lying on a thin curve (0 = perfectly). inf if they collapse to a line."""
    sv = np.linalg.svd(Q - Q.mean(0), compute_uv=False)
    if len(Q) < 4 or sv[1] < 0.05 * sv[0]:
        return np.inf
    d, ix = cKDTree(Q).query(Q, k=3)
    B, C = Q[ix[:, 1]], Q[ix[:, 2]]
    cr = np.abs((B[:, 0] - Q[:, 0]) * (C[:, 1] - Q[:, 1]) - (B[:, 1] - Q[:, 1]) * (C[:, 0] - Q[:, 0]))
    longest = np.maximum(np.maximum(d[:, 1], d[:, 2]), np.linalg.norm(B - C, axis=1))
    return float(np.mean(cr / np.maximum(longest, 1e-30)) / (sv[0] / np.sqrt(len(Q)) + 1e-30))


def _proj(a):
    u, w = _basis(a)
    return np.c_[u, w]


def _find_outline_axis(X):
    sub = X if len(X) <= 120 else X[np.random.default_rng(0).choice(len(X), 120, replace=False)]
    axes = _fib_axes(250)
    cands = sorted((_outline_score(sub @ _proj(a)), i) for i, a in enumerate(axes))
    cands = [c for c in cands if np.isfinite(c[0])][:3]
    if not cands:
        raise ValueError("points don't outline a shape - pick points all around it, on at least two sides")
    best = None
    for _, i in cands:
        a, cur, step = axes[i].copy(), _outline_score(X @ _proj(axes[i])), 0.15
        for _ in range(16):
            u, w = _basis(a)
            for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                a2 = _unit(a + step * (np.cos(ang) * u + np.sin(ang) * w))
                sc = _outline_score(X @ _proj(a2))
                if sc < cur:
                    a, cur = a2, sc
                    break
            else:
                step *= 0.5
        if best is None or cur < best[0]:
            best = (cur, a)
    return best[1], best[0]


def _tour(Q):
    """Order 2D points into a closed loop of shortest length (nearest neighbour + 2-opt)."""
    n = len(Q)
    if n <= 3:
        return list(range(n))
    D = np.linalg.norm(Q[:, None] - Q[None], axis=2)
    tour, left = [int(np.argmin(Q[:, 0]))], set(range(n))
    left.discard(tour[0])
    while left:
        nxt = min(left, key=lambda j: D[tour[-1], j])
        tour.append(nxt)
        left.discard(nxt)
    for _ in range(60):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b, c, d = tour[i], tour[i + 1], tour[j], tour[(j + 1) % n]
                if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


def _resample_closed(R, m):
    Rc = np.vstack([R, R[:1]])
    t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Rc, axis=0), axis=1))]
    ts = np.linspace(0, t[-1], m, endpoint=False)
    return np.c_[np.interp(ts, t, Rc[:, 0]), np.interp(ts, t, Rc[:, 1])]


def _profile(Q, maxn):
    """Closed polygon through the 2D points (near-duplicates merged); more than maxn points are
    resampled evenly along it, which also smooths picking noise."""
    R = Q[_tour(Q)]
    tol = 1e-3 * float(np.ptp(R, axis=0).max())
    keep = [0]
    for i in range(1, len(R)):
        if np.linalg.norm(R[i] - R[keep[-1]]) > tol:
            keep.append(i)
    if len(keep) > 1 and np.linalg.norm(R[keep[-1]] - R[keep[0]]) <= tol:
        keep.pop()
    R = R[keep]
    if len(R) < 3:
        raise ValueError("points collapse onto a line - pick points all around the shape")
    return _resample_closed(R, maxn) if len(R) > maxn else R


def _dist_to_polygon(Q, R):
    A, B = R, np.roll(R, -1, axis=0)
    AB = B - A
    t = np.clip(((Q[:, None] - A[None]) * AB[None]).sum(2) / ((AB ** 2).sum(1)[None] + 1e-30), 0, 1)
    return np.linalg.norm(Q[:, None] - (A[None] + t[..., None] * AB[None]), axis=2).min(1)


def _close_sweep(rings, caps=True):
    """Quad strips between consecutive rings plus end caps, wound outward."""
    m, n = len(rings), len(rings[0])
    V = np.vstack(rings)
    F = [[r * n + k, r * n + (k + 1) % n, (r + 1) * n + (k + 1) % n, (r + 1) * n + k]
         for r in range(m - 1) for k in range(n)]
    if caps:
        F += [list(range(n))[::-1], [(m - 1) * n + k for k in range(n)]]
    vol = sum(float(V[f[0]] @ np.cross(V[f[i]], V[f[i + 1]])) for f in F for i in range(1, len(f) - 1))
    return V, ([f[::-1] for f in F] if vol < 0 else F)


def _poly_off(tn, coef):
    """In-plane offset of the centre line at tn in [0,1]; coef rows = powers 1..deg, cols = (u, w)."""
    return np.vander(tn, len(coef) + 1, increasing=True)[:, 1:] @ coef


def _bend_fit(Q, tn, size, n):
    """Polynomial in-plane offset of the centre line that makes the projected outline thinnest.
    Starts from the slice centroids, then refines. Returns (coef or None, score)."""
    deg = int(np.clip(n // 10, 1, 3))
    f = lambda x: min(_outline_score(Q - _poly_off(tn, x.reshape(deg, 2))), 1e3)
    zero = np.zeros(deg * 2)
    s0 = f(zero)
    starts = [zero]
    groups = [g for g in np.array_split(np.argsort(tn), int(np.clip(n // 8, 3, 8))) if len(g) >= 3]
    if len(groups) > deg:
        pc = np.polyfit([tn[g].mean() for g in groups], np.array([Q[g].mean(0) for g in groups]), deg)
        starts.append(pc[::-1][1:].ravel() if pc.ndim == 2 else pc[::-1][1:])
    best = (s0, None)
    for x0 in starts:
        simplex = np.vstack([x0] + [x0 + 0.1 * size * np.eye(deg * 2)[i] for i in range(deg * 2)])
        res = minimize(f, x0, method="Nelder-Mead",
                       options=dict(initial_simplex=simplex, maxiter=300, xatol=1e-4 * size, fatol=1e-7))
        if res.fun < best[0]:
            best = (float(res.fun), res.x.reshape(deg, 2))
    return (best[1], best[0]) if best[0] < 0.6 * s0 else (None, s0)


def fit_extrusion(P, mode="linear", lseg=0, seg=24):
    n = len(P)
    c = P.mean(0)
    X = P - c
    a, score = _find_outline_axis(X)
    cands = [a]
    if mode == "fitted" and n >= 12:                    # a bent shape's best straight axis is its long direction
        cands.append(np.linalg.svd(X, full_matrices=False)[2][0])
    cands = [(x if x @ cands[0] >= 0 else -x) for x in cands]
    best = None
    for a in cands:
        u, w = _basis(a)
        Q, t = np.c_[X @ u, X @ w], X @ a
        t0, t1 = float(t.min()), float(t.max())
        L = t1 - t0
        size = float(np.sqrt((Q.var(0)).sum())) + 1e-30
        tn = (t - t0) / max(L, 1e-30)
        if mode == "fitted" and n >= 12 and L > 1e-9:   # let the centre line bend to keep the outline thin
            coef, sc = _bend_fit(Q, tn, size, n)
        else:
            coef, sc = None, _outline_score(Q)
        if best is None or sc < best[0]:
            best = (sc, a, u, w, Q, t, t0, t1, L, tn, coef)
    _, a, u, w, Q, t, t0, t1, L, tn, coef = best
    Qf = Q - _poly_off(tn, coef) if coef is not None else Q
    prof = _profile(Qf, max(seg, 8))
    rms = float(np.sqrt(np.mean(_dist_to_polygon(Qf, prof) ** 2)))
    note = ""
    if L < 0.05 * float(np.ptp(prof, axis=0).max()):    # all picks on one rim -> give it some body
        mid = (t0 + t1) / 2
        L = 0.5 * float(np.ptp(prof, axis=0).max())
        t0, note = mid - L / 2, " (short - pick points further along its length)"
    M = int(lseg) or (12 if coef is not None else 1)          # segments along the length (0 = automatic)
    tq = np.linspace(0, 1, M + 1)
    offs = _poly_off(tq, coef) if coef is not None else np.zeros((M + 1, 2))
    rings = [c + a * (t0 + q * L) + u * (o[0] + prof[:, 0, None]) + w * (o[1] + prof[:, 1, None])
             for q, o in zip(tq, offs)]
    rings = [r.reshape(len(prof), 3) for r in rings]
    V, F = _close_sweep(rings)
    return V, F, "profile %d pts x %d along, length %.3g, RMS=%.2g%s%s" % (
        len(prof), M, L, rms, ", curved" if coef is not None else "", note)


# -------------------------------------------------------------------- loft
def _grid_matrix(theta, sj, N, Ns):
    """Rows interpolate a radius grid R[Ns, N] (periodic in angle, linear in station) at the points."""
    n = len(theta)
    th = (theta % (2 * np.pi)) / (2 * np.pi) * N
    k0 = np.floor(th).astype(int)
    fk, k0, k1 = th - k0, k0 % N, (k0 + 1) % N
    sj = np.clip(sj, 0, Ns - 1 - 1e-9)
    j0 = np.floor(sj).astype(int)
    fj = sj - j0
    A = np.zeros((n, Ns * N))
    rows = np.arange(n)
    for jj, wj in ((j0, 1 - fj), (j0 + 1, fj)):
        for kk, wk in ((k0, 1 - fk), (k1, fk)):
            np.add.at(A, (rows, jj * N + kk), wj * wk)
    return A


def _grid_fit(theta, sj, r, N, Ns):
    A = _grid_matrix(theta, sj, N, Ns)
    rm = float(np.mean(r))
    reg, rhs = [], []
    idx = lambda j, k: j * N + k % N
    def row(items, wgt, target=0.0):
        v = np.zeros(Ns * N)
        for (j, k), c in items:
            v[idx(j, k)] += c * wgt
        reg.append(v); rhs.append(target)
    for j in range(Ns):
        for k in range(N):
            row([((j, k - 1), 1), ((j, k), -2), ((j, k + 1), 1)], 0.25)
            row([((j, k), 1)], 1e-3, 1e-3 * rm)
            if j >= 1 and Ns > 2 and j < Ns - 1:
                row([((j - 1, k), 1), ((j, k), -2), ((j + 1, k), 1)], 0.25)
            if j < Ns - 1:
                row([((j, k), -1), ((j + 1, k), 1)], 0.05)
    R = np.linalg.lstsq(np.vstack([A, reg]), np.r_[r, rhs], rcond=None)[0]
    return R.reshape(Ns, N), A @ R - r


def _kasa_center(Q):
    x = np.linalg.lstsq(np.c_[2 * Q, np.ones(len(Q))], (Q ** 2).sum(1), rcond=None)[0]
    return x[:2]


def fit_loft(P, mode="linear", lseg=0, seg=24):
    n = len(P)
    base, a, _, _, _ = fit_cylinder(P)
    u, w = _basis(a)
    X = P - base
    s, Q = X @ a, np.c_[X @ u, X @ w]
    s0, s1 = float(s.min()), float(s.max())
    L = s1 - s0
    if L < 1e-9 * (np.ptp(P, axis=0).max() + 1e-30):
        raise ValueError("all points are at the same height along the axis - spread them along the shape")
    sn = (s - s0) / L
    scale = float(np.ptp(P, axis=0).max())
    if float(np.median(np.linalg.norm(Q, axis=1))) < 1e-4 * scale:
        raise ValueError("points lie on a line - pick points around the surface of the shape")
    coef = None
    if mode == "fitted" and n >= 15:               # bend the centre line through the section centres
        ts, cs = [], []
        for g in np.array_split(np.argsort(s), int(np.clip(n // 8, 3, 8))):
            if len(g) < 5:
                continue
            cen = _kasa_center(Q[g])
            th = np.sort(np.arctan2(Q[g, 1] - cen[1], Q[g, 0] - cen[0]))
            if np.max(np.diff(np.r_[th, th[0] + 2 * np.pi])) > np.radians(200):
                continue                            # too little of this section was picked to trust
            ts.append(sn[g].mean()); cs.append(cen)
        if len(ts) >= 3:
            coef = np.polyfit(ts, np.array(cs), min(3, len(ts) - 1))
            path = np.c_[np.polyval(coef[:, 0], np.linspace(0, 1, 20)), np.polyval(coef[:, 1], np.linspace(0, 1, 20))]
            if np.ptp(path, axis=0).max() < 0.02 * scale:      # essentially straight: keep it straight
                coef = None
    off = (lambda t: np.c_[np.polyval(coef[:, 0], t), np.polyval(coef[:, 1], t)]) if coef is not None \
        else (lambda t: np.zeros((len(np.atleast_1d(t)), 2)))
    Qc = Q - off(sn)
    N = int(np.clip(seg, 8, 48))
    Ns = int(np.clip(round(n / 6), 2, 7))
    R, resid = _grid_fit(np.arctan2(Qc[:, 1], Qc[:, 0]), sn * (Ns - 1), np.linalg.norm(Qc, axis=1), N, Ns)
    if mode == "fitted" and Ns >= 3:
        tq = np.linspace(0, 1, (int(lseg) or 12) + 1)
        Rout = CubicSpline(np.arange(Ns), R, axis=0)(tq * (Ns - 1))
    elif lseg:                                      # linear, but with the requested number of segments
        tq = np.linspace(0, 1, int(lseg) + 1)
        Rout = np.array([np.interp(tq, np.linspace(0, 1, Ns), R[:, k]) for k in range(N)]).T
    else:
        tq, Rout = np.linspace(0, 1, Ns), R
    Rout = np.maximum(Rout, 1e-3 * scale)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    dirs = np.outer(np.cos(ang), u) + np.outer(np.sin(ang), w)
    rings = []
    for q, rr in zip(tq, Rout):
        o = off(q)[0]
        rings.append(base + a * (s0 + q * L) + u * o[0] + w * o[1] + rr[:, None] * dirs)
    V, F = _close_sweep(rings)
    return V, F, "%d segments along x %d around, RMS=%.2g%s" % (
        len(rings) - 1, N, float(np.sqrt(np.mean(resid ** 2))), ", curved" if coef is not None else "")
