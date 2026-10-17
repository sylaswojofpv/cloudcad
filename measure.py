"""
measure.py -- measurements on points: distance, path length, angle, circle / sphere / cylinder diameter,
angle between faces, and size info for a selection of mesh faces.  numpy only (solids.py for round fits).

A measurement is stored as (kind, pts).  build(kind, pts) turns that into the numbers to show and the lines to
draw, so measurements can be rebuilt after the scene origin moves or a model is loaded.

  build(kind, pts)         -> dict(title, items=[(name, value, type)], anchor, prims=[("line"|"loop", points)],
                                   points=[(name, position)])   # e.g. the center of a circle: part of the result
                              type is "len", "area", "vol", "deg" or "int"; lengths are in cloud units
  from_clicks(kind, pts)   -> (kind, pts, note): turns clicked points into a storable measurement (fits circle / sphere)
  fit_round(V)             -> candidate sphere / cylinder fits to the points of a shape
  selection_info(...)      -> size, area and volume of selected mesh faces
  ray_segment_closest      -> point on a segment closest to a view ray (snapping to edges)
"""
import numpy as np

KINDS = ("distance", "path", "angle", "circle", "sphere", "cylinder", "edge", "faceangle")
TITLES = {"distance": "Distance", "path": "Path", "angle": "Angle", "circle": "Circle", "sphere": "Sphere",
          "cylinder": "Cylinder", "edge": "Edge", "faceangle": "Face angle"}


# ------------------------------------------------------------------ helpers
def _unit(v):
    l = np.linalg.norm(v)
    return v / l if l > 1e-30 else v * 0.0


def basis(n):
    """Two unit vectors perpendicular to n."""
    n = _unit(np.asarray(n, float))
    t = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = _unit(np.cross(n, t))
    return u, np.cross(n, u)


def ring(c, n, r, m=64):
    """Points of a circle of radius r around centre c, in the plane with normal n."""
    u, w = basis(n)
    a = np.linspace(0, 2 * np.pi, m, endpoint=False)
    return c + r * (np.outer(np.cos(a), u) + np.outer(np.sin(a), w))


def ray_segment_closest(o, d, a, b):
    """Point on segment a-b closest to the ray o + s*d (d unit)."""
    v = b - a
    w0 = o - a
    bb, cc, dd, ee = float(d @ v), float(v @ v), float(d @ w0), float(v @ w0)
    den = cc - bb * bb
    t = 0.0 if den < 1e-12 * max(cc, 1e-30) else (ee - bb * dd) / den
    return a + float(np.clip(t, 0.0, 1.0)) * v


# --------------------------------------------------------------------- fits
def circle_fit(P):
    """Circle through 3+ points in space (least squares). -> centre, normal, radius, rms error."""
    P = np.asarray(P, float)
    if len(P) < 3:
        raise ValueError("A circle needs at least 3 points")
    c0 = P.mean(0)
    X = P - c0
    _, sv, vt = np.linalg.svd(X, full_matrices=False)
    if len(sv) < 2 or sv[1] < 1e-9 * max(sv[0], 1e-30):
        raise ValueError("The points are on a straight line - pick points around the circle")
    u, w = vt[0], vt[1]
    n = np.cross(u, w)
    Q = np.c_[X @ u, X @ w]
    sol = np.linalg.lstsq(np.c_[2 * Q, np.ones(len(Q))], (Q ** 2).sum(1), rcond=None)[0]
    cq, r = sol[:2].copy(), float(np.sqrt(max(sol[2] + sol[:2] @ sol[:2], 0.0)))
    for _ in range(12):                                    # polish: minimise the true radial error
        d = Q - cq
        rr = np.maximum(np.linalg.norm(d, axis=1), 1e-30)
        J = np.c_[-d / rr[:, None], -np.ones(len(Q))]
        step = np.linalg.lstsq(J, -(rr - r), rcond=None)[0]
        cq, r = cq + step[:2], r + float(step[2])
        if np.linalg.norm(step) < 1e-12 * max(abs(r), 1e-30):
            break
    r = abs(r)
    if not np.isfinite(r) or r < 1e-12:
        raise ValueError("Could not fit a circle to these points")
    rms = float(np.sqrt(np.mean((np.linalg.norm(Q - cq, axis=1) - r) ** 2)))
    return c0 + cq[0] * u + cq[1] * w, n, r, rms


def sphere_fit(P):
    """Sphere through 4+ points (least squares). -> centre, radius, rms error."""
    P = np.asarray(P, float)
    if len(P) < 4:
        raise ValueError("A sphere needs at least 4 points")
    c0 = P.mean(0)
    X = P - c0
    sv = np.linalg.svd(X, compute_uv=False)
    if sv[2] < 1e-6 * max(sv[0], 1e-30):
        raise ValueError("The points are in one plane - pick points all around the sphere")
    sol = np.linalg.lstsq(np.c_[2 * X, np.ones(len(X))], (X ** 2).sum(1), rcond=None)[0]
    c, r = sol[:3].copy(), float(np.sqrt(max(sol[3] + sol[:3] @ sol[:3], 0.0)))
    for _ in range(12):
        d = X - c
        rr = np.maximum(np.linalg.norm(d, axis=1), 1e-30)
        J = np.c_[-d / rr[:, None], -np.ones(len(X))]
        step = np.linalg.lstsq(J, -(rr - r), rcond=None)[0]
        c, r = c + step[:3], r + float(step[3])
        if np.linalg.norm(step) < 1e-12 * max(abs(r), 1e-30):
            break
    r = abs(r)
    if not np.isfinite(r) or r < 1e-12:
        raise ValueError("Could not fit a sphere to these points")
    rms = float(np.sqrt(np.mean((np.linalg.norm(X - c, axis=1) - r) ** 2)))
    return c0 + c, r, rms


def fit_round(V, max_pts=3000):
    """Sphere and cylinder fits to the points of a shape, best (lowest relative error) first.
    Each: dict(kind, pts (storable measurement points), r, rms, rel = rms / r). Fits whose radius is absurdly
    large (a flat set of points fits a huge sphere) are dropped."""
    from solids import fit_sphere, fit_cylinder
    V = np.unique(np.round(np.asarray(V, float), 12), axis=0)
    if len(V) < 6:
        raise ValueError("Select a whole shape (6 or more points) to fit its diameter")
    if len(V) > max_pts:
        V = V[np.random.default_rng(0).choice(len(V), max_pts, replace=False)]
    ext = float(np.ptp(V, axis=0).max())
    out = []
    try:
        c, r, rms = fit_sphere(V)
        if np.isfinite(r) and 0 < r <= 10 * ext:
            out.append(dict(kind="sphere", pts=np.array([c, c + [r, 0.0, 0.0]]), r=float(r), rms=float(rms), rel=float(rms / r)))
    except Exception:
        pass
    try:
        base, a, r, h, rms = fit_cylinder(V)
        if np.isfinite(r) and 0 < r <= 10 * ext and np.isfinite(h) and h > 0:
            u, _ = basis(a)
            out.append(dict(kind="cylinder", pts=np.array([base, base + a * h, base + r * u]),
                            r=float(r), rms=float(rms), rel=float(rms / r)))
    except Exception:
        pass
    out.sort(key=lambda d: d["rel"])
    return out


# ------------------------------------------------------------------- builders
def _distance(P, title="Distance"):
    if len(P) != 2:
        raise ValueError("A distance needs 2 points")
    a, b = P
    d = b - a
    return dict(title=title,
                items=[("Length", float(np.linalg.norm(d)), "len"), ("ΔX", float(d[0]), "len"), ("ΔY", float(d[1]), "len"),
                       ("ΔZ", float(d[2]), "len"), ("Horizontal (XY)", float(np.hypot(d[0], d[1])), "len")],
                anchor=(a + b) / 2, prims=[("line", P)])


def _edge(P):
    return _distance(P, "Edge")


def _path(P):
    if len(P) < 2:
        raise ValueError("A path needs at least 2 points")
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.r_[0, np.cumsum(seg)]
    total = float(cum[-1])
    if total < 1e-30:
        raise ValueError("The points are all in one place")
    k = int(np.searchsorted(cum, total / 2))
    k = min(max(k, 1), len(P) - 1)
    f = (total / 2 - cum[k - 1]) / max(seg[k - 1], 1e-30)
    anchor = P[k - 1] + f * (P[k] - P[k - 1])
    return dict(title="Path", items=[("Length", total, "len"), ("Segments", len(P) - 1, "int"),
                                     ("Straight line, first to last", float(np.linalg.norm(P[-1] - P[0])), "len")],
                anchor=anchor, prims=[("line", P)])


def _angle(P):
    if len(P) != 3:
        raise ValueError("An angle needs 3 points")
    a, b, c = P
    v1, v2 = a - b, c - b
    l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if l1 < 1e-12 or l2 < 1e-12:
        raise ValueError("Two of the points are in the same place")
    u1, u2 = v1 / l1, v2 / l2
    th = float(np.arccos(np.clip(u1 @ u2, -1.0, 1.0)))
    r = 0.3 * min(l1, l2)
    t = np.linspace(0, 1, 24)[:, None]
    if th < 1e-6:
        dirs = np.repeat(u1[None], len(t), axis=0)
    elif np.pi - th < 1e-4:                                # straight line: swing through a perpendicular
        w = basis(u1)[0]
        dirs = np.cos(t * th) * u1 + np.sin(t * th) * w
    else:
        dirs = (np.sin((1 - t) * th) * u1 + np.sin(t * th) * u2) / np.sin(th)
    arc = b + r * dirs
    anchor = b + 1.3 * r * _unit(dirs[len(dirs) // 2])
    deg = float(np.degrees(th))
    return dict(title="Angle", items=[("Angle", deg, "deg"), ("Other side", 360.0 - deg, "deg"),
                                      ("Arm 1", l1, "len"), ("Arm 2", l2, "len")],
                anchor=anchor, prims=[("line", np.array([a, b, c])), ("line", arc)])


def _circle(P):
    c, n, r, rms = circle_fit(P)
    d0 = P[0] - c
    d0 = d0 - (d0 @ n) * n
    d0 = _unit(d0) if np.linalg.norm(d0) > 1e-12 * r else basis(n)[0]
    return dict(title="Circle", items=[("Diameter", 2 * r, "len"), ("Radius", r, "len"),
                                       ("Circumference", 2 * np.pi * r, "len"), ("Fit error", rms, "len")],
                anchor=c, prims=[("loop", ring(c, n, r)), ("line", np.array([c - d0 * r, c + d0 * r]))],
                points=[("Center", c)])


def _sphere(P):
    if len(P) != 2:
        raise ValueError("A sphere measurement stores a centre and a surface point")
    c = P[0]
    r = float(np.linalg.norm(P[1] - P[0]))
    return dict(title="Sphere", items=[("Diameter", 2 * r, "len"), ("Radius", r, "len"),
                                       ("Surface area", 4 * np.pi * r * r, "area"), ("Volume", 4 / 3 * np.pi * r ** 3, "vol")],
                points=[("Center", c)],
                anchor=c, prims=[("loop", ring(c, [1, 0, 0], r)), ("loop", ring(c, [0, 1, 0], r)), ("loop", ring(c, [0, 0, 1], r)),
                                 ("line", np.array([c - [r, 0, 0], c + [r, 0, 0]]))])


def _cylinder(P):
    if len(P) != 3:
        raise ValueError("A cylinder measurement stores two axis points and a surface offset")
    c0, c1, p = P
    ax = c1 - c0
    L = float(np.linalg.norm(ax))
    if L < 1e-12:
        raise ValueError("The cylinder has no length")
    a = ax / L
    q = p - c0
    r = float(np.linalg.norm(q - (q @ a) * a))
    r0, r1 = ring(c0, a, r), ring(c1, a, r)
    prims = [("loop", r0), ("loop", r1), ("line", np.array([c0, c1]))]
    prims += [("line", np.array([r0[k], r1[k]])) for k in (0, 16, 32, 48)]
    return dict(title="Cylinder", items=[("Diameter", 2 * r, "len"), ("Radius", r, "len"), ("Length", L, "len"),
                                         ("Volume", np.pi * r * r * L, "vol"), ("Side area", 2 * np.pi * r * L, "area")],
                anchor=(c0 + c1) / 2, prims=prims, points=[("Center, end 1", c0), ("Center, end 2", c1)])


def _faceangle(P):
    if len(P) != 4:
        raise ValueError("A face angle needs two faces")
    p1, p2 = P[0], P[2]
    n1, n2 = _unit(P[1] - P[0]), _unit(P[3] - P[2])
    th = float(np.degrees(np.arccos(np.clip(n1 @ n2, -1.0, 1.0))))
    return dict(title="Face angle",
                items=[("Between planes", min(th, 180.0 - th), "deg"), ("Between normals", th, "deg"),
                       ("Distance between picks", float(np.linalg.norm(p2 - p1)), "len")],
                anchor=(p1 + p2) / 2, prims=[("line", np.array([P[0], P[1]])), ("line", np.array([P[2], P[3]])),
                                             ("line", np.array([p1, p2]))])


_BUILD = {"distance": _distance, "path": _path, "angle": _angle, "circle": _circle, "sphere": _sphere,
          "cylinder": _cylinder, "edge": _edge, "faceangle": _faceangle}


def build(kind, pts):
    """Numbers and drawing for a stored measurement. Raises ValueError with a readable reason."""
    if kind not in _BUILD:
        raise ValueError("unknown measurement %r" % kind)
    P = np.asarray(pts, float).reshape(-1, 3)
    if not np.isfinite(P).all():
        raise ValueError("The points are not valid")
    res = _BUILD[kind](P)
    res.setdefault("points", [])      # points the measurement produces (circle / sphere centers...): other measurements can snap to them
    return res


def from_clicks(kind, pts):
    """Clicked points -> (kind, pts, note) ready to store. Circle and sphere are fitted to all the clicks;
    a sphere is stored as its centre plus one surface point."""
    P = np.asarray(pts, float).reshape(-1, 3)
    if kind == "sphere":
        c, r, rms = sphere_fit(P)
        return "sphere", np.array([c, c + [r, 0.0, 0.0]]), "fitted to %d points, error %.1f%% of radius" % (len(P), 100 * rms / r)
    if kind == "circle":
        c, n, r, rms = circle_fit(P)
        return "circle", P, "fitted to %d points, error %.1f%% of radius" % (len(P), 100 * rms / r)
    return kind, P, ""


# -------------------------------------------------------------- selection info
def selection_info(V, F, faces, verts):
    """Size of a selection. faces: face ids (may be empty), verts: vertex ids.
    -> dict(n_faces, n_verts, size (dx, dy, dz), diagonal, centroid, area, volume (None when not a closed shape))."""
    V = np.asarray(V, float)
    verts = sorted(set(verts) | {v for i in faces for v in F[i]})
    P = V[verts]
    size = np.ptp(P, axis=0)
    area, vol, closed = 0.0, None, False
    if faces:
        cen = P.mean(0)
        directed, signed = {}, 0.0
        for i in faces:
            f = F[i]
            Q = V[f] - cen
            area += 0.5 * float(np.linalg.norm(_newell(Q)))
            for k in range(1, len(f) - 1):
                signed += float(Q[0] @ np.cross(Q[k], Q[k + 1]))
            for k in range(len(f)):
                e = (f[k], f[(k + 1) % len(f)])
                directed[e] = directed.get(e, 0) + 1
        closed = all(c == 1 for c in directed.values()) and all((b, a) in directed for (a, b) in directed)
        if closed:
            vol = abs(signed) / 6.0
    return dict(n_faces=len(faces), n_verts=len(verts), size=size, diagonal=float(np.linalg.norm(size)),
                centroid=P.mean(0), area=area, volume=vol)


def _newell(P):
    Q = np.roll(P, -1, axis=0)
    return np.array([((P[:, 1] - Q[:, 1]) * (P[:, 2] + Q[:, 2])).sum(),
                     ((P[:, 2] - Q[:, 2]) * (P[:, 0] + Q[:, 0])).sum(),
                     ((P[:, 0] - Q[:, 0]) * (P[:, 1] + Q[:, 1])).sum()])
