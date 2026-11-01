"""
mirror.py -- mirror a body through an adjustable plane and fit that plane to a point cloud.  Pure numpy / scipy.

The plane is stored as a base plane (c0, n0) plus three adjustments p = (d, a, b):
    d  slide along n0 (model units)      a  tilt about the plane's first axis (radians)
    b  tilt about its second axis (radians)
    plane_from(c0, n0, p) -> (c, n)
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def unit(v):
    l = np.linalg.norm(v)
    return v / l if l > 1e-30 else v


def basis(n):
    """Two unit vectors lying in the plane with normal n."""
    n = unit(np.asarray(n, float))
    t = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = unit(np.cross(n, t))
    return u, np.cross(n, u)


def rot(k, ang):
    k = unit(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def plane_from(c0, n0, p):
    n0 = unit(np.asarray(n0, float))
    u, w = basis(n0)
    n = unit(rot(w, p[2]) @ (rot(u, p[1]) @ n0))
    return np.asarray(c0, float) + n0 * p[0], n


def reflect(P, c, n):
    """Mirror point(s) P through the plane that contains c and has unit normal n."""
    P = np.asarray(P, float)
    d = (P - c) @ n
    return P - 2.0 * d * n if P.ndim == 1 else P - 2.0 * d[:, None] * n


def side_plane(P, axis_dir, margin=0.0):
    """A plane touching the +axis_dir side of the points P: (c, n) with n = axis_dir."""
    n = unit(np.asarray(axis_dir, float))
    return P[np.argmax(P @ n)] + n * margin if len(P) else np.zeros(3), n


def sample_surface(tri, n=2500, seed=0):
    """Area-weighted random points on triangles tri (T, 3, 3)."""
    if len(tri) == 0:
        return np.zeros((0, 3))
    a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if a.sum() < 1e-30:
        return tri.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    i = rng.choice(len(tri), n, p=a / a.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    t = tri[i]
    S = (1 - s)[:, None] * t[:, 0] + (s * (1 - r2))[:, None] * t[:, 1] + (s * r2)[:, None] * t[:, 2]
    return np.vstack([S, tri.reshape(-1, 3)])                  # plus the corners, so edges / corners are represented


def _trim(d, f):
    k = max(1, int(len(d) * f))
    return float(np.partition(d, k - 1)[:k].mean())


def gap(S, cloud, c, n, side=None, tree=None):
    """Mean distance (trimmed) between the mirrored sample points and the cloud, both ways (0 = perfect)."""
    tree = tree or cKDTree(cloud)
    R = reflect(S, c, n)
    d1 = tree.query(R)[0]
    P = cloud
    if side is not None:
        P = cloud[((cloud - c) @ n) * side < 0]                 # only cloud on the far side takes part
    if len(P) < 5:
        return _trim(d1, 0.8) * 2
    d2 = cKDTree(R).query(P)[0]
    return _trim(d1, 0.8) + _trim(d2, 0.7)


def fit_plane(S, cloud, c0, n0, p0=(0, 0, 0), max_cloud=20000, seed=0):
    """Adjust the mirror plane so the mirrored sample points S sit on the point cloud.
    Starts from the user's rough plane (c0, n0, adjustments p0); returns (p, before, after) where p are the
    new adjustments relative to (c0, n0) and before / after the mean gap (model units)."""
    S, cloud = np.asarray(S, float), np.asarray(cloud, float)
    if len(S) < 4 or len(cloud) < 20:
        raise ValueError("Not enough points - show the point cloud near the mirrored body")
    ext = float(np.linalg.norm(np.ptp(S, axis=0))) or 1.0
    c, n = plane_from(c0, n0, p0)
    side = float(np.sign((S.mean(0) - c) @ n)) or 1.0           # which side the original body is on
    R = reflect(S, c, n)
    lo, hi = R.min(0) - 0.35 * ext, R.max(0) + 0.35 * ext
    P = cloud[(cloud >= lo).all(1) & (cloud <= hi).all(1)]
    if len(P) < 20:
        raise ValueError("No point cloud near the mirrored body - move the plane closer first")
    if len(P) > max_cloud:
        P = P[np.random.default_rng(seed).choice(len(P), max_cloud, replace=False)]
    tree = cKDTree(P)

    def f(x):
        cc, nn = plane_from(c0, n0, (x[0] * ext, x[1], x[2]))
        return gap(S, P, cc, nn, side, tree)

    x0 = np.array([p0[0] / ext, p0[1], p0[2]], float)
    before = f(x0)
    best = (before, x0)
    for step in (0.04, 0.012, 0.004):                            # coarse to fine
        simplex = np.vstack([best[1], best[1] + [step, 0, 0], best[1] + [0, step * 1.5, 0], best[1] + [0, 0, step * 1.5]])
        r = minimize(f, best[1], method="Nelder-Mead",
                     options=dict(initial_simplex=simplex, maxiter=220, xatol=1e-7, fatol=1e-10 * ext))
        if r.fun < best[0]:
            best = (float(r.fun), r.x)
    x = best[1]
    return (float(x[0] * ext), float(x[1]), float(x[2])), before, best[0]
