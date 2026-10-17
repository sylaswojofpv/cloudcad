"""
reconstruct.py -- Poisson surface reconstruction of a lasso-selected set of cloud points.
Pure numpy (+ Open3D for the reconstruction itself:  pip install open3d).

  points_in_polygon   screen-space lasso test
  estimate_normals    normals for the points: estimated from neighbours, made consistent, then oriented
  poisson_mesh        triangle mesh (V, F) with outward / consistent winding; detail and triangle count adjustable
"""
import numpy as np

MAX_INPUT = 200_000          # points fed to the reconstruction (more are thinned evenly)
ORIENTS = ("auto", "outward", "inward")


def _lib():
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError("Mesh from points needs Open3D:  pip install open3d")
    return o3d


def points_in_polygon(xy, poly):
    """Even-odd test of points xy (N, 2) against a polygon (M, 2) (closed implicitly)."""
    xy, poly = np.asarray(xy, float), np.asarray(poly, float)
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi, xj, yj = poly[i, 0], poly[i, 1], poly[j, 0], poly[j, 1]
        cond = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= cond & (x < xint)
        j = i
    return inside


def _orient_consistent(P, N, k=12):
    """Make the normal signs agree across the surface. Neighbouring points (k nearest) are linked, edges between
    similar normals are cheap, and the signs are propagated outward from one point along the cheapest spanning
    tree. Works per connected piece, never fails on flat or degenerate data (no triangulation involved).
    Returns (N with consistent signs, component label of every point)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, connected_components, minimum_spanning_tree
    from scipy.spatial import cKDTree
    n = len(P)
    k = int(min(k, n - 1))
    _, idx = cKDTree(P).query(P, k=k + 1)
    rows, cols = np.repeat(np.arange(n), k), idx[:, 1:].ravel()
    w = 1.0 - np.abs((N[rows] * N[cols]).sum(1)) + 1e-6      # strictly positive: zero entries would vanish from the graph
    G = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    G = G.maximum(G.T)
    T = minimum_spanning_tree(G)
    T = T + T.T
    ncomp, label = connected_components(T, directed=False)
    N = N.copy()
    for comp in range(ncomp):
        root = int(np.flatnonzero(label == comp)[0])
        order, pred = breadth_first_order(T, root, directed=False, return_predecessors=True)
        for v in order[1:]:
            if N[v] @ N[pred[v]] < 0:
                N[v] = -N[v]
    return N, label


def estimate_normals(P, eye=None, orient="auto", k=30):
    """Unit normals for points P (N, 3).
    1. A plane is fitted to each point's k nearest neighbours (its normal, sign still arbitrary).
    2. The signs are made consistent by propagating along a neighbour graph, so each surface is oriented as a whole.
    3. Each connected surface is flipped once to face the right way. 'auto' decides from its points: one that wraps
       around its centre (an object) points outward; a one-sided scan (terrain, a wall) points toward `eye`
       (or +Z when no eye is given). 'outward' / 'inward' force the direction.
    Returns (normals, description of what was done)."""
    o3d = _lib()
    P = np.ascontiguousarray(P, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(int(min(k, max(len(P) - 1, 3)))))
    N, label = _orient_consistent(P, np.asarray(pcd.normals).copy())
    hows = set()
    for comp in np.unique(label):
        m = label == comp
        if m.sum() < 10:
            continue
        Pc, Nc = P[m], N[m]
        c = Pc.mean(0)
        R = Pc - c
        s = float((Nc * R).sum(1).mean() / (np.linalg.norm(R, axis=1).mean() + 1e-30))   # +1: away from the centre
        if orient == "outward":
            flip, how = s < 0, "outward"
        elif orient == "inward":
            flip, how = s > 0, "inward"
        elif abs(s) > 0.2:                                 # the points enclose a volume
            flip, how = s < 0, "outward"
        else:                                              # open, one-sided surface
            ref = (np.asarray(eye, float) - c) if eye is not None else np.array([0.0, 0.0, 1.0])
            flip, how = float(Nc.mean(0) @ ref) < 0, "toward the view" if eye is not None else "up"
        if flip:
            N[m] = -Nc
        hows.add(how)
    return N, " / ".join(sorted(hows)) or "up"


def poisson_mesh(P, detail=8, max_tris=50_000, trim=True, eye=None, orient="auto", seed=0):
    """Poisson surface reconstruction of the points P (N, 3).
    detail   octree depth, roughly 5 (coarse, smooth) to 11 (fine, many triangles)
    max_tris upper bound on triangles (0 = unlimited); the mesh is decimated down to it
    trim     cut away the low-density skin Poisson invents where there are no points
    Returns (V, F, info). F holds triangles wound so that the face normals follow the oriented point normals.
    Raises ValueError with a readable reason when no surface can be made."""
    o3d = _lib()
    P = np.asarray(P, float)
    P = P[np.isfinite(P).all(axis=1)]
    if len(P) < 30:
        raise ValueError("Select more points (at least 30) - a surface needs something to fit")
    if len(np.unique(np.round(P, 9), axis=0)) < 30:
        raise ValueError("The selected points are almost all identical")
    n_in = len(P)
    if n_in > MAX_INPUT:
        P = P[np.random.default_rng(seed).choice(n_in, MAX_INPUT, replace=False)]
    if float(np.ptp(P, axis=0).max()) < 1e-12:
        raise ValueError("The selected points are all in one place")
    N, how = estimate_normals(P, eye, orient)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(P)
    pcd.normals = o3d.utility.Vector3dVector(N)
    depth = int(np.clip(detail, 4, 12))
    mesh, _dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, scale=1.1, linear_fit=False)
    if trim and len(mesh.vertices):
        # cut away the skin Poisson invents where there are no points: vertices far from every input point
        # (far = several typical point spacings, and more than a couple of octree cells)
        from scipy.spatial import cKDTree
        tree = cKDTree(P)
        spacing = float(np.median(tree.query(P, k=2)[0][:, 1]))
        cell = float(np.ptp(P, axis=0).max()) * 1.1 / (2 ** depth)
        far = tree.query(np.asarray(mesh.vertices))[0] > max(4.0 * spacing, 2.0 * cell)
        if far.any() and not far.all():
            mesh.remove_vertices_by_mask(far)
    lo, hi = P.min(0), P.max(0)
    pad = 0.05 * float(np.linalg.norm(hi - lo))
    mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(lo - pad, hi + pad))   # Poisson can balloon past the data
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    n_full = len(mesh.triangles)
    if n_full == 0:
        raise ValueError("No surface came out - try more detail or a different selection")
    if max_tris and n_full > max_tris:
        mesh = mesh.simplify_quadric_decimation(int(max_tris))
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.triangles, int)
    if len(F) == 0:
        raise ValueError("No surface came out - try more detail or a different selection")
    info = "%s points -> %s triangles (detail %d), normals point %s" % (
        format(len(P), ","), format(len(F), ","), depth, how)
    if n_in > len(P):
        info += "; %s of %s points used" % (format(len(P), ","), format(n_in, ","))
    return V, F.tolist(), info
