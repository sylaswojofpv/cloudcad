"""
booleans.py -- union / difference / intersection of closed solids (shells) in a Mesh.

The heavy lifting (robust mesh intersection) is done by manifold3d (pip install manifold3d).
Our polygon faces are triangulated for it, and the triangle result is turned back into
quads / n-gons: coplanar triangles are merged and needless straight vertices removed, so the
output stays pleasant to edit. Surfaces that genuinely curve stay triangulated.
"""
import numpy as np

from mesh_core import _newell, _edge_occ, _merge_faces

KINDS = ("union", "difference", "intersect")


# ------------------------------------------------------------------ shells
def shells_of(mesh):
    """Connected solids: list of face-id lists (ordered by lowest face id)."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for f in mesh.F:
        r = find(f[0])
        for v in f[1:]:
            parent[find(v)] = r
    groups = {}
    for i, f in enumerate(mesh.F):
        groups.setdefault(find(f[0]), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def split_loose(mesh, fids):
    """Separate a shell into its closed solid part and loose faces attached to it (a plane or a
    single face sharing an edge / corner with the solid). Faces that have an open edge are peeled
    off repeatedly, so what is left is closed. -> (solid face ids, loose face ids).
    If nothing closed is left, the whole shell is returned as the solid so the usual error shows."""
    fids = list(fids)
    de = {}
    for i in fids:
        f = mesh.F[i]
        for k in range(len(f)):
            de.setdefault((f[k], f[(k + 1) % len(f)]), set()).add(i)

    def is_open(i):
        f = mesh.F[i]
        if len(set(f)) < 3:
            return True
        return any(not de.get((f[(k + 1) % len(f)], f[k])) for k in range(len(f)))

    alive, queue = set(fids), list(fids)
    while queue:
        i = queue.pop()
        if i not in alive or not is_open(i):
            continue
        alive.discard(i)
        f = mesh.F[i]
        for k in range(len(f)):
            a, bb = f[k], f[(k + 1) % len(f)]
            de[(a, bb)].discard(i)
            queue.extend(de.get((bb, a), ()))       # faces that were closed by this one may now be open
    if not alive:
        return fids, []
    return [i for i in fids if i in alive], [i for i in fids if i not in alive]


def signed_volume(V, F):
    vol = 0.0
    for f in F:
        P = V[list(f)]
        for k in range(1, len(f) - 1):
            vol += float(P[0] @ np.cross(P[k], P[k + 1]))
    return vol / 6.0


def check_shell(mesh, fids):
    """-> (problem message or None, flip). flip=True when the shell is inside out."""
    de = {}
    for i in fids:
        f = mesh.F[i]
        if len(set(f)) < 3:
            return "it has a degenerate face", False
        for k in range(len(f)):
            e = (f[k], f[(k + 1) % len(f)])
            de[e] = de.get(e, 0) + 1
    n_open = sum(1 for (a, b) in de if (b, a) not in de)
    if n_open:
        return "it is not closed (%d open edge%s)" % (n_open, "" if n_open == 1 else "s"), False
    if any(c > 1 for c in de.values()):
        return "it has overlapping faces or a fin (non-manifold edges)", False
    vol = signed_volume(mesh.V, [mesh.F[i] for i in fids])
    if abs(vol) < 1e-15:
        return "it has no volume", False
    return None, vol < 0


# ----------------------------------------------------------- triangulation
def triangulate_polygon(P):
    """Ear clipping of a (possibly concave, possibly slightly non-planar) polygon.
    Returns index triples that keep the polygon's winding; collinear vertices are kept
    without creating zero-area triangles."""
    P = np.asarray(P, float)
    n = len(P)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]
    nrm = _newell(P)
    if np.linalg.norm(nrm) < 1e-18:
        return [(0, i, i + 1) for i in range(1, n - 1)]
    k = int(np.argmax(np.abs(nrm)))
    u, v = (k + 1) % 3, (k + 2) % 3
    if nrm[k] < 0:
        u, v = v, u
    Q = P[:, [u, v]]
    scale = max(float(np.ptp(Q, axis=0).max()), 1e-30)
    eps = 1e-12 * scale * scale

    def cross(a, b, c):
        return (Q[b, 0] - Q[a, 0]) * (Q[c, 1] - Q[a, 1]) - (Q[b, 1] - Q[a, 1]) * (Q[c, 0] - Q[a, 0])

    def covered(p, a, b, c):          # on or inside triangle a,b,c (counter-clockwise)
        return cross(a, b, p) >= -eps and cross(b, c, p) >= -eps and cross(c, a, p) >= -eps

    idx, tris = list(range(n)), []
    while len(idx) > 3:
        m, ear = len(idx), None
        for strict in (True, False):
            for ii in range(m):
                a, b, c = idx[ii - 1], idx[ii], idx[(ii + 1) % m]
                cr = cross(a, b, c)
                if cr <= (eps if strict else -eps):
                    continue
                if any(covered(p, a, b, c) for p in idx if p not in (a, b, c)):
                    continue
                ear = ii
                break
            if ear is not None:
                break
        if ear is None:                                # numerical trouble: fan what is left
            tris += [(idx[0], idx[i], idx[i + 1]) for i in range(1, len(idx) - 1)]
            return tris
        tris.append((idx[ear - 1], idx[ear], idx[(ear + 1) % m]))
        idx.pop(ear)
    tris.append((idx[0], idx[1], idx[2]))
    return tris


# ------------------------------------------------------------ manifold glue
def _lib():
    try:
        import manifold3d as mf
    except ImportError:
        raise ImportError("Boolean operations need manifold3d:  pip install manifold3d")
    return mf


def _to_manifold(mf, mesh, fids, flip):
    used = sorted({v for i in fids for v in mesh.F[i]})
    mp = {v: k for k, v in enumerate(used)}
    tris = []
    for i in fids:
        f = mesh.F[i][::-1] if flip else mesh.F[i]
        for a, b, c in triangulate_polygon(mesh.V[f]):
            tris.append((mp[f[a]], mp[f[b]], mp[f[c]]))
    P = np.ascontiguousarray(mesh.V[used], dtype=np.float64)
    if hasattr(mf, "Mesh64"):
        m = mf.Manifold(mf.Mesh64(vert_properties=P, tri_verts=np.array(tris, np.uint64)))
    else:
        m = mf.Manifold(mf.Mesh(vert_properties=P.astype(np.float32), tri_verts=np.array(tris, np.uint32)))
    return m


def _from_manifold(m):
    out = m.to_mesh64() if hasattr(m, "to_mesh64") else m.to_mesh()
    return np.asarray(out.vert_properties, float)[:, :3], np.asarray(out.tri_verts, int)


# ------------------------------------------------- triangles -> quads / n-gons
def _convex(P, nrm, tol=1e-3):
    n = len(P)
    for i in range(n):
        d1, d2 = P[i] - P[i - 1], P[(i + 1) % n] - P[i]
        l = np.linalg.norm(d1) * np.linalg.norm(d2)
        if l > 1e-30 and float(np.cross(d1, d2) @ nrm) < -tol * l:
            return False
    return True


def _unit(v):
    l = np.linalg.norm(v)
    return v / l if l > 1e-30 else v * 0.0


def merge_coplanar(V, T, ang_deg=0.5):
    """Greedily merge adjacent coplanar triangles into convex polygons."""
    V = np.asarray(V, float)
    diag = float(np.linalg.norm(np.ptp(V, axis=0))) or 1.0
    dist_tol, cos_t = 1e-6 * diag, np.cos(np.radians(ang_deg))
    F = [[int(a), int(b), int(c)] for a, b, c in T]
    alive = [True] * len(F)
    nrm = [_unit(_newell(V[f])) for f in F]
    cen = [V[f].mean(0) for f in F]
    for _ in range(25):
        emap = {}
        for k, f in enumerate(F):
            if alive[k]:
                for i in range(len(f)):
                    a, b = f[i], f[(i + 1) % len(f)]
                    emap.setdefault((min(a, b), max(a, b)), []).append(k)
        keys = sorted((e for e, fs in emap.items() if len(fs) == 2),
                      key=lambda e: -np.linalg.norm(V[e[0]] - V[e[1]]))
        changed = False
        for a, b in keys:
            k1, k2 = emap[(a, b)]
            if not (alive[k1] and alive[k2]):
                continue
            if _edge_occ(F[k1], a, b) != 1 or _edge_occ(F[k2], a, b) != 1:
                continue
            if nrm[k1] @ nrm[k2] < cos_t:
                continue
            if (np.abs((V[F[k2]] - cen[k1]) @ nrm[k1]).max() > dist_tol or
                    np.abs((V[F[k1]] - cen[k2]) @ nrm[k2]).max() > dist_tol):
                continue
            m = _merge_faces(F[k1], F[k2], a, b)
            if m is None:
                continue
            mn = _unit(_newell(V[m]))
            if not _convex(V[m], mn):
                continue
            F[k1], alive[k2], nrm[k1], cen[k1] = m, False, mn, V[m].mean(0)
            changed = True
        if not changed:
            break
    return [f for k, f in enumerate(F) if alive[k]]


def collapse_straight(V, F, ang_deg=1.0):
    """Remove vertices that sit on a straight line with exactly two neighbours."""
    F = [list(f) for f in F]
    vf = {}
    for k, f in enumerate(F):
        for v in f:
            vf.setdefault(v, []).append(k)
    cos_min = np.cos(np.radians(ang_deg))
    for v, fl in vf.items():
        if any(len(F[k]) <= 3 for k in fl):
            continue
        nb = set()
        for k in fl:
            f = F[k]
            i = f.index(v)
            nb.add(f[i - 1])
            nb.add(f[(i + 1) % len(f)])
        if len(nb) != 2:
            continue
        p, q = sorted(nb)
        d1, d2 = V[v] - V[p], V[q] - V[v]
        l = np.linalg.norm(d1) * np.linalg.norm(d2)
        if l > 1e-30 and float(d1 @ d2) / l >= cos_min:
            for k in fl:
                F[k] = [u for u in F[k] if u != v]
    used = sorted({u for f in F for u in f})
    mp = {old: new for new, old in enumerate(used)}
    return np.asarray(V)[used], [[mp[u] for u in f] for f in F]


# ------------------------------------------------------------------ the API
def boolean_op(mesh, kind, shells):
    """Combine solids. `shells` = face-id lists; the first is the target for 'difference'.
    Returns (V_new, F_new, info). Raises ValueError with a readable reason on failure."""
    if kind not in KINDS:
        raise ValueError("unknown boolean %r" % kind)
    mf = _lib()
    ops = []
    for n, fids in enumerate(shells, 1):
        problem, flip = check_shell(mesh, fids)
        if problem:
            raise ValueError("Solid %d can't be used: %s." % (n, problem))
        m = _to_manifold(mf, mesh, fids, flip)
        if str(m.status()) not in ("Error.NoError", "NoError"):
            raise ValueError("Solid %d isn't a valid closed solid for booleans (%s)." % (n, m.status()))
        ops.append(m)
    R = ops[0]
    for B in ops[1:]:
        R = R + B if kind == "union" else (R - B if kind == "difference" else R ^ B)
    if R.is_empty():
        raise ValueError({"union": "The result is empty.",
                          "difference": "Nothing is left - the first solid is completely removed.",
                          "intersect": "The solids don't overlap, so the intersection is empty."}[kind])
    V, T = _from_manifold(R)
    F = merge_coplanar(V, T)
    V, F = collapse_straight(V, F)
    tri = sum(1 for f in F if len(f) == 3)
    name = {"union": "Union", "difference": "Difference", "intersect": "Intersect"}[kind]
    return V, F, "%s of %d solids: %d faces (%d triangles)" % (name, len(shells), len(F), tri)


def replace_faces(mesh, remove_ids, V_new, F_new, tag=-1):
    """Swap the given faces for new geometry (tagged with body id `tag`). Returns the new faces' ids."""
    rm = set(remove_ids)
    keepF = [f for i, f in enumerate(mesh.F) if i not in rm]
    keepT = [t for i, t in enumerate(mesh.T) if i not in rm]
    used = {u for f in keepF for u in f}
    # vertices used only by the removed faces go; everything else (even loose points) stays
    in_removed = {u for i in rm for u in mesh.F[i]} - used
    keep = [i for i in range(len(mesh.V)) if i not in in_removed]
    mp = {old: new for new, old in enumerate(keep)}
    off = len(keep)
    mesh.V = np.vstack([mesh.V[keep], np.asarray(V_new, float).reshape(-1, 3)])
    mesh.F = [[mp[u] for u in f] for f in keepF] + [[off + u for u in f] for f in F_new]
    mesh.T = keepT + [int(tag)] * len(F_new)
    mesh.touch()
    return list(range(len(keepF), len(keepF) + len(F_new)))
