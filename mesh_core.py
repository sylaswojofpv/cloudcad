"""
mesh_core.py  --  pure numpy (no Qt / OpenGL).
  * load_xyz       : robust .xyz reader (extra columns such as nx ny nz are ignored)
  * Mesh           : editable polygon mesh (n-gon faces) with freeform-style operations
  * Camera         : orbit camera + projection / ray helpers (single source of truth for
                     both rendering and picking)
  * pick_*         : screen-space picking helpers
"""
import re
import json
import numpy as np

# =============================================================== point cloud ==
_SPLIT = re.compile(r"[,\s;]+")


def _first_three_floats(line):
    toks = [t for t in _SPLIT.split(line.strip()) if t]
    if len(toks) < 3:
        return None
    try:
        return [float(t) for t in toks[:3]]
    except ValueError:
        return None


def _slow_load(path):
    rows = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("//"):
                continue
            r = _first_three_floats(s)
            if r is not None:
                rows.append(r)
    if not rows:
        raise ValueError("No numeric 'x y z' rows found in file")
    return np.asarray(rows, dtype=np.float64)


def load_xyz(path):
    """Read x y z from the first three columns. Anything after that (nx ny nz, rgb,
    intensity ...) is ignored, so files with or without normals both work. Header lines,
    comments, blank lines and space / tab / comma / semicolon separators are tolerated."""
    skip, delim, found = 0, None, False
    with open(path, "r", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if i > 5000:
                break
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("//"):
                continue
            if _first_three_floats(s) is not None:
                skip, found = i, True
                delim = "," if "," in s else (";" if ";" in s else None)
                break
    if not found:
        raise ValueError("No numeric 'x y z' rows found in file")
    try:  # fast C parser
        pts = np.loadtxt(path, delimiter=delim, usecols=(0, 1, 2), skiprows=skip,
                         comments=["#", "//"], dtype=np.float64, ndmin=2)
    except (ValueError, IndexError):  # ragged / dirty file -> tolerant slow path
        pts = _slow_load(path)
    return pts[np.isfinite(pts).all(axis=1)]


# ======================================================================= mesh ==
def _newell(P):
    Q = np.roll(P, -1, axis=0)
    return np.array([((P[:, 1] - Q[:, 1]) * (P[:, 2] + Q[:, 2])).sum(),
                     ((P[:, 2] - Q[:, 2]) * (P[:, 0] + Q[:, 0])).sum(),
                     ((P[:, 0] - Q[:, 0]) * (P[:, 1] + Q[:, 1])).sum()])


def _seg_cross(p1, p2, p3, p4):
    def cr(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return cr(p3, p4, p1) * cr(p3, p4, p2) < 0 and cr(p1, p2, p3) * cr(p1, p2, p4) < 0


def order_polygon(P):
    """Keep the click order unless it makes a bow-tie, then sort around the centroid."""
    n = len(P)
    if n < 4:
        return list(range(n))
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    xy = np.c_[(P - c) @ vt[0], (P - c) @ vt[1]]
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            if _seg_cross(xy[i], xy[(i + 1) % n], xy[j], xy[(j + 1) % n]):
                return [int(k) for k in np.argsort(np.arctan2(xy[:, 1], xy[:, 0]))]
    return list(range(n))


def _edge_occ(f, a, b):
    """How many times face loop f runs along edge a-b (either direction)."""
    n = len(f)
    return sum(1 for i in range(n)
               if (f[i] == a and f[(i + 1) % n] == b) or (f[i] == b and f[(i + 1) % n] == a))


def _clean_loop(loop):
    """Drop consecutive duplicate vertices (cyclically). None if fewer than 3 remain."""
    out = []
    for v in loop:
        if not out or out[-1] != v:
            out.append(v)
    while len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out if len(out) >= 3 else None


def _despike(loop):
    """Remove dangling spikes (x, y, x) left when a merge closes a slit in a polygon."""
    loop = list(loop)
    changed = True
    while changed and len(loop) >= 3:
        changed = False
        n = len(loop)
        for i in range(n):
            if loop[i - 1] == loop[(i + 1) % n]:
                j = (i + 1) % n
                loop = [v for k, v in enumerate(loop) if k not in (i, j)]
                changed = True
                break
    return loop


def _merge_faces(f1, f2, a, b):
    """Merge two faces across edge a-b into one loop (None if not a simple polygon)."""
    n1, n2 = len(f1), len(f2)
    i = next((k for k in range(n1) if f1[k] == a and f1[(k + 1) % n1] == b), None)
    if i is None:                       # f1 runs b->a, so swap names
        a, b = b, a
        i = next((k for k in range(n1) if f1[k] == a and f1[(k + 1) % n1] == b), None)
    j = next((k for k in range(n2) if f2[k] == b and f2[(k + 1) % n2] == a), None)
    if j is None:                       # f2 runs the same way: flip it
        f2 = f2[::-1]
        j = next((k for k in range(n2) if f2[k] == b and f2[(k + 1) % n2] == a), None)
    if i is None or j is None:
        return None
    m = _despike((f1[i + 1:] + f1[:i + 1]) + (f2[j + 1:] + f2[:j + 1])[1:-1])
    return m if len(m) >= 3 and len(set(m)) == len(m) else None


class Mesh:
    def __init__(self, V=None, F=None, T=None):
        self.V = np.zeros((0, 3)) if V is None else np.asarray(V, float).reshape(-1, 3)
        self.F = [] if F is None else [[int(i) for i in f] for f in F]
        self.T = [-1] * len(self.F) if T is None or len(T) != len(self.F) else [int(t) for t in T]   # body id per face
        self._tri = None

    # ---- basics
    @classmethod
    def plane(cls, center, size):
        h = size / 2.0
        c = np.asarray(center, float)
        return cls(c + np.array([[-h, -h, 0], [h, -h, 0], [h, h, 0], [-h, h, 0]]), [[0, 1, 2, 3]])

    def copy(self):
        return Mesh(self.V.copy(), [f[:] for f in self.F], self.T[:])

    def touch(self):
        self._tri = None

    def add_vertex(self, p):
        self.V = np.vstack([self.V, np.asarray(p, float)[None, :]])
        self._tri = None
        return len(self.V) - 1

    def add_face(self, idx, tag=-1):
        self.F.append([int(i) for i in idx])
        self.T.append(int(tag))
        self._tri = None
        return len(self.F) - 1

    def edge_faces(self):
        d = {}
        for fi, f in enumerate(self.F):
            n = len(f)
            for i in range(n):
                a, b = f[i], f[(i + 1) % n]
                d.setdefault((min(a, b), max(a, b)), []).append(fi)
        return d

    def face_normal(self, fi):
        n = _newell(self.V[self.F[fi]])
        l = np.linalg.norm(n)
        return n / l if l > 1e-12 else np.array([0.0, 0.0, 1.0])

    def face_area(self, fi):
        return 0.5 * float(np.linalg.norm(_newell(self.V[self.F[fi]])))

    def triangulate(self):
        """Centroid-fan triangulation -> (tri (T,3,3), tri_face (T,), face_normals (F,3))."""
        if self._tri is None:
            if not self.F:
                self._tri = (np.zeros((0, 3, 3)), np.zeros(0, int), np.zeros((0, 3)))
            else:
                tris, tf = [], []
                fn = np.zeros((len(self.F), 3))
                lens = np.fromiter((len(f) for f in self.F), int, len(self.F))
                t3 = np.flatnonzero(lens == 3)
                if len(t3):                                    # triangles (e.g. reconstructed meshes): vectorised
                    A = self.V[np.array([self.F[i] for i in t3])]
                    nr = np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0])
                    l = np.linalg.norm(nr, axis=1)
                    fn[t3] = np.where((l > 1e-12)[:, None], nr / np.maximum(l, 1e-12)[:, None], [0.0, 0.0, 1.0])
                    tris.append(A)
                    tf.append(t3)
                for fi in np.flatnonzero(lens != 3):
                    f = self.F[fi]
                    P = self.V[f]
                    c = P.mean(0)
                    tris.append(np.stack([np.broadcast_to(c, P.shape), P, np.roll(P, -1, 0)], 1))
                    tf.append(np.full(len(f), fi, int))
                    fn[fi] = self.face_normal(fi)
                self._tri = (np.concatenate(tris), np.concatenate(tf).astype(int), fn)
        return self._tri

    # ---- freeform-style operations
    def subdivide(self, face_ids):
        """Split each face into quads (centre + edge midpoints). Neighbouring faces that are
        not subdivided get the shared edge midpoint inserted so the mesh stays watertight.
        Returns the indices of the new faces."""
        S = set(face_ids)
        base, new, mid = len(self.V), [], {}

        def addv(p):
            new.append(p)
            return base + len(new) - 1

        def midpoint(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                mid[k] = addv((self.V[a] + self.V[b]) / 2.0)
            return mid[k]

        sub, subT = [], []
        for fi in sorted(S):
            f = self.F[fi]
            n = len(f)
            c = addv(self.V[f].mean(0))
            m = [midpoint(f[i], f[(i + 1) % n]) for i in range(n)]
            for i in range(n):
                sub.append([f[i], m[i], c, m[i - 1]])
                subT.append(self.T[fi])
        keep, keepT = [], []
        for fi, f in enumerate(self.F):
            if fi in S:
                continue
            keepT.append(self.T[fi])
            g, n = [], len(f)
            for i in range(n):
                a, b = f[i], f[(i + 1) % n]
                g.append(a)
                k = (min(a, b), max(a, b))
                if k in mid:
                    g.append(mid[k])
            keep.append(g)
        self.F = keep + sub
        self.T = keepT + subT
        if new:
            self.V = np.vstack([self.V, np.array(new)])
        self._tri = None
        return list(range(len(keep), len(keep) + len(sub)))

    def extrude_faces(self, face_ids, dist):
        """Duplicate the selected faces' vertices (moved along their normals by dist) and
        add side walls on the selection boundary. Face ids stay valid (they become the caps).
        Returns the new vertex ids."""
        S = set(face_ids)
        cnt, directed = {}, []
        for fi in S:
            f = self.F[fi]
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                k = (min(a, b), max(a, b))
                cnt[k] = cnt.get(k, 0) + 1
                directed.append((a, b, k, fi))
        vn = {}
        for fi in S:
            n = self.face_normal(fi)
            for v in self.F[fi]:
                vn[v] = vn.get(v, 0) + n
        base, remap, new = len(self.V), {}, []
        for v in sorted(vn):
            n = vn[v]
            l = np.linalg.norm(n)
            n = n / l if l > 1e-12 else np.array([0.0, 0.0, 1.0])
            remap[v] = base + len(new)
            new.append(self.V[v] + n * dist)
        for fi in S:
            self.F[fi] = [remap[v] for v in self.F[fi]]
        for a, b, k, fi in directed:
            if cnt[k] == 1:
                self.F.append([a, b, remap[b], remap[a]])
                self.T.append(self.T[fi])
        self.V = np.vstack([self.V, np.array(new)])
        self._tri = None
        return list(remap.values())

    def extrude_edges(self, edges, dist=None, fallback_normal=(0, 0, 1)):
        """Pull a new quad out of each selected edge (outward, in the plane of the adjacent
        face). Returns (new_edges, new_faces)."""
        ef = self.edge_faces()
        fb = np.asarray(fallback_normal, float)
        items, lens = [], []
        for (a, b) in edges:
            fs = ef.get((a, b), [])
            p, q, n, tg = a, b, fb, -1
            if fs:
                tg = self.T[fs[0]]
                f = self.F[fs[0]]
                n = self.face_normal(fs[0])
                for i in range(len(f)):
                    if f[i] == b and f[(i + 1) % len(f)] == a:
                        p, q = b, a
                        break
            e = self.V[q] - self.V[p]
            lens.append(np.linalg.norm(e))
            d = np.cross(e, n)
            if np.linalg.norm(d) < 1e-12:
                d = np.cross(e, [1.0, 0.0, 0.0])
            items.append((p, q, d / (np.linalg.norm(d) + 1e-12), tg))
        L = dist if dist else 0.5 * float(np.mean(lens))
        vdir = {}
        for p, q, d, _ in items:
            for v in (p, q):
                vdir[v] = vdir.get(v, 0) + d
        base, remap, new = len(self.V), {}, []
        for v, d in vdir.items():
            l = np.linalg.norm(d)
            d = d / l if l > 1e-12 else np.array([0.0, 0.0, 1.0])
            remap[v] = base + len(new)
            new.append(self.V[v] + d * L)
        self.V = np.vstack([self.V, np.array(new)])
        nf, ne = [], []
        for p, q, d, tg in items:
            self.F.append([q, p, remap[p], remap[q]])
            self.T.append(tg)
            nf.append(len(self.F) - 1)
            a, b = remap[p], remap[q]
            ne.append((min(a, b), max(a, b)))
        self._tri = None
        return ne, nf

    def delete_faces(self, face_ids):
        S = set(face_ids)
        self.T = [t for i, t in enumerate(self.T) if i not in S]
        self.F = [f for i, f in enumerate(self.F) if i not in S]
        self._tri = None

    def remove_vertices(self, drop):
        """Remove vertices and every face using them. Returns old->new index map."""
        drop = set(drop)
        alive = [not (set(f) & drop) for f in self.F]
        self.T = [t for t, a in zip(self.T, alive) if a]
        self.F = [f for f, a in zip(self.F, alive) if a]
        keep = [i for i in range(len(self.V)) if i not in drop]
        mp = {old: new for new, old in enumerate(keep)}
        self.F = [[mp[v] for v in f] for f in self.F]
        self.V = self.V[keep]
        self._tri = None
        return mp

    # ---- dissolve: delete elements without tearing holes in the surface
    @staticmethod
    def _neighbors(F, alive, v):
        nb = set()
        for k, f in enumerate(F):
            if alive[k] and v in f:
                n = len(f)
                for i in range(n):
                    if f[i] == v:
                        nb.add(f[i - 1])
                        nb.add(f[(i + 1) % n])
        return nb

    @staticmethod
    def _strip_vertex(F, alive, v):
        """Remove v from every face loop (its two edges join into one)."""
        for k, f in enumerate(F):
            if alive[k] and v in f:
                g = _clean_loop([u for u in f if u != v])
                if g is None:
                    alive[k] = False
                else:
                    F[k] = g

    def _straight(self, v, nb, cos_min):
        p, q = sorted(nb)
        d1, d2 = self.V[v] - self.V[p], self.V[q] - self.V[v]
        l = np.linalg.norm(d1) * np.linalg.norm(d2)
        return l < 1e-18 or float(d1 @ d2) / l >= cos_min

    @staticmethod
    def _fan_merge(F, fs, v):
        """Merge all faces around v into one loop that no longer contains v."""
        rot = {}
        for k in fs:
            f = F[k]
            if f.count(v) != 1:
                return None
            i = f.index(v)
            rot[k] = f[i:] + f[:i]                      # [v, a1, ..., am]
        first = {k: r[1] for k, r in rot.items()}
        last = {k: r[-1] for k, r in rot.items()}
        starts = [k for k in fs if not any(last[g] == first[k] for g in fs if g != k)]
        start = starts[0] if starts else fs[0]
        order, cur = [start], start
        while True:
            nxt = [g for g in fs if g != cur and first[g] == last[cur]]
            if len(nxt) != 1 or nxt[0] == start:
                break
            if nxt[0] in order:
                return None
            order.append(nxt[0])
            cur = nxt[0]
        if len(order) != len(fs):
            return None
        loop = []
        for k in order:
            p = rot[k][1:]
            if loop and loop[-1] == p[0]:
                p = p[1:]
            loop += p
        if len(loop) > 1 and loop[0] == loop[-1]:
            loop.pop()
        loop = _despike(loop)
        return (order, loop) if len(loop) >= 3 and len(set(loop)) == len(loop) else None

    def _compact(self, F, alive, drop):
        keep = [i for i in range(len(self.V)) if i not in drop]
        mp = {old: new for new, old in enumerate(keep)}
        self.T = [self.T[k] for k in range(len(F)) if alive[k]]
        self.F = [[mp[v] for v in f] for k, f in enumerate(F) if alive[k]]
        self.V = self.V[keep]
        self._tri = None

    def dissolve_edges(self, edges, max_bend_deg=45.0):
        """Delete edges: the two faces on either side merge into one. An end vertex left with
        only two roughly straight edges (bend under max_bend_deg) is removed so those edges
        join into one; sharp corners stay. A boundary edge (one face) removes that face.
        Returns {'merged', 'faces_removed', 'skipped'}."""
        F = [list(f) for f in self.F]
        alive = [True] * len(F)
        ends, seen = set(), set()
        st = dict(merged=0, faces_removed=0, skipped=0)
        for a, b in edges:
            hits = [(k, _edge_occ(f, a, b)) for k, f in enumerate(F) if alive[k] and _edge_occ(f, a, b)]
            total = sum(o for _, o in hits)
            if total == 0:                                # already gone (e.g. closed up by an earlier merge)
                ends.update((a, b))
                continue
            if total == 1:                                # boundary: nothing to merge with
                alive[hits[0][0]] = False
                seen.update(F[hits[0][0]])
                st["faces_removed"] += 1
            elif total == 2 and len(hits) == 1:           # dangling slit, tidied below
                ends.update((a, b))
                st["merged"] += 1
            elif total == 2:
                m = _merge_faces(F[hits[0][0]], F[hits[1][0]], a, b)
                if m is None:
                    st["skipped"] += 1
                    continue
                F[hits[0][0]] = m
                alive[hits[1][0]] = False
                ends.update((a, b))
                st["merged"] += 1
            else:
                st["skipped"] += 1
        cos_min = np.cos(np.radians(max_bend_deg))
        changed = True
        while changed:                                    # removing one vertex may free another
            changed = False
            for v in sorted(ends):
                nb = self._neighbors(F, alive, v)
                if len(nb) == 1 or (len(nb) == 2 and self._straight(v, nb, cos_min)):
                    self._strip_vertex(F, alive, v)
                    if len(nb) == 1:                      # a dangling end went away: check its neighbour too
                        ends.update(nb)
                    changed = True
        used = {u for k, f in enumerate(F) if alive[k] for u in f}
        self._compact(F, alive, {v for v in (ends | seen) if v not in used})
        return st

    def dissolve_vertices(self, verts):
        """Delete vertices: the faces around a vertex merge into one polygon; a vertex with just
        two edges is removed and its edges join. Vertices that cannot be removed cleanly are
        kept. Returns {'merged', 'skipped'}."""
        F = [list(f) for f in self.F]
        alive = [True] * len(F)
        st = dict(merged=0, skipped=0)
        gone = set()
        for v in sorted(set(verts)):
            fs = [k for k, f in enumerate(F) if alive[k] and v in f]
            if not fs:
                gone.add(v)
                continue
            if len(self._neighbors(F, alive, v)) > 2:
                res = self._fan_merge(F, fs, v)
                if res is None:
                    st["skipped"] += 1
                    continue
                order, loop = res
                gone.update(set().union(*(F[k] for k in order)) - set(loop))   # v and any closed-off spike
                F[order[0]] = loop
                for k in order[1:]:
                    alive[k] = False
                st["merged"] += len(order) - 1
            else:
                self._strip_vertex(F, alive, v)
            gone.add(v)
        used = {u for k, f in enumerate(F) if alive[k] for u in f}
        self._compact(F, alive, {v for v in gone if v not in used})
        return st

    def weld(self, verts):
        """Merge vertices into one at their centroid. Returns the merged vertex id."""
        vs = sorted(set(verts))
        if len(vs) < 2:
            return vs[0] if vs else None
        keep = vs[0]
        self.V[keep] = self.V[vs].mean(0)
        m = {v: keep for v in vs[1:]}
        newF, newT = [], []
        for fi, f in enumerate(self.F):
            h = []
            for v in (m.get(v, v) for v in f):
                if not h or h[-1] != v:
                    h.append(v)
            if len(h) > 1 and h[0] == h[-1]:
                h.pop()
            if len(h) >= 3 and len(set(h)) == len(h):
                newF.append(h)
                newT.append(self.T[fi])
        self.F = newF
        self.T = newT
        mp = self.remove_vertices(set(m))
        return mp[keep]

    def flip_faces(self, face_ids):
        for i in face_ids:
            self.F[i] = self.F[i][::-1]
        self._tri = None

    def orient_toward(self, idx, toward):
        """Return idx, reversed if its normal points away from `toward`."""
        if float(_newell(self.V[idx]) @ np.asarray(toward, float)) < 0:
            return idx[::-1]
        return idx

    # ---- file io
    def to_dict(self, origin):
        return {"origin": list(map(float, origin)), "vertices": self.V.tolist(), "faces": self.F, "tags": self.T}

    def export_obj(self, path, offset):
        with open(path, "w") as fh:
            fh.write("# Point Cloud Modeler\n")
            for v in self.V + np.asarray(offset, float):
                fh.write("v %.6f %.6f %.6f\n" % tuple(v))
            for f in self.F:
                fh.write("f " + " ".join(str(i + 1) for i in f) + "\n")


def save_model(path, mesh, origin, extra=None):
    d = mesh.to_dict(origin)
    if extra:
        d.update(extra)
    with open(path, "w") as fh:
        json.dump(d, fh)


def load_model(path):
    with open(path) as fh:
        d = json.load(fh)
    return Mesh(np.array(d["vertices"], float), d["faces"], d.get("tags")), np.array(d.get("origin", [0, 0, 0]), float)


# ===================================================================== camera ==
def _unit(v):
    return v / np.linalg.norm(v)


class Camera:
    """Z-up orbit camera."""

    def __init__(self):
        self.target = np.zeros(3)
        self.dist = 10.0
        self.yaw = np.radians(-60.0)
        self.pitch = np.radians(35.0)
        self.fov = np.radians(45.0)
        self.width, self.height = 800, 600
        self.radius = 10.0  # scene radius, only used for the far plane

    near = property(lambda s: max(s.dist * 0.02, 1e-6))
    far = property(lambda s: s.dist + 10.0 * s.radius)

    def _dir(self):  # target -> eye
        cp = np.cos(self.pitch)
        return np.array([cp * np.cos(self.yaw), cp * np.sin(self.yaw), np.sin(self.pitch)])

    def eye(self):
        return self.target + self.dist * self._dir()

    def forward(self):
        return -self._dir()

    def right(self):
        return _unit(np.cross(self.forward(), [0, 0, 1.0]))

    def up(self):
        return np.cross(self.right(), self.forward())

    def view_matrix(self):
        s, u, f, e = self.right(), self.up(), self.forward(), self.eye()
        M = np.eye(4)
        M[0, :3], M[1, :3], M[2, :3] = s, u, -f
        M[0, 3], M[1, 3], M[2, 3] = -s @ e, -u @ e, f @ e
        return M

    def proj_matrix(self):
        n, f = self.near, self.far
        t = 1.0 / np.tan(self.fov / 2)
        a = self.width / max(self.height, 1)
        M = np.zeros((4, 4))
        M[0, 0], M[1, 1] = t / a, t
        M[2, 2], M[2, 3] = (f + n) / (n - f), 2 * f * n / (n - f)
        M[3, 2] = -1.0
        return M

    def mvp(self):
        return self.proj_matrix() @ self.view_matrix()

    def world_per_pixel(self, depth):
        return 2 * np.tan(self.fov / 2) * depth / max(self.height, 1)

    def frame(self, center, radius):
        self.target = np.asarray(center, float)
        self.radius = max(float(radius), 1e-6)
        self.dist = self.radius / np.tan(self.fov / 2) * 1.2

    def orbit(self, dx, dy):
        self.yaw -= dx * 0.01
        self.pitch = float(np.clip(self.pitch + dy * 0.01, np.radians(-89.5), np.radians(89.5)))

    def pan(self, dx, dy):
        s = self.world_per_pixel(self.dist)
        self.target = self.target - self.right() * dx * s + self.up() * dy * s

    def zoom(self, steps, mx, my):
        """Zoom keeping the point under the cursor (on the target plane) fixed."""
        f = 0.87 ** steps
        o, d = self.ray(mx, my)
        p = ray_plane(o, d, self.target, self.forward())
        if p is not None:
            self.target = self.target + (p - self.target) * (1 - f)
        self.dist = max(self.dist * f, 1e-6)

    def project(self, P):
        """-> screen x, y (pixels, y down), ndc z, view depth w. Works for float32/64 arrays."""
        P = np.asarray(P)
        if P.dtype not in (np.float32, np.float64):
            P = P.astype(np.float64)
        M = self.mvp().astype(P.dtype)
        w = P @ M[3, :3] + M[3, 3]
        with np.errstate(divide="ignore", invalid="ignore"):
            iw = 1.0 / w
            sx = ((P @ M[0, :3] + M[0, 3]) * iw * 0.5 + 0.5) * self.width
            sy = (1.0 - ((P @ M[1, :3] + M[1, 3]) * iw * 0.5 + 0.5)) * self.height
            z = (P @ M[2, :3] + M[2, 3]) * iw
        return sx, sy, z, w

    def ray(self, mx, my):
        inv = np.linalg.inv(self.mvp())
        nx, ny = 2 * mx / self.width - 1, 1 - 2 * my / self.height
        p = inv @ np.array([nx, ny, 1.0, 1.0])
        p = p[:3] / p[3]
        o = self.eye()
        return o, _unit(p - o)


def ray_plane(o, d, p, n):
    den = float(d @ n)
    if abs(den) < 1e-12:
        return None
    return o + d * (float((p - o) @ n) / den)


def ray_axis_param(o, d, p, a):
    """Parameter s such that p + s*a is the point on the axis line closest to the ray."""
    w0 = p - o
    b = float(a @ d)
    den = 1 - b * b
    if den < 1e-8:
        return None
    return (b * float(d @ w0) - float(a @ w0)) / den


# ==================================================================== picking ==
def pick_cloud_point(cam, P, mx, my, radius):
    """Cloud point closest to the cursor on screen. Points overlapping it (within 1.5 px)
    are resolved in favour of the one nearest the camera."""
    sx, sy, z, w = cam.project(P)
    dx, dy = sx - mx, sy - my
    m = (np.abs(dx) <= radius) & (np.abs(dy) <= radius) & (w > cam.near)
    idx = np.flatnonzero(m)
    if idx.size == 0:
        return None
    d2 = dx[idx] ** 2 + dy[idx] ** 2
    idx, d2 = idx[d2 <= radius * radius], d2[d2 <= radius * radius]
    if idx.size == 0:
        return None
    idx = idx[d2 <= (np.sqrt(d2.min()) + 1.5) ** 2]
    return int(idx[np.argmin(w[idx])])


def pick_mesh_vertex(cam, V, mx, my, radius):
    if len(V) == 0:
        return None
    sx, sy, z, w = cam.project(V)
    d = np.hypot(sx - mx, sy - my)
    d[w <= cam.near] = np.inf
    i = int(np.argmin(d))
    return i if d[i] <= radius else None


def pick_mesh_edge(cam, V, edges, mx, my, radius):
    if not edges:
        return None
    sx, sy, z, w = cam.project(V)
    S = np.c_[sx, sy]
    E = np.asarray(edges)
    A, B = S[E[:, 0]], S[E[:, 1]]
    AB = B - A
    p = np.array([mx, my])
    t = np.clip(((p - A) * AB).sum(1) / np.maximum((AB * AB).sum(1), 1e-12), 0, 1)
    d = np.linalg.norm(A + AB * t[:, None] - p, axis=1)
    d[(w[E[:, 0]] <= cam.near) | (w[E[:, 1]] <= cam.near)] = np.inf
    i = int(np.argmin(d))
    return i if d[i] <= radius else None


def pick_mesh_face(cam, mesh, mx, my):
    tri, tf, _ = mesh.triangulate()
    if len(tri) == 0:
        return None
    sx, sy, z, w = cam.project(tri.reshape(-1, 3))
    X, Y, W = sx.reshape(-1, 3), sy.reshape(-1, 3), w.reshape(-1, 3)
    x0, x1, x2 = X.T
    y0, y1, y2 = Y.T
    det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    with np.errstate(divide="ignore", invalid="ignore"):
        l0 = ((y1 - y2) * (mx - x2) + (x2 - x1) * (my - y2)) / det
        l1 = ((y2 - y0) * (mx - x2) + (x0 - x2) * (my - y2)) / det
    l2 = 1 - l0 - l1
    hit = (np.abs(det) > 1e-9) & (l0 >= 0) & (l1 >= 0) & (l2 >= 0) & (W.min(1) > cam.near)
    idx = np.flatnonzero(hit)
    if idx.size == 0:
        return None
    depth = (l0 * W[:, 0] + l1 * W[:, 1] + l2 * W[:, 2])[idx]
    return int(tf[idx[np.argmin(depth)]])


# ============================================================ surface following ==
class SurfaceIndex:
    """Nearest-neighbour index over the cloud. Fits a local plane to the k nearest points
    and moves vertices onto it, so a mesh can slide along the scanned surface."""

    def __init__(self, pts, max_pts=3_000_000):
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            raise ImportError("Surface following needs SciPy:  pip install scipy")
        pts = np.asarray(pts)
        if len(pts) > max_pts:
            pts = pts[::int(np.ceil(len(pts) / max_pts))]
        self.pts = np.ascontiguousarray(pts, dtype=np.float32)
        self.tree = cKDTree(self.pts, balanced_tree=False, compact_nodes=False)

    def local_planes(self, Q, k):
        k = int(min(max(k, 3), len(self.pts)))
        d, idx = self.tree.query(Q, k=k, workers=-1)
        nb = self.pts[idx].astype(np.float64)              # (N, k, 3)
        c = nb.mean(axis=1)
        X = nb - c[:, None, :]
        _, v = np.linalg.eigh(np.einsum("nki,nkj->nij", X, X))
        return c, v[:, :, 0], d[:, -1]                      # centroid, normal, radius

    def project(self, Q, eye=None, use_ray=False, offset=0.0, k=24, iters=3):
        """Move points Q onto the local cloud surface.
        use_ray: slide along the line from `eye` through each point (the surface under
                 the cursor); otherwise take the closest point on the local plane.
        offset : distance above the surface, along the normal facing `eye` (or +Z)."""
        Q = np.atleast_2d(np.asarray(Q, float))
        P = Q.copy()
        rays = None
        if use_ray and eye is not None:
            rays = Q - eye
            rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
        for _ in range(iters):
            c, n, _ = self.local_planes(P, k)
            foot = P - ((P - c) * n).sum(1)[:, None] * n
            if rays is None:
                P = foot
                continue
            den = (n * rays).sum(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((c - eye) * n).sum(1) / den
            ok = (np.abs(den) > 0.1) & (t > 0)
            P = np.where(ok[:, None], eye + t[:, None] * rays, foot)
        if offset:
            c, n, _ = self.local_planes(P, k)
            ref = (eye - P) if eye is not None else np.array([0.0, 0.0, 1.0])
            n[(n * ref).sum(1) < 0] *= -1
            P = P + n * offset
        return P


# ============================================================ orientation frames ==
def frame_from_axes(z, t=None):
    """Right-handed orthonormal frame (columns x, y, z). z is given; x is t projected
    into the plane perpendicular to z (any perpendicular if t is missing or parallel)."""
    z = np.asarray(z, float)
    z = z / np.linalg.norm(z)
    x = None
    if t is not None:
        t = np.asarray(t, float)
        x = t - (t @ z) * z
        if np.linalg.norm(x) <= 1e-9 * max(np.linalg.norm(t), 1e-30):
            x = None
    if x is None:
        ref = np.eye(3)[int(np.argmin(np.abs(z)))]
        x = ref - (ref @ z) * z
    x = x / np.linalg.norm(x)
    return np.c_[x, np.cross(z, x), z]


def pca_frame(P):
    """Frame aligned with the principal axes of a point set (box, cylinder, flat patch...).
    None when there is no meaningful orientation (under 3 points, or round like a sphere)."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return None
    X = P - P.mean(0)
    w, v = np.linalg.eigh(X.T @ X)                       # ascending eigenvalues
    if w[2] <= 1e-18 or (w[2] - w[0]) < 0.05 * w[2]:
        return None
    canon = lambda a: a if a[int(np.argmax(np.abs(a)))] >= 0 else -a
    tie_hi = (w[2] - w[1]) < 0.05 * w[2]
    tie_lo = (w[1] - w[0]) < 0.05 * w[2]
    if tie_hi and tie_lo:
        return None
    if tie_lo:                                           # rod / cylinder: z = its axis
        return frame_from_axes(canon(v[:, 2]))
    if tie_hi:                                           # disc / plate: z = its normal
        return frame_from_axes(canon(v[:, 0]))
    return frame_from_axes(canon(v[:, 0]), canon(v[:, 2]))   # box-like: x longest, z shortest
