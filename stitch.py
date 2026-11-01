"""
stitch.py -- merge several bodies into one by stitching them together at their vertices, edges and faces
(instead of re-meshing everything the way a boolean union does).  numpy + scipy; shapely (optional) for faces
that only partly overlap.

    merge(V, F, body, tol, bridge=0.0) -> (V2, F2, body2, stats)

  1. weld      vertices of different bodies that sit within `tol` of each other become one vertex
  2. T-joints  a vertex that lies on another body's edge is inserted into that edge (no cracks)
  3. contacts  faces of two bodies that touch face-to-face (opposite normals, same plane) disappear; where they only
               partly cover each other the uncovered parts are cut out and triangulated so the outline matches
  4. bridge    optional: open rims that face each other across a small gap are joined with a strip of triangles
Everything not involved in a contact keeps its own vertices and polygons.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def _unit(v):
    l = np.linalg.norm(v)
    return v / l if l > 1e-30 else v


def _newell(P):
    n = np.zeros(3)
    for i in range(len(P)):
        n += np.cross(P[i], P[(i + 1) % len(P)])
    return n


def _basis(n):
    t = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = _unit(np.cross(n, t))
    return u, np.cross(n, u)


def _area2(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


# ------------------------------------------------------------------ 2D triangulation (polygon with holes)
def _ring_area(Q, ring):
    return 0.5 * sum(Q[ring[i]][0] * Q[ring[(i + 1) % len(ring)]][1] - Q[ring[(i + 1) % len(ring)]][0] * Q[ring[i]][1]
                     for i in range(len(ring)))


def _seg_cross(p1, p2, p3, p4):
    d1, d2 = _area2(p3, p4, p1), _area2(p3, p4, p2)
    d3, d4 = _area2(p1, p2, p3), _area2(p1, p2, p4)
    return ((d1 > 1e-14 and d2 < -1e-14) or (d1 < -1e-14 and d2 > 1e-14)) and \
           ((d3 > 1e-14 and d4 < -1e-14) or (d3 < -1e-14 and d4 > 1e-14))


def _inside(Q, ring, p):
    x, y, c = p[0], p[1], False
    for i in range(len(ring)):
        a, b = Q[ring[i]], Q[ring[(i + 1) % len(ring)]]
        if (a[1] > y) != (b[1] > y) and x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]:
            c = not c
    return c


def triangulate(Q, outer, holes=()):
    """Ear-clip a polygon (vertex ids `outer`) with holes. Q[id] -> (x, y). Returns [(i, j, k)] counter-clockwise."""
    ring = list(outer) if _ring_area(Q, outer) > 0 else list(outer)[::-1]
    hs = [list(h) if _ring_area(Q, h) < 0 else list(h)[::-1] for h in holes]
    for h in sorted(hs, key=lambda h: -max(Q[i][0] for i in h)):
        others = [x for x in hs if x is not h]
        best = None
        for hi in sorted(range(len(h)), key=lambda i: -Q[h[i]][0]):
            for oi in sorted(range(len(ring)), key=lambda i: (Q[ring[i]][0] - Q[h[hi]][0]) ** 2 + (Q[ring[i]][1] - Q[h[hi]][1]) ** 2):
                a, b = Q[h[hi]], Q[ring[oi]]
                mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
                if not _inside(Q, ring, mid) or _inside(Q, h, mid):
                    continue
                bad = False
                for rr in [ring, h] + others:
                    for k in range(len(rr)):
                        c, d = rr[k], rr[(k + 1) % len(rr)]
                        if _seg_cross(a, b, Q[c], Q[d]):
                            bad = True
                            break
                    if bad:
                        break
                if not bad:
                    best = (hi, oi)
                    break
            if best:
                break
        if best is None:
            raise ValueError("could not connect a hole")
        hi, oi = best
        ring = ring[:oi + 1] + h[hi:] + h[:hi + 1] + [ring[oi]] + ring[oi + 1:]
    return _ear_clip(Q, ring)


def _ear_clip(Q, ring, depth=0):
    """Ear clipping of a (weakly simple) ring of vertex ids. Vertices on a straight stretch are kept, never dropped, so the
    outline of the result matches the neighbours exactly."""
    tris, idx = [], list(range(len(ring)))
    guard = 0
    while len(idx) > 3 and guard < 10 * len(ring) + 50:
        guard += 1
        n, found, straight = len(idx), False, None
        for k in range(n):
            ia, ib, ic = ring[idx[k - 1]], ring[idx[k]], ring[idx[(k + 1) % n]]
            a, b, c = Q[ia], Q[ib], Q[ic]
            ar = _area2(a, b, c)
            if abs(ar) < 1e-14:
                if straight is None and ia != ic:
                    straight = k
                continue
            if ar < 0:
                continue
            ok = True
            for m in idx:
                v = ring[m]
                if v in (ia, ib, ic):
                    continue
                p = Q[v]
                if (abs(p[0] - a[0]) + abs(p[1] - a[1]) < 1e-12 or abs(p[0] - b[0]) + abs(p[1] - b[1]) < 1e-12
                        or abs(p[0] - c[0]) + abs(p[1] - c[1]) < 1e-12):
                    continue
                if _area2(a, b, p) >= -1e-14 and _area2(b, c, p) >= -1e-14 and _area2(c, a, p) >= -1e-14:
                    ok = False
                    break
            if ok:
                tris.append((ia, ib, ic)); idx.pop(k); found = True
                break
        if found:
            continue
        if straight is not None and depth < 40:               # only straight stretches left: split the ring at a diagonal
            n = len(idx)
            for off in range(2, n - 1):
                kd = (straight + off) % n
                b, d = Q[ring[idx[straight]]], Q[ring[idx[kd]]]
                if ring[idx[straight]] == ring[idx[kd]]:
                    continue
                mid = ((b[0] + d[0]) / 2, (b[1] + d[1]) / 2)
                sub = [ring[i] for i in idx]
                if not _inside(Q, sub, mid):
                    continue
                if any(_seg_cross(b, d, Q[sub[q]], Q[sub[(q + 1) % n]]) for q in range(n)):
                    continue
                r1 = [ring[idx[(straight + q) % n]] for q in range(off + 1)]
                r2 = [ring[idx[(kd + q) % n]] for q in range(n - off + 1)]
                return tris + _ear_clip(Q, r1, depth + 1) + _ear_clip(Q, r2, depth + 1)
        areas = [_area2(Q[ring[idx[k - 1]]], Q[ring[idx[k]]], Q[ring[idx[(k + 1) % len(idx)]]]) for k in range(len(idx))]
        k = int(np.argmax(areas))                               # numerically stuck: clip the most convex corner
        tris.append((ring[idx[k - 1]], ring[idx[k]], ring[idx[(k + 1) % len(idx)]])); idx.pop(k)
    if len(idx) == 3:
        t = tuple(ring[i] for i in idx)
        if abs(_area2(Q[t[0]], Q[t[1]], Q[t[2]])) > 1e-14:
            tris.append(t)
    return tris


# ------------------------------------------------------------------ the working mesh
class _Work:
    def __init__(self, V, F, body, tol):
        self.V = np.array(V, float)
        self.F = [list(map(int, f)) for f in F]
        self.B = list(body)
        self.tol = tol
        self.stats = dict(welded=0, tjoints=0, removed=0, contacts=0, unresolved=0, bridged=0)

    # -- helpers
    def used(self):
        u = np.zeros(len(self.V), bool)
        for f in self.F:
            u[f] = True
        return u

    def vsets(self):
        s = {}
        for f, b in zip(self.F, self.B):
            for v in f:
                s.setdefault(v, set()).add(b)
        return s

    def edges(self):
        e = {}
        for fi, f in enumerate(self.F):
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                e.setdefault((min(a, b), max(a, b)), []).append(fi)
        return e

    def add_vertex(self, p):
        self.V = np.vstack([self.V, p])
        return len(self.V) - 1

    # -- 1. weld
    def weld(self):
        V, tol = self.V, self.tol
        used = np.flatnonzero(self.used())
        if len(used) < 2:
            return
        vb = {}
        for f, b in zip(self.F, self.B):
            for v in f:
                vb.setdefault(v, set()).add(b)
        pairs = cKDTree(V[used]).query_pairs(tol, output_type="ndarray")
        if not len(pairs):
            return
        a, b = used[pairs[:, 0]], used[pairs[:, 1]]
        keep = np.array([vb[x] != vb[y] for x, y in zip(a, b)])
        a, b = a[keep], b[keep]
        if not len(a):
            return
        n = len(V)
        _, lab = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)
        mp = np.arange(n)
        members = {}
        for v in np.unique(np.r_[a, b]):
            members.setdefault(lab[v], []).append(v)
        for g in members.values():
            r = min(g)
            self.V[r] = V[g].mean(0)
            for v in g:
                mp[v] = r
            self.stats["welded"] += len(g) - 1
        self._remap(mp)

    def _remap(self, mp):
        newF, newB = [], []
        for f, b in zip(self.F, self.B):
            g = [int(mp[v]) for v in f]
            h = [v for i, v in enumerate(g) if v != g[i - 1]]
            if len(set(h)) >= 3:
                newF.append(h); newB.append(b)
        self.F, self.B = newF, newB

    # -- 2. T-joints
    def tjoints(self):
        V, tol = self.V, self.tol
        E = self.edges()
        used = np.flatnonzero(self.used())
        if len(used) < 3:
            return
        tree = cKDTree(V[used])
        ins = {}                                              # edge -> [(t, vertex)]
        for (a, b), fl in E.items():
            A, d = V[a], V[b] - V[a]
            L = float(np.linalg.norm(d))
            if L <= 3 * tol:
                continue
            for j in tree.query_ball_point((V[a] + V[b]) / 2, L / 2 + tol):
                v = int(used[j])
                if v == a or v == b or any(v in self.F[f] for f in fl):
                    continue
                t = float((V[v] - A) @ d) / (L * L)
                if t * L <= tol or (1 - t) * L <= tol:
                    continue
                if float(np.linalg.norm(V[v] - (A + t * d))) < tol:
                    ins.setdefault((a, b), []).append((t, v))
        if not ins:
            return
        newF = []
        for f in self.F:
            g = []
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                g.append(a)
                k = (min(a, b), max(a, b))
                if k in ins:
                    lst = sorted(ins[k])
                    if a > b:
                        lst = lst[::-1]
                    g += [v for _, v in lst if v not in f]
            newF.append(g)
        self.F = newF
        self.stats["tjoints"] += sum(len(v) for v in ins.values())

    # -- 3a. identical faces back to back
    def coincident(self):
        seen, kill = {}, set()
        for fi, f in enumerate(self.F):
            seen.setdefault(tuple(sorted(f)), []).append(fi)
        for fl in seen.values():
            for i in range(len(fl)):
                for j in range(i + 1, len(fl)):
                    a, b = fl[i], fl[j]
                    if a in kill or b in kill or self.B[a] == self.B[b]:
                        continue
                    if _newell(self.V[self.F[a]]) @ _newell(self.V[self.F[b]]) < 0:
                        kill.update((a, b))
        if kill:
            self.F = [f for i, f in enumerate(self.F) if i not in kill]
            self.B = [b for i, b in enumerate(self.B) if i not in kill]
            self.stats["removed"] += len(kill)

    # -- 3b. faces that touch face-to-face but only partly cover each other
    def contacts(self):
        try:
            from shapely.geometry import Polygon
            from shapely import make_valid
        except ImportError:
            self.stats["no_shapely"] = True
            return
        tol = self.tol
        done = set()
        for _ in range(400):
            nF = len(self.F)
            if nF < 2:
                return
            N = np.array([_unit(_newell(self.V[f])) for f in self.F])
            C = np.array([self.V[f].mean(0) for f in self.F])
            R = np.array([np.linalg.norm(self.V[f] - C[i], axis=1).max() for i, f in enumerate(self.F)])
            tree = cKDTree(C)
            hit = None
            for a in range(nF):
                for b in tree.query_ball_point(C[a], R[a] + R.max() + tol):
                    if b <= a or self.B[a] == self.B[b] or (a, b) in done:
                        continue
                    if N[a] @ N[b] > -0.98:
                        continue
                    if np.abs((self.V[self.F[b]] - C[a]) @ N[a]).max() > 1.5 * tol or np.abs((self.V[self.F[a]] - C[b]) @ N[b]).max() > 1.5 * tol:
                        continue
                    hit = (a, b)
                    break
                if hit:
                    break
            if hit is None:
                return
            a, b = hit
            done.add(hit)
            c0, n0 = C[a], N[a]
            u, w = _basis(n0)
            to2 = lambda P: np.c_[(P - c0) @ u, (P - c0) @ w]
            pa, pb = Polygon(to2(self.V[self.F[a]])), Polygon(to2(self.V[self.F[b]]))
            if not pa.is_valid:
                pa = make_valid(pa)
            if not pb.is_valid:
                pb = make_valid(pb)
            inter = pa.intersection(pb)
            if inter.is_empty or inter.area < 1e-6 * min(pa.area, pb.area) + 1e-12:
                continue
            pieces = []
            for poly, body, flip in ((pa.difference(inter), self.B[a], False), (pb.difference(inter), self.B[b], True)):
                geoms = [poly] if poly.geom_type == "Polygon" else [g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon"]
                for g in geoms:
                    if g.area > 1e-9 * max(pa.area, pb.area):
                        pieces.append((g, body, flip))
            grid = self._grid()
            newF, newB = [], []
            try:
                for g, body, flip in pieces:
                    ids = {}

                    def vid(xy):
                        p3 = c0 + u * xy[0] + w * xy[1]
                        v = self._find(grid, p3)
                        if v is None:
                            v = self.add_vertex(p3)
                            grid.setdefault(tuple(np.floor(p3 / (2 * tol)).astype(int)), []).append(v)
                        return v
                    ring = [vid(p) for p in list(g.exterior.coords)[:-1]]
                    holes = [[vid(p) for p in list(h.coords)[:-1]] for h in g.interiors]
                    Q = {}
                    for r in [ring] + holes:
                        for v in r:
                            Q[v] = tuple(to2(self.V[v][None, :])[0])
                    for t in triangulate(Q, ring, holes):
                        newF.append(list(t[::-1]) if flip else list(t)); newB.append(body)
            except ValueError:
                self.stats["unresolved"] += 1
                continue
            keep = [i for i in range(nF) if i not in (a, b)]
            self.F = [self.F[i] for i in keep] + newF
            self.B = [self.B[i] for i in keep] + newB
            self.stats["contacts"] += 1
            done = set()                                      # indices changed: faces may pair up differently now

    def _grid(self):
        g = {}
        for v in np.flatnonzero(self.used()):
            g.setdefault(tuple(np.floor(self.V[v] / (2 * self.tol)).astype(int)), []).append(int(v))
        return g

    def _find(self, grid, p):
        k = np.floor(p / (2 * self.tol)).astype(int)
        best, bd = None, self.tol
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for v in grid.get((k[0] + dx, k[1] + dy, k[2] + dz), ()):
                        d = float(np.linalg.norm(self.V[v] - p))
                        if d <= bd:
                            best, bd = v, d
        return best

    # -- 4. bridge open rims that face each other
    def bridge(self, gap):
        E = self.edges()
        nxt, ebody = {}, {}
        for fi, f in enumerate(self.F):
            for i in range(len(f)):
                a, b = f[i], f[(i + 1) % len(f)]
                if len(E[(min(a, b), max(a, b))]) == 1:
                    nxt.setdefault(a, []).append(b)
                    ebody[a] = self.B[fi]
        loops, seen = [], set()
        for s in list(nxt):
            if s in seen or len(nxt[s]) != 1:
                continue
            loop, v = [], s
            while v not in seen and v in nxt and len(nxt[v]) == 1:
                seen.add(v); loop.append(v); v = nxt[v][0]
            if v == s and len(loop) >= 3:
                loops.append(loop)
        info = []
        for lp in loops:
            P = self.V[lp]
            per = float(np.linalg.norm(P - np.roll(P, -1, 0), axis=1).sum())
            info.append((P.mean(0), per, ebody[lp[0]], cKDTree(P)))
        cand = []
        for i in range(len(loops)):
            for j in range(i + 1, len(loops)):
                if info[i][2] == info[j][2]:
                    continue
                d = float(info[i][3].query(self.V[loops[j]])[0].min())
                r = info[i][1] / max(info[j][1], 1e-30)
                if d <= gap and 0.3 <= r <= 3.3:
                    cand.append((d, i, j))
        taken = set()
        directed = {(f[i], f[(i + 1) % len(f)]) for f in self.F for i in range(len(f))}
        for _, i, j in sorted(cand):
            if i in taken or j in taken:
                continue
            P, Q = loops[i], loops[j]
            best = None
            for Qo in (Q, Q[::-1]):
                s0 = int(np.argmin(np.linalg.norm(self.V[Qo] - self.V[P[0]], axis=1)))
                Qr = Qo[s0:] + Qo[:s0]
                tris = self._zip(P, Qr)
                area = sum(0.5 * np.linalg.norm(_newell(self.V[list(t)])) for t in tris)
                if best is None or area < best[0]:
                    best = (area, tris)
            tris = best[1]
            if any((t[k], t[(k + 1) % 3]) in directed for t in tris for k in range(3)):
                tris = [t[::-1] for t in tris]
            for t in tris:
                self.F.append(list(t)); self.B.append(info[i][2])
            taken.update((i, j))
            self.stats["bridged"] += 1

    def _zip(self, P, Q):
        V, n, m = self.V, len(P), len(Q)
        i = j = 0
        tris = []
        while i < n or j < m:
            if i >= n:
                adv_p = False
            elif j >= m:
                adv_p = True
            else:
                dp = np.linalg.norm(V[P[(i + 1) % n]] - V[Q[j % m]])
                dq = np.linalg.norm(V[P[i % n]] - V[Q[(j + 1) % m]])
                adv_p = dp <= dq
            if adv_p:
                tris.append((P[i % n], P[(i + 1) % n], Q[j % m])); i += 1
            else:
                tris.append((P[i % n], Q[(j + 1) % m], Q[j % m])); j += 1
        return tris


def merge(V, F, body, tol, bridge=0.0):
    """Stitch the faces F (vertex-id lists into V, body id per face) into one seamless mesh.
    Returns (V2, F2, body2, stats); the vertices are compacted. tol = how close counts as touching."""
    w = _Work(V, F, body, tol)
    w.weld()
    w.tjoints()
    w.coincident()
    w.contacts()
    w.tjoints()
    w.coincident()
    if bridge and bridge > 0:
        w.bridge(bridge)
    used = np.flatnonzero(w.used())
    mp = np.full(len(w.V), -1, int)
    mp[used] = np.arange(len(used))
    return w.V[used], [[int(mp[v]) for v in f] for f in w.F], list(w.B), w.stats
