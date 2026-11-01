"""
constraints.py -- geometric constraints between mesh vertices, edges and faces.  Pure numpy / scipy.

An *entity* is a dict  {t: 'v'|'e'|'f', vs: [vertex ids], p: [last known positions]}
    vertex -> point, edge -> line (through its 2 vertices), face -> plane (Newell normal + centroid)
A *constraint* is a dict {kind, ents: [entity, ...], value, on, rigid}
    ents[0] is the ANCHOR: it stays where it is.  ents[1:] are moved until the constraint holds.

kinds (entities needed):
    parallel      line|plane + line|plane        (line vs plane: the line ends up parallel to the plane)
    perpendicular line|plane + line|plane        (line vs plane: the line ends up along the normal)
    collinear     any two of point/line/plane    (line-line: same line; plane-plane: same plane = coplanar;
                                                  line in plane; point on line / plane; point-point: coincident)
    distance      two entities (dimension)       custom distance; or ONE edge = its length
    angle         two lines / planes / line+plane (dimension) custom angle in degrees
    equal         edge + edge                    same length (the 2nd edge is resized about its middle)
    midpoint      edge + vertex                  vertex goes to the middle of the edge
    align         ONE line or plane              value = 0/1/2: line along X/Y/Z, plane normal along X/Y/Z
    fixed         ONE or more entities           the vertices are locked and never move
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

KINDS = ["parallel", "perpendicular", "collinear", "distance", "angle", "equal", "midpoint", "align", "fixed"]
NEEDS = dict(parallel=2, perpendicular=2, collinear=2, distance=(1, 2), angle=2, equal=2, midpoint=2, align=1, fixed=(1, 99))
LABEL = dict(parallel="Parallel", perpendicular="Perpendicular", collinear="Collinear / Coplanar", distance="Distance",
             angle="Angle", equal="Equal length", midpoint="Midpoint", align="Align to axis", fixed="Fixed")
HAS_VALUE = ("distance", "angle", "align")
ETYPE = dict(v="vertex", e="edge", f="face")


def _unit(v):
    l = np.linalg.norm(v)
    return v / l if l > 1e-30 else v


def _perp(a):
    t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    return _unit(np.cross(a, t))


def _newell(P):
    n = np.zeros(3)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        n += np.cross(a, b)
    return n


def make_entity(V, t, vs):
    vs = [int(i) for i in vs]
    return dict(t=t, vs=vs, p=[[float(x) for x in V[i]] for i in vs])


def geom(V, e):
    P = V[e["vs"]]
    if e["t"] == "v":
        return dict(r=0, c=P[0], a=None)
    if e["t"] == "e":
        return dict(r=1, c=(P[0] + P[1]) / 2, a=_unit(P[1] - P[0]), p0=P[0], p1=P[1])
    n = _unit(_newell(P))
    return dict(r=2, c=P.mean(0), a=n)


def rot_about(k, ang):
    k = _unit(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def rot_between(u, v):
    u, v = _unit(u), _unit(v)
    c = float(np.clip(u @ v, -1, 1))
    if c > 1 - 1e-14:
        return np.eye(3)
    k = np.cross(u, v)
    if np.linalg.norm(k) < 1e-12:
        k = _perp(u)
    return rot_about(k, np.arccos(c))


def angle_between(u, v):
    return float(np.arccos(np.clip(_unit(u) @ _unit(v), -1, 1)))


# ------------------------------------------------------------------ gap vector
def _gap_FS(F, S):
    """Shortest vector from feature F to the smaller (or equal rank) feature S."""
    if F["r"] == 0:
        return S["c"] - F["c"]
    if F["r"] == 1:
        if S["r"] == 0:
            d = S["c"] - F["c"]
            return d - (d @ F["a"]) * F["a"]
        cr = np.cross(F["a"], S["a"])
        if np.linalg.norm(cr) < 1e-9:                       # parallel lines
            d = S["c"] - F["c"]
            return d - (d @ F["a"]) * F["a"]
        n = _unit(cr)
        return ((S["c"] - F["c"]) @ n) * n                  # skew lines: along the common perpendicular
    n = F["a"]                                              # plane
    if S["r"] == 1 and abs(S["a"] @ n) > 1e-6:
        return np.zeros(3)                                  # a line crossing a plane touches it
    return ((S["c"] - F["c"]) @ n) * n


def gap(A, B):
    """Vector from A's feature to B's feature (shortest)."""
    return _gap_FS(A, B) if B["r"] <= A["r"] else -_gap_FS(B, A)


# ---------------------------------------------------------------- the rules
def _axis_rotation(kind, A, B):
    """Rotation taking B's axis to satisfy parallel / perpendicular against A (None when n/a)."""
    if A["r"] == 0 or B["r"] == 0:
        return None
    aA, aB = A["a"], B["a"]
    same = A["r"] == B["r"]
    want_par = (kind == "parallel") == same
    if want_par:
        tgt = aA if aA @ aB >= 0 else -aA
    else:
        t = aB - (aB @ aA) * aA
        tgt = _unit(t) if np.linalg.norm(t) > 1e-9 else _perp(aA)
    return rot_between(aB, tgt)


def _hold_dir(g, B):
    n = np.linalg.norm(g)
    if n > 1e-12:
        return g / n
    if B["r"] == 2:
        return B["a"]
    if B["r"] == 1:
        return _perp(B["a"])
    return np.array([1.0, 0, 0])


def step(kind, A, B, value):
    """One correction for the mover B against anchor A.  Returns (R, t) (rotate about pivot, then translate), or None."""
    I, z = np.eye(3), np.zeros(3)
    if kind in ("parallel", "perpendicular"):
        if A["r"] == 0 or B["r"] == 0:
            raise ValueError("needs lines or faces, not vertices")
        return _axis_rotation(kind, A, B), z
    if kind == "collinear":
        R = _axis_rotation("parallel", A, B) if A["r"] and B["r"] else None
        R = I if R is None else R
        B2 = dict(B)
        if B["a"] is not None:
            B2["a"] = R @ B["a"]
        g = gap(A, B2)
        return R, -g
    if kind == "midpoint":
        if A["r"] != 1 or B["r"] != 0:
            raise ValueError("needs an edge and then a vertex")
        return I, A["c"] - B["c"]
    if kind == "distance":
        R = I
        if A["r"] + B["r"] >= 3 or (A["r"] and B["r"] and A["r"] != B["r"]) or (A["r"] == 2 and B["r"] == 2):
            R = _axis_rotation("parallel", A, B)
            R = I if R is None else R
        B2 = dict(B)
        if B["a"] is not None:
            B2["a"] = R @ B["a"]
        g = gap(A, B2)
        d = float(np.linalg.norm(g))
        u = _hold_dir(g, B2)
        return R, u * (float(value) - d)
    if kind == "angle":
        if A["r"] == 0 or B["r"] == 0:
            raise ValueError("needs lines or faces, not vertices")
        aA, aB = A["a"], B["a"]
        if A["r"] == B["r"]:
            r, u, tgt = aA, aB, np.radians(value)
        elif A["r"] == 2:
            r, u, tgt = aA * (1 if aB @ aA >= 0 else -1), aB, np.radians(90 - value)
        else:
            r, u, tgt = aA * (1 if aA @ aB >= 0 else -1), aB, np.radians(90 - value)
        th = angle_between(r, u)
        k = np.cross(r, u)
        if np.linalg.norm(k) < 1e-9:
            k = _perp(r)
        return rot_about(k, tgt - th), z
    raise ValueError("unknown constraint " + kind)


def residual(kind, A, B, value):
    """How far the constraint is from holding (length units or radians)."""
    if kind in ("parallel", "perpendicular") or (kind == "collinear" and A["r"] and B["r"]):
        k = "parallel" if kind == "collinear" else kind
        same = A["r"] == B["r"]
        c = min(abs(float(A["a"] @ B["a"])), 1.0)
        ang = np.arcsin(np.sqrt(max(0.0, 1 - c * c))) if (k == "parallel") == same else np.arcsin(c)
        if kind == "collinear":
            return float(np.linalg.norm(gap(A, B))) + (1e3 if ang > 1e-5 else 0.0)
        return float(ang)
    if kind == "angle":
        if A["r"] == B["r"]:
            return abs(angle_between(A["a"], B["a"]) - np.radians(value))
        n, a = (A["a"], B["a"]) if A["r"] == 2 else (B["a"], A["a"])
        return abs(np.arcsin(min(abs(float(n @ a)), 1.0)) - np.radians(value))
    if kind == "midpoint":
        return float(np.linalg.norm(B["c"] - A["c"]))
    if kind in ("collinear", "distance"):
        d = float(np.linalg.norm(gap(A, B)))
        return d if kind == "collinear" else abs(d - float(value))
    return 0.0


# ------------------------------------------------------------------ the solver
def shell_labels(nv, F):
    """Connected-component label of every vertex (vertices joined through faces)."""
    if not len(F):
        return np.arange(nv)
    ln = np.fromiter((len(f) for f in F), int, len(F))
    flat = np.fromiter((i for f in F for i in f), int, int(ln.sum()))
    ends = np.cumsum(ln)
    nxt = np.roll(flat, -1)
    nxt[ends - 1] = flat[ends - ln]
    g = coo_matrix((np.ones(len(flat)), (flat, nxt)), shape=(nv, nv))
    return connected_components(g, directed=False)[1]


def check_types(kind, ents):
    """None when the picked entity types suit the constraint, else a message."""
    t = [e["t"] for e in ents]
    if kind in ("parallel", "perpendicular", "angle"):
        if len(t) != 2 or "v" in t:
            return "%s needs two objects that are edges or faces (not vertices)" % LABEL[kind]
    elif kind == "collinear":
        if len(t) != 2:
            return "Pick two objects (vertex, edge or face)"
    elif kind == "distance":
        if not (len(t) == 2 or t == ["e"]):
            return "Pick two objects, or a single edge to set its length"
    elif kind == "equal":
        if t != ["e", "e"]:
            return "Equal length needs two edges"
    elif kind == "midpoint":
        if t != ["e", "v"]:
            return "Midpoint: pick an edge first, then the vertex to put in its middle"
    elif kind == "align":
        if len(t) != 1 or t[0] == "v":
            return "Align needs one edge or one face"
    elif kind == "fixed":
        if not t:
            return "Pick what to fix"
    return None


def current_value(V, c):
    """The distance (model units) or angle (degrees) the entities have right now."""
    ents = c["ents"]
    if c["kind"] == "distance":
        if len(ents) == 1:
            a, b = ents[0]["vs"]
            return float(np.linalg.norm(V[b] - V[a]))
        return float(np.linalg.norm(gap(geom(V, ents[0]), geom(V, ents[1]))))
    if c["kind"] == "angle":
        A, B = geom(V, ents[0]), geom(V, ents[1])
        th = angle_between(A["a"], B["a"])
        if A["r"] == B["r"]:
            return float(np.degrees(th))
        n, a = (A["a"], B["a"]) if A["r"] == 2 else (B["a"], A["a"])
        return float(np.degrees(np.arcsin(min(abs(float(n @ a)), 1.0))))
    return 0.0


def validate(c, nv):
    kind, ents = c["kind"], c["ents"]
    need = NEEDS[kind]
    lo, hi = (need, need) if isinstance(need, int) else need
    if not lo <= len(ents) <= hi:
        return "needs %s object%s" % (lo if lo == hi else "%d-%d" % (lo, hi), "" if lo == 1 and hi == 1 else "s")
    if any(i >= nv or i < 0 for e in ents for i in e["vs"]):
        return "refers to deleted geometry"
    return None


def describe(c):
    return " + ".join(ETYPE[e["t"]] for e in c["ents"])


def _apply(V, group, R, t, pivot):
    V[group] = (V[group] - pivot) @ R.T + pivot + t


def _resize(V, lab, locked, a, b, L, rigid, a_fixed):
    """Make edge a-b length L.  Rigid: stretch the whole connected body along the edge (about vertex a, or the edge's middle);
    otherwise only move the edge's own vertices.  Returns how far anything moved."""
    d = V[b] - V[a]
    cur = float(np.linalg.norm(d))
    u = d / cur if cur > 1e-30 else np.array([1.0, 0, 0])
    piv = V[a].copy() if a_fixed else (V[a] + V[b]) / 2
    if rigid and cur > 1e-12:
        grp = np.nonzero(lab == lab[a])[0]
        if lab[a] == lab[b] and not (set(grp.tolist()) & locked):
            old = V[grp].copy()
            X = V[grp] - piv                          # stretch along the edge only - its cross-section stays the same
            V[grp] = piv + X + np.outer(X @ u, u) * (L / cur - 1.0)
            return float(np.abs(V[grp] - old).max())
    n0, n1 = (V[a].copy(), piv + u * L) if a_fixed else (piv - u * L / 2, piv + u * L / 2)
    moved = max(float(np.linalg.norm(n0 - V[a])), float(np.linalg.norm(n1 - V[b])))
    V[a], V[b] = n0, n1
    return moved


def solve(V, F, cons, passes=60):
    """Move vertices of V in place so the (enabled) constraints hold; the first entity of each constraint
    is the anchor.  Returns a list of status strings, one per constraint: 'ok' or a reason it failed."""
    n = len(V)
    status = [None] * len(cons)
    scale = float(np.linalg.norm(np.ptp(V, axis=0))) if n else 1.0
    tol_len, tol_ang = 1e-7 * max(scale, 1e-9), 1e-8
    locked = set()
    for c in cons:
        if c.get("on", True) and c["kind"] == "fixed" and validate(c, n) is None:
            for e in c["ents"]:
                locked.update(e["vs"])
                if len(e["p"]) == len(e["vs"]):                  # a Fixed vertex holds its stored position
                    V[e["vs"]] = np.asarray(e["p"], float)
    lab = shell_labels(n, F)
    order = []
    for k, c in enumerate(cons):
        if not c.get("on", True):
            status[k] = "off"
        elif (m := validate(c, n)):
            status[k] = m
        elif c["kind"] == "fixed":
            status[k] = "ok"
        else:
            order.append(k)

    def mover_group(c, B_vs, anchor_vs):
        group = set(B_vs)
        if c.get("rigid", True):
            group = set(np.nonzero(np.isin(lab, np.unique(lab[B_vs])))[0].tolist())
        if group & set(anchor_vs) or (group & locked and not set(B_vs) <= group - locked):
            group = set(B_vs)
        if group & locked and group != set(B_vs):
            group = set(B_vs)
        shared = [i for i in anchor_vs if i in group]
        return sorted(group - set(anchor_vs) - locked), shared

    for _ in range(passes):
        moved = 0.0
        for k in order:
            c = cons[k]
            kind, ents, val = c["kind"], c["ents"], c.get("value") or 0.0
            try:
                if kind == "align":
                    e = ents[0]
                    g = geom(V, e)
                    if g["r"] == 0:
                        raise ValueError("needs a line or a face")
                    ax = np.eye(3)[int(val)]
                    R = rot_between(g["a"], ax if g["a"] @ ax >= 0 else -ax)
                    if np.linalg.norm(R - np.eye(3)) < 1e-12:
                        status[k] = "ok"; continue
                    grp, _ = mover_group(c, e["vs"], [])
                    if set(e["vs"]) & locked:
                        status[k] = "blocked by a Fixed constraint"; continue
                    P0 = V[grp].copy(); _apply(V, grp, R, np.zeros(3), g["c"])
                    moved = max(moved, float(np.abs(V[grp] - P0).max()))
                    status[k] = "ok"; continue
                if kind == "distance" and len(ents) == 1:       # length of one edge: first vertex stays
                    e = ents[0]
                    if e["t"] != "e":
                        raise ValueError("one object must be an edge")
                    a, b = e["vs"]
                    if b in locked and a in locked:
                        status[k] = "blocked by a Fixed constraint"; continue
                    if b in locked:
                        a, b = b, a
                    moved = max(moved, _resize(V, lab, locked, a, b, float(val), c.get("rigid", True), a_fixed=True))
                    status[k] = "ok"; continue
                A, B0 = ents[0], ents[1]
                if kind == "equal":
                    if A["t"] != "e" or B0["t"] != "e":
                        raise ValueError("needs two edges")
                    L = float(np.linalg.norm(V[A["vs"][1]] - V[A["vs"][0]]))
                    a, b = B0["vs"]
                    if a in locked or b in locked:
                        status[k] = "blocked by a Fixed constraint"; continue
                    moved = max(moved, _resize(V, lab, locked, a, b, L, c.get("rigid", True) and lab[a] != lab[A["vs"][0]], a_fixed=False))
                    status[k] = "ok"; continue
                if set(B0["vs"]) <= locked:
                    ga, gb = geom(V, A), geom(V, B0)
                    s = step(kind, ga, gb, val)
                    if s is None or (np.abs(s[0] - np.eye(3)).max() < 1e-9 and np.abs(s[1]).max() < tol_len):
                        status[k] = "ok"
                    else:
                        status[k] = "second object is Fixed - can't be moved"
                    continue
                ga, gb = geom(V, A), geom(V, B0)
                s = step(kind, ga, gb, val)
                if s is None:
                    status[k] = "ok"; continue
                R, t = s
                if np.abs(R - np.eye(3)).max() < 1e-12 and np.abs(t).max() < tol_len * 1e-3:
                    status[k] = "ok"; continue
                grp, shared = mover_group(c, B0["vs"], A["vs"])
                if not grp:
                    status[k] = "nothing free to move"; continue
                pivot = V[shared[0]].copy() if shared else gb["c"]
                P0 = V[grp].copy(); _apply(V, grp, R, t, pivot)
                moved = max(moved, float(np.abs(V[grp] - P0).max()))
                status[k] = "ok"
            except ValueError as ex:
                status[k] = str(ex)
        if moved < tol_len * 1e-2:
            break
    # verify
    for k in order:
        if status[k] != "ok":
            continue
        c = cons[k]
        try:
            if len(c["ents"]) == 2 and c["kind"] in ("parallel", "perpendicular", "collinear", "distance", "angle", "midpoint"):
                A, B = geom(V, c["ents"][0]), geom(V, c["ents"][1])
                r = residual(c["kind"], A, B, c.get("value") or 0.0)
                lim = tol_ang * 1e3 if c["kind"] in ("parallel", "perpendicular", "angle") else tol_len * 10
                if r > lim:
                    status[k] = "conflicts with other constraints"
        except Exception:
            pass
    return status


def refresh(V, cons):
    """Store the current vertex positions in every entity (used to re-find vertices after topology changes)."""
    for c in cons:
        for e in c["ents"]:
            if all(0 <= i < len(V) for i in e["vs"]):
                e["p"] = [[float(x) for x in V[i]] for i in e["vs"]]


def remap(V, cons, tol):
    """After vertices were renumbered: point every entity at the vertex that sits at its stored position.
    Constraints whose geometry is gone are dropped; returns the surviving list."""
    from scipy.spatial import cKDTree
    if len(V) == 0:
        return []
    tree = cKDTree(V)
    keep = []
    for c in cons:
        ok = True
        for e in c["ents"]:
            P = np.asarray(e["p"], float)
            d, j = tree.query(P)
            if np.any(d > tol):
                ok = False
                break
            vs = [int(x) for x in j]
            for q, (i, p) in enumerate(zip(e["vs"], P)):    # keep the old index while it still holds that spot
                if 0 <= i < len(V) and np.linalg.norm(V[i] - p) <= tol:
                    vs[q] = int(i)
            e["vs"] = vs
        if ok and all(len(set(e["vs"])) == len(e["vs"]) for e in c["ents"]):
            keep.append(c)
    return keep


def shift(cons, delta):
    for c in cons:
        for e in c["ents"]:
            e["p"] = [[x + d for x, d in zip(p, delta)] for p in e["p"]]


def to_json(cons):
    return [dict(kind=c["kind"], value=c.get("value"), on=c.get("on", True), rigid=c.get("rigid", True),
                 ents=[dict(t=e["t"], vs=e["vs"], p=e["p"]) for e in c["ents"]]) for c in cons]


def from_json(data):
    out = []
    for d in data or []:
        if d.get("kind") in KINDS and d.get("ents"):
            out.append(dict(kind=d["kind"], value=d.get("value"), on=d.get("on", True), rigid=d.get("rigid", True),
                            ents=[dict(t=e["t"], vs=[int(i) for i in e["vs"]], p=e["p"]) for e in d["ents"]]))
    return out
