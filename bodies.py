"""
bodies.py -- bookkeeping for the Bodies list: bodies, boolean folders and how they are saved.

The mesh stays one Mesh; every face carries a body id (Mesh.T). A body that is
  live      has its faces in the mesh,
  hidden    is switched off in the list: its geometry is kept here (geom) and is not in the mesh,
  consumed  was combined by a boolean: its geometry is kept here (geom) inside the boolean's folder.
A Folder is one boolean operation: its operands (consumed bodies) and its result body.
Solids made by the Solid tool remember the picked points (src), so their points can be edited.
"""
import hashlib

import numpy as np

OP_NAMES = {"union": "Union", "difference": "Difference", "intersect": "Intersect"}


class Body:
    def __init__(self, id, name, kind="mesh", src=None):
        self.id, self.name, self.kind, self.src = id, name, kind, src
        self.state = "live"        # live | hidden | consumed
        self.geom = None           # (V, F) while the body is not in the mesh
        self.folder = None         # folder this body is an operand of (None = top level)


class Folder:
    def __init__(self, id, name, op, operands, result):
        self.id, self.name, self.op = id, name, op
        self.operands = list(operands)     # body ids, in boolean order
        self.result = result               # body id of the combined solid


class Registry:
    def __init__(self):
        self.bodies, self.folders = {}, {}
        self.next_id = 1
        self.counts = {}

    # ----------------------------------------------------------------- create
    def _name(self, prefix):
        self.counts[prefix] = self.counts.get(prefix, 0) + 1
        return "%s %d" % (prefix, self.counts[prefix])

    def new_body(self, prefix="Body", kind="mesh", src=None):
        b = Body(self.next_id, self._name(prefix), kind, src)
        self.next_id += 1
        self.bodies[b.id] = b
        return b

    def new_folder(self, op, operands, result):
        f = Folder(self.next_id, self._name(OP_NAMES[op]), op, operands, result)
        self.next_id += 1
        self.folders[f.id] = f
        for o in operands:
            self.bodies[o].folder = f.id
        return f

    # ----------------------------------------------------------------- queries
    def folder_of_result(self, bid):
        for f in self.folders.values():
            if f.result == bid:
                return f
        return None

    def is_result(self, bid):
        return self.folder_of_result(bid) is not None

    def parent_folder(self, fid):
        """Folder that contains folder `fid` as one of its operands (via fid's result body), or None."""
        r = self.folders[fid].result
        b = self.bodies.get(r)
        return self.folders.get(b.folder) if b is not None and b.folder is not None else None

    # ------------------------------------------------------------------ shift
    def shift(self, delta):
        """Move everything stored here (the scene origin changed)."""
        delta = np.asarray(delta, float)
        for b in self.bodies.values():
            if b.geom is not None:
                b.geom = (b.geom[0] + delta, b.geom[1])
            if b.src is not None:
                b.src["pts"] = b.src["pts"] + delta

    # --------------------------------------------------------------------- io
    def to_dict(self):
        def src(s):
            return None if s is None else dict(s, pts=np.asarray(s["pts"]).tolist())
        return {
            "next_id": self.next_id, "counts": self.counts,
            "bodies": [{"id": b.id, "name": b.name, "kind": b.kind, "state": b.state, "folder": b.folder,
                        "src": src(b.src),
                        "geom": None if b.geom is None else {"V": np.asarray(b.geom[0]).tolist(), "F": b.geom[1]}}
                       for b in self.bodies.values()],
            "folders": [{"id": f.id, "name": f.name, "op": f.op, "operands": f.operands, "result": f.result}
                        for f in self.folders.values()],
        }

    @classmethod
    def from_dict(cls, d):
        r = cls()
        if not d:
            return r
        r.next_id, r.counts = int(d.get("next_id", 1)), dict(d.get("counts", {}))
        for e in d.get("bodies", []):
            b = Body(e["id"], e["name"], e.get("kind", "mesh"))
            b.state, b.folder = e.get("state", "live"), e.get("folder")
            if e.get("src"):
                b.src = dict(e["src"], pts=np.array(e["src"]["pts"], float))
            if e.get("geom"):
                b.geom = (np.array(e["geom"]["V"], float).reshape(-1, 3), [list(f) for f in e["geom"]["F"]])
            r.bodies[b.id] = b
        for e in d.get("folders", []):
            r.folders[e["id"]] = Folder(e["id"], e["name"], e["op"], e["operands"], e["result"])
        return r


# ------------------------------------------------------------------ geometry
def extract_faces(mesh, face_ids):
    """Copy the given faces out of the mesh as compact (V, F)."""
    face_ids = list(face_ids)
    used = sorted({v for i in face_ids for v in mesh.F[i]})
    mp = {v: k for k, v in enumerate(used)}
    return mesh.V[used].copy(), [[mp[v] for v in mesh.F[i]] for i in face_ids]


def geom_sig(V):
    """Shape fingerprint that ignores where the body sits (so moving a solid isn't an 'edit')."""
    V = np.asarray(V, float)
    if len(V) == 0:
        return ""
    C = V - V.mean(0)
    scale = float(np.ptp(V, axis=0).max()) or 1.0
    Q = np.round(C / scale * 1e4).astype(np.int64)
    Q = Q[np.lexsort(Q.T[::-1])]
    return "%d:%s" % (len(Q), hashlib.md5(Q.tobytes()).hexdigest()[:12])
