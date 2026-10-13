"""
Point Cloud Modeler  --  simple freeform-style surface modeling over an .xyz point cloud.

    pip install PySide6 PyOpenGL numpy
    python modeler.py [optional_cloud.xyz]
"""
import os
import sys
import ctypes
import traceback
import numpy as np

from PySide6.QtCore import Qt, Signal, QEventLoop
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QSurfaceFormat
from PySide6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QDockWidget, QWidget,
                               QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QDoubleSpinBox,
                               QSpinBox, QCheckBox, QSlider, QPushButton, QMessageBox, QComboBox, QScrollArea)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *

from mesh_core import (Mesh, Camera, load_xyz, save_model, load_model, order_polygon,
                       ray_plane, ray_axis_param, pick_cloud_point, pick_mesh_vertex,
                       pick_mesh_edge, pick_mesh_face, SurfaceIndex)
from solids import SHAPES, SHAPE_NAMES, MIN_POINTS, fit_solid

MAX_DISPLAY = 3_000_000   # points actually drawn (stride-decimated above this)
MAX_SNAP = 400_000        # points tested during hover / drag snapping
TOOLS = {"select": "Select", "move": "Move / Transform", "points": "Face from cloud points",
         "solid": "Solid (best fit)"}
MODES = {"v": "Vertex", "e": "Edge", "f": "Face"}
HINTS = {
    "select": "Click to pick (Shift = add/remove)",
    "move": "Drag a handle or element. X/Y/Z lock axis, Ctrl = snap to cloud, Esc cancels",
    "points": "Click cloud points (Ctrl = ignore mesh verts). Enter = make face, Backspace = undo point, Esc = cancel",
    "solid": "Click points on the object's surface - more points = better fit. Enter = create, Backspace = undo point, Esc = cancel",
}


def height_colors(z):
    t = (z - z.min()) / max(float(z.max() - z.min()), 1e-12)
    stops = np.array([0, .25, .5, .75, 1.0])
    cols = np.array([[60, 90, 255], [40, 200, 255], [80, 230, 110], [250, 230, 60], [255, 110, 60]], float)
    return np.ascontiguousarray(np.stack([np.interp(t, stops, cols[:, i]) for i in range(3)], 1).astype(np.uint8))


class Viewer(QOpenGLWidget):
    status = Signal(str)
    selection_changed = Signal()
    state_changed = Signal()

    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.cam = Camera()
        self.mesh = Mesh.plane((0, 0, 0), 10.0)
        self.pristine = True
        self.origin = np.zeros(3)
        self.mode, self.sel, self.tool = "v", set(), "move"
        # cloud
        self.cloud_full = self.cloud_snap = self.cloud_disp = None
        self.cloud_cols = None
        self.vbo = None
        self._cloud_dirty = False
        self.n_disp = 0
        self.grid_z = 0.0
        # settings
        self.point_size, self.snap_radius, self.face_alpha = 2.0, 12.0, 0.6
        self.use_height_colors, self.auto_n = True, 4
        self.follow, self.surf_offset, self.surf_k, self.surf = False, 0.0, 24, None
        self.solid_shape, self.solid_seg, self.solid_preview, self.solid_info = "cylinder", 24, None, ""
        # interaction state
        self.pending, self.drag, self.nav = [], None, None
        self.axis_lock, self.snap, self.hover_axis = None, None, None
        self.last = (0, 0)
        self.undo, self.redo = [], []
        self.sel_trail = []   # elements picked by clicking, newest last (Ctrl+Z un-selects these first)
        self._arr = None
        self._gl_failed = False
        self.hud = QLabel(self)
        self.hud.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hud.setStyleSheet("color:#dfe6ee;background:rgba(20,22,26,150);padding:6px;border-radius:4px;")
        self.hud.move(8, 8)
        self.update_hud()

    # ------------------------------------------------------------ state helpers
    def changed(self):
        self.mesh.touch()
        self._arr = None
        self.update()

    def update_hud(self):
        extra = ""
        self.refit_solid()
        if self.tool == "points":
            extra = "  [%d/%s]" % (len(self.pending), self.auto_n or "manual")
        elif self.tool == "solid":
            extra = "  [%s: %d pts%s]" % (SHAPE_NAMES[self.solid_shape], len(self.pending), self.solid_info)
        self.hud.setText("<b>%s</b>   |   select: %s%s<br><span style='color:#9fb0c3'>%s<br>"
                         "MMB orbit · Shift+MMB / RMB pan · wheel zoom · Home frame</span>"
                         % (TOOLS[self.tool], MODES[self.mode], extra + ("  |  follow surface ON" if self.follow else ""), HINTS[self.tool]))
        self.hud.adjustSize()

    def msg(self, s):
        self.status.emit(s)

    def snapshot(self):
        return (self.mesh.copy(), self.mode, set(self.sel))

    def push_undo(self):
        self.undo.append(self.snapshot())
        del self.undo[:-100]
        self.sel_trail = []
        self.redo.clear()
        self.pristine = False

    def restore(self, snap):
        self.mesh, self.mode, self.sel = snap[0].copy(), snap[1], set(snap[2])
        self.sel_trail = []
        self.drag = None
        self.changed()
        self.selection_changed.emit()
        self.update_hud()

    def do_undo(self):
        if self.pending:                      # 1) un-pick the last cloud point
            self.pending.pop()
            self.update_hud(); self.update()
            return
        trail = [i for i in self.sel_trail if i in self.sel]
        if trail:                             # 2) un-select the last clicked element
            self.sel.discard(trail[-1])
            self.sel_trail = trail[:-1]
            self._arr = None
            self.update(); self.selection_changed.emit()
            return
        if self.undo:                         # 3) otherwise: normal undo, unchanged
            self.redo.append(self.snapshot())
            self.restore(self.undo.pop())

    def do_redo(self):
        if self.redo:
            self.undo.append(self.snapshot())
            self.restore(self.redo.pop())

    def refit_solid(self):
        """Live preview of the best-fit solid for the points picked so far."""
        self.solid_preview, self.solid_info = None, ""
        if self.tool != "solid" or not self.pending:
            return
        need = MIN_POINTS[self.solid_shape]
        if len(self.pending) < need:
            self.solid_info = ", need %d+" % need
            return
        try:
            V, F, info = fit_solid(self.solid_shape, np.array([p[2] for p in self.pending]), self.solid_seg)
            self.solid_preview, self.solid_info = (V, F), ", " + info
        except Exception as ex:
            self.solid_info = ", no fit (%s)" % ex

    def finish_solid(self):
        if self.solid_preview is None:
            return self.msg("Pick more points first")
        V, F = self.solid_preview
        self.push_undo()
        base = len(self.mesh.V)
        self.mesh.V = np.vstack([self.mesh.V, V])
        ids = [self.mesh.add_face([base + i for i in f]) for f in F]
        self.pending = []
        self.mode, self.sel, self.tool = "f", set(ids), "move"
        self.changed(); self.update_hud(); self.selection_changed.emit(); self.state_changed.emit()
        self.msg("%s created. Drag it, or select faces / vertices to move and extrude (E)" % SHAPE_NAMES[self.solid_shape])

    def get_surface(self):
        if self.surf is None:
            if self.cloud_full is None:
                self.msg("Load a point cloud first")
                return None
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.msg("Building surface index...")
                QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
                self.surf = SurfaceIndex(self.cloud_full)
            except ImportError as ex:
                self.msg(str(ex))
                return None
            finally:
                QApplication.restoreOverrideCursor()
        return self.surf

    def set_follow(self, on):
        self.follow = bool(on)
        if self.follow and self.get_surface() is None:
            self.follow = False
        self.state_changed.emit()
        self.update_hud()
        if self.drag is not None:   # re-evaluate the move in progress
            self.apply_drag(self.last[0], self.last[1], False)
        self.msg("Follow cloud surface: %s" % ("ON" if self.follow else "OFF"))

    def op_snap_surface(self):
        vs = self.sel_verts()
        if not vs:
            return self.need("something")
        si = self.get_surface()
        if si is None:
            return
        self.push_undo()
        self.mesh.V[vs] = si.project(self.mesh.V[vs], eye=self.cam.eye(), offset=self.surf_offset, k=self.surf_k)
        self.changed()
        self.selection_changed.emit()

    def set_tool(self, t):
        self.tool = t
        self.pending.clear()
        self.drag = None
        self.update_hud()
        self.update()

    # --------------------------------------------------------------- selection
    def sel_verts(self):
        F = self.mesh.F
        if self.mode == "v":
            return sorted(i for i in self.sel if i < len(self.mesh.V))
        if self.mode == "e":
            return sorted({v for e in self.sel for v in e})
        return sorted({v for f in self.sel if f < len(F) for v in F[f]})

    def sel_faces(self):
        if self.mode == "f":
            return sorted(f for f in self.sel if f < len(self.mesh.F))
        vs = set(self.sel_verts())
        return [i for i, f in enumerate(self.mesh.F) if vs and vs.issuperset(f)] if vs else []

    def set_mode(self, m):
        vs = set(self.sel_verts())
        self.sel_trail = []
        self.mode = m
        if m == "v":
            self.sel = vs
        elif m == "e":
            self.sel = {e for e in self.mesh.edge_faces() if e[0] in vs and e[1] in vs}
        else:
            self.sel = {i for i, f in enumerate(self.mesh.F) if vs.issuperset(f)}
        self._arr = None
        self.update_hud()
        self.update()
        self.selection_changed.emit()

    def selection_normal(self):
        fs = self.sel_faces()
        if not fs:
            vs = set(self.sel_verts())
            fs = [i for i, f in enumerate(self.mesh.F) if vs & set(f)]
        if not fs:
            return None
        w = [self.mesh.face_area(i) for i in fs]
        n = sum(self.mesh.face_normal(i) * a for i, a in zip(fs, w))
        l = np.linalg.norm(n)
        return n / l if l > 1e-6 * max(sum(w), 1e-12) else None

    def pick_element(self, x, y):
        m, r = self.mesh, self.snap_radius
        if self.mode == "v":
            return pick_mesh_vertex(self.cam, m.V, x, y, r)
        if self.mode == "e":
            edges = list(m.edge_faces().keys())
            i = pick_mesh_edge(self.cam, m.V, edges, x, y, r)
            return edges[i] if i is not None else None
        return pick_mesh_face(self.cam, m, x, y)

    def apply_pick(self, hit, additive):
        if hit is None:
            if not additive:
                self.sel, self.sel_trail = set(), []
        elif additive:
            self.sel ^= {hit}
            if hit in self.sel:
                self.sel_trail.append(hit)
            else:
                self.sel_trail = [i for i in self.sel_trail if i != hit]
        else:
            self.sel, self.sel_trail = {hit}, [hit]
        self._arr = None
        self.update()
        self.selection_changed.emit()

    def find_snap(self, x, y, full=False, cloud_only=False):
        if not cloud_only and len(self.mesh.V):
            i = pick_mesh_vertex(self.cam, self.mesh.V, x, y, self.snap_radius)
            if i is not None:
                return ("mesh", i, self.mesh.V[i].copy())
        if self.cloud_full is not None:
            arr = self.cloud_full if full else self.cloud_snap
            j = pick_cloud_point(self.cam, arr, x, y, self.snap_radius)
            if j is not None:
                return ("cloud", j, arr[j].astype(float))
        return None

    # ----------------------------------------------------------------- gizmo
    def gizmo(self):
        if self.tool != "move":
            return None
        vs = self.sel_verts()
        if not vs:
            return None
        c = self.mesh.V[vs].mean(0)
        w = float(self.cam.project(c[None, :])[3][0])
        if w <= self.cam.near:
            return None
        L = self.cam.world_per_pixel(w) * 90.0
        axes = [("X", np.array([1.0, 0, 0]), (0.95, 0.25, 0.25)),
                ("Y", np.array([0, 1.0, 0]), (0.3, 0.85, 0.3)),
                ("Z", np.array([0, 0, 1.0]), (0.35, 0.55, 1.0))]
        n = self.selection_normal()
        if n is not None:
            axes.append(("N", n, (1.0, 0.9, 0.2)))
        return c, L, axes

    def pick_gizmo(self, x, y):
        g = self.gizmo()
        if g is None:
            return None
        c, L, axes = g
        p = np.array([x, y])
        best, bd = None, 10.0
        for name, a, col in axes:
            sx, sy, _, w = self.cam.project(np.array([c, c + a * L]))
            if (w <= self.cam.near).any():
                continue
            A, B = np.array([sx[0], sy[0]]), np.array([sx[1], sy[1]])
            AB = B - A
            t = np.clip(((p - A) @ AB) / max(AB @ AB, 1e-9), 0, 1)
            d = np.linalg.norm(A + AB * t - p)
            if d < bd:
                best, bd = (name, a), d
        return best

    # ----------------------------------------------------------------- cloud
    def load_cloud(self, paths, append=False):
        """Load one or more files into a single cloud. append=True keeps what is loaded."""
        if isinstance(paths, str):
            paths = [paths]
        arrays, errors = [], []
        for p in paths:
            try:
                a = load_xyz(p)
                if len(a) == 0:
                    raise ValueError("no points")
                arrays.append(a)
            except Exception as ex:
                errors.append("%s: %s" % (os.path.basename(p), ex))
        if not arrays:
            raise ValueError("\n".join(errors) or "No files given")
        nfiles = len(arrays)
        if append and self.cloud_full is not None:
            arrays.insert(0, self.cloud_full.astype(np.float64) + self.origin)
        else:
            append = False
        pts = np.concatenate(arrays)
        self.surf = None
        lo, hi = pts.min(0), pts.max(0)
        old = self.origin.copy()
        self.origin = (lo + hi) / 2.0
        self.mesh.V = self.mesh.V + (old - self.origin)   # keep existing model aligned
        local = (pts - self.origin).astype(np.float32)
        n = len(local)
        stride = max(1, int(np.ceil(n / MAX_DISPLAY)))
        self.cloud_full = local
        self.cloud_disp = np.ascontiguousarray(local[::stride])
        s2 = max(1, int(np.ceil(n / MAX_SNAP)))
        self.cloud_snap = local if s2 == 1 else np.ascontiguousarray(local[::s2])
        self.cloud_cols = height_colors(self.cloud_disp[:, 2])
        self.n_disp = len(self.cloud_disp)
        self._cloud_dirty = True
        self.grid_z = float(local[:, 2].min())
        ext = hi - lo
        if self.pristine:
            self.mesh = Mesh.plane((0, 0, 0), max(0.4 * max(ext[0], ext[1]), 1e-3))
            self.sel = set()
        if not append:
            self.cam.frame((0, 0, 0), 0.5 * float(np.linalg.norm(ext)))
        self._arr = None
        self.update()
        self.selection_changed.emit()
        self.msg("Loaded %d file(s): %s points total (drawing %s)"
                 % (nfiles, format(n, ","), format(self.n_disp, ",")))
        return errors

    def frame_all(self):
        if self.cloud_full is not None:
            lo, hi = self.cloud_full.min(0), self.cloud_full.max(0)
        else:
            lo, hi = self.mesh.V.min(0), self.mesh.V.max(0)
        self.cam.frame((lo + hi) / 2, 0.5 * float(np.linalg.norm(hi - lo)))
        self.update()

    def view_preset(self, yaw_deg, pitch_deg):
        self.cam.yaw, self.cam.pitch = np.radians(yaw_deg), np.radians(pitch_deg)
        self.update()

    # ---------------------------------------------------------------- editing
    def need(self, what):
        self.msg("Select %s first" % what)

    def op_subdivide(self):
        fs = self.sel_faces()
        if not fs:
            return self.need("one or more faces (mode 3)")
        self.push_undo()
        new = self.mesh.subdivide(fs)
        self.mode, self.sel = "f", set(new)
        self.changed(); self.update_hud(); self.selection_changed.emit()

    def op_extrude(self):
        if self.mode == "e" and self.sel:
            self.push_undo()
            ne, nf = self.mesh.extrude_edges(sorted(self.sel))
            self.sel = set(ne)
        else:
            fs = self.sel_faces()
            if not fs:
                return self.need("faces (mode 3) or edges (mode 2)")
            self.push_undo()
            area = np.mean([self.mesh.face_area(i) for i in fs])
            self.mesh.extrude_faces(fs, 0.35 * np.sqrt(max(area, 1e-12)))
            self.mode, self.sel = "f", set(fs)
        self.tool = "move"
        self.changed(); self.update_hud(); self.selection_changed.emit()
        self.state_changed.emit()
        self.msg("Extruded - drag the gizmo (N = along normal) to shape it")

    def op_fill(self):
        vs = self.sel_verts()
        if len(vs) < 3:
            return self.need("3 or more vertices")
        order = order_polygon(self.mesh.V[vs])
        idx = self.mesh.orient_toward([vs[i] for i in order], -self.cam.forward())
        self.push_undo()
        fi = self.mesh.add_face(idx)
        self.mode, self.sel = "f", {fi}
        self.changed(); self.update_hud(); self.selection_changed.emit()

    def op_weld(self):
        vs = self.sel_verts()
        if len(vs) < 2:
            return self.need("2 or more vertices")
        self.push_undo()
        k = self.mesh.weld(vs)
        self.mode, self.sel = "v", {k}
        self.changed(); self.update_hud(); self.selection_changed.emit()

    def op_delete(self):
        if not self.sel:
            return self.need("something")
        self.push_undo()
        if self.mode == "v":
            self.mesh.remove_vertices(self.sel)
        elif self.mode == "e":
            ef = self.mesh.edge_faces()
            self.mesh.delete_faces({f for e in self.sel for f in ef.get(e, [])})
        else:
            self.mesh.delete_faces(self.sel)
        self.sel = set()
        self.changed(); self.selection_changed.emit()

    def op_flip(self):
        fs = self.sel_faces()
        if not fs:
            return self.need("faces")
        self.push_undo()
        self.mesh.flip_faces(fs)
        self.changed()

    def op_add_plane(self):
        self.push_undo()
        t = self.cam.target
        size = 0.15 * self.cam.radius * 2
        base = len(self.mesh.V)
        p = Mesh.plane(t, size)
        self.mesh.V = np.vstack([self.mesh.V, p.V])
        fi = self.mesh.add_face([base, base + 1, base + 2, base + 3])
        self.mode, self.sel = "f", {fi}
        self.changed(); self.update_hud(); self.selection_changed.emit()

    def set_selection_centroid(self, pos):
        vs = self.sel_verts()
        if not vs:
            return
        self.push_undo()
        self.mesh.V[vs] += np.asarray(pos, float) - self.mesh.V[vs].mean(0)
        self.changed()

    # ------------------------------------------------------------------ files
    def save(self, path):
        save_model(path, self.mesh, self.origin)

    def open_model(self, path):
        m, org = load_model(path)
        self.push_undo()
        if self.cloud_full is None:
            self.origin = org
            self.mesh = m
        else:
            m.V = m.V + (org - self.origin)
            self.mesh = m
        self.sel = set()
        self.pristine = False
        self.changed(); self.selection_changed.emit()

    def export_obj(self, path, original_coords):
        self.mesh.export_obj(path, self.origin if original_coords else np.zeros(3))

    # ---------------------------------------------------------- face from points
    def finish_points_face(self):
        P = self.pending
        if len(P) < 3:
            return self.msg("Need at least 3 points")
        pos = np.array([p[2] for p in P])
        if len({tuple(np.round(q, 9)) for q in pos}) < len(pos):
            return self.msg("Duplicate points - pick distinct points")
        self.push_undo()
        idx = [p[1] if p[0] == "mesh" else self.mesh.add_vertex(p[2]) for p in P]
        order = order_polygon(pos)
        idx = self.mesh.orient_toward([idx[i] for i in order], -self.cam.forward())
        fi = self.mesh.add_face(idx)
        self.pending = []
        self.mode, self.sel = "f", {fi}
        self.changed(); self.update_hud(); self.selection_changed.emit()
        self.msg("Face created from %d points" % len(idx))

    # ------------------------------------------------------------------ drag
    def begin_drag(self, x, y, axis=None, grab=None):
        vs = self.sel_verts()
        if not vs:
            return
        orig = self.mesh.V[vs].copy()
        c = orig.mean(0)
        o, d = self.cam.ray(x, y)
        fwd = self.cam.forward()
        self.drag = dict(verts=vs, orig=orig, c=c, o0=o, r0=d, fwd=fwd, axis=axis,
                         p0=ray_plane(o, d, c, fwd), grab=c if grab is None else grab,
                         moved=False, pushed=False, press=(x, y))

    def apply_drag(self, x, y, ctrl):
        dr = self.drag
        if dr is None:
            return
        if not dr["moved"]:
            if np.hypot(x - dr["press"][0], y - dr["press"][1]) < 3:
                return
            dr["moved"] = True
        if not dr["pushed"]:
            self.undo.append((self.mesh.copy(), self.mode, set(self.sel)))
            self.redo.clear()
            self.pristine = False
            self.sel_trail = []
            dr["pushed"] = True
        o, r = self.cam.ray(x, y)
        c = dr["c"]
        axis = dr["axis"] if dr["axis"] is not None else self.axis_lock
        self.snap = None
        if axis is not None:
            s1 = ray_axis_param(o, r, c, axis)
            s0 = ray_axis_param(dr["o0"], dr["r0"], c, axis)
            if s1 is None or s0 is None:
                return
            delta = (s1 - s0) * axis
        else:
            p = ray_plane(o, r, c, dr["fwd"])
            if p is None or dr["p0"] is None:
                return
            delta = p - dr["p0"]
            if ctrl:
                sn = self.find_snap(x, y, cloud_only=True)
                if sn is not None:
                    delta = sn[2] - dr["grab"]
                    self.snap = sn
        new = dr["orig"] + delta
        if self.follow and axis is None and self.snap is None:   # free drags only
            si = self.get_surface()
            if si is not None:
                new = si.project(new, eye=self.cam.eye(), use_ray=True,
                                 offset=self.surf_offset, k=self.surf_k)
        self.mesh.V[dr["verts"]] = new
        self.changed()
        self.selection_changed.emit()

    def cancel_drag(self):
        dr = self.drag
        if dr is None:
            return
        self.mesh.V[dr["verts"]] = dr["orig"]
        if dr["pushed"] and self.undo:
            self.undo.pop()
        self.drag = None
        self.axis_lock = None
        self.changed()
        self.selection_changed.emit()

    # ------------------------------------------------------------------ input
    def mousePressEvent(self, e):
        self.setFocus()
        x, y = e.position().x(), e.position().y()
        self.last = (x, y)
        b, m = e.button(), e.modifiers()
        shift = bool(m & Qt.KeyboardModifier.ShiftModifier)
        if b == Qt.MouseButton.MiddleButton or (b == Qt.MouseButton.LeftButton and m & Qt.KeyboardModifier.AltModifier):
            self.nav = "pan" if shift else "orbit"
        elif b == Qt.MouseButton.RightButton:
            self.nav = "pan"
        elif b == Qt.MouseButton.LeftButton:
            if self.tool == "select":
                self.apply_pick(self.pick_element(x, y), shift)
            elif self.tool == "move":
                h = self.pick_gizmo(x, y)
                if h is not None:
                    self.begin_drag(x, y, axis=h[1])
                    return
                hit = self.pick_element(x, y)
                if hit is None:
                    self.apply_pick(None, shift)
                    return
                if shift or hit not in self.sel:
                    self.apply_pick(hit, shift)
                grab = self.mesh.V[hit].copy() if self.mode == "v" and hit in self.sel else None
                self.begin_drag(x, y, grab=grab)
            elif self.tool == "points":
                sn = self.find_snap(x, y, full=True, cloud_only=bool(m & Qt.KeyboardModifier.ControlModifier))
                if sn is None:
                    return self.msg("No point under cursor")
                self.pending.append(sn)
                self.update_hud(); self.update()
                if self.auto_n and len(self.pending) >= self.auto_n:
                    self.finish_points_face()
            elif self.tool == "solid":
                sn = self.find_snap(x, y, full=True, cloud_only=True)
                if sn is None:
                    return self.msg("No cloud point under cursor")
                self.pending.append(sn)
                self.update_hud(); self.update()

    def mouseMoveEvent(self, e):
        x, y = e.position().x(), e.position().y()
        dx, dy = x - self.last[0], y - self.last[1]
        self.last = (x, y)
        if self.nav == "orbit":
            self.cam.orbit(dx, dy); self.update()
        elif self.nav == "pan":
            self.cam.pan(dx, dy); self.update()
        elif self.drag is not None:
            self.apply_drag(x, y, bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier))
        else:
            old = (self.snap, self.hover_axis)
            self.snap, self.hover_axis = None, None
            if self.tool in ("points", "solid"):
                self.snap = self.find_snap(x, y, cloud_only=self.tool == "solid" or bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier))
            elif self.tool == "move":
                h = self.pick_gizmo(x, y)
                self.hover_axis = h[0] if h else None
            if self.tool != "select":
                self.update()

    def mouseReleaseEvent(self, e):
        self.nav = None
        if self.drag is not None:
            self.drag = None
            self.axis_lock = None
            self.snap = None
            self.update()
            self.selection_changed.emit()

    def wheelEvent(self, e):
        p = e.position()
        self.cam.zoom(e.angleDelta().y() / 120.0, p.x(), p.y())
        self.update()

    def keyPressEvent(self, e):
        k = e.key()
        K = Qt.Key
        if k in (K.Key_X, K.Key_Y, K.Key_Z) and self.drag is not None and self.drag["axis"] is None:
            a = {K.Key_X: (1.0, 0, 0), K.Key_Y: (0, 1.0, 0), K.Key_Z: (0, 0, 1.0)}[k]
            a = np.array(a)
            self.axis_lock = None if (self.axis_lock is not None and np.allclose(self.axis_lock, a)) else a
            self.apply_drag(self.last[0], self.last[1], False)
        elif k in (K.Key_Return, K.Key_Enter) and self.tool == "points":
            self.finish_points_face()
        elif k in (K.Key_Return, K.Key_Enter) and self.tool == "solid":
            self.finish_solid()
        elif k == K.Key_Backspace and self.pending:
            self.pending.pop(); self.update_hud(); self.update()
        elif k == K.Key_Escape:
            if self.drag is not None:
                self.cancel_drag()
            elif self.pending:
                self.pending = []; self.update_hud(); self.update()
            else:
                self.apply_pick(None, False)
        else:
            super().keyPressEvent(e)

    # ------------------------------------------------------------- rendering
    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glEnable(GL_NORMALIZE)

    def _upload_cloud(self):
        if self.vbo is not None:
            glDeleteBuffers(2, self.vbo)
            self.vbo = None
        if self.cloud_disp is not None:
            b = glGenBuffers(2)
            glBindBuffer(GL_ARRAY_BUFFER, b[0])
            glBufferData(GL_ARRAY_BUFFER, self.cloud_disp.nbytes, self.cloud_disp, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, b[1])
            glBufferData(GL_ARRAY_BUFFER, self.cloud_cols.nbytes, self.cloud_cols, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            self.vbo = b
        self._cloud_dirty = False

    def _build_arrays(self):
        m = self.mesh
        a = {}
        tri, tf, fn = m.triangulate()
        selF = set(self.sel_faces())
        if len(tri):
            a["tri_pos"] = np.ascontiguousarray(tri.reshape(-1, 3), np.float32)
            a["tri_nrm"] = np.ascontiguousarray(np.repeat(fn[tf], 3, axis=0), np.float32)
            issel = np.isin(tf, list(selF)) if selF else np.zeros(len(tf), bool)
            base = np.array([0.36, 0.62, 0.92, self.face_alpha], np.float32)
            selc = np.array([1.0, 0.62, 0.2, self.face_alpha], np.float32)
            a["tri_col"] = np.ascontiguousarray(np.repeat(np.where(issel[:, None], selc, base), 3, axis=0), np.float32)
        ef = m.edge_faces()
        keys = list(ef.keys())
        if keys:
            if self.mode == "e":
                selE = set(self.sel)
            elif self.mode == "f":
                selE = {k for k, fs in ef.items() if any(f in selF for f in fs)}
            else:
                selE = set()
            E = np.array(keys)
            a["line_pos"] = np.ascontiguousarray(m.V[E].reshape(-1, 3), np.float32)
            cols = np.array([(1.0, 0.55, 0.1) if k in selE else (0.88, 0.93, 1.0) for k in keys], np.float32)
            a["line_col"] = np.ascontiguousarray(np.repeat(cols, 2, axis=0))
        if len(m.V):
            a["vert_pos"] = np.ascontiguousarray(m.V, np.float32)
            sv = self.sel_verts()
            if sv:
                a["selv_pos"] = np.ascontiguousarray(m.V[sv], np.float32)
        return a

    def _pts(self, pos, size, color):
        glPointSize(size * self.devicePixelRatioF())
        glColor3f(*color)
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, pos)
        glDrawArrays(GL_POINTS, 0, len(pos))
        glDisableClientState(GL_VERTEX_ARRAY)

    def _draw_grid(self):
        r = max(self.cam.radius, 1e-3)
        step = 10 ** np.floor(np.log10(r / 4))
        n = int(np.ceil(r * 1.5 / step))
        z, ext = self.grid_z, n * step
        glLineWidth(1.0)
        glBegin(GL_LINES)
        for i in range(-n, n + 1):
            c = (0.3, 0.3, 0.34) if i else (0.45, 0.45, 0.5)
            glColor3f(*c)
            glVertex3f(i * step, -ext, z); glVertex3f(i * step, ext, z)
            glVertex3f(-ext, i * step, z); glVertex3f(ext, i * step, z)
        glEnd()

    def _draw_cloud(self):
        if self.vbo is None:
            return
        glPointSize(self.point_size * self.devicePixelRatioF())
        glEnableClientState(GL_VERTEX_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo[0])
        glVertexPointer(3, GL_FLOAT, 0, ctypes.c_void_p(0))
        if self.use_height_colors:
            glEnableClientState(GL_COLOR_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo[1])
            glColorPointer(3, GL_UNSIGNED_BYTE, 0, ctypes.c_void_p(0))
        else:
            glColor3f(0.78, 0.8, 0.82)
        glDrawArrays(GL_POINTS, 0, self.n_disp)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    def _draw_faces(self, a):
        if "tri_pos" not in a:
            return
        glEnable(GL_LIGHTING); glEnable(GL_LIGHT0)
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.35, 0.35, 0.35, 1.0))
        glPushMatrix(); glLoadIdentity()
        glLightfv(GL_LIGHT0, GL_POSITION, (0.3, 0.5, 1.0, 0.0))
        glPopMatrix()
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_POLYGON_OFFSET_FILL); glPolygonOffset(1.0, 1.0)
        glDepthMask(GL_TRUE if self.face_alpha >= 0.99 else GL_FALSE)
        for st in (GL_VERTEX_ARRAY, GL_NORMAL_ARRAY, GL_COLOR_ARRAY):
            glEnableClientState(st)
        glVertexPointer(3, GL_FLOAT, 0, a["tri_pos"])
        glNormalPointer(GL_FLOAT, 0, a["tri_nrm"])
        glColorPointer(4, GL_FLOAT, 0, a["tri_col"])
        glDrawArrays(GL_TRIANGLES, 0, len(a["tri_pos"]))
        for st in (GL_VERTEX_ARRAY, GL_NORMAL_ARRAY, GL_COLOR_ARRAY):
            glDisableClientState(st)
        glDepthMask(GL_TRUE)
        glDisable(GL_POLYGON_OFFSET_FILL); glDisable(GL_COLOR_MATERIAL)
        glDisable(GL_LIGHTING); glDisable(GL_BLEND)

    def _draw_overlay(self, a):
        glDisable(GL_DEPTH_TEST)
        dpr = self.devicePixelRatioF()
        if "line_pos" in a:
            glLineWidth(1.6 * dpr)
            glEnableClientState(GL_VERTEX_ARRAY); glEnableClientState(GL_COLOR_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, a["line_pos"])
            glColorPointer(3, GL_FLOAT, 0, a["line_col"])
            glDrawArrays(GL_LINES, 0, len(a["line_pos"]))
            glDisableClientState(GL_VERTEX_ARRAY); glDisableClientState(GL_COLOR_ARRAY)
        if "vert_pos" in a:
            self._pts(a["vert_pos"], 6, (0.95, 0.97, 1.0))
        if "selv_pos" in a:
            self._pts(a["selv_pos"], 9, (1.0, 0.55, 0.1))
        # gizmo
        g = self.gizmo()
        if g is not None:
            c, L, axes = g
            glLineWidth(3.0 * dpr)
            for name, v, col in axes:
                hot = (name == self.hover_axis) or (self.drag is not None and self.drag["axis"] is not None and np.allclose(self.drag["axis"], v))
                glColor3f(*((1, 1, 1) if hot else col))
                glBegin(GL_LINES); glVertex3f(*c); glVertex3f(*(c + v * L)); glEnd()
                glPointSize(11 * dpr)
                glBegin(GL_POINTS); glVertex3f(*(c + v * L)); glEnd()
            if self.axis_lock is not None:   # long guide line for X/Y/Z lock
                glColor3f(1, 1, 1); glLineWidth(1.0)
                glBegin(GL_LINES); glVertex3f(*(c - self.axis_lock * L * 40)); glVertex3f(*(c + self.axis_lock * L * 40)); glEnd()
        # pending points for "face from cloud points"
        if self.pending:
            pos = np.array([p[2] for p in self.pending], np.float32)
            glColor3f(1, 0.9, 0.2); glLineWidth(2.0 * dpr)
            if self.tool == "points":
                glBegin(GL_LINE_STRIP)
                for q in pos:
                    glVertex3f(*q)
                glEnd()
            self._pts(pos, 12, (1.0, 0.9, 0.2))
        if self.solid_preview is not None:   # wireframe of the best-fit solid
            pv, pf = self.solid_preview
            keys = list(Mesh(pv, pf).edge_faces().keys())
            pos = np.ascontiguousarray(pv[np.array(keys)].reshape(-1, 3), np.float32)
            glColor3f(0.4, 1.0, 0.6); glLineWidth(1.5 * dpr)
            glEnableClientState(GL_VERTEX_ARRAY)
            glVertexPointer(3, GL_FLOAT, 0, pos)
            glDrawArrays(GL_LINES, 0, len(pos))
            glDisableClientState(GL_VERTEX_ARRAY)
        if self.snap is not None:
            self._pts(np.array([self.snap[2]], np.float32), 15, (0.3, 1.0, 0.4) if self.snap[0] == "mesh" else (1.0, 0.95, 0.2))
        glEnable(GL_DEPTH_TEST)

    def paintGL(self):
        if self._gl_failed:
            return
        try:
            self.cam.width, self.cam.height = max(self.width(), 1), max(self.height(), 1)
            dpr = self.devicePixelRatioF()
            glViewport(0, 0, int(self.width() * dpr), int(self.height() * dpr))
            glClearColor(0.13, 0.14, 0.16, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            if self._cloud_dirty:
                self._upload_cloud()
            glMatrixMode(GL_PROJECTION)
            glLoadMatrixd(np.ascontiguousarray(self.cam.proj_matrix().T))
            glMatrixMode(GL_MODELVIEW)
            glLoadMatrixd(np.ascontiguousarray(self.cam.view_matrix().T))
            glDisable(GL_LIGHTING)
            self._draw_grid()
            self._draw_cloud()
            if self._arr is None:
                self._arr = self._build_arrays()
            self._draw_faces(self._arr)
            self._draw_overlay(self._arr)
        except Exception:
            traceback.print_exc()
            self._gl_failed = True


# ============================================================== main window ==
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Point Cloud Modeler")
        self.resize(1400, 860)
        self.v = Viewer()
        self.setCentralWidget(self.v)
        self.setAcceptDrops(True)
        self.v.status.connect(lambda s: self.statusBar().showMessage(s, 8000))
        self._busy = False
        self._build_actions()
        self._build_dock()
        self.v.selection_changed.connect(self.refresh_coords)
        self.v.state_changed.connect(self.sync_checked)
        self.statusBar().showMessage("File > Open point cloud (Ctrl+O) to begin")

    def act(self, text, key, slot, menu, tip="", checkable=False, group=None):
        a = QAction(text, self)
        if key:
            a.setShortcut(QKeySequence(key))
        a.setToolTip(tip or text)
        a.setCheckable(checkable)
        a.triggered.connect(lambda *_: slot())
        menu.addAction(a)
        if group:
            group.addAction(a)
        return a

    def _build_actions(self):
        mb = self.menuBar()
        f, ed, t, vw = mb.addMenu("&File"), mb.addMenu("&Edit"), mb.addMenu("&Tools"), mb.addMenu("&View")
        self.act("Open point clouds…", "Ctrl+O", self.open_cloud, f)
        self.act("Add point clouds…", "Ctrl+Shift+O", lambda: self.open_cloud(append=True), f)
        self.act("Open model…", None, self.open_model, f)
        self.act("Save model…", "Ctrl+S", self.save_model, f)
        f.addSeparator()
        self.act("Export OBJ (original coordinates)…", None, lambda: self.export(True), f)
        self.act("Export OBJ (cloud-centered)…", None, lambda: self.export(False), f)
        f.addSeparator()
        self.act("Quit", "Ctrl+Q", self.close, f)

        self.act("Undo", "Ctrl+Z", self.v.do_undo, ed)
        self.act("Redo", "Ctrl+Y", self.v.do_redo, ed)
        ed.addSeparator()
        self.act("Add plane", "Shift+A", self.v.op_add_plane, ed)
        self.act("Subdivide faces", "S", self.v.op_subdivide, ed)
        self.act("Extrude faces / edges", "E", self.v.op_extrude, ed)
        self.act("Fill face from selected vertices", "F", self.v.op_fill, ed)
        self.act("Weld selected vertices", "M", self.v.op_weld, ed)
        self.act("Flip normals", "N", self.v.op_flip, ed)
        self.act("Snap selection to cloud surface", "Shift+S", self.v.op_snap_surface, ed)
        self.act("Delete selection", "Delete", self.v.op_delete, ed)

        tg = QActionGroup(self)
        self.tool_acts = {}
        for key, name, sc in (("select", "Select", "Q"), ("move", "Move / Transform", "W"), ("points", "Face from cloud points", "P"),
                        ("solid", "Solid from points (best fit)", "B")):
            self.tool_acts[key] = self.act(name, sc, lambda k=key: self.v.set_tool(k), t, checkable=True, group=tg)
        self.tool_acts["move"].setChecked(True)
        t.addSeparator()
        mg = QActionGroup(self)
        self.mode_acts = {}
        for key, name, sc in (("v", "Vertex mode", "1"), ("e", "Edge mode", "2"), ("f", "Face mode", "3")):
            self.mode_acts[key] = self.act(name, sc, lambda k=key: self.v.set_mode(k), t, checkable=True, group=mg)
        self.mode_acts["v"].setChecked(True)
        t.addSeparator()
        self.follow_act = self.act("Follow cloud surface while moving", "C",
                                   lambda: self.v.set_follow(self.follow_act.isChecked()), t, checkable=True)
        sm = t.addMenu("Solid shape")
        for key in SHAPES:
            self.act(SHAPE_NAMES[key], None, lambda k=key: self.pick_solid(k), sm)

        self.act("Frame all", "Home", self.v.frame_all, vw)
        self.act("Top", "Ctrl+1", lambda: self.v.view_preset(-90, 89.5), vw)
        self.act("Front", "Ctrl+2", lambda: self.v.view_preset(-90, 0), vw)
        self.act("Right", "Ctrl+3", lambda: self.v.view_preset(0, 0), vw)

    def pick_solid(self, key):
        self.v.solid_shape = key
        self.shape_cb.setCurrentIndex(SHAPES.index(key))
        if self.v.tool != "solid":
            self.v.set_tool("solid")
            self.sync_checked()
        self.v.update_hud()
        self.v.update()

    def sync_checked(self):
        self.tool_acts[self.v.tool].setChecked(True)
        self.mode_acts[self.v.mode].setChecked(True)
        self.follow_act.setChecked(self.v.follow)
        self.follow_cb.blockSignals(True)
        self.follow_cb.setChecked(self.v.follow)
        self.follow_cb.blockSignals(False)

    def _build_dock(self):
        dock = QDockWidget("Properties", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        w = QWidget(); lay = QVBoxLayout(w)

        g = QGroupBox("Point cloud"); gl = QVBoxLayout(g)
        ps = QDoubleSpinBox(); ps.setRange(1, 20); ps.setValue(self.v.point_size)
        ps.valueChanged.connect(lambda x: (setattr(self.v, "point_size", x), self.v.update()))
        hc = QCheckBox("Color by height"); hc.setChecked(True)
        hc.toggled.connect(lambda x: (setattr(self.v, "use_height_colors", x), self.v.update()))
        sr = QSpinBox(); sr.setRange(3, 60); sr.setValue(int(self.v.snap_radius))
        sr.valueChanged.connect(lambda x: setattr(self.v, "snap_radius", float(x)))
        for lbl, wd in (("Point size", ps), ("Pick radius (px)", sr)):
            r = QHBoxLayout(); r.addWidget(QLabel(lbl)); r.addWidget(wd); gl.addLayout(r)
        gl.addWidget(hc); lay.addWidget(g)

        g = QGroupBox("Surface"); gl = QVBoxLayout(g)
        sl = QSlider(Qt.Orientation.Horizontal); sl.setRange(10, 100); sl.setValue(60)
        sl.valueChanged.connect(lambda x: (setattr(self.v, "face_alpha", x / 100.0), setattr(self.v, "_arr", None), self.v.update()))
        r = QHBoxLayout(); r.addWidget(QLabel("Opacity")); r.addWidget(sl); gl.addLayout(r)
        an = QSpinBox(); an.setRange(0, 32); an.setValue(self.v.auto_n); an.setSpecialValueText("manual")
        an.valueChanged.connect(lambda x: (setattr(self.v, "auto_n", x), self.v.update_hud()))
        r = QHBoxLayout(); r.addWidget(QLabel("Point-face: auto-finish at")); r.addWidget(an); gl.addLayout(r)
        lay.addWidget(g)

        g = QGroupBox("Solids (best fit)"); gl = QVBoxLayout(g)
        self.shape_cb = QComboBox()
        for key in SHAPES:
            self.shape_cb.addItem(SHAPE_NAMES[key], key)
        self.shape_cb.setCurrentIndex(SHAPES.index(self.v.solid_shape))
        self.shape_cb.activated.connect(lambda i: self.pick_solid(self.shape_cb.itemData(i)))
        sg = QSpinBox(); sg.setRange(8, 96); sg.setValue(self.v.solid_seg)
        sg.valueChanged.connect(lambda x: (setattr(self.v, "solid_seg", x), self.v.update_hud(), self.v.update()))
        for lbl, wd in (("Shape (B)", self.shape_cb), ("Segments", sg)):
            r = QHBoxLayout(); r.addWidget(QLabel(lbl)); r.addWidget(wd); gl.addLayout(r)
        gl.addWidget(QLabel("Click points on the object's surface,\nthen press Enter in the viewport."))
        lay.addWidget(g)

        g = QGroupBox("Surface following"); gl = QVBoxLayout(g)
        self.follow_cb = QCheckBox("Follow cloud surface while moving (C)")
        self.follow_cb.toggled.connect(lambda x: (self.follow_act.setChecked(x), self.v.set_follow(x)))
        gl.addWidget(self.follow_cb)
        so = QDoubleSpinBox(); so.setRange(-1e9, 1e9); so.setDecimals(4); so.setSingleStep(0.05)
        so.valueChanged.connect(lambda x: setattr(self.v, "surf_offset", x))
        sk = QSpinBox(); sk.setRange(4, 400); sk.setValue(self.v.surf_k)
        sk.valueChanged.connect(lambda x: setattr(self.v, "surf_k", x))
        for lbl, wd in (("Offset above surface", so), ("Neighbors (smoothing)", sk)):
            r = QHBoxLayout(); r.addWidget(QLabel(lbl)); r.addWidget(wd); gl.addLayout(r)
        lay.addWidget(g)

        g = QGroupBox("Selection position"); gl = QVBoxLayout(g)
        self.sel_lbl = QLabel("nothing selected"); gl.addWidget(self.sel_lbl)
        self.spins = []
        for ax in "XYZ":
            sp = QDoubleSpinBox(); sp.setRange(-1e12, 1e12); sp.setDecimals(4); sp.setSingleStep(0.1)
            sp.setKeyboardTracking(False); sp.valueChanged.connect(self.coord_edited)
            self.spins.append(sp)
            r = QHBoxLayout(); r.addWidget(QLabel(ax)); r.addWidget(sp); gl.addLayout(r)
        self.world_cb = QCheckBox("Show original (world) coordinates")
        self.world_cb.toggled.connect(self.refresh_coords)
        gl.addWidget(self.world_cb); lay.addWidget(g)

        g = QGroupBox("Operations"); gl = QVBoxLayout(g)
        for text, fn in (("Subdivide (S)", self.v.op_subdivide), ("Extrude (E)", self.v.op_extrude),
                         ("Fill face from verts (F)", self.v.op_fill), ("Weld verts (M)", self.v.op_weld),
                         ("Flip normals (N)", self.v.op_flip), ("Snap to surface (Shift+S)", self.v.op_snap_surface),
                         ("Delete (Del)", self.v.op_delete)):
            b = QPushButton(text); b.clicked.connect(fn); b.setFocusPolicy(Qt.FocusPolicy.NoFocus); gl.addWidget(b)
        lay.addWidget(g)
        lay.addStretch(1)
        sa = QScrollArea()                    # the panel scrolls when the window is short
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setFrameShape(QScrollArea.Shape.NoFrame)
        sa.setWidget(w)
        sa.setMinimumWidth(w.sizeHint().width() + 24)
        dock.setWidget(sa)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    # -- coordinate panel
    def _off(self):
        return self.v.origin if self.world_cb.isChecked() else np.zeros(3)

    def refresh_coords(self, *_):
        vs = self.v.sel_verts()
        self._busy = True
        for s in self.spins:
            s.setEnabled(bool(vs))
        if vs:
            c = self.v.mesh.V[vs].mean(0) + self._off()
            for s, val in zip(self.spins, c):
                s.setValue(float(val))
            self.sel_lbl.setText("%d vertex(es)%s" % (len(vs), " - showing centroid" if len(vs) > 1 else ""))
        else:
            self.sel_lbl.setText("nothing selected")
        self._busy = False

    def coord_edited(self, *_):
        if self._busy:
            return
        pos = np.array([s.value() for s in self.spins]) - self._off()
        self.v.set_selection_centroid(pos)

    # -- files
    def _guard(self, fn):
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            fn()
        except Exception as ex:
            QMessageBox.critical(self, "Error", str(ex))
        finally:
            QApplication.restoreOverrideCursor()

    def open_cloud(self, paths=None, append=False):
        if not paths:
            paths, _ = QFileDialog.getOpenFileNames(self, "Open point clouds", "", "Point clouds (*.xyz *.txt *.csv *.pts);;All files (*)")
        if isinstance(paths, str):
            paths = [paths]
        if paths:
            self._guard(lambda: self._load(paths, append))

    def _load(self, paths, append):
        errs = self.v.load_cloud(paths, append)
        if errs:
            QMessageBox.warning(self, "Some files were skipped", "\n".join(errs))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):   # dropped files are added to the scene
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile() and not u.toLocalFile().lower().endswith(".json")]
        if paths:
            self.open_cloud(paths, append=True)

    def open_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open model", "", "Model (*.json)")
        if p:
            self._guard(lambda: self.v.open_model(p))

    def save_model(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save model", "model.json", "Model (*.json)")
        if p:
            self._guard(lambda: self.v.save(p))

    def export(self, world):
        p, _ = QFileDialog.getSaveFileName(self, "Export OBJ", "model.obj", "Wavefront OBJ (*.obj)")
        if p:
            self._guard(lambda: self.v.export_obj(p, world))


def main():
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        win.open_cloud(sys.argv[1:])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
