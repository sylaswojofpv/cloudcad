"""
Point Cloud Modeler  --  simple freeform-style surface modeling over an .xyz point cloud.

    pip install PySide6 PyOpenGL numpy
    python modeler.py [optional_cloud.xyz]
"""
import os
import sys
import copy
import json
import ctypes
import traceback
from collections import Counter
import numpy as np

from PySide6.QtCore import Qt, Signal, QEventLoop, QTimer, QRectF, QPointF
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QSurfaceFormat, QPainter, QColor, QFont, QPen
from PySide6.QtWidgets import (QApplication, QMainWindow, QFileDialog, QDockWidget, QWidget,
                               QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QDoubleSpinBox,
                               QSpinBox, QCheckBox, QSlider, QPushButton, QMessageBox, QComboBox, QScrollArea,
                               QToolButton, QSizePolicy, QButtonGroup, QGridLayout, QListWidget)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import *

from mesh_core import (Mesh, Camera, load_xyz, save_model, load_model, order_polygon,
                       ray_plane, ray_axis_param, pick_cloud_point, pick_mesh_vertex,
                       pick_mesh_edge, pick_mesh_face, SurfaceIndex, frame_from_axes, pca_frame)
from solids import SHAPES, SHAPE_NAMES, MIN_POINTS, SWEEP_SHAPES, fit_solid
from booleans import shells_of, boolean_op, replace_faces, split_loose
from bodies import Registry, OP_NAMES, extract_faces, geom_sig
from bodies_panel import BodiesPanel
from reconstruct import poisson_mesh, points_in_polygon
from measure import build as meas_build, from_clicks, fit_round, selection_info, ray_segment_closest, TITLES as MEAS_TITLES

MAX_DISPLAY = 3_000_000   # points actually drawn (stride-decimated above this)
MAX_SNAP = 400_000        # points tested during hover / drag snapping
TOOLS = {"select": "Select", "move": "Move / Transform", "points": "Face from cloud points",
         "solid": "Solid (best fit)", "mesh": "Mesh from points (Poisson)", "measure": "Measure"}
MEAS_MODES = ("distance", "path", "angle", "circle", "sphere", "edge", "faceangle")
MEAS_LABELS = {"distance": "Distance (2 points)", "path": "Path length (many points)", "angle": "Angle (3 points)",
               "circle": "Diameter / radius (circle)", "sphere": "Diameter (sphere)", "edge": "Edge length",
               "faceangle": "Angle between faces"}
MEAS_AUTO = {"distance": 2, "angle": 3, "faceangle": 2}                  # picks that complete a measurement by themselves
MEAS_MIN = {"distance": 2, "path": 2, "angle": 3, "circle": 3, "sphere": 4, "faceangle": 2}
MEAS_HELP = {
    "distance": "Click two points: cloud points, shape vertices, points on edges (midpoints snap) or on faces, or the center of a measured circle / sphere. Ctrl = cloud points only.",
    "path": "Click points along the path, then press Enter. Gives the total length.",
    "angle": "Click three points - the middle one is the corner of the angle.",
    "circle": "Click 3 or more points on a circle or rim, then press Enter. A circle is fitted to all of them, and its center becomes a point you can measure from.",
    "sphere": "Click 4 or more points all around a sphere, then press Enter. A sphere is fitted to all of them, and its center becomes a point you can measure from.",
    "edge": "Click a mesh edge to get its length.",
    "faceangle": "Click two faces of a shape to get the angle between them.",
}
MEAS_SYMBOL = {"Diameter": "Ø ", "Radius": "R "}
MODES = {"v": "Vertex", "e": "Edge", "f": "Face"}
HINTS = {
    "select": "Click to pick (Shift = add/remove)",
    "move": "Drag a handle or element. X/Y/Z lock axis, Ctrl = snap to cloud, Esc cancels",
    "points": "Click cloud points (Ctrl = ignore mesh verts). Enter = make face, Backspace = undo point, Esc = cancel",
    "solid": "Click points on the object's surface - more points = better fit. Enter = create, Backspace = undo point, Esc = cancel",
    "mesh": "Drag a loop (lasso) around cloud points to build a mesh (Shift = add to selection). Change detail in the panel. Enter = add to model, Esc = clear",
}
SOLID_HINTS = {
    "plane": "Click points on the flat surface (3+). Enter = create, Backspace = undo point, Esc = cancel",
    "extrude": "Click many points all around the shape's sides (not just one face). The 2D profile and direction are fitted to all of them. Enter = create, Backspace = undo, Esc = cancel",
    "loft": "Click many points over the surface, all around and along its length. Sections are fitted to all of them. Enter = create, Backspace = undo, Esc = cancel",
}


def height_colors(z):
    t = (z - z.min()) / max(float(z.max() - z.min()), 1e-12)
    stops = np.array([0, .25, .5, .75, 1.0])
    cols = np.array([[60, 90, 255], [40, 200, 255], [80, 230, 110], [250, 230, 60], [255, 110, 60]], float)
    return np.ascontiguousarray(np.stack([np.interp(t, stops, cols[:, i]) for i in range(3)], 1).astype(np.uint8))


class MeasureOverlay(QWidget):
    """Transparent layer over the viewport that writes the measurement labels (the lines are drawn by OpenGL)."""

    def __init__(self, viewer):
        super().__init__(viewer)
        self.v = viewer
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def paintEvent(self, e):
        v = self.v
        p = QPainter(self)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            if v.meas_show:
                for i, m in enumerate(v.measures):
                    hot = i == v.meas_sel
                    self._label(p, m["res"]["anchor"], v.meas_label_lines(m["res"], hot or v.meas_details), hot, False)
            live = v.meas_live()
            if live is not None:
                self._label(p, live[0], live[1], True, True)
        finally:
            p.end()

    def _label(self, p, world, lines, hot, live):
        cam = self.v.cam
        sx, sy, _, w = cam.project(np.asarray(world, float)[None, :])
        if not np.isfinite(sx[0]) or not np.isfinite(sy[0]) or w[0] <= cam.near:
            return
        f = QFont(self.font())
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)
        fm = p.fontMetrics()
        wd = max(fm.horizontalAdvance(t) for t in lines) + 14
        ht = fm.height() * len(lines) + 8
        x, y = float(sx[0]) - wd / 2, float(sy[0]) - ht - 10
        p.setBrush(QColor(20, 22, 26, 225))
        p.setPen(QPen(QColor(255, 255, 255) if hot else (QColor(255, 220, 60) if live else QColor(255, 110, 200)), 1.5))
        p.drawRoundedRect(QRectF(x, y, wd, ht), 5, 5)
        p.setPen(QColor(240, 244, 248))
        for k, t in enumerate(lines):
            p.drawText(QPointF(x + 7, y + 4 + fm.ascent() + k * fm.height()), t)


class Viewer(QOpenGLWidget):
    status = Signal(str)
    selection_changed = Signal()
    state_changed = Signal()
    measures_changed = Signal()    # the list of measurements (or how they are shown) changed
    bodies_changed = Signal()      # the Bodies list needs a refresh

    def __init__(self):
        super().__init__()
        self._body_timer = QTimer(self)          # bodies are re-synced shortly after the mesh stops changing
        self._body_timer.setSingleShot(True)
        self._body_timer.setInterval(200)
        self._body_timer.timeout.connect(self._refresh_bodies)
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
        self.sweep_mode, self.sweep_lseg = "linear", 0       # loft / extrusion: linear or fitted, segments along the length (0 = auto)
        self.local_axes = True   # gizmo / axis lock follow the selection's orientation
        self.reg = Registry()    # bodies + boolean folders (see bodies.py)
        self.clouds = []         # loaded point clouds: dicts name / pts (local float32) / visible
        self.editing = None      # id of the body whose points are being edited
        # measurements
        self.measures = []                       # dicts: kind, pts (local coords), note, res (built numbers + lines)
        self.meas_kind, self.meas_sel, self.meas_hover = "distance", None, None
        self.meas_show, self.meas_details = True, False
        self.meas_units, self.meas_scale, self.meas_dec = "", 1.0, 3
        self.cloud_ext = None                    # size of the visible cloud
        self._edge_key, self._edge_arr = None, None
        # mesh from points (Poisson)
        self.mp_pts, self.mp_preview, self.mp_lines, self.mp_info = None, None, None, ""
        self.mp_detail, self.mp_max_tris, self.mp_trim, self.mp_orient = 8, 50_000, True, "auto"
        self.lasso, self.lasso_w, self.lasso_on, self.lasso_add = [], [], False, False
        self._mp_timer = QTimer(self)            # rebuild the preview shortly after a setting stops changing
        self._mp_timer.setSingleShot(True)
        self._mp_timer.setInterval(350)
        self._mp_timer.timeout.connect(self.build_mp_preview)
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
        self.overlay = MeasureOverlay(self)
        self.overlay.setGeometry(self.rect())
        self.hud.raise_()
        self.update_hud()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if hasattr(self, "overlay"):
            self.overlay.setGeometry(0, 0, self.width(), self.height())
            self.hud.raise_()

    # ------------------------------------------------------------ state helpers
    def changed(self):
        self.mesh.touch()
        self._arr = None
        self.update()
        self._body_timer.start()

    def _refresh_bodies(self):
        self.sync_bodies()
        self.bodies_changed.emit()

    def update_hud(self):
        extra = ""
        self.refit_solid()
        if self.tool == "points":
            extra = "  [%d/%s]" % (len(self.pending), self.auto_n or "manual")
        elif self.tool == "solid":
            extra = "  [%s: %d pts%s]" % (SHAPE_NAMES[self.solid_shape], len(self.pending), self.solid_info)
        elif self.tool == "measure":
            extra = "  [%s: %d picked]" % (self.meas_kind, len(self.pending))
        elif self.tool == "mesh":
            extra = "  [%s pts selected%s]" % (format(0 if self.mp_pts is None else len(self.mp_pts), ","),
                                              "" if self.mp_preview is None else ", %s triangles" % format(len(self.mp_preview[1]), ","))
        self.hud.setText("<b>%s</b>   |   select: %s%s<br><span style='color:#9fb0c3'>%s<br>"
                         "MMB orbit · Shift+MMB / RMB pan · wheel zoom · Home frame</span>"
                         % (TOOLS[self.tool], MODES[self.mode], extra + ("  |  follow surface ON" if self.follow else ""),
                            self.hint_text()))
        self.hud.adjustSize()

    def hint_text(self):
        if self.tool == "solid":
            return SOLID_HINTS.get(self.solid_shape, HINTS["solid"])
        if self.tool == "measure":
            return MEAS_HELP[self.meas_kind] + " Enter = finish, Backspace = undo point, Esc = cancel"
        return HINTS[self.tool]

    def msg(self, s):
        self.status.emit(s)

    def snapshot(self):
        return (self.mesh.copy(), self.mode, set(self.sel), copy.deepcopy(self.reg))

    def push_undo(self):
        self.undo.append(self.snapshot())
        del self.undo[:-100]
        self.sel_trail = []
        self.redo.clear()
        self.pristine = False

    def restore(self, snap):
        self.mesh, self.mode, self.sel = snap[0].copy(), snap[1], set(snap[2])
        self.reg = copy.deepcopy(snap[3])
        self.sel_trail = []
        self.drag = None
        self.changed()
        self.selection_changed.emit()
        self.bodies_changed.emit()
        self.update_hud()

    # The redo stack holds snapshots (mesh edits) and small dicts (a picked point / a clicked selection that was
    # taken back). Those small entries only apply while the state is exactly as the undo left it; otherwise they are dropped.
    def _pend_token(self):
        return tuple(tuple(np.round(np.asarray(p[2], float), 9)) for p in self.pending)

    def _sel_token(self):
        return (frozenset(self.sel), tuple(self.sel_trail))

    def undo_pick(self):
        """Take back the last picked point (redoable)."""
        pt = self.pending.pop()
        self.redo.append(dict(kind="pend", tool=self.tool, pt=pt, token=self._pend_token()))
        self.update_hud(); self.update(); self.state_changed.emit()

    def do_undo(self):
        if self.pending:                      # 1) un-pick the last cloud point
            return self.undo_pick()
        trail = [i for i in self.sel_trail if i in self.sel]
        if trail:                             # 2) un-select the last clicked element
            self.sel.discard(trail[-1])
            self.sel_trail = trail[:-1]
            self.redo.append(dict(kind="sel", mode=self.mode, item=trail[-1], token=self._sel_token()))
            self._arr = None
            self.update(); self.selection_changed.emit()
            return
        if self.undo:                         # 3) otherwise: normal undo
            self.redo.append(self.snapshot())
            self.restore(self.undo.pop())

    def _redo_small(self, e):
        """Redo a taken-back pick / selection if nothing else changed since. -> True when it was applied."""
        if e["kind"] == "pend":
            if self.tool != e["tool"] or self._pend_token() != e["token"]:
                return False
            self.pending.append(e["pt"])
            self.update_hud(); self.update(); self.state_changed.emit()
            return True
        if self.mode != e["mode"] or self._sel_token() != e["token"]:
            return False
        self.sel.add(e["item"])
        self.sel_trail.append(e["item"])
        self._arr = None
        self.update(); self.selection_changed.emit()
        return True

    def do_redo(self):
        while self.redo and isinstance(self.redo[-1], dict):     # picks / clicked selections first (most recent undo)
            if self._redo_small(self.redo.pop()):
                return
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
            V, F, info = fit_solid(self.solid_shape, np.array([p[2] for p in self.pending]), self.solid_seg,
                                    self.sweep_mode, self.sweep_lseg)
            self.solid_preview, self.solid_info = (V, F), ", " + info
        except Exception as ex:
            self.solid_info = ", no fit (%s)" % ex

    def finish_solid(self):
        if self.solid_preview is None:
            return self.msg("Pick more points first")
        V, F = self.solid_preview
        src = dict(shape=self.solid_shape, pts=np.array([p[2] for p in self.pending], float), seg=self.solid_seg,
                   mode=self.sweep_mode, lseg=self.sweep_lseg, sig=geom_sig(V))
        self.sync_bodies()
        if self.editing is not None and self.editing in self.reg.bodies:
            return self._finish_edit(V, F, src)
        self.push_undo()
        body = self.reg.new_body(SHAPE_NAMES[self.solid_shape], "solid", src)
        ids = replace_faces(self.mesh, [], V, F, tag=body.id)
        self.pending = []
        self.mode, self.sel, self.tool = "f", set(ids), "move"
        self.changed(); self.update_hud(); self.selection_changed.emit(); self.state_changed.emit()
        self.bodies_changed.emit()
        self.msg("%s created. Drag it, or select faces / vertices to move and extrude (E)" % SHAPE_NAMES[self.solid_shape])

    # ------------------------------------------------------------------ measuring
    def meas_fmt(self, v, t):
        """A measured value as text: lengths x scale, with the unit name; areas and volumes scale accordingly."""
        s, d, u = self.meas_scale, self.meas_dec, self.meas_units
        if t == "deg":
            return "%.*f°" % (min(d, 4), v)
        if t == "int":
            return "%d" % v
        k = {"len": 1, "area": 2, "vol": 3}[t]
        return "%.*f%s" % (d, v * s ** k, (" " + u + ("", "²", "³")[k - 1]) if u else "")

    def meas_label_lines(self, res, detail):
        """Text of a label: the main value, plus every other value when detail is on."""
        n0, v0, t0 = res["items"][0]
        lines = [MEAS_SYMBOL.get(n0, "") + self.meas_fmt(v0, t0)]
        if detail:
            lines += ["%s: %s" % (n, self.meas_fmt(v, t)) for n, v, t in res["items"][1:]]
        return lines

    def measure_summary(self, i, full=False):
        m = self.measures[i]
        items = m["res"]["items"]
        use = items if full else items[:1]
        s = "%s: %s" % (m["res"]["title"], ", ".join("%s %s" % (n, self.meas_fmt(v, t)) if full else self.meas_fmt(v, t)
                                                      for n, v, t in use))
        return s + ("  [%s]" % m["note"] if full and m["note"] else "")

    def set_meas_kind(self, k):
        if k in MEAS_MODES and k != self.meas_kind:
            self.meas_kind = k
            self.pending.clear()
            self.snap, self.meas_hover = None, None
        self.update_hud(); self.update(); self.state_changed.emit()

    def set_meas_sel(self, i):
        self.meas_sel = i if (i is not None and 0 <= i < len(self.measures)) else None
        self.update()

    def set_meas_format(self, units=None, scale=None, dec=None, show=None, details=None):
        if units is not None:
            self.meas_units = units
        if scale is not None:
            self.meas_scale = float(scale)
        if dec is not None:
            self.meas_dec = int(dec)
        if show is not None:
            self.meas_show = bool(show)
        if details is not None:
            self.meas_details = bool(details)
        self.measures_changed.emit()
        self.state_changed.emit()
        self.update()

    def add_measure(self, kind, pts, note=""):
        """Store a measurement. Raises ValueError (readable) when it can't be built."""
        res = meas_build(kind, pts)
        self.measures.append(dict(kind=kind, pts=np.asarray(pts, float).reshape(-1, 3).copy(), note=note, res=res))
        self.meas_sel = len(self.measures) - 1
        self.update_hud(); self.update(); self.measures_changed.emit(); self.state_changed.emit()
        extra = ("  -  center point added: click it to measure from it" if len(res["points"]) == 1 else
                 "  -  center points added: click them to measure from them") if res["points"] else ""
        self.msg(self.measure_summary(self.meas_sel, full=True) + extra)

    def delete_measure(self, i=None):
        i = self.meas_sel if i is None else i
        if i is None or not (0 <= i < len(self.measures)):
            return self.msg("Select a measurement in the list first")
        del self.measures[i]
        self.meas_sel = min(i, len(self.measures) - 1) if self.measures else None
        self.update(); self.measures_changed.emit()

    def clear_measures(self):
        self.measures, self.meas_sel = [], None
        self.update(); self.measures_changed.emit()

    def measures_text(self):
        return "\n".join(self.measure_summary(i, full=True) for i in range(len(self.measures)))

    def shift_measures(self, delta):
        """The scene origin moved: move every measurement with it."""
        delta = np.asarray(delta, float)
        for m in self.measures:
            m["pts"] = m["pts"] + delta
            m["res"] = meas_build(m["kind"], m["pts"])

    def measures_to_json(self):
        return [dict(kind=m["kind"], pts=m["pts"].tolist(), note=m["note"]) for m in self.measures]

    def measures_from_json(self, data, shift):
        self.measures, self.meas_sel = [], None
        for d in data or []:
            try:
                pts = np.array(d["pts"], float).reshape(-1, 3) + shift
                self.measures.append(dict(kind=d["kind"], pts=pts, note=d.get("note", ""), res=meas_build(d["kind"], pts)))
            except (ValueError, KeyError):
                continue

    # --- snapping: cloud points, shape vertices, points on edges / faces
    def mesh_edges(self):
        """All mesh edges as an (E, 2) array (cached until the mesh changes)."""
        tri = self.mesh.triangulate()[0]
        if self._edge_key is not tri:
            self._edge_arr = np.array(list(self.mesh.edge_faces().keys()), int).reshape(-1, 2)
            self._edge_key = tri
        return self._edge_arr

    def _pick_edge(self, x, y):
        """-> (edge index, t along it, edge length in pixels) of the edge under the cursor, or None."""
        E = self.mesh_edges()
        if not len(E):
            return None
        sx, sy, _, w = self.cam.project(self.mesh.V)
        S = np.c_[sx, sy]
        A, B = S[E[:, 0]], S[E[:, 1]]
        AB = B - A
        p = np.array([x, y])
        t = np.clip(((p - A) * AB).sum(1) / np.maximum((AB * AB).sum(1), 1e-12), 0, 1)
        d = np.linalg.norm(A + AB * t[:, None] - p, axis=1)
        d[(w[E[:, 0]] <= self.cam.near) | (w[E[:, 1]] <= self.cam.near)] = np.inf
        i = int(np.argmin(d))
        return (i, float(t[i]), float(np.linalg.norm(AB[i]))) if d[i] <= self.snap_radius else None

    def _face_point(self, fi, x, y):
        o, d = self.cam.ray(x, y)
        f = self.mesh.F[fi]
        p = ray_plane(o, d, self.mesh.V[f[0]], self.mesh.face_normal(fi))
        return self.mesh.V[f].mean(0) if p is None else p

    def derived_points(self):
        """Points that measurements produced (circle / sphere centers, cylinder end centers) as an (N, 3) array.
        They belong to their measurement: delete the measurement and they are gone. Hidden with the measurements."""
        pts = [p for m in self.measures for _, p in m["res"]["points"]] if self.meas_show else []
        return np.array(pts, float).reshape(-1, 3)

    def find_measure_snap(self, x, y, ctrl=False, full=False):
        """Best thing to measure from under the cursor: ('mesh'|'cloud'|'edge'|'face', index, position) or None."""
        if not ctrl:                                       # centers produced by other measurements come first
            P = self.derived_points()
            if len(P):
                sx, sy, _, w = self.cam.project(P)
                d = np.hypot(sx - x, sy - y)
                d[w <= self.cam.near] = np.inf
                k = int(np.argmin(d))
                if d[k] <= self.snap_radius:
                    return ("center", k, P[k].copy())
        sn = self.find_snap(x, y, full=full, cloud_only=ctrl)
        if sn is not None or ctrl:
            return sn
        hit = self._pick_edge(x, y)
        if hit is not None:
            i, t, px = hit
            a, b = self.mesh_edges()[i]
            A, B = self.mesh.V[a], self.mesh.V[b]
            if abs(t - 0.5) * px <= 0.6 * self.snap_radius:
                return ("edge", i, (A + B) / 2)
            o, d = self.cam.ray(x, y)
            return ("edge", i, ray_segment_closest(o, d, A, B))
        fi = pick_mesh_face(self.cam, self.mesh, x, y)
        if fi is not None:
            return ("face", fi, self._face_point(fi, x, y))
        return None

    def measure_hover(self, x, y, ctrl):
        """Update the snap marker (and the highlighted edge in Edge mode) for the cursor position."""
        self.snap, self.meas_hover = None, None
        k = self.meas_kind
        if k == "edge":
            hit = self._pick_edge(x, y)
            if hit is not None:
                a, b = self.mesh_edges()[hit[0]]
                self.meas_hover = (self.mesh.V[a].copy(), self.mesh.V[b].copy())
        elif k == "faceangle":
            fi = pick_mesh_face(self.cam, self.mesh, x, y)
            if fi is not None:
                self.snap = ("face", fi, self._face_point(fi, x, y))
        else:
            self.snap = self.find_measure_snap(x, y, ctrl, full=False)

    def measure_click(self, x, y, ctrl):
        k = self.meas_kind
        if k == "edge":
            hit = self._pick_edge(x, y)
            if hit is None:
                return self.msg("Click on a mesh edge")
            a, b = self.mesh_edges()[hit[0]]
            return self.add_measure("edge", [self.mesh.V[a], self.mesh.V[b]])
        if k == "faceangle":
            fi = pick_mesh_face(self.cam, self.mesh, x, y)
            if fi is None:
                return self.msg("Click on a face of a shape")
            self.pending.append(("face", fi, self._face_point(fi, x, y)))
        else:
            sn = self.find_measure_snap(x, y, ctrl, full=True)
            if sn is None:
                return self.msg("Nothing to snap to here - click a cloud point, a shape's vertex, edge or face (Ctrl = cloud points only)")
            self.pending.append(sn)
        need = MEAS_AUTO.get(k)
        if need and len(self.pending) >= need:
            return self.finish_measure()
        self.update_hud(); self.update(); self.state_changed.emit()
        n, lo = len(self.pending), MEAS_MIN[k]
        self.msg("%d point%s picked%s" % (n, "" if n == 1 else "s",
                                          "" if n >= lo or k in MEAS_AUTO else " - need %d+, then Enter" % lo)
                 if n < lo else "%d points picked - press Enter to finish (or keep clicking)" % n)

    def finish_measure(self):
        k, P = self.meas_kind, self.pending
        if k == "edge" or not P:
            return
        if len(P) < MEAS_MIN[k]:
            return self.msg("Pick at least %d points (%d so far)" % (MEAS_MIN[k], len(P)))
        pts = np.array([p[2] for p in P], float)
        try:
            if k == "faceangle":
                L = 0.1 * max(self.cam.radius, 1e-9)
                n1, n2 = self.mesh.face_normal(P[0][1]), self.mesh.face_normal(P[1][1])
                pts = np.array([pts[0], pts[0] + n1 * L, pts[1], pts[1] + n2 * L])
            kind, pts, note = from_clicks(k, pts)
            self.add_measure(kind, pts, note)
        except ValueError as ex:
            return self.msg(str(ex))                       # keep the picks so the user can add more points
        self.pending = []
        self.update_hud(); self.update(); self.state_changed.emit()

    def meas_live(self):
        """(anchor, label lines) of the measurement as it would be if the cursor position were picked next."""
        if self.tool != "measure" or not self.pending or self.snap is None:
            return None
        if self.meas_kind not in ("distance", "path", "angle", "circle", "sphere"):
            return None
        try:
            kind, pts, _ = from_clicks(self.meas_kind, [p[2] for p in self.pending] + [self.snap[2]])
            res = meas_build(kind, pts)
        except ValueError:
            return None
        return res["anchor"], self.meas_label_lines(res, False)

    def fit_selection_diameter(self):
        """Diameter of the selected shape: the best sphere or cylinder through its vertices and edge midpoints."""
        fs = self.sel_faces()
        vs = self.sel_verts()
        if len(vs) < 6:
            return self.msg("Select a round shape first (Select tool, Face mode, or click the body in the Bodies list)")
        V, F = self.mesh.V, self.mesh.F
        mid = {tuple(sorted((f[k], f[(k + 1) % len(f)]))) for i in fs for f in [F[i]] for k in range(len(f))}
        P = V[vs]
        if mid:
            E = np.array(sorted(mid))
            P = np.vstack([P, (V[E[:, 0]] + V[E[:, 1]]) / 2])   # edge midpoints: a box's corners alone fit a sphere
        try:
            cands = fit_round(P)
        except ValueError as ex:
            return self.msg(str(ex))
        if not cands or cands[0]["rel"] > 0.06:
            best = "" if not cands else " (best fit: %s, error %.0f%% of its radius)" % (cands[0]["kind"], 100 * cands[0]["rel"])
            return self.msg("The selection doesn't look like a sphere or a cylinder%s. Use the circle measurement on its rim instead" % best)
        b = cands[0]
        try:
            self.add_measure(b["kind"], b["pts"], "fitted to the selection, error %.1f%% of radius" % (100 * b["rel"]))
        except ValueError as ex:
            self.msg(str(ex))

    def meas_selection_info(self):
        vs = self.sel_verts()
        return selection_info(self.mesh.V, self.mesh.F, self.sel_faces(), vs) if vs else None

    def meas_info_text(self):
        """Size of the selected shape and of the cloud, for the panel."""
        f, lines = self.meas_fmt, []
        info = self.meas_selection_info()
        if info is None:
            lines.append("Select a shape (Select tool or the Bodies list) to see its size here.")
        else:
            sz = info["size"]
            lines.append("Selected: %d faces, %d vertices" % (info["n_faces"], info["n_verts"]))
            lines.append("Size: %s × %s × %s" % tuple(f(s, "len") for s in sz))
            lines.append("Diagonal: %s" % f(info["diagonal"], "len"))
            if info["n_faces"]:
                lines.append("Surface area: %s" % f(info["area"], "area"))
                lines.append("Volume: %s" % (f(info["volume"], "vol") if info["volume"] is not None else "n/a (not a closed shape)"))
        if self.cloud_ext is not None:
            lines.append("Cloud size: %s × %s × %s" % tuple(f(s, "len") for s in self.cloud_ext))
        return "\n".join(lines)

    def _draw_measures(self, dpr):
        if self.meas_hover is not None:                    # the edge that would be measured
            glColor3f(1.0, 0.95, 0.2); glLineWidth(4.0 * dpr)
            glBegin(GL_LINES); glVertex3f(*self.meas_hover[0]); glVertex3f(*self.meas_hover[1]); glEnd()
        if not self.meas_show:
            return
        for i, m in enumerate(self.measures):
            hot = i == self.meas_sel
            col = (1.0, 1.0, 1.0) if hot else (1.0, 0.43, 0.78)
            glColor3f(*col); glLineWidth((3.0 if hot else 2.0) * dpr)
            for kind, P in m["res"]["prims"]:
                glBegin(GL_LINE_LOOP if kind == "loop" else GL_LINE_STRIP)
                for q in P:
                    glVertex3f(*q)
                glEnd()
            self._pts(np.ascontiguousarray(m["pts"], np.float32), 8 if hot else 6, col)
            for _, c in m["res"]["points"]:                # the center point(s) this measurement produced: a cross
                w = float(self.cam.project(np.asarray(c, float)[None, :])[3][0])
                if w <= self.cam.near:
                    continue
                s = self.cam.world_per_pixel(w) * 8.0
                ccol = (1.0, 1.0, 1.0) if hot else (1.0, 0.78, 0.2)
                glColor3f(*ccol); glLineWidth(2.0 * dpr)
                glBegin(GL_LINES)
                for ax in np.eye(3):
                    glVertex3f(*(c - ax * s)); glVertex3f(*(c + ax * s))
                glEnd()
                self._pts(np.asarray(c, np.float32)[None, :], 9, ccol)

    # --------------------------------------------------- mesh from points (Poisson)
    def mp_clear(self):
        self.mp_pts = self.mp_preview = self.mp_lines = None
        self.mp_info = ""
        self.lasso, self.lasso_w, self.lasso_on = [], [], False
        self._mp_timer.stop()

    def _lasso_world(self, x, y):
        """Where a lasso point sits in space (on the plane through the view target) - only used to draw the loop."""
        o, d = self.cam.ray(x, y)
        p = ray_plane(o, d, self.cam.target, self.cam.forward())
        return o + d * self.cam.dist if p is None else p

    def finish_lasso(self):
        poly, add = np.array(self.lasso, float), self.lasso_add
        self.lasso, self.lasso_w, self.lasso_on = [], [], False
        self.update()
        if len(poly) < 3:
            return self.msg("Drag a loop around the cloud points to build a mesh from")
        if self.cloud_full is None:
            return self.msg("Load a point cloud first (and make sure it is visible)")
        pts = self.cloud_full
        sx, sy, _, w = self.cam.project(pts)
        box = ((sx >= poly[:, 0].min()) & (sx <= poly[:, 0].max()) & (sy >= poly[:, 1].min())
               & (sy <= poly[:, 1].max()) & (w > self.cam.near))
        idx = np.flatnonzero(box)
        if idx.size:
            idx = idx[points_in_polygon(np.c_[sx[idx], sy[idx]], poly)]
        if idx.size < 30:
            return self.msg("Only %d points inside the loop - lasso more of the cloud (30+ needed)" % idx.size)
        sel = pts[idx].astype(np.float64)
        if add and self.mp_pts is not None:
            sel = np.unique(np.vstack([self.mp_pts, sel]), axis=0)
        self.mp_pts, self.mp_eye = sel, self.cam.eye()
        self.build_mp_preview()

    def schedule_mp(self):
        """A setting changed: rebuild the preview once things settle (only if points are selected)."""
        if self.mp_pts is not None:
            self._mp_timer.start()

    def build_mp_preview(self):
        if self.mp_pts is None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.msg("Building mesh from %s points..." % format(len(self.mp_pts), ","))
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            V, F, info = poisson_mesh(self.mp_pts, self.mp_detail, self.mp_max_tris, self.mp_trim,
                                      getattr(self, "mp_eye", None), self.mp_orient)
        except (ValueError, ImportError) as ex:
            self.mp_preview = self.mp_lines = None
            self.update_hud(); self.update(); self.state_changed.emit()
            return self.msg(str(ex))
        finally:
            QApplication.restoreOverrideCursor()
        Fa = np.asarray(F, int)
        E = np.sort(np.concatenate([Fa[:, [0, 1]], Fa[:, [1, 2]], Fa[:, [2, 0]]]), axis=1)
        E = np.unique(E, axis=0)
        self.mp_preview, self.mp_info = (V, F), info
        self.mp_lines = np.ascontiguousarray(V[E].reshape(-1, 3), np.float32)
        self.update_hud(); self.update(); self.state_changed.emit()
        self.msg(info + "  -  Enter / 'Add mesh to model' keeps it; change Detail to refine")

    def commit_mp(self):
        if self.mp_preview is None:
            return self.msg("Lasso some cloud points first")
        V, F = self.mp_preview
        self.sync_bodies()
        self.push_undo()
        body = self.reg.new_body("Poisson mesh", "mesh")
        replace_faces(self.mesh, [], V, F, tag=body.id)
        n = len(F)
        self.mp_clear()
        self.mode, self.sel, self.tool = "f", set(), "move"
        self.changed(); self.update_hud(); self.selection_changed.emit(); self.state_changed.emit()
        self.bodies_changed.emit()
        self.msg("Mesh added (%s triangles). Select it in the Bodies list to move or edit it; flip normals with N" % format(n, ","))

    # ------------------------------------------------------------- bodies list
    def sync_bodies(self):
        """Give every untagged face a body (connected solids that carry no tag become new bodies)
        and forget bodies that have no faces left (boolean results are kept so they can be undone)."""
        m, reg = self.mesh, self.reg
        for i, t in enumerate(m.T):
            if t not in reg.bodies or reg.bodies[t].state != "live":
                m.T[i] = -1
        if any(t < 0 for t in m.T):
            for fl in shells_of(m):
                tags = [m.T[i] for i in fl]
                if -1 not in tags:
                    continue
                known = [t for t in tags if t >= 0]
                t = Counter(known).most_common(1)[0][0] if known else reg.new_body("Body", "mesh").id
                for i in fl:
                    if m.T[i] < 0:
                        m.T[i] = t
        used = set(m.T)
        for bid in [b.id for b in reg.bodies.values() if b.state == "live"]:
            if bid not in used and not reg.is_result(bid):
                del reg.bodies[bid]

    def body_faces(self, bid):
        return [i for i, t in enumerate(self.mesh.T) if t == bid]

    def select_body(self, bid):
        b = self.reg.bodies.get(bid)
        if b is None or b.state != "live":
            return
        self.mode, self.sel, self.sel_trail = "f", set(self.body_faces(bid)), []
        self._arr = None
        self.update_hud(); self.update(); self.selection_changed.emit(); self.state_changed.emit()

    def set_body_visible(self, bid, vis):
        self.sync_bodies()
        b = self.reg.bodies.get(bid)
        if b is None or b.state == "consumed" or (b.state == "live") == vis:
            return
        self.push_undo()
        if vis:
            replace_faces(self.mesh, [], b.geom[0], b.geom[1], tag=bid)
            b.geom, b.state = None, "live"
        else:
            faces = self.body_faces(bid)
            b.geom = extract_faces(self.mesh, faces)
            replace_faces(self.mesh, faces, [], [], -1)
            b.state = "hidden"
        self.sel, self.sel_trail = set(), []
        self.changed(); self.selection_changed.emit(); self.bodies_changed.emit()

    def set_cloud_visible(self, idx, vis):
        if 0 <= idx < len(self.clouds) and self.clouds[idx]["visible"] != vis:
            self.clouds[idx]["visible"] = vis
            self._rebuild_cloud()

    # --- booleans as folders
    def _folder_boolean(self, f, geoms):
        """Run folder f's boolean on its operands' stored geometry (geoms overrides by body id)."""
        tmp, shells = Mesh(), []
        for o in f.operands:
            V, F = geoms[o] if o in geoms else self.reg.bodies[o].geom
            base, first = len(tmp.V), len(tmp.F)
            tmp.V = np.vstack([tmp.V, V])
            for fc in F:
                tmp.add_face([base + k for k in fc])
            shells.append(list(range(first, len(tmp.F))))
        return boolean_op(tmp, f.kind if hasattr(f, "kind") else f.op, shells)

    def undo_boolean(self, fid):
        self.sync_bodies()
        reg, m = self.reg, self.mesh
        f = reg.folders.get(fid)
        if f is None:
            return
        if reg.parent_folder(fid) is not None:
            return self.msg("This boolean is inside another one - undo the outer boolean first")
        self.push_undo()
        r = reg.bodies.get(f.result)
        if r is not None:
            faces = self.body_faces(r.id)
            if faces:
                replace_faces(m, faces, [], [], -1)
            del reg.bodies[r.id]
        ids = []
        for o in f.operands:
            b = reg.bodies[o]
            V, F = b.geom
            b.geom, b.state, b.folder = None, "live", None
            ids += replace_faces(m, [], V, F, tag=o)
        del reg.folders[fid]
        self.mode, self.sel, self.sel_trail = "f", set(ids), []
        self.changed(); self.selection_changed.emit(); self.bodies_changed.emit()
        self.msg("%s undone - the original solids are back" % f.name)

    # --- editing the points a solid was made from
    def edit_body_points(self, bid):
        self.sync_bodies()
        b = self.reg.bodies.get(bid)
        if b is None or not b.src:
            return self.msg("This body has no stored points (only solids made with the Solid tool do)")
        if b.state == "live":
            V = extract_faces(self.mesh, self.body_faces(bid))[0]
        else:
            V = b.geom[0]
        if geom_sig(V) != b.src.get("sig") and QMessageBox.question(
                self, "Edit points",
                "'%s' was reshaped by hand since it was created.\nApplying new points will replace those edits. Continue?" % b.name
                ) != QMessageBox.StandardButton.Yes:
            return
        s = b.src
        self.editing, self.tool, self.drag = bid, "solid", None
        self.solid_shape, self.solid_seg = s["shape"], s["seg"]
        self.sweep_mode, self.sweep_lseg = s["mode"], s["lseg"]
        self.pending = [("cloud", -1, np.array(p, float)) for p in s["pts"]]
        self.update_hud(); self.update(); self.state_changed.emit()
        self.msg("Editing the points of '%s': click a point to remove it, click the cloud to add points, "
                 "Enter = apply, Esc = cancel" % b.name)

    def _finish_edit(self, V, F, src):
        reg, m = self.reg, self.mesh
        b = reg.bodies[self.editing]
        ids = []
        if b.state == "consumed":                      # operand of a boolean: redo the boolean(s) above it
            geoms, updates, f = {b.id: (V, F)}, [], reg.folders.get(b.folder)
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                while f is not None:
                    rV, rF, _ = self._folder_boolean(f, geoms)
                    r = reg.bodies[f.result]
                    updates.append((r, rV, rF))
                    geoms[r.id] = (rV, rF)
                    f = reg.folders.get(r.folder) if r.state == "consumed" else None
            except (ValueError, ImportError) as ex:
                return self.msg("The boolean can't be redone with these points: %s" % ex)
            finally:
                QApplication.restoreOverrideCursor()
            self.push_undo()
            b.geom = (V, F)
            for r, rV, rF in updates:
                if r.state == "live":
                    ids = replace_faces(m, self.body_faces(r.id), rV, rF, tag=r.id)
                else:
                    r.geom = (rV, rF)
        elif b.state == "live":
            self.push_undo()
            ids = replace_faces(m, self.body_faces(b.id), V, F, tag=b.id)
        else:
            self.push_undo()
            b.geom = (V, F)
        b.src = src
        self.editing, self.pending = None, []
        self.mode, self.sel, self.sel_trail, self.tool = "f", set(ids), [], "move"
        self.changed(); self.update_hud(); self.selection_changed.emit(); self.state_changed.emit()
        self.bodies_changed.emit()
        self.msg("Points applied to '%s'" % b.name)

    def get_surface(self):
        if self.surf is None:
            if self.cloud_full is None:
                self.msg("Load a point cloud first (and make sure it is visible)")
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
        if t != "solid":
            self.editing = None
        self.pending.clear()
        self.drag = None
        self.snap, self.meas_hover = None, None
        if t != "mesh":
            self.mp_clear()
        self.update_hud()
        self.update()
        self.state_changed.emit()

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

    def selection_frame(self):
        """Orthonormal frame (columns x, y, z) matching the selection's orientation, or None
        (= world axes). Faces / edges / vertices: z = surface normal, x = first selected edge
        or face edge. Selections with no normal (a closed solid): principal axes of the points."""
        vs = self.sel_verts()
        if not vs:
            return None
        m, V = self.mesh, self.mesh.V
        z = self.selection_normal()
        if z is None:
            return pca_frame(V[vs])
        if self.mode == "e" and self.sel:
            a, b = sorted(self.sel)[0]
            t = V[b] - V[a]
        else:
            fs = self.sel_faces()
            if not fs:
                vset = set(vs)
                fs = [i for i, f in enumerate(m.F) if vset & set(f)]
            f = m.F[fs[0]]
            t = V[f[1]] - V[f[0]]
        return frame_from_axes(z, t)

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
        fr = self.selection_frame() if self.local_axes else None
        if fr is not None:                    # handles follow the selection's orientation
            axes = [(nm, fr[:, i], col) for i, (nm, _, col) in enumerate(axes)]
        else:
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
            if np.linalg.norm(p - A) < 12:    # on the gizmo origin: free-drag the element instead
                continue
            AB = B - A
            t = np.clip(((p - A) @ AB) / max(AB @ AB, 1e-9), 0, 1)
            d = np.linalg.norm(A + AB * t - p)
            if d < bd:
                best, bd = (name, a), d
        return best

    # ----------------------------------------------------------------- cloud
    def load_cloud(self, paths, append=False):
        """Load files, one point cloud per file. append=True keeps the clouds already loaded."""
        if isinstance(paths, str):
            paths = [paths]
        loaded, errors = [], []
        for p in paths:
            try:
                a = load_xyz(p)
                if len(a) == 0:
                    raise ValueError("no points")
                loaded.append((os.path.basename(p), a))
            except Exception as ex:
                errors.append("%s: %s" % (os.path.basename(p), ex))
        if not loaded:
            raise ValueError("\n".join(errors) or "No files given")
        if not append:
            self.clouds = []
        lo = np.min([a.min(0) for _, a in loaded], axis=0)
        hi = np.max([a.max(0) for _, a in loaded], axis=0)
        for c in self.clouds:                              # everything already loaded counts for the bounds
            lo = np.minimum(lo, c["pts"].min(0) + self.origin)
            hi = np.maximum(hi, c["pts"].max(0) + self.origin)
        old = self.origin.copy()
        self.origin = (lo + hi) / 2.0
        delta = old - self.origin
        for c in self.clouds:
            c["pts"] = (c["pts"].astype(np.float64) + delta).astype(np.float32)
        self.mesh.V = self.mesh.V + delta                  # keep existing model aligned
        self.reg.shift(delta)
        self.shift_measures(delta)
        taken = {c["name"] for c in self.clouds}
        for name, a in loaded:
            base, n = name, 2
            while name in taken:
                name, n = "%s (%d)" % (base, n), n + 1
            taken.add(name)
            self.clouds.append(dict(name=name, pts=(a - self.origin).astype(np.float32), visible=True))
        ext = hi - lo
        if self.pristine:
            self.mesh = Mesh.plane((0, 0, 0), max(0.4 * max(ext[0], ext[1]), 1e-3))
            self.reg = Registry()
            self.sel = set()
            if not append:                                 # a fresh start: old measurements belong to the old cloud
                self.measures, self.meas_sel = [], None
                self.measures_changed.emit()
        self._rebuild_cloud()
        if not append:
            self.cam.frame((0, 0, 0), 0.5 * float(np.linalg.norm(ext)))
        self._arr = None
        self.update()
        self.selection_changed.emit()
        self.bodies_changed.emit()
        n = sum(len(c["pts"]) for c in self.clouds)
        self.msg("Loaded %d file(s): %s points total (drawing %s)"
                 % (len(loaded), format(n, ","), format(self.n_disp, ",")))
        return errors

    def _rebuild_cloud(self):
        """Display / snap / surface data come from the visible clouds only."""
        vis = [c["pts"] for c in self.clouds if c["visible"]]
        self.surf = None
        self._cloud_dirty = True
        if not vis:
            self.cloud_full = self.cloud_snap = self.cloud_disp = self.cloud_cols = None
            self.n_disp = 0
            self.cloud_ext = None
            self.update()
            return
        local = np.concatenate(vis)
        n = len(local)
        stride = max(1, int(np.ceil(n / MAX_DISPLAY)))
        self.cloud_full = local
        self.cloud_disp = np.ascontiguousarray(local[::stride])
        s2 = max(1, int(np.ceil(n / MAX_SNAP)))
        self.cloud_snap = local if s2 == 1 else np.ascontiguousarray(local[::s2])
        self.cloud_cols = height_colors(self.cloud_disp[:, 2])
        self.n_disp = len(self.cloud_disp)
        self.grid_z = float(local[:, 2].min())
        self.cloud_ext = np.ptp(local, axis=0).astype(float)
        self.update()

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
        self.msg("Extruded - drag the gizmo to shape it (blue Z = along the face normal)")

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
        note = ""
        if self.mode == "v":                  # dissolve: surrounding faces merge, edges join
            st = self.mesh.dissolve_vertices(self.sel)
            if st["skipped"]:
                note = "%d vertex(es) kept - removing them would tear the surface" % st["skipped"]
        elif self.mode == "e":                # dissolve: the two faces beside the edge merge
            st = self.mesh.dissolve_edges(sorted(self.sel))
            if st["faces_removed"]:
                note = "Border edge: nothing to merge with, so its face was removed"
            elif st["skipped"]:
                note = "%d edge(s) kept - they cannot be dissolved cleanly" % st["skipped"]
        else:
            self.mesh.delete_faces(self.sel)
        self.sel = set()
        self.changed(); self.selection_changed.emit()
        if note:
            self.msg(note)

    def boolean_shells(self):
        """Solids touched by the selection, in the order they were picked (click, Shift+click...)."""
        m = self.mesh
        sh = shells_of(m)
        face_shell = {f: k for k, fl in enumerate(sh) for f in fl}
        vert_shell = {v: k for f, k in face_shell.items() for v in m.F[f]}

        def shell_of(item):
            if self.mode == "f":
                return face_shell.get(item)
            return vert_shell.get(item if self.mode == "v" else item[0])

        order = []
        for it in [i for i in self.sel_trail if i in self.sel] + sorted(self.sel):
            k = shell_of(it)
            if k is not None and k not in order:
                order.append(k)
        return [sh[k] for k in order]

    def op_boolean(self, kind):
        self.sync_bodies()
        solids = self.boolean_shells()
        if len(solids) < 2:
            return self.msg("Boolean needs two or more solids: click one, then Shift+click the others")
        parts = [split_loose(self.mesh, s_) for s_ in solids]     # loose faces stuck to a solid stay out of the boolean
        solids, n_loose = [p[0] for p in parts], sum(len(p[1]) for p in parts)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            V, F, info = boolean_op(self.mesh, kind, solids)
        except (ValueError, ImportError) as ex:
            return self.msg(str(ex))
        finally:
            QApplication.restoreOverrideCursor()
        self.push_undo()
        reg, m = self.reg, self.mesh
        operands = []
        for shell in solids:                               # one body per solid, in pick order
            bid = Counter(m.T[i] for i in shell).most_common(1)[0][0]
            if bid in operands or bid not in reg.bodies:   # two solids sharing a body id: split it
                bid = reg.new_body("Body", "mesh").id
            for i in shell:
                m.T[i] = bid
            reg.bodies[bid].geom = extract_faces(m, shell)
            operands.append(bid)
        result = reg.new_body("Combined", "boolean")
        reg.new_folder(kind, operands, result.id)          # puts the operands inside the folder
        for bid in operands:
            reg.bodies[bid].state = "consumed"
        ids = replace_faces(m, [i for s_ in solids for i in s_], V, F, tag=result.id)
        self.mode, self.sel, self.sel_trail = "f", set(ids), []
        self.changed(); self.update_hud(); self.selection_changed.emit(); self.state_changed.emit()
        self.bodies_changed.emit()
        note = ("  (%d loose face%s stuck to the solids kept as a separate body)"
                % (n_loose, "" if n_loose == 1 else "s")) if n_loose else ""
        self.msg(info + ("  [first-picked solid minus the others]" if kind == "difference" else "") + note
                 + "  - undo it from its folder in the Bodies list")

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
        self.sync_bodies()
        save_model(path, self.mesh, self.origin, extra={
            "bodies": self.reg.to_dict(), "measurements": self.measures_to_json(),
            "measure_format": dict(units=self.meas_units, scale=self.meas_scale, dec=self.meas_dec)})

    def open_model(self, path):
        m, org = load_model(path)
        with open(path) as fh:
            data = json.load(fh)
        reg_data = data.get("bodies")
        self.push_undo()
        if not self.clouds:
            self.origin = org
            shift = np.zeros(3)
        else:
            shift = org - self.origin
            m.V = m.V + shift
        self.mesh = m
        self.reg = Registry.from_dict(reg_data)
        self.reg.shift(shift)
        self.measures_from_json(data.get("measurements"), shift)
        fmt = data.get("measure_format") or {}
        self.meas_units, self.meas_scale, self.meas_dec = fmt.get("units", ""), float(fmt.get("scale", 1.0)), int(fmt.get("dec", 3))
        self.sel, self.editing, self.pending = set(), None, []
        self.pristine = False
        self.sync_bodies()
        self.changed(); self.selection_changed.emit(); self.bodies_changed.emit()
        self.measures_changed.emit(); self.state_changed.emit()

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
                         moved=False, pushed=False, press=(x, y),
                         frame=self.selection_frame() if self.local_axes else None)

    def apply_drag(self, x, y, ctrl):
        dr = self.drag
        if dr is None:
            return
        if not dr["moved"]:
            if np.hypot(x - dr["press"][0], y - dr["press"][1]) < 3:
                return
            dr["moved"] = True
        if not dr["pushed"]:
            self.undo.append(self.snapshot())
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
            if self.tool == "measure":
                self.measure_click(x, y, bool(m & Qt.KeyboardModifier.ControlModifier))
            elif self.tool == "mesh":
                self.lasso, self.lasso_on, self.lasso_add = [(x, y)], True, shift
                self.lasso_w = [self._lasso_world(x, y)]
                self.update()
            elif self.tool == "select":
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
                if self.pending:                           # clicking near a picked point removes it
                    P = np.array([p[2] for p in self.pending])
                    sx, sy, _, w = self.cam.project(P)
                    d = np.hypot(sx - x, sy - y)
                    d[w <= self.cam.near] = np.inf
                    k = int(np.argmin(d))
                    if d[k] <= max(self.snap_radius, 8.0):
                        del self.pending[k]
                        self.update_hud(); self.update()
                        return self.msg("Point removed (%d left)" % len(self.pending))
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
        elif self.lasso_on:
            lx, ly = self.lasso[-1]
            if np.hypot(x - lx, y - ly) >= 3:
                self.lasso.append((x, y))
                self.lasso_w.append(self._lasso_world(x, y))
                self.update()
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
            elif self.tool == "measure":
                self.measure_hover(x, y, bool(e.modifiers() & Qt.KeyboardModifier.ControlModifier))
            if self.tool != "select":
                self.update()

    def mouseReleaseEvent(self, e):
        self.nav = None
        if self.lasso_on and e.button() == Qt.MouseButton.LeftButton:
            self.finish_lasso()
            return
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
            if self.drag["frame"] is not None:    # X/Y/Z lock uses the selection's own axes
                a = self.drag["frame"][:, {K.Key_X: 0, K.Key_Y: 1, K.Key_Z: 2}[k]]
            self.axis_lock = None if (self.axis_lock is not None and np.allclose(self.axis_lock, a)) else a
            self.apply_drag(self.last[0], self.last[1], False)
        elif k in (K.Key_Return, K.Key_Enter) and self.tool == "points":
            self.finish_points_face()
        elif k in (K.Key_Return, K.Key_Enter) and self.tool == "solid":
            self.finish_solid()
        elif k in (K.Key_Return, K.Key_Enter) and self.tool == "mesh":
            self.commit_mp()
        elif k in (K.Key_Return, K.Key_Enter) and self.tool == "measure":
            self.finish_measure()
        elif k == K.Key_Backspace and self.pending:
            self.undo_pick()
        elif k == K.Key_Escape:
            if self.tool == "mesh" and (self.lasso_on or self.mp_pts is not None):
                self.mp_clear()
                self.update_hud(); self.update(); self.state_changed.emit()
            elif self.drag is not None:
                self.cancel_drag()
            elif self.pending or self.editing is not None:
                self.pending, self.editing = [], None
                self.update_hud(); self.update(); self.state_changed.emit()
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
            if self.tool == "points" or (self.tool == "measure" and self.meas_kind in ("distance", "path", "angle")):
                glBegin(GL_LINE_STRIP)
                for q in pos:
                    glVertex3f(*q)
                if self.tool == "measure" and self.snap is not None:      # rubber band to the cursor
                    glVertex3f(*self.snap[2])
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
            self._pts(np.array([self.snap[2]], np.float32), 15,
                      {"mesh": (0.3, 1.0, 0.4), "edge": (0.3, 0.9, 1.0), "face": (0.3, 0.9, 1.0), "center": (1.0, 0.78, 0.2)}.get(self.snap[0], (1.0, 0.95, 0.2)))
        self._draw_measures(dpr)
        if self.tool == "mesh":
            if self.mp_pts is not None:                    # the lassoed points
                self._pts(np.ascontiguousarray(self.mp_pts[::max(1, len(self.mp_pts) // 200_000)], np.float32),
                          3, (0.3, 1.0, 0.85))
            if self.mp_lines is not None:                  # wireframe of the mesh that would be added
                glColor3f(0.4, 1.0, 0.6); glLineWidth(1.0 * dpr)
                glEnableClientState(GL_VERTEX_ARRAY)
                glVertexPointer(3, GL_FLOAT, 0, self.mp_lines)
                glDrawArrays(GL_LINES, 0, len(self.mp_lines))
                glDisableClientState(GL_VERTEX_ARRAY)
        if len(self.lasso_w) > 1:                          # the lasso being drawn
            glColor3f(1.0, 0.9, 0.2); glLineWidth(2.0 * dpr)
            glBegin(GL_LINE_LOOP)
            for q in self.lasso_w:
                glVertex3f(*q)
            glEnd()
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
            self.overlay.update()                          # labels follow the camera
        except Exception:
            traceback.print_exc()
            self._gl_failed = True


# ============================================================== main window ==
SEG_SHAPES = ("cylinder", "cone", "sphere", "loft", "extrude")   # shapes whose smoothness "Segments" controls
TOOL_NAMES = {"select": "Select", "move": "Move / Transform", "points": "Face from points", "solid": "Best-fit solid",
              "mesh": "Mesh from points", "measure": "Measure"}
# edit buttons: (label, Viewer method, modes in which the button is shown)
EDIT_OPS = (("Subdivide", "op_subdivide", "f", "Subdivide faces (S)"), ("Extrude", "op_extrude", "ef", "Extrude (E)"),
            ("Fill face", "op_fill", "v", "Fill a face from the selected vertices (F)"),
            ("Weld", "op_weld", "v", "Weld the selected vertices into one (M)"),
            ("Flip", "op_flip", "f", "Flip normals (N)"),
            ("Snap", "op_snap_surface", "vef", "Snap the selection to the cloud surface (Shift+S)"),
            ("Delete", "op_delete", "vef", "Delete (Del)"))


class Section(QWidget):
    """A header that folds / unfolds its content - used for options most people rarely need."""
    def __init__(self, title, collapsed=True):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self.btn = QToolButton()
        self.btn.setText(title)
        self.btn.setCheckable(True)
        self.btn.setChecked(not collapsed)
        self.btn.setAutoRaise(True)
        self.btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.btn.setArrowType(Qt.ArrowType.DownArrow if not collapsed else Qt.ArrowType.RightArrow)
        self.btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn.setStyleSheet("QToolButton{text-align:left;font-weight:bold;}")
        self.body = QWidget()
        self.body.setVisible(not collapsed)
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(14, 0, 0, 0)
        self.body_lay.setSpacing(3)
        self.btn.toggled.connect(self._toggled)
        lay.addWidget(self.btn)
        lay.addWidget(self.body)

    def _toggled(self, on):
        self.body.setVisible(on)
        self.btn.setArrowType(Qt.ArrowType.DownArrow if on else Qt.ArrowType.RightArrow)


def _row(label, widget):
    r = QHBoxLayout()
    r.setSpacing(4)
    r.addWidget(QLabel(label))
    r.addStretch(1)
    r.addWidget(widget)
    return r


def _small(text):
    l = QLabel(text)
    l.setWordWrap(True)
    l.setStyleSheet("color:#8a97a6;")
    return l


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
        self._build_toolbar()
        self._build_dock()
        self.bodies_dock = QDockWidget("Bodies", self)       # left: point clouds, bodies, boolean folders
        self.bodies_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                                     | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.bodies_panel = BodiesPanel(self.v)
        self.bodies_dock.setWidget(self.bodies_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.bodies_dock)
        self.bodies_dock.setMinimumWidth(150)
        self.resizeDocks([self.bodies_dock, self.props_dock], [185, 225], Qt.Orientation.Horizontal)
        self._build_view_menu()
        self.v.selection_changed.connect(self.refresh_coords)
        self.v.selection_changed.connect(self.update_panel)
        self.v.state_changed.connect(self.sync_checked)
        self.v.state_changed.connect(self.sync_solid_controls)
        self.v.measures_changed.connect(self.refresh_meas_list)
        self.refresh_coords()
        self.update_panel()
        self.statusBar().showMessage("File > Open point clouds (Ctrl+O) to begin, or drop files onto the window")

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

    # ------------------------------------------------------------------ menus
    def _build_actions(self):
        mb = self.menuBar()
        f, ed, t, cr, md, self.view_menu = (mb.addMenu("&File"), mb.addMenu("&Edit"), mb.addMenu("&Tools"),
                                           mb.addMenu("&Create"), mb.addMenu("&Modify"), mb.addMenu("&View"))
        self.act("Open point clouds…", "Ctrl+O", self.open_cloud, f)
        self.act("Add point clouds…", "Ctrl+Shift+O", lambda: self.open_cloud(append=True), f)
        f.addSeparator()
        self.act("Open model…", None, self.open_model, f)
        self.act("Save model…", "Ctrl+S", self.save_model, f)
        f.addSeparator()
        ex = f.addMenu("Export OBJ")
        self.act("Original coordinates…", None, lambda: self.export(True), ex)
        self.act("Cloud-centered…", None, lambda: self.export(False), ex)
        f.addSeparator()
        self.act("Quit", "Ctrl+Q", self.close, f)

        self.undo_act = self.act("Undo", "Ctrl+Z", self.v.do_undo, ed)
        self.redo_act = self.act("Redo", "Ctrl+Y", self.v.do_redo, ed)
        self.redo_act.setShortcuts([QKeySequence("Ctrl+Y"), QKeySequence("Ctrl+Shift+Z")])
        ed.addSeparator()
        self.act("Delete selection", "Delete", self.v.op_delete, ed)

        tg = QActionGroup(self)
        self.tool_acts = {}
        for key, sc in (("select", "Q"), ("move", "W"), ("points", "P"), ("solid", "B"), ("mesh", "G"), ("measure", "R")):
            self.tool_acts[key] = self.act(TOOL_NAMES[key], sc, lambda k=key: self.v.set_tool(k), t, checkable=True, group=tg)
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

        self.act("Plane", "Shift+A", self.v.op_add_plane, cr)
        self.act("Face from cloud points", None, lambda: self.v.set_tool("points"), cr)
        self.act("Mesh from points (lasso, Poisson)", None, lambda: self.v.set_tool("mesh"), cr)
        sm = cr.addMenu("Best-fit solid")
        for key in SHAPES:
            self.act(SHAPE_NAMES[key], None, lambda k=key: self.pick_solid(k), sm)

        self.act("Subdivide faces", "S", self.v.op_subdivide, md)
        self.act("Extrude faces / edges", "E", self.v.op_extrude, md)
        self.act("Fill face from selected vertices", "F", self.v.op_fill, md)
        self.act("Weld selected vertices", "M", self.v.op_weld, md)
        self.act("Flip normals", "N", self.v.op_flip, md)
        self.act("Snap selection to cloud surface", "Shift+S", self.v.op_snap_surface, md)
        md.addSeparator()
        bm = md.addMenu("Combine solids (select 2+ first)")
        for kind, label in (("union", "Union"), ("difference", "Difference (first picked - others)"), ("intersect", "Intersect")):
            self.act(label, None, lambda k=kind: self.v.op_boolean(k), bm)

        vw = self.view_menu
        self.act("Frame all", "Home", self.v.frame_all, vw)
        self.act("Top", "Ctrl+1", lambda: self.v.view_preset(-90, 89.5), vw)
        self.act("Front", "Ctrl+2", lambda: self.v.view_preset(-90, 0), vw)
        self.act("Right", "Ctrl+3", lambda: self.v.view_preset(0, 0), vw)
        vw.addSeparator()

    def _build_view_menu(self):
        """Panel toggles - added once both docks exist."""
        vw = self.view_menu
        for dock, key in ((self.bodies_dock, "F2"), (self.props_dock, "F3")):
            a = dock.toggleViewAction()
            a.setShortcut(QKeySequence(key))
            vw.addAction(a)
        self.act("Hide / show both panels", "F4", self.toggle_panels, vw)
        self.hud_act = self.act("Show hints overlay", "H", lambda: self.v.hud.setVisible(self.hud_act.isChecked()), vw,
                                checkable=True)
        self.hud_act.setChecked(True)

    def toggle_panels(self):
        docks = (self.bodies_dock, self.props_dock)
        show = not any(d.isVisible() for d in docks)
        for d in docks:
            d.setVisible(show)

    def _build_toolbar(self):
        tb = self.addToolBar("Tools")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        for key in ("select", "move", "points", "solid", "mesh", "measure"):
            a = self.tool_acts[key]
            a.setToolTip("%s (%s)" % (TOOL_NAMES[key], a.shortcut().toString()))
            tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.undo_act)
        tb.addAction(self.redo_act)

    # ----------------------------------------------------------- panel logic
    def pick_solid(self, key):
        self.v.solid_shape = key
        self.shape_cb.setCurrentIndex(SHAPES.index(key))
        if self.v.tool != "solid":
            self.v.set_tool("solid")
        self.update_panel()
        self.v.update_hud()
        self.v.update()

    def sync_solid_controls(self):
        """Show the viewer's solid settings in the panel (they change when a body's points are edited)."""
        v = self.v
        for w, fn in ((self.shape_cb, lambda: self.shape_cb.setCurrentIndex(SHAPES.index(v.solid_shape))),
                      (self.seg_sp, lambda: self.seg_sp.setValue(v.solid_seg)),
                      (self.sweep_mc, lambda: self.sweep_mc.setCurrentIndex(0 if v.sweep_mode == "linear" else 1)),
                      (self.sweep_st, lambda: self.sweep_st.setValue(v.sweep_lseg))):
            w.blockSignals(True); fn(); w.blockSignals(False)
        self.update_panel()

    def sync_checked(self):
        self.tool_acts[self.v.tool].setChecked(True)
        self.mode_acts[self.v.mode].setChecked(True)
        self.follow_act.setChecked(self.v.follow)
        self.follow_cb.blockSignals(True)
        self.follow_cb.setChecked(self.v.follow)
        self.follow_cb.blockSignals(False)
        self.update_panel()

    def update_panel(self, *_):
        """Show only the controls that matter for the current tool, mode and selection."""
        v = self.v
        edit = v.tool in ("select", "move")
        has_sel = bool(v.sel)
        self.p_title.setText(TOOL_NAMES[v.tool])
        self.sec_sel.setVisible(edit)
        self.sec_points.setVisible(v.tool == "points")
        self.sec_solid.setVisible(v.tool == "solid")
        self.sec_mesh.setVisible(v.tool == "mesh")
        self.sec_meas.setVisible(v.tool == "measure")
        if v.tool == "measure":
            i = self.meas_kc.findData(v.meas_kind)
            if i >= 0 and self.meas_kc.currentIndex() != i:
                self.meas_kc.blockSignals(True); self.meas_kc.setCurrentIndex(i); self.meas_kc.blockSignals(False)
            self.meas_help.setText(MEAS_HELP[v.meas_kind])
            self.meas_info.setText(v.meas_info_text())
            self.meas_finish_btn.setVisible(v.meas_kind not in ("edge",) and v.meas_kind not in MEAS_AUTO)
            self.meas_finish_btn.setEnabled(len(v.pending) >= MEAS_MIN.get(v.meas_kind, 99))
        if v.tool == "mesh":
            n = 0 if v.mp_pts is None else len(v.mp_pts)
            self.mp_lbl.setText("%s points selected" % format(n, ",") if n else "Nothing selected - drag a loop around cloud points")
            self.mp_info_lbl.setText(v.mp_info)
            self.mp_add_btn.setEnabled(v.mp_preview is not None)
        if edit:
            for k, b in self.mode_btns.items():
                b.blockSignals(True); b.setChecked(v.mode == k); b.blockSignals(False)
            self.pos_box.setVisible(bool(v.sel_verts()))
            self.move_opts.setVisible(v.tool == "move")
            self.follow_sub.setVisible(v.follow)
            self.ops_box.setVisible(has_sel)
            for btn, modes in self.op_btns:
                btn.setVisible(v.mode in modes)
        two = False
        if edit and len(v.sel) >= 2:
            try:
                two = len(v.boolean_shells()) >= 2
            except Exception:
                two = False
        self.sec_bool.setVisible(two)
        if v.tool == "solid":
            self.seg_row.setVisible(v.solid_shape in SEG_SHAPES)
            self.sweep_box.setVisible(v.solid_shape in SWEEP_SHAPES)    # linear / fitted: loft + extrusion only

    # ------------------------------------------------------------------ panel
    def _group(self, title):
        g = QGroupBox(title)
        gl = QVBoxLayout(g)
        gl.setContentsMargins(6, 4, 6, 6)
        gl.setSpacing(4)
        return g, gl

    def _build_dock(self):
        self.props_dock = QDockWidget("Properties", self)
        self.props_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable
                                    | QDockWidget.DockWidgetFeature.DockWidgetClosable)
        w = QWidget(); lay = QVBoxLayout(w)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(6)
        self.p_title = QLabel()
        self.p_title.setStyleSheet("font-weight:bold;font-size:13px;")
        lay.addWidget(self.p_title)

        # ---- combine solids: only while 2+ solids are selected
        self.sec_bool, gl = self._group("Combine solids")
        for text, kind in (("Union", "union"), ("Difference (first - others)", "difference"), ("Intersect", "intersect")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, k=kind: self.v.op_boolean(k))
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            gl.addWidget(b)
        gl.addWidget(_small("The result goes into a folder in the Bodies list - use its Undo button to split it again."))
        lay.addWidget(self.sec_bool)

        # ---- select / move
        self.sec_sel, gl = self._group("Selection")
        row = QHBoxLayout(); row.setSpacing(2)
        self.mode_btns, grp = {}, QButtonGroup(self)
        for key, text, tip in (("v", "Vertex", "Vertex mode (1)"), ("e", "Edge", "Edge mode (2)"), ("f", "Face", "Face mode (3)")):
            b = QToolButton(); b.setText(text); b.setToolTip(tip); b.setCheckable(True); b.setAutoRaise(False)
            b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, k=key: self.v.set_mode(k))
            grp.addButton(b); row.addWidget(b); self.mode_btns[key] = b
        gl.addLayout(row)
        self.sel_lbl = QLabel("nothing selected")
        gl.addWidget(self.sel_lbl)
        self.pos_box = QWidget(); pl = QVBoxLayout(self.pos_box)
        pl.setContentsMargins(0, 0, 0, 0); pl.setSpacing(2)
        self.spins = []
        for ax in "XYZ":
            sp = QDoubleSpinBox(); sp.setRange(-1e12, 1e12); sp.setDecimals(4); sp.setSingleStep(0.1)
            sp.setKeyboardTracking(False); sp.valueChanged.connect(self.coord_edited)
            sp.setMinimumWidth(60)
            self.spins.append(sp)
            pl.addLayout(_row(ax, sp))
        self.world_cb = QCheckBox("World coordinates")
        self.world_cb.setToolTip("Show the position in the original coordinates of the point cloud")
        self.world_cb.toggled.connect(self.refresh_coords)
        pl.addWidget(self.world_cb)
        gl.addWidget(self.pos_box)
        self.move_opts = QWidget(); ml = QVBoxLayout(self.move_opts)      # move tool only
        ml.setContentsMargins(0, 0, 0, 0); ml.setSpacing(3)
        self.follow_cb = QCheckBox("Follow surface (C)")
        self.follow_cb.toggled.connect(lambda x: (self.follow_act.setChecked(x), self.v.set_follow(x)))
        ml.addWidget(self.follow_cb)
        self.follow_sub = QWidget(); fl = QVBoxLayout(self.follow_sub)    # only while following is ticked
        fl.setContentsMargins(14, 0, 0, 0); fl.setSpacing(2)
        so = QDoubleSpinBox(); so.setRange(-1e9, 1e9); so.setDecimals(4); so.setSingleStep(0.05)
        so.setToolTip("Height above the cloud surface while following it")
        so.valueChanged.connect(lambda x: setattr(self.v, "surf_offset", x))
        sk = QSpinBox(); sk.setRange(4, 400); sk.setValue(self.v.surf_k)
        sk.setToolTip("Neighbouring cloud points averaged when following the surface (more = smoother)")
        sk.valueChanged.connect(lambda x: setattr(self.v, "surf_k", x))
        fl.addLayout(_row("Offset", so))
        fl.addLayout(_row("Smoothing", sk))
        ml.addWidget(self.follow_sub)
        self.local_cb = QCheckBox("Aligned handles")
        self.local_cb.setToolTip("On: handles align with the selected faces / edges / vertices "
                                 "(blue Z = surface normal). Off: world X / Y / Z.")
        self.local_cb.setChecked(self.v.local_axes)
        self.local_cb.toggled.connect(lambda x: (setattr(self.v, "local_axes", x), self.v.update()))
        ml.addWidget(self.local_cb)
        gl.addWidget(self.move_opts)
        self.ops_box = QWidget(); og = QGridLayout(self.ops_box)          # edit buttons: only what fits the mode
        og.setContentsMargins(0, 2, 0, 0); og.setSpacing(3)
        self.op_btns = []
        for n, (text, meth, modes, tip) in enumerate(EDIT_OPS):
            b = QPushButton(text); b.setToolTip(tip); b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, m=meth: getattr(self.v, m)())
            og.addWidget(b, n // 2, n % 2)
            self.op_btns.append((b, modes))
        gl.addWidget(self.ops_box)
        lay.addWidget(self.sec_sel)

        # ---- face from points
        self.sec_points, gl = self._group("Options")
        an = QSpinBox(); an.setRange(0, 32); an.setValue(self.v.auto_n); an.setSpecialValueText("manual")
        an.valueChanged.connect(lambda x: (setattr(self.v, "auto_n", x), self.v.update_hud()))
        gl.addLayout(_row("Finish after", an))
        gl.addWidget(_small("Click cloud points. The face is made after that many clicks, or press Enter (manual)."))
        lay.addWidget(self.sec_points)

        # ---- best-fit solids
        self.sec_solid, gl = self._group("Options")
        self.shape_cb = QComboBox()
        for key in SHAPES:
            self.shape_cb.addItem(SHAPE_NAMES[key], key)
        self.shape_cb.setCurrentIndex(SHAPES.index(self.v.solid_shape))
        self.shape_cb.activated.connect(lambda i: self.pick_solid(self.shape_cb.itemData(i)))
        gl.addLayout(_row("Shape", self.shape_cb))
        sg = QSpinBox(); sg.setRange(8, 96); sg.setValue(self.v.solid_seg)
        sg.setToolTip("Points around the shape (smoothness)")
        sg.valueChanged.connect(lambda x: (setattr(self.v, "solid_seg", x), self.v.update_hud(), self.v.update()))
        self.seg_row = QWidget(); sl = QVBoxLayout(self.seg_row)          # box and plane have no round parts
        sl.setContentsMargins(0, 0, 0, 0); sl.addLayout(_row("Segments", sg))
        gl.addWidget(self.seg_row)
        self.sweep_box = QWidget(); bl = QVBoxLayout(self.sweep_box)      # loft + extrusion only
        bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(3)
        mc = QComboBox()
        mc.addItem("Linear", "linear")
        mc.addItem("Fitted (curved)", "fitted")
        mc.setCurrentIndex(0 if self.v.sweep_mode == "linear" else 1)
        st = QSpinBox(); st.setRange(0, 200); st.setValue(self.v.sweep_lseg); st.setSpecialValueText("auto")
        st.setToolTip("Segments along the length of the loft / extrusion")
        def set_mode(i):
            self.v.sweep_mode = mc.itemData(i)
            self.v.update_hud(); self.v.update()
        mc.activated.connect(set_mode)
        st.valueChanged.connect(lambda x: (setattr(self.v, "sweep_lseg", x), self.v.update_hud(), self.v.update()))
        bl.addLayout(_row("Type", mc))
        bl.addLayout(_row("Along length", st))
        gl.addWidget(self.sweep_box)
        self.seg_sp, self.sweep_mc, self.sweep_st = sg, mc, st
        gl.addWidget(_small("Click points on the object's surface, then press Enter."))
        lay.addWidget(self.sec_solid)

        # ---- mesh from points (Poisson reconstruction)
        self.sec_mesh, gl = self._group("Mesh from points")
        self.mp_lbl = QLabel("")
        self.mp_lbl.setWordWrap(True)
        gl.addWidget(self.mp_lbl)
        dl = QSlider(Qt.Orientation.Horizontal); dl.setRange(5, 11); dl.setValue(self.v.mp_detail)
        dl.setToolTip("Less detail = fewer, larger triangles and a smoother surface. More detail = finer surface, more triangles.")
        dtxt = QLabel()
        def detail_changed(x):
            self.v.mp_detail = x
            dtxt.setText({5: "very low", 6: "low", 7: "medium-low", 8: "medium", 9: "medium-high", 10: "high", 11: "very high"}[x])
            self.v.schedule_mp()
        dl.valueChanged.connect(detail_changed)
        detail_changed(self.v.mp_detail)
        gl.addLayout(_row("Detail", dtxt))
        gl.addWidget(dl)
        mt = QSpinBox(); mt.setRange(0, 1_000_000); mt.setSingleStep(5000); mt.setValue(self.v.mp_max_tris)
        mt.setSpecialValueText("no limit"); mt.setKeyboardTracking(False)
        mt.setToolTip("Upper limit on triangles - the mesh is simplified down to it. Very large meshes make the viewport slow.")
        mt.valueChanged.connect(lambda x: (setattr(self.v, "mp_max_tris", x), self.v.schedule_mp()))
        gl.addLayout(_row("Max triangles", mt))
        oc = QComboBox()
        for label, key in (("Auto", "auto"), ("Outward", "outward"), ("Inward", "inward")):
            oc.addItem(label, key)
        oc.setToolTip("Which way the surface faces. Auto: outward for an object that wraps around, "
                      "toward the view for a one-sided scan (terrain, wall).")
        oc.activated.connect(lambda i: (setattr(self.v, "mp_orient", oc.itemData(i)), self.v.schedule_mp()))
        gl.addLayout(_row("Normals", oc))
        tc = QCheckBox("Trim stray surface"); tc.setChecked(self.v.mp_trim)
        tc.setToolTip("Cut away the skin that Poisson reconstruction invents where there are no points")
        tc.toggled.connect(lambda x: (setattr(self.v, "mp_trim", x), self.v.schedule_mp()))
        gl.addWidget(tc)
        self.mp_info_lbl = _small("")
        gl.addWidget(self.mp_info_lbl)
        self.mp_add_btn = QPushButton("Add mesh to model (Enter)")
        self.mp_add_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.mp_add_btn.clicked.connect(lambda: self.v.commit_mp())
        gl.addWidget(self.mp_add_btn)
        cb = QPushButton("Clear selection (Esc)"); cb.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        cb.clicked.connect(lambda: (self.v.mp_clear(), self.v.update_hud(), self.v.update(), self.update_panel()))
        gl.addWidget(cb)
        gl.addWidget(_small("Drag a loop around cloud points; Shift+drag adds more. The green wireframe is the mesh that will be added."))
        lay.addWidget(self.sec_mesh)

        # ---- measure
        self.sec_meas, gl = self._group("Measure")
        kc = QComboBox()
        for key in MEAS_MODES:
            kc.addItem(MEAS_LABELS[key], key)
        kc.activated.connect(lambda i: self.v.set_meas_kind(kc.itemData(i)))
        self.meas_kc = kc
        gl.addWidget(kc)
        self.meas_help = _small("")
        gl.addWidget(self.meas_help)
        self.meas_finish_btn = QPushButton("Finish measurement (Enter)")
        self.meas_finish_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.meas_finish_btn.clicked.connect(lambda: self.v.finish_measure())
        gl.addWidget(self.meas_finish_btn)
        gl.addWidget(_small("Selected shape"))
        self.meas_info = _small("")
        self.meas_info.setStyleSheet("color:#cfd8e3;")
        gl.addWidget(self.meas_info)
        fb = QPushButton("Diameter of selected shape")
        fb.setToolTip("Fits the best sphere or cylinder to the selected shape and measures its diameter, radius and length")
        fb.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        fb.clicked.connect(lambda: self.v.fit_selection_diameter())
        gl.addWidget(fb)
        gl.addWidget(_small("Results"))
        self.meas_list = QListWidget()
        self.meas_list.setMinimumHeight(110)
        self.meas_list.setToolTip("Click a measurement to highlight it in the view")
        self.meas_list.currentRowChanged.connect(lambda r: self.v.set_meas_sel(r))
        gl.addWidget(self.meas_list)
        row = QHBoxLayout(); row.setSpacing(3)
        for text, fn in (("Delete", lambda: self.v.delete_measure()), ("Clear all", lambda: self.v.clear_measures()),
                         ("Copy", lambda: (QApplication.clipboard().setText(self.v.measures_text()), self.v.msg("Measurements copied")))):
            b = QPushButton(text); b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, f=fn: f())
            row.addWidget(b)
        gl.addLayout(row)
        sc_cb = QCheckBox("Show measurements"); sc_cb.setChecked(self.v.meas_show)
        sc_cb.toggled.connect(lambda x: self.v.set_meas_format(show=x))
        dt_cb = QCheckBox("All values on the labels"); dt_cb.setChecked(self.v.meas_details)
        dt_cb.toggled.connect(lambda x: self.v.set_meas_format(details=x))
        gl.addWidget(sc_cb); gl.addWidget(dt_cb)
        uc = QComboBox()
        for u in ("", "mm", "cm", "m", "km", "in", "ft"):
            uc.addItem(u or "(none)", u)
        uc.setToolTip("Name of the unit shown after the numbers")
        uc.activated.connect(lambda i: self.v.set_meas_format(units=uc.itemData(i)))
        gl.addLayout(_row("Unit", uc))
        ss = QDoubleSpinBox(); ss.setRange(1e-9, 1e9); ss.setDecimals(6); ss.setValue(self.v.meas_scale)
        ss.setKeyboardTracking(False)
        ss.setToolTip("Displayed value = cloud units x this. Cloud in metres and you want mm: 1000")
        ss.valueChanged.connect(lambda x: self.v.set_meas_format(scale=x))
        gl.addLayout(_row("Scale", ss))
        ds = QSpinBox(); ds.setRange(0, 8); ds.setValue(self.v.meas_dec)
        ds.valueChanged.connect(lambda x: self.v.set_meas_format(dec=x))
        gl.addLayout(_row("Decimals", ds))
        self.meas_unit_cb, self.meas_scale_sp, self.meas_dec_sp, self.meas_show_cb = uc, ss, ds, sc_cb
        lay.addWidget(self.sec_meas)

        # ---- display (rarely changed -> folded away)
        disp = Section("Display", collapsed=True)
        ps = QDoubleSpinBox(); ps.setRange(1, 20); ps.setValue(self.v.point_size)
        ps.valueChanged.connect(lambda x: (setattr(self.v, "point_size", x), self.v.update()))
        sr = QSpinBox(); sr.setRange(3, 60); sr.setValue(int(self.v.snap_radius))
        sr.setToolTip("How close (in pixels) a click must be to a cloud point to pick it")
        sr.valueChanged.connect(lambda x: setattr(self.v, "snap_radius", float(x)))
        hc = QCheckBox("Color points by height"); hc.setChecked(True)
        hc.toggled.connect(lambda x: (setattr(self.v, "use_height_colors", x), self.v.update()))
        sl_ = QSlider(Qt.Orientation.Horizontal); sl_.setRange(10, 100); sl_.setValue(int(self.v.face_alpha * 100))
        sl_.valueChanged.connect(lambda x: (setattr(self.v, "face_alpha", x / 100.0), setattr(self.v, "_arr", None), self.v.update()))
        disp.body_lay.addLayout(_row("Point size", ps))
        disp.body_lay.addLayout(_row("Pick radius", sr))
        disp.body_lay.addWidget(hc)
        disp.body_lay.addLayout(_row("Opacity", sl_))
        lay.addWidget(disp)
        lay.addStretch(1)

        sa = QScrollArea()                    # the panel scrolls when the window is short
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sa.setFrameShape(QScrollArea.Shape.NoFrame)
        sa.setWidget(w)
        sa.setMinimumWidth(190)
        self.props_dock.setWidget(sa)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.props_dock)

    def refresh_meas_list(self):
        v = self.v
        self.meas_list.blockSignals(True)
        self.meas_list.clear()
        for i in range(len(v.measures)):
            self.meas_list.addItem(v.measure_summary(i))
        if v.meas_sel is not None and 0 <= v.meas_sel < len(v.measures):
            self.meas_list.setCurrentRow(v.meas_sel)
        self.meas_list.blockSignals(False)
        for w, fn in ((self.meas_unit_cb, lambda: self.meas_unit_cb.setCurrentIndex(max(0, self.meas_unit_cb.findData(v.meas_units)))),
                      (self.meas_scale_sp, lambda: self.meas_scale_sp.setValue(v.meas_scale)),
                      (self.meas_dec_sp, lambda: self.meas_dec_sp.setValue(v.meas_dec)),
                      (self.meas_show_cb, lambda: self.meas_show_cb.setChecked(v.meas_show))):
            w.blockSignals(True); fn(); w.blockSignals(False)

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
