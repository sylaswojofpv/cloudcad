"""
bodies_panel.py -- the Bodies list (left dock).

  Point clouds   one row per loaded cloud, tick box = visible
  Bodies         one row per body, tick box = visible; click a row to select that body
  boolean folder one per boolean operation: the combined result (tick box) plus the solids it was
                 made from (greyed). The Undo button on the folder gives the original solids back.
                 Right-click a body made with the Solid tool to edit the points it was fitted to.
"""
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (QHeaderView, QLabel, QMenu, QPushButton, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from bodies import OP_NAMES

KIND, ID = Qt.ItemDataRole.UserRole, Qt.ItemDataRole.UserRole + 1
GREY = QBrush(QColor(140, 148, 158))


class BodiesPanel(QWidget):
    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.v = viewer
        self._building = False
        self._collapsed = set()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        h = self.tree.header()
        h.setStretchLastSection(False)
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemClicked.connect(self._item_clicked)
        self.tree.itemExpanded.connect(lambda it: self._collapsed.discard(self._key(it)))
        self.tree.itemCollapsed.connect(lambda it: self._collapsed.add(self._key(it)))
        self.tree.customContextMenuRequested.connect(self._menu)
        lay.addWidget(self.tree, 1)
        viewer.bodies_changed.connect(lambda: QTimer.singleShot(0, self.refresh))
        self.refresh()

    # ------------------------------------------------------------------ build
    @staticmethod
    def _key(it):
        return (it.data(0, KIND), it.data(0, ID))

    def _item(self, parent, text, kind, ident, checked=None, grey=False, bold=False):
        it = QTreeWidgetItem(parent, [text, ""])
        it.setData(0, KIND, kind)
        it.setData(0, ID, ident)
        if checked is None:
            it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
        else:
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        if grey:
            it.setForeground(0, GREY)
        if bold or grey:
            f = QFont(it.font(0))
            f.setBold(bold)
            f.setItalic(grey)
            it.setFont(0, f)
        return it

    def _body_item(self, parent, b):
        label = b.name + ("" if b.state == "live" else "  (hidden)")
        it = self._item(parent, label, "body", b.id, checked=(b.state == "live"), grey=(b.state == "hidden"))
        if b.src:
            it.setToolTip(0, "Made from picked points - right-click to edit them")
        return it

    def _add_folder(self, parent, f, as_operand=False):
        reg = self.v.reg
        it = self._item(parent, "%s  [%s]" % (f.name, OP_NAMES[f.op]), "folder", f.id, bold=True)
        inner = reg.parent_folder(f.id) is not None
        btn = QPushButton("Undo")
        btn.setToolTip("Undo this %s - the %d original solids come back"
                       % (OP_NAMES[f.op].lower(), len(f.operands)) if not inner
                       else "This boolean is inside another one - undo the outer one first")
        btn.setEnabled(not inner)
        btn.clicked.connect(lambda _=False, fid=f.id: self.v.undo_boolean(fid))
        self.tree.setItemWidget(it, 1, btn)
        if not as_operand:                                 # (an operand's own result row is its folder row)
            res = reg.bodies.get(f.result)
            if res is not None:
                self._body_item(it, res)
        for o in f.operands:
            b = reg.bodies.get(o)
            if b is None:
                continue
            nested = reg.folder_of_result(o)
            if nested is not None:
                self._add_folder(it, nested, as_operand=True)
            else:
                self._item(it, b.name + "  (combined)", "operand", b.id, grey=True)
        it.setExpanded(self._key(it) not in self._collapsed)
        return it

    def refresh(self):
        v, reg = self.v, self.v.reg
        self._building = True
        self.tree.setUpdatesEnabled(False)
        try:
            self.tree.clear()
            g = self._item(self.tree, "Point clouds (%d)" % len(v.clouds), "group", "clouds", bold=True)
            for i, c in enumerate(v.clouds):
                self._item(g, "%s  (%s pts)" % (c["name"], format(len(c["pts"]), ",")), "cloud", i, checked=c["visible"])
            g.setExpanded(self._key(g) not in self._collapsed)

            top = []                                       # creation order: bodies and folders by id
            for b in reg.bodies.values():
                if b.state != "consumed" and b.folder is None and not reg.is_result(b.id):
                    top.append((b.id, "body", b))
            for f in reg.folders.values():
                if reg.parent_folder(f.id) is None:
                    top.append((f.id, "folder", f))
            top.sort(key=lambda t: t[0])
            g = self._item(self.tree, "Bodies (%d)" % len(top), "group", "bodies", bold=True)
            for _, kind, obj in top:
                if kind == "body":
                    self._body_item(g, obj)
                else:
                    self._add_folder(g, obj)
            g.setExpanded(self._key(g) not in self._collapsed)
        finally:
            self.tree.setUpdatesEnabled(True)
            self._building = False

    # ----------------------------------------------------------------- events
    def _item_changed(self, it, col):
        if self._building or col != 0 or not (it.flags() & Qt.ItemFlag.ItemIsUserCheckable):
            return
        on = it.checkState(0) == Qt.CheckState.Checked
        kind, ident = it.data(0, KIND), it.data(0, ID)
        if kind == "cloud":
            self.v.set_cloud_visible(ident, on)
        elif kind == "body":
            self.v.set_body_visible(ident, on)

    def _item_clicked(self, it, col):
        kind, ident = it.data(0, KIND), it.data(0, ID)
        if kind == "body":
            self.v.select_body(ident)
        elif kind == "folder":
            f = self.v.reg.folders.get(ident)
            if f is not None:
                self.v.select_body(f.result)

    def _menu(self, pos):
        it = self.tree.itemAt(pos)
        if it is None:
            return
        kind, ident = it.data(0, KIND), it.data(0, ID)
        reg = self.v.reg
        m = QMenu(self)
        if kind in ("body", "operand"):
            b = reg.bodies.get(ident)
            if b is not None and b.state == "live":
                m.addAction("Select", lambda: self.v.select_body(ident))
            if b is not None and b.src:
                m.addAction("Edit points…", lambda: self.v.edit_body_points(ident))
        elif kind == "folder":
            f = reg.folders.get(ident)
            if f is not None and reg.parent_folder(ident) is None:
                m.addAction("Undo %s" % f.name, lambda: self.v.undo_boolean(ident))
        if m.actions():
            m.exec(self.tree.viewport().mapToGlobal(pos))
