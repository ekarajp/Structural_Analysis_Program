"""Start page: New Model / Browse / Import and the recent models with a picture
of each (like the start window of Archicad), shown when the program starts."""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from . import theme
from .sketch import model_sketch, paint_sketch

MAX_RECENT = 16
KEY_RECENT = "start/recent"
KEY_SHOW = "start/show_at_startup"


def cache_dir() -> str:
    d = os.environ.get("STRUCTSTUDIO_THUMBS") or os.path.join(
        QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.AppLocalDataLocation) or os.path.expanduser("~"),
        "thumbnails")
    return d


class RecentModels:
    """Recently opened / saved models (QSettings) and their sketches (small JSON
    files in the program's local data folder)."""

    def __init__(self, settings: QtCore.QSettings, folder: Optional[str] = None):
        self.settings = settings
        self.folder = folder or cache_dir()

    # -------------------------------------------------------------- list
    def entries(self) -> List[dict]:
        raw = self.settings.value(KEY_RECENT, "")
        try:
            data = json.loads(raw) if raw else []
        except (TypeError, ValueError):
            data = []
        return [e for e in data if isinstance(e, dict) and e.get("path")]

    def _store(self, entries: List[dict]):
        self.settings.setValue(KEY_RECENT, json.dumps(entries[:MAX_RECENT]))
        self.settings.sync()

    @staticmethod
    def _same(a: str, b: str) -> bool:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))

    def add(self, path: str, model=None, analysed: bool = False):
        """Put ``path`` first; with a model, its summary and sketch are kept too."""
        old = next((e for e in self.entries() if self._same(e["path"], path)), {})
        e = {"path": os.path.abspath(path), "name": os.path.basename(path), "opened": time.time(),
             "info": old.get("info", "")}
        for ext in (".ssj.json", ".json"):
            if e["name"].lower().endswith(ext):
                e["name"] = e["name"][:-len(ext)]
                break
        if model is not None:
            e["info"] = self.describe(model, analysed)
            self._save_sketch(path, model)
        self._store([e] + [x for x in self.entries() if not self._same(x["path"], path)])

    def remove(self, path: str):
        self._store([x for x in self.entries() if not self._same(x["path"], path)])
        try:
            os.remove(self.sketch_file(path))
        except OSError:
            pass

    @staticmethod
    def describe(model, analysed: bool = False) -> str:
        n_st = max(0, len(model.stories) - 1)
        parts = [f"{n_st} stor{'y' if n_st == 1 else 'ies'}" if n_st else "no stories",
                 f"{len(model.frames):,} frames"]
        if model.areas:
            parts.append(f"{len(model.areas):,} shells")
        if not model.frames and not model.areas:
            parts[-1] = "grid only" if model.grid_lines else "empty"
        if analysed:
            parts.append("analysed")
        return " · ".join(parts)

    # ------------------------------------------------------------ sketch
    def sketch_file(self, path: str) -> str:
        key = hashlib.sha1(os.path.normcase(os.path.abspath(path)).encode("utf-8")).hexdigest()[:20]
        return os.path.join(self.folder, key + ".json")

    def _save_sketch(self, path: str, model):
        try:
            os.makedirs(self.folder, exist_ok=True)
            with open(self.sketch_file(path), "w", encoding="utf-8") as fh:
                json.dump(model_sketch(model), fh, separators=(",", ":"))
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            pass  # a picture is a convenience: never stop a save for it

    def sketch(self, path: str) -> Optional[Dict[str, list]]:
        try:
            with open(self.sketch_file(path), encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None


# =============================================================================
def when_text(t: float, now: Optional[float] = None) -> str:
    now = now or time.time()
    d0 = time.localtime(now)
    d = time.localtime(t)
    hm = time.strftime("%H:%M", d)
    if d.tm_year == d0.tm_year and d.tm_yday == d0.tm_yday:
        return f"Today {hm}"
    if now - t < 2 * 86400 and (d0.tm_yday - d.tm_yday) % 366 == 1:
        return f"Yesterday {hm}"
    return time.strftime("%d %b %Y", d).lstrip("0")


class _TileDelegate(QtWidgets.QStyledItemDelegate):
    """A card: the model picture on top, name / summary / date below."""

    TILE = QtCore.QSize(236, 214)
    PIC = 150

    def __init__(self, dialog: "StartDialog"):
        super().__init__(dialog)
        self.dialog = dialog

    def sizeHint(self, option, index):
        if self.dialog.list.viewMode() == QtWidgets.QListView.IconMode:
            return self.TILE
        return QtCore.QSize(400, 64)

    def paint(self, p: QtGui.QPainter, option, index):
        e = index.data(QtCore.Qt.UserRole)
        if not e:
            return
        pal = option.palette
        tiles = self.dialog.list.viewMode() == QtWidgets.QListView.IconMode
        r = QtCore.QRectF(option.rect).adjusted(4, 4, -4, -4)
        sel = bool(option.state & QtWidgets.QStyle.State_Selected)
        hover = bool(option.state & QtWidgets.QStyle.State_MouseOver)
        p.save()
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setPen(QtGui.QPen(pal.color(QtGui.QPalette.Highlight) if sel else pal.color(QtGui.QPalette.Mid),
                            2.0 if sel else 1.0))
        p.setBrush(pal.color(QtGui.QPalette.Base))
        p.drawRoundedRect(r, 3, 3)
        if hover and not sel:
            p.setPen(QtGui.QPen(pal.color(QtGui.QPalette.Highlight), 1.0))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawRoundedRect(r, 3, 3)
        colors = theme.active().colors()
        missing = not os.path.exists(e["path"])
        sk = self.dialog.recent.sketch(e["path"])
        if tiles:
            pic = QtCore.QRectF(r.left() + 1, r.top() + 1, r.width() - 2, self.PIC)
            text = QtCore.QRectF(r.left() + 9, pic.bottom() + 6, r.width() - 18, r.bottom() - pic.bottom() - 8)
        else:
            pic = QtCore.QRectF(r.left() + 1, r.top() + 1, 70, r.height() - 2)
            text = QtCore.QRectF(pic.right() + 10, r.top() + 4, r.width() - pic.width() - 18, r.height() - 8)
        p.save()
        p.setClipRect(pic)
        paint_sketch(p, pic, sk, colors, pad=12 if tiles else 5)
        if sk is None:
            p.setPen(pal.color(QtGui.QPalette.PlaceholderText))
            p.drawText(pic, QtCore.Qt.AlignCenter, "No picture yet" if tiles else "")
        p.restore()
        p.setPen(QtGui.QPen(pal.color(QtGui.QPalette.Midlight), 1.0))
        if tiles:
            p.drawLine(QtCore.QPointF(r.left() + 1, pic.bottom()), QtCore.QPointF(r.right() - 1, pic.bottom()))
        f = QtGui.QFont(option.font)
        f.setBold(True)
        p.setFont(f)
        fm = QtGui.QFontMetrics(f)
        txt = pal.color(QtGui.QPalette.Disabled if missing else QtGui.QPalette.Normal, QtGui.QPalette.Text)
        muted = pal.color(QtGui.QPalette.PlaceholderText)
        p.setPen(txt)
        lh = fm.height() + 1
        p.drawText(QtCore.QRectF(text.left(), text.top(), text.width(), lh), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                   fm.elidedText(e.get("name") or os.path.basename(e["path"]), QtCore.Qt.ElideRight, int(text.width())))
        p.setFont(option.font)
        fm2 = QtGui.QFontMetrics(option.font)
        p.setPen(muted)
        info = "File not found" if missing else (e.get("info") or "")
        line2 = info if tiles else f"{info}   ·   {when_text(e.get('opened', 0))}"
        p.drawText(QtCore.QRectF(text.left(), text.top() + lh, text.width(), lh), QtCore.Qt.AlignLeft,
                   fm2.elidedText(line2, QtCore.Qt.ElideRight, int(text.width())))
        line3 = f"{when_text(e.get('opened', 0))} · {os.path.dirname(e['path'])}" if tiles else os.path.dirname(e["path"])
        p.drawText(QtCore.QRectF(text.left(), text.top() + 2 * lh, text.width(), lh), QtCore.Qt.AlignLeft,
                   fm2.elidedText(line3, QtCore.Qt.ElideMiddle, int(text.width())))
        p.restore()


class StartDialog(QtWidgets.QDialog):
    """The start window.  After exec(), ``choice`` is ("new",), ("open",),
    ("import",), ("recent", path) or None."""

    def __init__(self, settings: QtCore.QSettings, parent=None, recent: Optional[RecentModels] = None):
        super().__init__(parent)
        self.settings = settings
        self.recent = recent or RecentModels(settings)
        self.choice = None
        self.setWindowTitle("Structural Studio")
        self.setObjectName("startPage")
        self.resize(1100, 720)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # ---- header: title and the three ways in
        head = QtWidgets.QFrame()
        head.setObjectName("startHeader")
        head.setAutoFillBackground(True)
        hp = head.palette()
        hp.setColor(QtGui.QPalette.Window, hp.color(QtGui.QPalette.Base))
        head.setPalette(hp)
        hl = QtWidgets.QHBoxLayout(head)
        hl.setContentsMargins(18, 12, 18, 12)
        titles = QtWidgets.QVBoxLayout()
        titles.setSpacing(2)
        t = QtWidgets.QLabel("Welcome to Structural Studio")
        tf = t.font()
        tf.setPointSizeF(tf.pointSizeF() * 1.35)
        tf.setBold(True)
        t.setFont(tf)
        sub = QtWidgets.QLabel("Start a new model, open one from disk, or continue a recent model.")
        sub.setForegroundRole(QtGui.QPalette.PlaceholderText)
        titles.addWidget(t)
        titles.addWidget(sub)
        hl.addLayout(titles)
        hl.addStretch(1)
        self.btn_new = self._big_button("New Model...", "file_new", "Settings, grid and stories, starting structure",
                                        lambda: self._done(("new",)), primary=True)
        self.btn_open = self._big_button("Browse...", "file_open", "Open a Structural Studio model (.ssj.json)",
                                         lambda: self._done(("open",)))
        self.btn_import = self._big_button("Import ETABS / SAP2000 / SAFE...", None,
                                           ".e2k .$et .edb .s2k .$2k .sdb .f2k .fdb", lambda: self._done(("import",)))
        for b in (self.btn_new, self.btn_open, self.btn_import):
            hl.addWidget(b)
        lay.addWidget(head)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Plain)
        line.setForegroundRole(QtGui.QPalette.Mid)
        lay.addWidget(line)

        # ---- recent models
        body = QtWidgets.QVBoxLayout()
        body.setContentsMargins(18, 12, 18, 10)
        bar = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QLabel("Recent models")
        lf = lab.font()
        lf.setBold(True)
        lab.setFont(lf)
        bar.addWidget(lab)
        bar.addSpacing(4)
        self.count = QtWidgets.QLabel()
        self.count.setForegroundRole(QtGui.QPalette.PlaceholderText)
        bar.addWidget(self.count)
        bar.addStretch(1)
        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Find a model")
        self.search.setClearButtonEnabled(True)
        self.search.setFixedWidth(240)
        self.search.textChanged.connect(self._fill)
        bar.addWidget(self.search)
        bar.addWidget(QtWidgets.QLabel("Sort"))
        self.sort = QtWidgets.QComboBox()
        self.sort.addItems(["Last opened", "Name", "Folder"])
        self.sort.currentIndexChanged.connect(self._fill)
        bar.addWidget(self.sort)
        self.btn_tiles = self._view_button("view_tiles", "Show as pictures", True)
        self.btn_list = self._view_button("view_list", "Show as a list", False)
        bar.addWidget(self.btn_tiles)
        bar.addWidget(self.btn_list)
        body.addLayout(bar)
        self.list = QtWidgets.QListWidget()
        self.list.setObjectName("recentModels")
        self.list.setMouseTracking(True)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.list.setUniformItemSizes(True)
        self.list.setResizeMode(QtWidgets.QListView.Adjust)
        self.list.setMovement(QtWidgets.QListView.Static)
        self.list.setSpacing(4)
        self.list.setItemDelegate(_TileDelegate(self))
        self.list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._context_menu)
        self.list.itemDoubleClicked.connect(lambda it: self._open_item(it))
        self.list.itemSelectionChanged.connect(self._selection)
        lp = self.list.palette()
        lp.setColor(QtGui.QPalette.Base, lp.color(QtGui.QPalette.Window))
        self.list.setPalette(lp)
        body.addWidget(self.list, 1)
        self.empty = QtWidgets.QLabel("No recent models yet. Models you open or save are listed here.")
        self.empty.setAlignment(QtCore.Qt.AlignCenter)
        self.empty.setForegroundRole(QtGui.QPalette.PlaceholderText)
        body.addWidget(self.empty, 1)
        lay.addLayout(body, 1)

        # ---- footer
        foot = QtWidgets.QHBoxLayout()
        foot.setContentsMargins(14, 8, 14, 10)
        self.show_check = QtWidgets.QCheckBox("Show this window at startup")
        self.show_check.setChecked(self.show_at_startup(settings))
        self.show_check.toggled.connect(lambda on: settings.setValue(KEY_SHOW, bool(on)))
        self.show_check.setToolTip("Turned off, it can be turned on again: Options > Show Start Page at Startup "
                                   "(or open it any time: File > Start Page...)")
        foot.addWidget(self.show_check)
        self.show_note = QtWidgets.QLabel("To show it again: Options > Show Start Page at Startup")
        self.show_note.setStyleSheet("color: palette(placeholder-text);")
        self.show_note.setVisible(not self.show_check.isChecked())
        self.show_check.toggled.connect(lambda on: self.show_note.setVisible(not on))
        foot.addWidget(self.show_note)
        foot.addStretch(1)
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        self.btn_go = QtWidgets.QPushButton("Open")
        self.btn_go.setDefault(True)
        self.btn_go.clicked.connect(lambda: self._open_item(self.list.currentItem()))
        foot.addWidget(self.btn_close)
        foot.addWidget(self.btn_go)
        lay.addLayout(foot)
        self._set_view(settings.value("start/view", "tiles") != "list")
        self._fill()

    # ------------------------------------------------------------ widgets
    def _big_button(self, text, icon, tip, slot, primary=False) -> QtWidgets.QPushButton:
        from .icons import icon as make_icon
        b = QtWidgets.QPushButton(text)
        if icon:
            b.setIcon(make_icon(icon))
        b.setIconSize(QtCore.QSize(20, 20))
        b.setMinimumHeight(36)
        b.setToolTip(tip)
        b.clicked.connect(slot)
        if primary:
            b.setObjectName("primaryButton")
            pal = b.palette()
            pal.setColor(QtGui.QPalette.Button, pal.color(QtGui.QPalette.Highlight))
            pal.setColor(QtGui.QPalette.ButtonText, pal.color(QtGui.QPalette.HighlightedText))
            b.setPalette(pal)
            f = b.font()
            f.setBold(True)
            b.setFont(f)
        return b

    def _view_button(self, icon, tip, checked) -> QtWidgets.QToolButton:
        b = QtWidgets.QToolButton()
        b.setCheckable(True)
        b.setChecked(checked)
        b.setToolTip(tip)
        b.setAutoRaise(True)
        b.setText("▦" if icon == "view_tiles" else "☰")
        b.clicked.connect(lambda: self._set_view(icon == "view_tiles"))
        return b

    def _set_view(self, tiles: bool):
        self.btn_tiles.setChecked(tiles)
        self.btn_list.setChecked(not tiles)
        if tiles:
            self.list.setViewMode(QtWidgets.QListView.IconMode)
            self.list.setFlow(QtWidgets.QListView.LeftToRight)
            self.list.setWrapping(True)
        else:
            self.list.setViewMode(QtWidgets.QListView.ListMode)
            self.list.setFlow(QtWidgets.QListView.TopToBottom)
            self.list.setWrapping(False)
        self.settings.setValue("start/view", "tiles" if tiles else "list")
        self.list.doItemsLayout()

    # -------------------------------------------------------------- items
    def _fill(self):
        entries = self.recent.entries()
        q = self.search.text().strip().lower()
        if q:
            entries = [e for e in entries if q in e["path"].lower() or q in (e.get("info") or "").lower()]
        mode = self.sort.currentIndex()
        if mode == 1:
            entries.sort(key=lambda e: (e.get("name") or "").lower())
        elif mode == 2:
            entries.sort(key=lambda e: (os.path.dirname(e["path"]).lower(), (e.get("name") or "").lower()))
        self.list.clear()
        for e in entries:
            it = QtWidgets.QListWidgetItem(e.get("name") or os.path.basename(e["path"]))
            it.setData(QtCore.Qt.UserRole, e)
            it.setToolTip(e["path"])
            self.list.addItem(it)
        total = len(self.recent.entries())
        self.count.setText(f"({total})" if total else "")
        self.list.setVisible(bool(total))
        self.empty.setVisible(not total)
        if self.list.count():
            self.list.setCurrentRow(0)
        self._selection()

    def _selection(self):
        it = self.list.currentItem()
        self.btn_go.setEnabled(it is not None and os.path.exists(it.data(QtCore.Qt.UserRole)["path"]))

    def _open_item(self, it):
        if it is None:
            return
        path = it.data(QtCore.Qt.UserRole)["path"]
        if not os.path.exists(path):
            ok = QtWidgets.QMessageBox.question(
                self, "File not found", f"{path}\n\nwas moved or deleted. Remove it from the list?")
            if ok == QtWidgets.QMessageBox.Yes:
                self.recent.remove(path)
                self._fill()
            return
        self._done(("recent", path))

    def _context_menu(self, pos):
        it = self.list.itemAt(pos)
        if it is None:
            return
        path = it.data(QtCore.Qt.UserRole)["path"]
        m = QtWidgets.QMenu(self)
        a_open = m.addAction("Open")
        a_open.setEnabled(os.path.exists(path))
        a_dir = m.addAction("Open Containing Folder")
        a_dir.setEnabled(os.path.isdir(os.path.dirname(path)))
        m.addSeparator()
        a_rm = m.addAction("Remove from List")
        got = m.exec(self.list.viewport().mapToGlobal(pos))
        if got is a_open:
            self._open_item(it)
        elif got is a_dir:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(os.path.dirname(path)))
        elif got is a_rm:
            self.recent.remove(path)
            self._fill()

    def _done(self, choice):
        self.choice = choice
        self.accept()

    @staticmethod
    def show_at_startup(settings: QtCore.QSettings) -> bool:
        v = settings.value(KEY_SHOW, True)
        return v not in (False, "false", "0", 0)
