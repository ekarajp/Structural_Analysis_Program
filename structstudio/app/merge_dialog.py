"""Merge Joints - choose which joint's value is kept when the two joints being
merged have different values (springs, masses, diaphragm, local axes ...)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtWidgets


class MergeChoiceDialog(QtWidgets.QDialog):
    """One row per conflict: the two values side by side, a choice between them.
    Identical values and values set on one joint only are merged without asking."""

    def __init__(self, items, parent=None):
        super().__init__(parent)
        self.items = list(items)
        self.setWindowTitle("Merge Joints - choose the values to keep")
        self.resize(760, 160 + 34 * min(len(self.items), 10))
        lay = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "These joints are at the same place but have <b>different values</b>. Choose, for each, the value the "
            "merged joint keeps. (Loads of both joints are always kept; a support restrained on either joint stays "
            "restrained.)")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.table = QtWidgets.QTableWidget(len(self.items), 4)
        self.table.setHorizontalHeaderLabels(["What", "Joint", "Keep this value", "or this value"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        hh.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.buttons: List[Tuple[QtWidgets.QRadioButton, QtWidgets.QRadioButton]] = []
        for r, it in enumerate(self.items):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(it.label))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{it.keep} + {it.gone}"))
            group = QtWidgets.QButtonGroup(self)
            a = QtWidgets.QRadioButton(f"{it.keep}: {it.keep_value}")
            b = QtWidgets.QRadioButton(f"{it.gone}: {it.gone_value}")
            a.setToolTip(a.text())
            b.setToolTip(b.text())
            a.setChecked(True)
            group.addButton(a)
            group.addButton(b)
            for c, btn in ((2, a), (3, b)):
                cell = QtWidgets.QWidget()
                h = QtWidgets.QHBoxLayout(cell)
                h.setContentsMargins(6, 0, 6, 0)
                h.addWidget(btn)
                self.table.setCellWidget(r, c, cell)
            self.buttons.append((a, b))
        lay.addWidget(self.table, 1)
        row = QtWidgets.QHBoxLayout()
        all_a = QtWidgets.QPushButton("All: the joint that stays")
        all_b = QtWidgets.QPushButton("All: the merged joint")
        all_a.clicked.connect(lambda: [p[0].setChecked(True) for p in self.buttons])
        all_b.clicked.connect(lambda: [p[1].setChecked(True) for p in self.buttons])
        row.addWidget(all_a)
        row.addWidget(all_b)
        row.addStretch(1)
        lay.addLayout(row)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Merge")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def choose(self, row: int, take_merged: bool):
        self.buttons[row][1 if take_merged else 0].setChecked(True)

    def choices(self) -> Dict[Tuple[str, str, str], str]:
        out = {}
        for it, (a, b) in zip(self.items, self.buttons):
            out[(it.keep, it.gone, it.group)] = it.gone if b.isChecked() else it.keep
        return out


def ask_merge_choices(items, parent=None) -> Optional[Dict[Tuple[str, str, str], str]]:
    """The window's resolver for model.point_ops: None when the user cancels."""
    d = MergeChoiceDialog(items, parent)
    if not d.exec():
        return None
    return d.choices()
