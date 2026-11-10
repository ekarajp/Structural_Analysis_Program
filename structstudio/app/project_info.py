"""File > Project Information: the project data used in reports and drawing title
blocks (name, number, location, owner, engineer and licence no., checker, company and
logo, revision and date, design codes, notes).  Kept with the model; also reachable
from the New Model window."""
from __future__ import annotations

import dataclasses
from typing import Dict, Optional

from PySide6 import QtWidgets

from ..model.model import ProjectInfo
from .forms import FormDialog

FIELDS = (
    ("-", "Project", None),
    ("name", "Project name", None),
    ("number", "Project number", None),
    ("location", "Location", None),
    ("owner", "Owner / client", None),
    ("-", "People", None),
    ("engineer", "Design engineer", None),
    ("license_no", "Engineer's licence / registration no.", None),
    ("checker", "Checked by", None),
    ("company", "Company", None),
    ("logo", "Company logo (image file)", None),
    ("-", "Document", None),
    ("revision", "Revision", None),
    ("date", "Date", None),
    ("design_codes", "Design codes / standards", None),
    ("notes", "Notes", None),
)


def info_dict(info: ProjectInfo) -> Dict[str, str]:
    return {f.name: str(getattr(info, f.name) or "") for f in dataclasses.fields(ProjectInfo)}


class ProjectInfoDialog(FormDialog):
    def __init__(self, values: Dict[str, str], parent=None):
        fields = [(k, lab) if k == "-" else (k, lab, str(values.get(k, "") or "")) for k, lab, _ in FIELDS]
        super().__init__("Project Information", fields, parent,
                         note="Used in the report and on printed drawings (title block).  Kept with the model.")
        logo = self._widgets["logo"]
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self._browse_logo)
        form = self.findChild(QtWidgets.QFormLayout)
        if form is not None:
            i, _role = form.getWidgetPosition(logo)
            box = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(box)
            h.setContentsMargins(0, 0, 0, 0)
            form.removeWidget(logo)
            h.addWidget(logo, 1)
            h.addWidget(browse)
            form.setWidget(i, QtWidgets.QFormLayout.FieldRole, box)
        self.resize(520, self.sizeHint().height())

    def _browse_logo(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Company logo", "",
                                                        "Images (*.png *.jpg *.jpeg *.bmp *.svg)")
        if path:
            self._widgets["logo"].setText(path)

    def result_values(self) -> Dict[str, str]:
        v = self.values()
        return {k: str(v.get(k, "")).strip() for k, _lab, _ in FIELDS if k != "-"}


def edit_project_info(values: Dict[str, str], parent=None) -> Optional[Dict[str, str]]:
    d = ProjectInfoDialog(values, parent)
    if d.exec() != QtWidgets.QDialog.Accepted:
        return None
    out = d.result_values()
    if not out.get("name"):
        out["name"] = "Untitled"
    return out
