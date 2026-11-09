"""Display themes (dark / light) and user graphics colours.

Like the *Options > Graphics Colors* of ETABS / SAP2000: every kind of object
and output has a colour, each theme (dark background or white background)
has its own defaults chosen to read well on that background, and the user can
override any colour per theme.  Objects can be coloured by object type (the
default), by section or by material.

This module has no Qt dependency except :func:`qt_palette` (imported lazily),
so the colour logic is testable off-screen.  :func:`apply` writes the active
colours into :data:`structstudio.app.scene.COLORS`, which every scene builder
reads at build time.
"""
from __future__ import annotations

import colorsys
import json
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

RGB = Tuple[float, float, float]

THEMES = ("light", "dark")
THEME_LABELS = {"dark": "Dark", "light": "Light"}
COLOR_BY = ("object", "section", "material")
COLOR_BY_LABELS = {"object": "Object type", "section": "Section property", "material": "Material"}

# (key, label, group) in the order shown in the colours dialog
COLOR_ITEMS: List[Tuple[str, str, str]] = [
    ("background_top", "Background (top)", "Window"),
    ("background_bottom", "Background (bottom)", "Window"),
    ("grid", "Grid lines", "Window"),
    ("grid_text", "Grid and story labels", "Window"),
    ("text", "Text", "Window"),
    ("story_plane", "Plane of the active plan / elevation shown in 3D", "Window"),
    ("beam", "Beams", "Objects"),
    ("column", "Columns", "Objects"),
    ("brace", "Braces", "Objects"),
    ("null_line", "Null lines", "Objects"),
    ("floor", "Floors / slabs", "Objects"),
    ("wall", "Walls", "Objects"),
    ("opening", "Openings / null areas", "Objects"),
    ("area_edge", "Area edges", "Objects"),
    ("support", "Supports", "Objects"),
    ("spring", "Spring supports", "Objects"),
    ("axis_1", "Local axis 1 (arrow and number)", "Objects"),
    ("axis_2", "Local axis 2 (arrow and number)", "Objects"),
    ("axis_3", "Local axis 3 (arrow and number)", "Objects"),
    ("selection", "Selected objects (dashed highlight)", "Interaction"),
    ("select_box", "Selection box (drag)", "Interaction"),
    ("preview", "Drawing preview (the line / outline being drawn)", "Interaction"),
    ("free_point", "Drawing cursor - point with no snap", "Interaction"),
    ("snap", "Snap marker", "Interaction"),
    ("tracking", "Tracking / extension guide lines (dashed)", "Interaction"),
    ("reference", "Reference lines (through joints)", "Window"),
    ("deformed", "Deformed shape", "Output"),
    ("mesh", "Analysis mesh", "Output"),
    ("moment_pos", "Diagram / contour positive", "Output"),
    ("moment_neg", "Diagram / contour negative", "Output"),
    ("load", "Loads", "Output"),
]
COLOR_KEYS = tuple(k for k, _, _ in COLOR_ITEMS)

# opacities (0..1) the user can change next to the colours; (key, label, default)
OPACITY_ITEMS: List[Tuple[str, str, float]] = [
    ("select_box_fill", "Selection box fill opacity", 0.18),
    ("select_box_border", "Selection box border opacity", 0.85),
    ("floor_fill", "Floors / ramps (fill)", 0.45),
    ("wall_fill", "Walls (fill)", 0.60),
    ("wall_plan", "Walls in plan (thickness fill)", 0.85),
    ("column_plan", "Columns in plan (section fill, not extruded)", 0.55),
    ("column_lines", "Column lines (not extruded)", 0.75),
    ("view_plane", "Plane of the active plan / elevation in 3D (blue)", 0.13),
    ("grid_lines", "Grid lines", 0.60),
    ("load_fill", "Load diagrams (fill)", 0.18),
    ("label_box", "Boxes behind load values", 0.88),
    ("diagram_fill", "Result diagrams (fill)", 0.55),
]
OPACITY_KEYS = tuple(k for k, _, _ in OPACITY_ITEMS)
OPACITY_DEFAULTS = {k: v for k, _, v in OPACITY_ITEMS}

# symbol sizes and line weights (theme independent), Options > Graphics Sizes:
# (key, label, default, minimum, maximum, step, suffix)
SIZE_ITEMS: List[Tuple[str, str, float, float, float, float, str]] = [
    ("support_size", "Support / spring symbols", 2.0, 0.25, 10.0, 0.25, " x"),
    ("joint_size", "Joint dots", 1.0, 0.25, 6.0, 0.25, " x"),
    ("frame_width", "Beam / brace lines", 3.0, 0.5, 12.0, 0.5, " px"),
    ("column_width", "Column lines", 3.6, 0.5, 12.0, 0.5, " px"),
    ("null_width", "Null lines", 1.5, 0.5, 8.0, 0.5, " px"),
    ("area_edge_width", "Area edges", 1.8, 0.5, 8.0, 0.25, " px"),
    ("grid_width", "Grid lines", 1.0, 0.5, 6.0, 0.5, " px"),
    ("deformed_width", "Deformed shape lines", 2.5, 0.5, 10.0, 0.5, " px"),
    ("load_size", "Load arrows and diagrams", 1.0, 0.25, 5.0, 0.25, " x"),
    ("load_text", "Load values (text)", 13.0, 6.0, 36.0, 1.0, " pt"),
    ("label_text", "Object labels and grid bubbles (text)", 11.0, 6.0, 36.0, 1.0, " pt"),
    ("bubble_size", "Grid bubble circle (diameter)", 1.0, 0.25, 4.0, 0.25, " x"),
    ("bubble_gap", "Grid bubble distance from the grid line", 1.0, 0.0, 6.0, 0.25, " x"),
    ("spring_width", "Spring support symbols (line)", 2.0, 0.5, 8.0, 0.5, " px"),
    ("free_point_size", "Drawing cursor - point with no snap (size)", 1.4, 0.25, 5.0, 0.25, " x"),
    ("free_point_width", "Drawing cursor - point with no snap (line)", 2.5, 0.5, 8.0, 0.5, " px"),
    ("tracking_width", "Tracking / extension guide lines (dashed)", 1.5, 0.5, 6.0, 0.5, " px"),
    ("local_axes_length", "Local axes arrows (length)", 1.5, 0.25, 6.0, 0.25, " x"),
    ("local_axes_width", "Local axes arrows (line)", 2.5, 0.5, 10.0, 0.5, " px"),
    ("local_axes_text", "Local axes numbers 1 2 3 (text)", 16.0, 6.0, 40.0, 1.0, " pt"),
    ("local_axes_gap", "Local axes numbers: distance past the arrow tip", 1.0, 0.0, 4.0, 0.25, " x"),
]
# on / off drawing options (theme independent): (key, label, default)
OPTION_ITEMS: List[Tuple[str, str, bool]] = [
    ("moment_tension_side", "Moment diagrams on the tension side "
                            "(positive moment drawn below a beam, negative above)", True),
    ("plan_positive_below", "In plan views, diagrams turned into the plan: positive below horizontal "
                            "members and right of vertical ones (off: above / left)", True),
]
OPTION_KEYS = tuple(k for k, _, _ in OPTION_ITEMS)

# pick-one drawing settings (theme independent): (key, label, default, ((value, text), ...))
CHOICE_ITEMS: List[Tuple[str, str, str, Tuple[Tuple[str, str], ...]]] = [
    ("free_point_shape", "Drawing cursor - point with no snap: shape", "cross",
     (("cross", "X cross"), ("plus", "+ plus"), ("cross_circle", "X in a circle"), ("plus_circle", "+ in a circle"),
      ("square", "Square"), ("diamond", "Diamond"), ("circle", "Circle"), ("dot", "Dot"))),
]
CHOICE_KEYS = tuple(it[0] for it in CHOICE_ITEMS)
CHOICE_DEFAULTS = {it[0]: it[2] for it in CHOICE_ITEMS}
CHOICE_VALUES = {it[0]: tuple(v for v, _ in it[3]) for it in CHOICE_ITEMS}
OPTION_DEFAULTS = {k: v for k, _, v in OPTION_ITEMS}

SIZE_KEYS = tuple(it[0] for it in SIZE_ITEMS)
SIZE_DEFAULTS = {it[0]: it[2] for it in SIZE_ITEMS}
SIZE_RANGES = {it[0]: (it[3], it[4]) for it in SIZE_ITEMS}

DEFAULTS: Dict[str, Dict[str, RGB]] = {
    # dark background (the classic black CSI model window)
    "dark": {
        # one flat colour (no gradient): the dark counterpart of the white page
        "background_top": (0.12, 0.13, 0.16),
        "background_bottom": (0.12, 0.13, 0.16),
        "grid": (0.45, 0.47, 0.52),
        "grid_text": (0.75, 0.78, 0.85),
        "text": (0.92, 0.93, 0.95),
        "story_plane": (0.35, 0.65, 1.00),
        "beam": (0.36, 0.68, 1.00),
        "column": (0.12, 0.66, 0.28),  # deep green: strong, apart from the red supports and the blue beams
        "brace": (0.80, 0.55, 1.00),  # violet
        "null_line": (0.55, 0.57, 0.62),
        "floor": (0.75, 0.80, 0.86),
        "wall": (0.86, 0.74, 0.46),  # sand: not green, apart from the floors, loads and supports
        "opening": (0.40, 0.40, 0.40),
        "area_edge": (0.66, 0.70, 0.78),
        "support": (1.00, 0.42, 0.42),
        "spring": (1.00, 0.42, 0.42),  # the support colour: the coil tells it from a rigid support
        "selection": (1.00, 1.00, 0.30),
        "select_box": (0.45, 0.70, 1.00),
        "preview": (1.00, 0.45, 0.78),
        "free_point": (1.00, 0.45, 0.78),
        "snap": (1.00, 1.00, 0.30),
        "tracking": (1.00, 0.45, 0.78),
        "reference": (0.35, 0.80, 0.85),
        "deformed": (0.98, 0.85, 0.30),
        "mesh": (0.62, 0.68, 0.78),
        "moment_pos": (0.30, 0.85, 0.62),
        "moment_neg": (1.00, 0.42, 0.38),
        "load": (1.00, 0.60, 0.05),  # orange: apart from the yellow selection and the red supports
        "axis_1": (1.00, 0.32, 0.32),
        "axis_2": (0.35, 0.90, 0.42),
        "axis_3": (0.40, 0.62, 1.00),
    },
    # white background (the light model window of the newer CSI releases):
    # deeper, more saturated colours so thin lines stay visible on white
    "light": {
        # a white page, like a drawing sheet (no grey gradient)
        "background_top": (1.00, 1.00, 1.00),
        "background_bottom": (1.00, 1.00, 1.00),
        "grid": (0.66, 0.69, 0.74),
        "grid_text": (0.25, 0.29, 0.36),
        "text": (0.10, 0.12, 0.16),
        "story_plane": (0.15, 0.45, 0.90),
        "beam": (0.02, 0.30, 0.80),
        "column": (0.00, 0.40, 0.13),  # deep green: strong, apart from the red supports and the blue beams
        "brace": (0.50, 0.20, 0.78),  # violet
        "null_line": (0.50, 0.52, 0.56),
        "floor": (0.66, 0.76, 0.90),
        "wall": (0.70, 0.56, 0.28),  # sand / ochre
        "opening": (0.62, 0.62, 0.62),
        "area_edge": (0.22, 0.28, 0.40),
        "support": (0.80, 0.10, 0.12),
        "spring": (0.80, 0.10, 0.12),  # the support colour: the coil tells it from a rigid support
        "selection": (0.90, 0.08, 0.55),
        "select_box": (0.00, 0.47, 0.84),  # the Windows / CAD selection-window blue
        "preview": (0.93, 0.20, 0.62),  # pink: green circles are the pin symbols
        "free_point": (0.93, 0.20, 0.62),
        "snap": (0.90, 0.08, 0.55),
        "tracking": (0.93, 0.20, 0.62),
        "reference": (0.00, 0.60, 0.65),
        "deformed": (0.18, 0.18, 0.22),
        "mesh": (0.30, 0.34, 0.42),
        "moment_pos": (0.00, 0.58, 0.42),
        "moment_neg": (0.86, 0.16, 0.16),
        "load": (0.76, 0.42, 0.00),  # amber
        "axis_1": (0.85, 0.12, 0.12),
        "axis_2": (0.10, 0.60, 0.20),
        "axis_3": (0.15, 0.35, 0.90),
    },
}

# Qt widget chrome per theme (not user editable)
UI: Dict[str, Dict[str, str]] = {
    "dark": {"window": "#2b2f36", "base": "#1f2228", "alt": "#2f343c", "text": "#e6e8eb", "mid": "#4a505a",
             "button": "#353a42", "disabled": "#80868f", "accent": "#3d82d6", "accent_text": "#ffffff",
             "header_off": "#3a4049", "header_off_text": "#d5d9de", "tab": "#3a4049", "tab_text": "#e6e8eb",
             "ink": "#d7dbe0", "tooltip": "#383d45"},
    "light": {"window": "#f0f2f5", "base": "#ffffff", "alt": "#f4f6f8", "text": "#20242a", "mid": "#c4cad2",
              "button": "#e6e9ee", "disabled": "#9aa0a8", "accent": "#2f6db3", "accent_text": "#ffffff",
              "header_off": "#c9ced5", "header_off_text": "#20242a", "tab": "#e6e9ee", "tab_text": "#20242a",
              "ink": "#282c34", "tooltip": "#ffffe1"},
}


# ----------------------------------------------------------------------------
def to_hex(rgb: RGB) -> str:
    r, g, b = (max(0, min(255, int(round(c * 255)))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def from_hex(text: str) -> RGB:
    t = str(text).strip().lstrip("#")
    if len(t) == 3:
        t = "".join(c * 2 for c in t)
    if len(t) != 6:
        raise ValueError(f"Not a colour: {text!r}")
    return tuple(int(t[k:k + 2], 16) / 255.0 for k in (0, 2, 4))  # type: ignore[return-value]


def auto_color(name: str, theme: str = "dark", index: Optional[int] = None) -> RGB:
    """A stable, distinct colour for a section / material: golden-ratio hue
    spacing on its definition ``index`` (distinct neighbours), else on a hash of
    the name; lighter on dark backgrounds and deeper on white."""
    seed = index if index is not None else zlib.crc32(name.encode("utf-8"))
    h = (0.08 + seed * 0.618033988749895) % 1.0
    if theme == "light":
        return colorsys.hls_to_rgb(h, 0.42, 0.75)
    return colorsys.hls_to_rgb(h, 0.62, 0.70)


@dataclass
class ColorSettings:
    theme: str = "light"
    color_by: str = "object"
    overrides: Dict[str, Dict[str, str]] = field(default_factory=lambda: {t: {} for t in THEMES})
    opacities: Dict[str, float] = field(default_factory=dict)  # theme independent
    sizes: Dict[str, float] = field(default_factory=dict)  # theme independent
    options: Dict[str, bool] = field(default_factory=dict)  # theme independent, non-defaults only
    choices: Dict[str, str] = field(default_factory=dict)  # theme independent, non-defaults only

    def __post_init__(self):
        if self.theme not in THEMES:
            self.theme = "light"
        if self.color_by not in COLOR_BY:
            self.color_by = "object"
        for t in THEMES:
            self.overrides.setdefault(t, {})
        self.opacities = {k: float(v) for k, v in self.opacities.items() if k in OPACITY_KEYS}
        self.sizes = {k: float(v) for k, v in self.sizes.items() if k in SIZE_KEYS}
        self.options = {k: bool(v) for k, v in self.options.items() if k in OPTION_KEYS}
        self.choices = {k: str(v) for k, v in self.choices.items() if k in CHOICE_KEYS and v in CHOICE_VALUES[k]}

    def choice(self, key: str) -> str:
        return self.choices.get(key, CHOICE_DEFAULTS[key])

    def set_choice(self, key: str, value: str):
        if key not in CHOICE_KEYS or value not in CHOICE_VALUES[key]:
            raise KeyError(f"{key}={value}")
        if value == CHOICE_DEFAULTS[key]:
            self.choices.pop(key, None)
        else:
            self.choices[key] = value

    def option(self, key: str) -> bool:
        return bool(self.options.get(key, OPTION_DEFAULTS[key]))

    def set_option(self, key: str, on: bool):
        if key not in OPTION_KEYS:
            raise KeyError(key)
        if bool(on) == OPTION_DEFAULTS[key]:
            self.options.pop(key, None)
        else:
            self.options[key] = bool(on)

    def size(self, key: str) -> float:
        lo, hi = SIZE_RANGES[key]
        return max(lo, min(hi, self.sizes.get(key, SIZE_DEFAULTS[key])))

    def set_size(self, key: str, value: float):
        if key not in SIZE_KEYS:
            raise KeyError(key)
        lo, hi = SIZE_RANGES[key]
        value = max(lo, min(hi, float(value)))
        if abs(value - SIZE_DEFAULTS[key]) < 1e-9:
            self.sizes.pop(key, None)
        else:
            self.sizes[key] = value

    def opacity(self, key: str) -> float:
        return max(0.0, min(1.0, self.opacities.get(key, OPACITY_DEFAULTS[key])))

    def set_opacity(self, key: str, value: float):
        if key not in OPACITY_KEYS:
            raise KeyError(key)
        value = max(0.0, min(1.0, float(value)))
        if abs(value - OPACITY_DEFAULTS[key]) < 1e-9:
            self.opacities.pop(key, None)
        else:
            self.opacities[key] = value

    # ------------------------------------------------------------------ api
    def colors(self, theme: Optional[str] = None) -> Dict[str, RGB]:
        theme = theme or self.theme
        out = dict(DEFAULTS[theme])
        for k, v in self.overrides.get(theme, {}).items():
            if k in out:
                try:
                    out[k] = from_hex(v)
                except ValueError:
                    pass
        return out

    def set_color(self, key: str, rgb: RGB, theme: Optional[str] = None):
        if key not in COLOR_KEYS:
            raise KeyError(key)
        theme = theme or self.theme
        if to_hex(rgb) == to_hex(DEFAULTS[theme][key]):
            self.overrides[theme].pop(key, None)
        else:
            self.overrides[theme][key] = to_hex(rgb)

    def reset(self, theme: Optional[str] = None):
        """Everything back to the defaults: the colours of ``theme`` (the active
        one by default), the transparencies and the sizes."""
        self.overrides[theme or self.theme] = {}
        self.opacities = {}
        self.sizes = {}
        self.options = {}
        self.choices = {}

    def reset_all(self):
        """Colours of every theme, transparencies, sizes and options back to the defaults."""
        for t in THEMES:
            self.overrides[t] = {}
        self.opacities = {}
        self.sizes = {}
        self.options = {}

    def reset_color(self, key: str, theme: Optional[str] = None):
        self.overrides.get(theme or self.theme, {}).pop(key, None)

    def reset_colors(self, keys, theme: Optional[str] = None):
        for k in keys:
            self.reset_color(k, theme)

    def reset_opacity(self, key: str):
        self.opacities.pop(key, None)

    def reset_size(self, key: str):
        self.sizes.pop(key, None)

    def is_default(self) -> bool:
        return (not any(self.overrides.get(t) for t in THEMES) and not self.opacities and not self.sizes
                and not self.options and not self.choices)

    def size_is_custom(self, key: str) -> bool:
        return key in self.sizes

    def opacity_is_custom(self, key: str) -> bool:
        return key in self.opacities

    def is_custom(self, key: str, theme: Optional[str] = None) -> bool:
        return key in self.overrides.get(theme or self.theme, {})

    def ui(self) -> Dict[str, str]:
        return UI[self.theme]

    def copy(self) -> "ColorSettings":
        return ColorSettings.from_json(self.to_json())

    # ---------------------------------------------------------- persistence
    def to_json(self) -> str:
        return json.dumps({"theme": self.theme, "color_by": self.color_by, "overrides": self.overrides,
                           "opacities": self.opacities, "sizes": self.sizes,
                           "options": self.options, "choices": self.choices})

    @classmethod
    def from_json(cls, text: Optional[str]) -> "ColorSettings":
        try:
            d = json.loads(text) if text else {}
        except (TypeError, ValueError):
            d = {}
        if not isinstance(d, dict):
            d = {}
        ov = d.get("overrides") if isinstance(d.get("overrides"), dict) else {}
        overrides = {t: {k: v for k, v in (ov.get(t) or {}).items() if k in COLOR_KEYS and isinstance(v, str)}
                     for t in THEMES}
        op = d.get("opacities") if isinstance(d.get("opacities"), dict) else {}
        opacities = {k: float(v) for k, v in op.items() if k in OPACITY_KEYS and isinstance(v, (int, float))}
        sz = d.get("sizes") if isinstance(d.get("sizes"), dict) else {}
        sizes = {k: float(v) for k, v in sz.items() if k in SIZE_KEYS and isinstance(v, (int, float))}
        oo = d.get("options") if isinstance(d.get("options"), dict) else {}
        options = {k: bool(v) for k, v in oo.items() if k in OPTION_KEYS and isinstance(v, bool)}
        ch = d.get("choices") if isinstance(d.get("choices"), dict) else {}
        choices = {k: v for k, v in ch.items() if k in CHOICE_KEYS and v in CHOICE_VALUES[k]}
        return cls(theme=str(d.get("theme", "light")), color_by=str(d.get("color_by", "object")), overrides=overrides,
                   opacities=opacities, sizes=sizes, options=options, choices=choices)


# ----------------------------------------------------------------------------
_ACTIVE = ColorSettings()


def active() -> ColorSettings:
    return _ACTIVE


def apply(settings: ColorSettings) -> None:
    """Make ``settings`` the active colours of every scene builder."""
    global _ACTIVE
    from . import scene

    _ACTIVE = settings
    scene.COLORS.clear()
    scene.COLORS.update(settings.colors())
    scene.DISPLAY["color_by"] = settings.color_by
    scene.DISPLAY["theme"] = settings.theme
    scene.DISPLAY["opacity"] = {k: settings.opacity(k) for k in OPACITY_KEYS}
    scene.DISPLAY["sizes"] = {k: settings.size(k) for k in SIZE_KEYS}
    scene.DISPLAY["options"] = {k: settings.option(k) for k in OPTION_KEYS}
    scene.DISPLAY["choices"] = {k: settings.choice(k) for k in CHOICE_KEYS}


def object_color(name: str, explicit: Optional[str], theme: Optional[str] = None,
                 index: Optional[int] = None) -> RGB:
    """Colour of a section / material: its own colour if set, else automatic."""
    if explicit:
        try:
            return from_hex(explicit)
        except ValueError:
            pass
    return auto_color(name, theme or _ACTIVE.theme, index)


# ----------------------------------------------------------------------------
# Qt helpers
# ----------------------------------------------------------------------------
def qt_palette(theme: str):
    from PySide6 import QtGui

    u = UI[theme]
    c = QtGui.QColor
    pal = QtGui.QPalette()
    R = QtGui.QPalette
    pal.setColor(R.Window, c(u["window"]))
    pal.setColor(R.WindowText, c(u["text"]))
    pal.setColor(R.Base, c(u["base"]))
    pal.setColor(R.AlternateBase, c(u["alt"]))
    pal.setColor(R.ToolTipBase, c(u["tooltip"]))
    pal.setColor(R.ToolTipText, c(u["text"]))
    pal.setColor(R.Text, c(u["text"]))
    pal.setColor(R.Button, c(u["button"]))
    pal.setColor(R.ButtonText, c(u["text"]))
    pal.setColor(R.BrightText, c("#ff5050"))
    pal.setColor(R.Highlight, c(u["accent"]))
    pal.setColor(R.HighlightedText, c(u["accent_text"]))
    pal.setColor(R.Link, c(u["accent"]))
    pal.setColor(R.Mid, c(u["mid"]))
    # bevel roles (Fusion frames, toolbars, splitters): derived from the button colour
    # so no stray white / black edges appear in either mode
    btn = c(u["button"])
    pal.setColor(R.Light, btn.lighter(125 if theme == "dark" else 110))
    pal.setColor(R.Midlight, btn.lighter(112 if theme == "dark" else 105))
    pal.setColor(R.Dark, c(u["mid"]).darker(120))
    pal.setColor(R.Shadow, c(u["mid"]).darker(160))
    pal.setColor(R.PlaceholderText, c(u["disabled"]))
    for role in (R.WindowText, R.Text, R.ButtonText):
        pal.setColor(R.Disabled, role, c(u["disabled"]))
    return pal


def apply_qt(app, settings: ColorSettings) -> None:
    """Fusion style + the theme palette for the whole application."""
    from PySide6 import QtWidgets

    if app is None:
        return
    if isinstance(app, QtWidgets.QApplication):
        app.setStyle("Fusion")
    app.setPalette(qt_palette(settings.theme))
