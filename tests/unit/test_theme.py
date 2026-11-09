"""Display themes and graphics colours (no Qt needed)."""
import pytest

from structstudio.app import theme


def test_defaults_cover_every_item_and_roundtrip():
    for t in theme.THEMES:
        assert set(theme.DEFAULTS[t]) == set(theme.COLOR_KEYS)
    s = theme.ColorSettings(theme="light", color_by="material")
    s.set_color("beam", (1.0, 0.0, 0.0))
    s.set_color("column", theme.DEFAULTS["light"]["column"])  # same as default: no override
    assert s.overrides["light"] == {"beam": "#ff0000"} and s.is_custom("beam")
    r = theme.ColorSettings.from_json(s.to_json())
    assert r.theme == "light" and r.color_by == "material" and r.colors()["beam"] == (1.0, 0.0, 0.0)
    assert r.colors("dark")["beam"] == theme.DEFAULTS["dark"]["beam"]
    r.reset()
    assert r.colors()["beam"] == theme.DEFAULTS["light"]["beam"]
    bad = theme.ColorSettings.from_json('{"theme": "neon", "overrides": {"dark": {"beam": "zz", "x": "#fff"}}}')
    assert bad.theme == "light" and "x" not in bad.overrides["dark"]  # unknown theme -> default (light)
    assert bad.colors("dark")["beam"] == theme.DEFAULTS["dark"]["beam"]
    assert theme.ColorSettings().theme == "light"


def test_hex_and_auto_colors_are_stable():
    assert theme.from_hex("#0f0") == (0.0, 1.0, 0.0)
    assert theme.to_hex((0.0, 0.5, 1.0)) == "#0080ff"
    a = theme.auto_color("C50x50", "dark")
    assert a == theme.auto_color("C50x50", "dark") and a != theme.auto_color("B30x60", "dark")
    assert sum(theme.auto_color("C50x50", "light")) < sum(a)  # deeper on white
    with pytest.raises(ValueError):
        theme.from_hex("12")


def test_apply_and_color_by_section():
    pytest.importorskip("vtk")
    from structstudio.app import scene
    from structstudio.app.templates import new_building
    m = new_building(x_spacings=(6,), y_spacings=(5,), story_heights=(3,), grid_only=False)
    try:
        theme.apply(theme.ColorSettings(theme="light"))
        assert scene.COLORS["background_top"] == (1.0, 1.0, 1.0)
        keys = scene.build_model_actors(m).actors
        assert "frames_Beam" in keys and "frames_Column" in keys
        theme.apply(theme.ColorSettings(theme="light", color_by="section"))
        keys = scene.build_model_actors(m).actors
        secs = {f.section for f in m.frames.values()}
        assert {k for k in keys if k.startswith("frames_") and not k.endswith(("_sel", "_edges"))} == {
            f"frames_{f.type}_sec:{f.section}" for f in m.frames.values()}
        assert len(secs) >= 1
        theme.apply(theme.ColorSettings(color_by="material"))
        assert any("mat:" in k for k in scene.build_model_actors(m).actors)
    finally:
        theme.apply(theme.ColorSettings())


def test_opacities_roundtrip_and_selection_box_defaults():
    s = theme.ColorSettings()
    assert s.opacity("select_box_fill") == pytest.approx(0.18)
    s.set_opacity("select_box_fill", 0.5)
    s2 = theme.ColorSettings.from_json(s.to_json())
    assert s2.opacity("select_box_fill") == pytest.approx(0.5)
    assert s2.opacity("select_box_border") == pytest.approx(0.85)
    s2.reset()
    assert s2.opacity("select_box_fill") == pytest.approx(0.18)
    # the light theme uses the CAD / Windows blue for the drag box, distinct from the highlight
    assert theme.DEFAULTS["light"]["select_box"] != theme.DEFAULTS["light"]["selection"]
    assert "reference" in theme.DEFAULTS["dark"] and "reference" in theme.DEFAULTS["light"]
    with pytest.raises(KeyError):
        s.set_opacity("nope", 0.1)


def test_apply_writes_opacities_to_scene():
    from structstudio.app import scene
    s = theme.ColorSettings()
    s.set_opacity("select_box_border", 0.3)
    theme.apply(s)
    assert scene.DISPLAY["opacity"]["select_box_border"] == pytest.approx(0.3)
    assert scene.COLORS["select_box"] == theme.DEFAULTS["light"]["select_box"]


def test_sizes_and_reset_helpers():
    from structstudio.app import theme as th
    s = th.ColorSettings()
    assert s.is_default()
    s.set_size("support_size", 99)  # clamped to the range
    assert s.size("support_size") == th.SIZE_RANGES["support_size"][1]
    s.set_size("joint_size", th.SIZE_DEFAULTS["joint_size"])  # the default is not stored
    assert "joint_size" not in s.sizes
    s.set_color("beam", (1, 0, 0), "light")
    s.set_color("beam", (0, 1, 0), "dark")
    s.set_opacity("floor_fill", 0.9)
    again = th.ColorSettings.from_json(s.to_json())
    assert again.size("support_size") == s.size("support_size") and again.opacity("floor_fill") == 0.9
    s.reset_color("beam", "dark")
    assert s.is_custom("beam", "light") and not s.is_custom("beam", "dark")
    s.reset_size("support_size")
    s.reset_opacity("floor_fill")
    assert not s.sizes and not s.opacities
    s.reset_all()
    assert s.is_default()
    # every size item has a sane default inside its range
    for key, _label, default, lo, hi, step, _sfx in th.SIZE_ITEMS:
        assert lo <= default <= hi and step > 0


def test_dark_and_light_palettes_keep_contrast():
    """Members, loads and diagrams stay distinguishable from the background in both modes."""
    from structstudio.app import theme as th

    def lum(c):
        def ch(v):
            return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
        r, g, b = (ch(x) for x in c)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def contrast(a, b):
        la, lb = sorted((lum(a), lum(b)), reverse=True)
        return (la + 0.05) / (lb + 0.05)

    for t in th.THEMES:
        c = th.DEFAULTS[t]
        for key in ("beam", "column", "load", "support", "deformed", "moment_pos", "moment_neg", "text", "grid_text"):
            assert contrast(c[key], c["background_top"]) >= 2.5, (t, key)
        # diagrams are not drawn in the member colours
        for a, b in (("moment_pos", "beam"), ("moment_neg", "column"), ("deformed", "moment_neg")):
            assert sum(abs(x - y) for x, y in zip(c[a], c[b])) > 0.3, (t, a, b)


def test_drawing_cursor_shape_choice_and_pink_default():
    from structstudio.app import scene, theme as th
    from structstudio.app.snap_icons import FREE_SHAPES
    s = th.ColorSettings()
    # the point with no snap is pink in both themes (green circles are the pin symbols)
    for t in th.THEMES:
        r, g, b = s.colors(t)["free_point"]
        assert r > 0.8 and g < 0.5 and b > 0.5
    assert s.choice("free_point_shape") == "cross"
    assert set(th.CHOICE_VALUES["free_point_shape"]) <= set(FREE_SHAPES)
    s.set_choice("free_point_shape", "plus")
    with pytest.raises(KeyError):
        s.set_choice("free_point_shape", "banana")
    again = th.ColorSettings.from_json(s.to_json())
    assert again.choice("free_point_shape") == "plus" and not again.is_default()
    assert th.ColorSettings.from_json('{"choices": {"free_point_shape": "banana"}}').choice("free_point_shape") == "cross"
    th.apply(again)
    try:
        assert scene.gchoice("free_point_shape") == "plus"
    finally:
        th.apply(th.ColorSettings())
    s.set_choice("free_point_shape", "cross")  # the default is not stored
    assert s.is_default()


def test_local_axes_colours_and_sizes_are_settings():
    from structstudio.app import theme as th
    for t in th.THEMES:
        assert {"axis_1", "axis_2", "axis_3"} <= set(th.DEFAULTS[t])
    assert {"local_axes_length", "local_axes_width", "local_axes_text", "local_axes_gap"} <= set(th.SIZE_KEYS)


def test_object_colours_are_distinct_and_the_page_is_flat():
    """Columns, supports, beams, braces, walls, loads and the selection read apart."""
    from structstudio.app import theme as th
    keys = ("beam", "column", "brace", "wall", "support", "load", "selection")
    for t in th.THEMES:
        c = th.DEFAULTS[t]
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                assert sum(abs(x - y) for x, y in zip(c[a], c[b])) > 0.45, (t, a, b)
        assert c["background_top"] == c["background_bottom"]  # one flat colour
    assert th.DEFAULTS["light"]["background_top"] == (1.0, 1.0, 1.0)
