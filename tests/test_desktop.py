from sandbox.desktop import Desktop


def test_icons_are_targets_and_open_windows():
    d = Desktop()
    ids = {t.id for t in d.targets}
    assert ids >= {"icon:notepad", "icon:calculator", "icon:settings", "taskbar:start", "taskbar:desktop"}
    assert d.click(*d.icons[0].center) == "icon:notepad"
    assert [w.id for w in d.windows] == ["notepad"]
    assert "taskbar:win:notepad" in {t.id for t in d.targets}


def test_window_close_button_and_show_desktop():
    d = Desktop()
    d.open_window("calculator")
    assert d.click(*d.windows[-1].close_btn.center) == "calculator:close"
    assert d.windows == [] and d.task_buttons == []
    d.open_window("notepad")
    d.open_window("settings")
    d.click(*d.show_desktop_btn.center)
    assert d.windows == []


def test_covered_icon_is_not_a_target():
    d = Desktop()
    d.open_window("notepad")
    assert "icon:notepad" in {t.id for t in d.targets}
    win = d.windows[-1]               # slide the window over the icons
    win.x, win.y, win.w, win.h = 0, 0, 400, 400
    assert "icon:notepad" not in {t.id for t in d.targets}


def test_only_top_window_controls_are_targets():
    d = Desktop()
    d.open_window("notepad")
    d.open_window("settings")
    ids = {t.id for t in d.targets}
    assert "settings:close" in ids and "notepad:close" not in ids


def test_start_menu_opens_apps_and_click_away_closes():
    d = Desktop()
    assert d.click(*d.start_btn.center) == "taskbar:start"
    assert d.menu_open
    ids = {t.id for t in d.targets}
    assert "start:calculator" in ids and "icon:notepad" not in ids   # menu is modal
    item = next(i for i in d.menu_items if i.id == "start:calculator")
    assert d.click(*item.center) == "start:calculator"
    assert not d.menu_open and [w.id for w in d.windows] == ["calculator"]
    d.click(*d.start_btn.center)
    d.click(700, 300)                  # click away
    assert not d.menu_open


def test_taskbar_button_raises_window():
    d = Desktop()
    d.open_window("notepad")
    d.open_window("settings")
    d.click(*d.task_buttons[0].center)   # notepad's taskbar button
    assert d.windows[-1].id == "notepad"


def test_scattered_icons_are_unaligned_clickable_targets():
    d = Desktop()
    extra = d.icons[4:]
    assert [i.id for i in extra] == ["icon:files", "icon:browser", "icon:photos"]
    for a in extra:
        for b in extra:
            if a is not b:
                assert abs(a.x - b.x) >= 40 and abs(a.y - b.y) >= 40
    assert d.click(*extra[1].center) == "icon:browser"
    assert d.windows[-1].id == "browser"
    assert Desktop(icon_seed=7).icons[3].x == Desktop(icon_seed=7).icons[3].x   # same seed, same layout
