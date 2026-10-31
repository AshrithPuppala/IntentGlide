from sandbox.calculator import Calculator
from sandbox.desktop import Desktop


def run(*keys):
    calc = Calculator()
    for key in keys:
        calc.press(key)
    return calc


def test_basic_arithmetic_and_chaining():
    assert run("2", "plus", "3", "eq").display == "5"
    assert run("2", "plus", "3", "mul", "4", "eq").display == "20"      # immediate execution
    assert run("9", "minus", "4", "eq").display == "5"
    assert run("8", "div", "4", "eq").display == "2"


def test_decimals_sign_percent_delete_clear():
    assert run("1", "dot", "5", "plus", "1", "dot", "5", "eq").display == "3"
    assert run("5", "neg").display == "-5"
    assert run("5", "0", "pct").display == "0.5"
    assert run("1", "2", "3", "del").display == "12"
    assert run("1", "2", "clear").display == "0"


def test_divide_by_zero_is_an_error_until_cleared():
    calc = run("5", "div", "0", "eq")
    assert calc.display == "Error"
    calc.press("7")
    assert calc.display == "Error"
    calc.press("clear")
    assert calc.display == "0"


def test_changing_operator_does_not_apply_twice():
    assert run("5", "plus", "minus", "2", "eq").display == "3"


def test_calculator_window_keypad_is_clickable_and_targetable():
    d = Desktop()
    d.click(*d.icons[1].center)                       # open Calculator via its icon
    win = d.windows[-1]
    keys = {w.id: w for w in win.widgets}
    assert len(keys) == 20
    for name in ("7", "mul", "6", "eq"):
        assert d.click(*keys[f"calculator:key:{name}"].center) == f"calculator:key:{name}"
    assert win.calc.display == "42"
    assert len([t for t in d.targets if t.id.startswith("calculator:key:")]) == 20
