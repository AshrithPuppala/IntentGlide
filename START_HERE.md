# Start Here

This ZIP contains the repository scaffold derived from the supplied `README.md`.

## Install and run

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
python -m app.main
```

The initial sandbox uses synthetic input so the application can run without the ESP32 rig.

## Run tests

```bash
pytest -q
```

## Notes

- The Python modules are starter implementations to establish interfaces and a working baseline; they are not a clinically validated system.
- The fixed Butterworth baseline assumes a uniform 120 Hz analysis stream. Real mouse events are timestamped and nonuniform; resampling should be explicit before using this filter with live mouse events.
- The ESP32 sketch is a hardware starting point. Verify the exact PWM API for your board/core version and use a suitable motor driver.
- `README.md` is the full team workflow document supplied with the request.
