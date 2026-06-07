# ThermoCell — Battery Pack Thermal Simulation

A 2D thermal model of a cylindrical lithium-ion battery pack simulating heat generation, conduction, and liquid cooling across a full discharge cycle.

**Author:** Evan McIntyre

---

## Files

- `thermocell.py` — Python simulation using NumPy and Matplotlib. Outputs an animated GIF and summary plot.
- `thermocell.html` — Interactive browser demo with live heat map, real-time charts, and adjustable parameters.

---

## Physics

| Equation | Description |
|---|---|
| Q = I² × R_eff | Joule heating per cell |
| R_eff = R × (1 + 0.4×(1−SoC)) | State-of-charge dependent resistance |
| q = k × ΔT / Δx | Fourier conduction between neighbours |
| Q_cool = h × (T − T∞) | Convective cooling on edge rows |
| ΔT = (Q_net / Cm) × dt | Temperature update per step |

---

## Python Usage

```bash
pip install numpy matplotlib pillow
python thermocell.py                   # baseline scenario
python thermocell.py --scenario worst  # worst case (4C, 40°C ambient, no cooling)
```

**Scenarios:** `baseline` · `nocooling` · `hotclimate` · `worst`

---

## Web Demo

Open `thermocell.html` in any browser — no install required.
