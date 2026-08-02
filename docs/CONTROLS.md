# Simulator controls

- `W`, `A`, `S`, `D`: move along the horizontal camera heading
- Mouse: look around
- `Left Shift`: move faster while held
- `Q`, `E`: move down and up
- `G`: switch between free-flight and terrain-following camera-operator mode
- `Space`: launch the currently selected shell profile by hand
- `[`, `]`: step backwards and forwards through the shell library
- `V`: switch between Human Vision Mode (default) and Physical Camera Mode
- `Tab`: release or recapture the mouse
- `Escape`: release the captured mouse; press again to exit
- Left click: recapture the mouse after it has been released

Movement velocity accelerates and decelerates smoothly. Free-flight resolves
the camera body against the rendered terrain and water datum, so the lens can
no longer pass underground. Camera-operator mode uses a 1.68 m optical-centre
height, rejects non-walkable upward steps/slopes, and will not walk onto the
river. `Q`/`E` remain free-flight controls. The active mode and current position
in local East-Up-South metres are displayed in the window title.

When the loaded scenario carries a firing timeline, shells launch automatically
on the absolute event clock and the title shows the shot count. The shipped
historical scenario has no timeline — no dated firing record for
2024-10-05 has been obtained — so it launches one development shell and leaves
the rest to the manual key.

To watch the shell library instead:

```bash
python run_simulator.py --scenario assets/scenario_demo_synthetic.json
```

That scenario is a **synthetic demonstration sequence, not the 2024
performance**. Every one of its events is confidence grade D.
