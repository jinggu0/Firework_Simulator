# Simulator controls

- `W`, `A`, `S`, `D`: move along the horizontal camera heading
- Mouse: look around
- `Left Shift`: move faster while held
- `Q`, `E`: move down and up
- `Space`: launch the currently selected shell profile by hand
- `[`, `]`: step backwards and forwards through the shell library
- `V`: switch between Human Vision Mode (default) and Physical Camera Mode
- `Tab`: release or recapture the mouse
- `Escape`: release the captured mouse; press again to exit
- Left click: recapture the mouse after it has been released

Movement velocity accelerates and decelerates smoothly. The current position in
local East-Up-South metres is displayed in the window title.

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
