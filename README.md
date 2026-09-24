# 3DSPMR project page

Static site for **Vision to Geometry: 3D Spatial Memory for Sequential Embodied MLLM
Reasoning and Exploration** (NeurIPS 2026, under review).

The page is built around the three feasible-vs-infeasible EQA comparisons discussed in
`latex/Neurips2026/supp.tex` §"Comparison of Feasible and Infeasible Tasks"
(Figs. `incorrect_attribute`, `object_not_present`, `room_not_present`).
Each case shows the two runs side by side with a shared, **normalized** progress slider —
the two trajectories differ a lot in absolute length (e.g. 64 vs 158 steps), so aligning
them by raw step number would park the short one at its end while the long one is still
exploring, which is exactly the contrast the figure is meant to show.

## Layout

    index.html              the whole page (no build step, no external JS)
    assets/cases.json       per-case Q / GT / prediction / step counts + trajectory pixel coords
    assets/scene/<scene>.webp        one textured orthographic top-down per scene
    assets/fp/<case>/<kind>/NNN.webp first-person frames along the recorded path
    tools/replay_render.py  regenerates assets/ from ../visualization/ + the HM3D meshes
    .nojekyll               GitHub Pages: serve files as-is

## Regenerating the assets

    /egr/research-actionlab/caizhon2/miniconda3/envs/3dmem/bin/python tools/replay_render.py

**No re-inference.** The agent's positions come from the `Current position: [x, y, z]` lines in
each run's log; the script only puts a camera at those positions and re-renders. For each scene it
renders one textured orthographic top-down with a fixed viewport, so the world-to-pixel mapping is
analytic (`u = W/2 + (x-cx)/half * W/2`, same for `z`) and the trajectory can be projected exactly —
validated by checking the path lands on walkable floor and threads through doorways.

Two things worth knowing:

* **Heading is not recorded.** The logs contain position but no rotation, so the replay camera faces
  the direction of travel. The first-person frames show *where the agent went*, not necessarily
  where it was looking.
* **The runs' own `map/step_*.png` are not used.** They are re-framed per step (image size changes
  from 901x978 to 959x931 within one episode, and the px-per-metre ratio swings between 21 and 76),
  so scrubbing through them makes the map drift and the trajectory cannot be recovered from them.
  Rendering our own fixed viewport avoids both problems.

Trajectories are stored as pixel coordinates in `cases.json` and drawn on a `<canvas>` in the
browser rather than baked into one image per step — a few hundred numbers instead of a few hundred
images.

## Source runs

| Case | Feasible | Infeasible |
|---|---|---|
| Incorrect Attribute | `00824-Dd4bFSTQ8gi_4` (64 steps) | `00824-Dd4bFSTQ8gi_3` (158 steps) |
| Object not Present  | `00848-ziup5kvtCCR_3` (10 steps) | `00255-NGyoyh91xXJ_3` (120 steps) |
| Room not Present    | `00166-RaYrxWt5pR1_2` (8 steps)  | `00164-XfUxBGTFQQb_2` (24 steps) |

## Ground-truth corrections

`tools/build_assets.py` carries a small `GT_FIX` table keyed by `(episode, qa_id)`.
Currently one entry: `00824-Dd4bFSTQ8gi_4` Q1's answer is shown as
*"A toilet paper holder"* instead of the string recorded in the run.
The recorded `result.json` / `total_result.json` are deliberately **not** edited — they
are the record of what was actually evaluated. If a correction also needs to hold for the
benchmark itself, it has to be made at the annotation source and the affected gradings
re-run, which changes the reported scores and is out of this script's scope.

The author block is intentionally anonymous — the paper is under review.
