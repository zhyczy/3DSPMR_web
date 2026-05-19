# 3DSPMR project page

Project page scaffold for **"Vision to Geometry: 3D Spatial Memory for Sequential Embodied MLLM Reasoning and Exploration"**, adapted from the [HiSpatial](https://microsoft.github.io/HiSpatial/) template.

## Layout

```
3DSPMR_web/
├── index.html         # single-page site (Tailwind + Chart.js, CDN)
├── imgs/              # teaser, pipeline, video poster
├── qa_pairs/          # per-tab question images + sequential carousel frames
├── videos/            # demo.mp4 (or swap for YouTube embed)
└── .nojekyll          # tells GitHub Pages to serve files as-is
```

## Sections in `index.html`

| anchor                    | what it is                                                 |
|---------------------------|------------------------------------------------------------|
| `#teaser`                 | Hero figure + one-line pitch                               |
| `#abstract`               | Paper abstract                                             |
| `#qa-samples`             | 4 tabs of question types (Level 0–3) with QA cards         |
| `#method`                 | Pipeline figure + 3-step "vision → geometry → reasoning"   |
| `#inference-visual`       | Sequential reasoning carousel (User / Agent / GT bubbles)  |
| `#video`                  | Embedded video demo                                        |
| `#results`                | Quantitative results table                                 |
| `#ablation`               | Two Chart.js bar charts                                    |
| `#citation`               | BibTeX block with copy button                              |

## Things to fill in

Search `index.html` for these placeholders:

- **Authors / affiliations** — in the `<header>` block.
- **Links** — five `<a href="#">` buttons (Paper, arXiv, Code, Model, Dataset).
- **Abstract** — paragraph inside `#abstract`.
- **Result numbers** — `–` placeholders in the `<table>` inside `#results`.
- **Ablation data** — `[0, 0, 0, 0]` arrays in the `Chart` constructors at the bottom of `<script>`.
- **Inference carousel** — edit the `inferenceData = [ … ]` array.
- **Figures** — drop files into `imgs/` (`teaser.jpg`, `pipeline.jpg`, `video_poster.jpg`) and `qa_pairs/...` per the structure in `qa_pairs/README.txt`.
- **BibTeX** — update the `<code id="bibtex-content">` block.

## Preview locally

```bash
cd 3DSPMR_web
python -m http.server 8000
# open http://localhost:8000
```

## Deploy to GitHub Pages

1. Push this directory to a repo (or to the `gh-pages` branch of an existing one).
2. In repo Settings → Pages, set source to the branch/folder where `index.html` lives.
3. The `.nojekyll` file ensures underscore-prefixed paths aren't filtered.

## Credits

Layout adapted from the HiSpatial project page (Tailwind CSS + Chart.js).
# 3DSPMR_web
