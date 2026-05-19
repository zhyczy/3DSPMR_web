Question-answer sample images.

Structure expected by index.html:

  qa_pairs/
    level0/depth/img.png
    level0/pointmap/img.png
    level1/size/img.png
    level1/grounding/img.png
    level1/counting/img.png
    level2/distance/img.png
    level2/relative/img.png
    level2/room/img.png
    level3/active_search/img.png
    level3/planning/img.png
    level3/counterfactual/img.png
    sequential/step1.png
    sequential/step2.png
    sequential/step3.png
    sequential/spatial_distance.png
    sequential/planning.png

Each `img.png` is the visual that pairs with a Q/A card in the corresponding
tab on the page. The carousel under "Sequential Reasoning & Exploration" cycles
through the `sequential/` images plus other examples; edit the `inferenceData`
array in index.html to add/remove cards.

Tip: the QA card auto-switches between portrait and landscape layouts based on
the image's natural aspect ratio (see adjustCardLayout in index.html).
