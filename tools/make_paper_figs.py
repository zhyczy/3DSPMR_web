#!/usr/bin/env python
"""渲染论文里的 teaser(Illustration) 和 method(pipeline) 两张图给网页用。

和 make_figure_heros.py 不同, 这两张整幅都要, 只去白边。
"""
import os
import subprocess

import numpy as np
from PIL import Image

SRC = '/egr/research-actionlab/caizhon2/codes/EQA/SPMR/latex/images'
OUT = '/egr/research-actionlab/caizhon2/codes/EQA/SPMR/3DSPMR_web/assets/figure'
FIGS = {'teaser': ('Illustration', 1600), 'pipeline': ('pipeline', 1800)}
DPI = 200

os.makedirs(OUT, exist_ok=True)
for name, (stem, width) in FIGS.items():
    tmp = f'/tmp/_pf_{name}'
    subprocess.run(['pdftoppm', '-r', str(DPI), '-png', '-f', '1', '-l', '1',
                    f'{SRC}/{stem}.pdf', tmp], check=True)
    im = Image.open(f'{tmp}-1.png').convert('RGB')
    g = np.asarray(im.convert('L'))
    ys, xs = np.nonzero(g < 245)
    im = im.crop((max(0, xs.min() - 10), max(0, ys.min() - 10),
                  min(im.width, xs.max() + 11), min(im.height, ys.max() + 11)))
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(f'{OUT}/{name}.webp', 'WEBP', quality=88, method=5)
    os.remove(f'{tmp}-1.png')
    print(f'  {name:10s} {stem+".pdf":22s} -> {im.width}x{im.height}  '
          f'{os.path.getsize(f"{OUT}/{name}.webp")/1024:.0f} KB')
