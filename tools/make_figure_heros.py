#!/usr/bin/env python
"""把论文三张 figure 的图像带(场景总览 + 第一视角 + 目标特写)裁出来当网页 hero。

上边界: 图里的标题("Feasible vs. Infeasible Task" + case 名)和图例(Start/End/Trajectory)
在网页上已经分别由 case 标题和 legend 卡片给过了, 留着就是重复 —— 按行墨水占比切掉。
文字行稀疏(<0.2), 图像行密(>0.7), 用"连续 30 行平均 > 0.45"定位图像带的起点。

下边界: 图像带与下面问答区之间是一条横贯整幅的虚线, 按"跨幅 + 多次断开"两个特征
定位 —— 单看暗像素占比会把标题行和文字段落一起选进来。
"""
import os
import subprocess

import numpy as np
from PIL import Image

SRC = '/egr/research-actionlab/caizhon2/codes/EQA/SPMR/latex/images'
OUT = '/egr/research-actionlab/caizhon2/codes/EQA/SPMR/3DSPMR_web/assets/figure'
KEYS = ['incorrect_attribute', 'object_not_present', 'room_not_present']
DPI, WIDTH = 150, 1500


def image_row(a):
    """图像带的第一行 —— 上面的标题/图例都是稀疏文字行, 这里按密度切掉。"""
    ink = (a < 245).mean(1)
    for y in range(len(ink) - 30):
        if ink[y:y + 30].mean() > 0.45:
            return max(0, y - 6)
    return 0


def dashed_row(a):
    H, W = a.shape
    for y in range(int(H * 0.15), int(H * 0.65)):
        d = np.nonzero(a[y] < 120)[0]
        if len(d) < 0.25 * W or len(d) > 0.8 * W:
            continue
        if d[0] < 0.04 * W and d[-1] > 0.96 * W and (np.diff(d) > 1).sum() > 15:
            return y
    return None


os.makedirs(OUT, exist_ok=True)
for k in KEYS:
    tmp = f'/tmp/_fig_{k}'
    subprocess.run(['pdftoppm', '-r', str(DPI), '-png', '-f', '1', '-l', '1',
                    f'{SRC}/{k}.pdf', tmp], check=True)
    src = f'{tmp}-1.png'
    im = Image.open(src).convert('RGB')
    y = dashed_row(np.asarray(im.convert('L')))
    if y is None:
        print(f'  ★ {k}: 找不到虚线, 跳过裁剪')
        y = im.height
    y_top = image_row(np.asarray(im.convert('L')))
    im = im.crop((0, y_top, im.width, max(y_top + 1, y - 6)))
    # 去掉四周白边
    g = np.asarray(im.convert('L'))
    ys, xs = np.nonzero(g < 245)
    im = im.crop((max(0, xs.min() - 8), max(0, ys.min() - 8),
                  min(im.width, xs.max() + 9), min(im.height, ys.max() + 9)))
    if im.width > WIDTH:
        im = im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS)
    im.save(f'{OUT}/{k}.webp', 'WEBP', quality=86, method=5)
    os.remove(src)
    print(f'  {k:22s} 图像带 y={y_top}..{y}  ->  {im.width}x{im.height}  '
          f'{os.path.getsize(f"{OUT}/{k}.webp")/1024:.0f} KB')
