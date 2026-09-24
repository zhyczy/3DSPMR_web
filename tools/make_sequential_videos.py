#!/usr/bin/env python
"""三个环境规模各一段视频: 同一个 episode 上五种方法的前 100 步同步回放。

一个 episode = 5 个问题依次到来, 共用一份空间记忆和一份步数预算。
五格并排推进到第 100 步, 轨迹按"当时在做第几问"上色, 于是
"预算烧在哪一问上"和"谁已经答完了"边看边出来。

三段视频步长完全一致(都是 100 步 × PACE 帧), 帧率也一致, 可以横向比。
起点/终点/当前位置的配色与 feasible-vs-infeasible 那三段一致。
"""
import json
import os
import re
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from extract_baseline_paths import ARMS, paths_for
from topdown_render import render_scene

EXP = '/egr/research-actionlab/caizhon2/codes/EQA/SPMR/3DSPMR_code/ExploreVQA/extended_exp'
OUT = '/egr/research-actionlab/caizhon2/codes/EQA/SPMR/3DSPMR_web/videos'
ASSETS = '/egr/research-actionlab/caizhon2/codes/EQA/SPMR/3DSPMR_web/assets'

# 两个中环境 + 一个大环境 —— 规模取判分侧那张 SCALE_DICT(与步数预算同一口径)。
# 都是住宅(00034 指标更好看, 但那是个车库/仓库, 不能代表 apartment)。
# 不可行问题落在 3&5 / 2&4 / 4&5, 位置各不相同, 也都不在第 1 问。
# (slug, 规模, episode) —— slug 决定输出文件名, 两个 middle 不能重名
EPISODES = [('1', 'middle', '00706-YHmAkqgwe2p_1'),
            ('2', 'middle', '00217-qz3829g1Lzf_1'),
            ('3', 'large',  '00202-yVbpFay8gTU_0')]
FULL_BUDGET = {'small': 100, 'middle': 400, 'large': 640}

SHOW = 100          # 只回放前 100 步 —— 每格同样的预算
FPS, PACE = 15, 2   # 全站统一: 15 fps, 每步 2 帧
PANE, PAD, GAP, HEAD, FOOT = 360, 22, 14, 46, 78

TASK_COLS = [(47, 109, 246), (0, 179, 164), (123, 192, 67), (244, 163, 0), (232, 69, 60)]
# 与 feasible/infeasible 三段视频一致: 绿=起点, 浅黄=终点, 红=当前位置
C_START, C_END, C_NOW = (22, 179, 100), (255, 230, 128), (255, 59, 107)
OURS = '3DSPMR (ours)'
SKEY = {'Explore-EQA': 'explore_p345', '3D-Mem': '3d_mem_p345', 'GraphEQA': 'GraphEQA_p345',
        'Pred-EQA': 'Pred-EQA_p345', OURS: 'sg_snap_p345'}
INF = re.compile(r"^\s*(i don'?t know|there is no|no,? there is no)", re.I)


def font(sz, bold=False):
    p = ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else
         '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    return ImageFont.truetype(p, sz) if os.path.exists(p) else ImageFont.load_default()


F_H, F_M, F_S, F_T = font(21, True), font(17, True), font(14), font(15, True)


def fit(im, box):
    small = im.copy(); small.thumbnail((box, box), Image.LANCZOS)
    c = Image.new('RGB', (box, box), (250, 250, 252))
    off = ((box - small.width) // 2, (box - small.height) // 2)
    c.paste(small, off)
    return c, off, small.width / im.width


S = json.load(open(f'{EXP}/scores/merged_60/gpt-5/scores_results.json'))
QA = json.load(open(f'{EXP}/merged_60/sg_snap_p345.json'))
os.makedirs(OUT, exist_ok=True)
meta = {'show_steps': SHOW, 'fps': FPS, 'frames_per_step': PACE, 'episodes': {}}

NCOL = len(ARMS)
W = PAD * 2 + PANE * NCOL + GAP * (NCOL - 1)
H = HEAD + PANE + FOOT

for slug, scale, ep in EPISODES:
    tag = ep.rsplit('_', 1)[0]
    print(f'[{slug} {scale}] {ep}')
    # 先取 agent 所在楼层 —— 多层场景不传这个, 画的就是它没走过的那一层
    _p, _, _ = paths_for(ep, dict(ARMS)[OURS])
    floor_y = float(np.median([q[1] for q in _p])) if _p else None
    top, to_px = render_scene(tag, floor_y)
    qs = sorted(QA[ep])
    inf = [bool(INF.search(QA[ep][q]['answer'])) for q in qs]

    runs = {}
    for name, logs in ARMS:
        P, steps, bounds = paths_for(ep, logs)
        pts = to_px(P)
        keep = [i for i, s in enumerate(steps) if s < SHOW]
        edges = [0] + [b for b in bounds[1:] if b is not None] + [10 ** 9]
        ts = [q['step'] for q in S[SKEY[name]][ep]]
        # 第 k 步时已经答完几问 —— 累计步数不超过 k 的那些
        done_at, c, n = [], 0, 0
        for k in ts:
            c += k
            n += 1
            done_at.append((c, n))
        runs[name] = dict(pts=[pts[i] for i in keep], steps=[steps[i] for i in keep],
                          edges=edges, per_task=ts, done_at=done_at)

    meta['episodes'][ep] = {
        'scale': scale, 'full_budget': FULL_BUDGET[scale], 'infeasible': inf,
        'methods': {n: {'steps_per_subtask': r['per_task'],
                        'answered_in_100': max([0] + [d for c, d in r['done_at'] if c <= SHOW])}
                    for n, r in runs.items()}}

    tmp = f'/tmp/seqvid_{slug}'
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))

    NF = SHOW * PACE
    for fi in range(NF):
        cut = int(round((fi + 1) / PACE))          # 当前放到第几步
        cv = Image.new('RGB', (W, H), (255, 255, 255))
        d = ImageDraw.Draw(cv)

        lx = PAD
        d.text((lx, 8), 'Question', font=F_T, fill=(20, 22, 26))
        lx += int(d.textlength('Question', font=F_T)) + 12
        for i, col in enumerate(TASK_COLS):
            d.rectangle([lx, 13, lx + 24, 20], fill=col)
            d.text((lx + 29, 7), str(i + 1), font=F_S, fill=(90, 96, 108))
            lx += 48
        hdr = (f'{scale.upper()}  ·  Scene {tag.split("-")[0]}  ·  step budget {FULL_BUDGET[scale]}'
               f'  ·  infeasible questions: '
               f'{", ".join(str(i + 1) for i, x in enumerate(inf) if x)}')
        d.text((W - PAD - d.textlength(hdr, font=F_S), 8), hdr, font=F_S, fill=(120, 128, 140))

        for c, (name, _) in enumerate(ARMS):
            r = runs[name]
            px = PAD + c * (PANE + GAP)
            py = HEAD
            ours = (name == OURS)

            pane, (ox, oy), sc = fit(top, PANE)
            pd = ImageDraw.Draw(pane)
            sel = [i for i, s in enumerate(r['steps']) if s < cut]
            sp = [(ox + r['pts'][i][0] * sc, oy + r['pts'][i][1] * sc) for i in sel]
            ks = [r['steps'][i] for i in sel]
            for t in range(5):
                seg = [sp[i] for i, s in enumerate(ks) if r['edges'][t] <= s < r['edges'][t + 1]]
                if len(seg) < 2:
                    continue
                pd.line(seg, fill=(0, 0, 0), width=7, joint='curve')
                pd.line(seg, fill=TASK_COLS[t], width=4, joint='curve')

            def dot(p, col, rad=8):
                a, b = p
                pd.ellipse([a - rad, b - rad, a + rad, b + rad], fill=col,
                           outline=(0, 0, 0), width=2)
            if sp:
                dot(sp[0], C_START)
                last = len(sel) == len(r['steps'])          # 这条轨迹已经放完
                dot(sp[-1], C_END if last else C_NOW)

            cv.paste(pane, (px, py))
            d.rectangle([px, py, px + PANE - 1, py + PANE - 1],
                        outline=(47, 109, 246) if ours else (224, 228, 234),
                        width=3 if ours else 1)
            d.text((px + (PANE - d.textlength(name, font=F_H)) / 2, py - 26), name,
                   font=F_H, fill=(47, 109, 246) if ours else (20, 22, 26))

            done = max([0] + [n for cc, n in r['done_at'] if cc <= cut])
            lab = f'{done}/5 answered'
            d.text((px + (PANE - d.textlength(lab, font=F_M)) / 2, py + PANE + 8), lab,
                   font=F_M, fill=(47, 109, 246) if ours else (20, 22, 26))

            by, bh = py + PANE + 34, 13
            d.rectangle([px, by, px + PANE - 1, by + bh], fill=(238, 240, 244))
            xx, left, spent = px, cut, 0
            for t, k in enumerate(r['per_task']):
                k = max(0, min(k, left)); left -= k; spent += k
                w = k / SHOW * PANE
                if w >= 1:
                    d.rectangle([xx, by, xx + w, by + bh], fill=TASK_COLS[t])
                xx += w
            d.rectangle([px, by, px + PANE - 1, by + bh], outline=(214, 219, 227), width=1)
            sub = f'{min(spent, SHOW)} steps used'
            d.text((px + (PANE - d.textlength(sub, font=F_S)) / 2, by + bh + 3), sub,
                   font=F_S, fill=(120, 128, 140))

        cv.save(f'{tmp}/{fi:04d}.png')

    mp4 = f'{OUT}/sequential_{slug}.mp4'
    if os.path.exists(mp4):
        os.remove(mp4)          # 否则失败时旧文件会被当成新结果
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'warning', '-framerate', str(FPS),
                    '-i', f'{tmp}/%04d.png', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20',
                    '-movflags', '+faststart', mp4], check=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', mp4, '-frames:v', '1',
                    '-q:v', '4', f'{ASSETS}/poster_sequential_{slug}.jpg'], check=True)
    print(f'  -> {mp4}  {os.path.getsize(mp4)/1e6:.1f} MB  {NF} 帧 / {NF/FPS:.1f}s')

json.dump(meta, open(f'{ASSETS}/figure/sequential.json', 'w'), indent=1)
print('\ndone')
os._exit(0)
