#!/usr/bin/env python
"""把每个 case 合成一段对比视频: 一行四列 ——
   [feasible 第一视角][feasible 轨迹][infeasible 第一视角][infeasible 轨迹]。

两侧按**归一化进度**推进(不是按绝对步号) —— 两条轨迹步数差很大, 按步号对齐
会让短的那条早早停住, 正好看不出对比。
"""
import json, os, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WEB = '/egr/research-actionlab/caizhon2/codes/EQA/SPMR/3DSPMR_web'
OUT = f'{WEB}/videos'
A = f'{WEB}/assets'
FPS, PACE = 15, 2      # 全站统一: 15 fps, 每步 2 帧
# 帧数 = 最长那条臂的步数 × PACE。**不夹上下限** —— 夹了以后短 case 的每帧
# 步进会被拉慢, 三段视频推进速度就不一致, 没法横向比。代价是 24 步那个 case
# 只有 3 秒多, 但那本来就是条只走了 24 步的轨迹。
def nframe(v):
    return max(2, max(x['steps'] for x in v.values()) * PACE)
# 绿=起点, 浅黄=终点, 红=当前位置 —— 与论文图例和 sequential 那三段一致
C_START, C_END, C_NOW = (22, 179, 100), (255, 230, 128), (255, 59, 107)
PANE = 430          # 每个窗格边长
PAD, HEAD, GAP = 18, 40, 12

def font(sz, bold=False):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()

F_T, F_L, F_S = font(30, True), font(19, True), font(17)

def cap(d, text, x0, y):
    """窗格下方的说明, 在该窗格里居中。"""
    w = d.textlength(text, font=F_S)
    d.text((x0 + (PANE - w) / 2, y), text, font=F_S, fill=(120, 128, 140))


def fit(im, box):
    im = im.copy(); im.thumbnail((box, box), Image.LANCZOS)
    c = Image.new('RGB', (box, box), (248, 248, 250))
    c.paste(im, ((box - im.width) // 2, (box - im.height) // 2))
    return c

def draw_map(bg, path, upto):
    im = bg.copy(); d = ImageDraw.Draw(im)
    n = max(1, min(len(path), upto + 1))
    pts = [tuple(p) for p in path[:n]]
    if len(pts) > 1:
        d.line(pts, fill=(0, 0, 0), width=11, joint='curve')
        d.line(pts, fill=(255, 214, 0), width=6, joint='curve')
    def dot(p, r, f):
        d.ellipse([p[0]-r, p[1]-r, p[0]+r, p[1]+r], fill=f, outline=(20, 20, 20), width=3)
    # 与论文图例一致: 绿 = Start, 浅黄 = End
    dot(pts[0], 11, C_START)
    dot(pts[-1], 11, C_END if n >= len(path) else C_NOW)
    return im

meta = json.load(open(f'{A}/cases.json'))
os.makedirs(OUT, exist_ok=True)
W = PAD * 2 + PANE * 4 + GAP * 3
H = HEAD + PANE + PAD + 36

for c in meta['cases']:
    tmp = f'/tmp/vid_{c["key"]}'
    os.makedirs(tmp, exist_ok=True)
    V = c['variants']
    bgs = {k: Image.open(f'{A}/scene/{V[k]["scene"]}.webp').convert('RGB') for k in V}
    NF = nframe(V)
    for fi in range(NF):
        # 按**真实步号**推进, 不按归一化进度 —— 归一化会让 64 步那条臂和 158 步
        # 那条臂在同样的帧数里走完, 两条臂、三段视频的推进速度全不一样, 看起来
        # 就是"帧率不同"。现在每条臂一律 PACE 帧走一步, 短的那条先走完就停在
        # 终点等着, 长短差距反而直接看得出来。
        cut = (fi + 1) / PACE
        cv = Image.new('RGB', (W, H), (255, 255, 255))
        d = ImageDraw.Draw(cv)
        for j, (kind, label, col) in enumerate(
                [('feasible', 'FEASIBLE', (15, 157, 88)), ('infeasible', 'INFEASIBLE', (209, 116, 26))]):
            v = V[kind]
            x0 = PAD + j * 2 * (PANE + GAP)
            # 该臂占两列, 标签横跨两列
            d.text((x0, 10), label, font=F_L, fill=col)
            pi = min(len(v['path']) - 1, max(0, int(cut)))
            # 第一视角是从整段轨迹里抽帧存的, 按步号比例找对应那一帧
            frac = pi / max(1, len(v['path']) - 1)
            k = min(v['frames'] - 1, round(frac * (v['frames'] - 1)))
            fp = Image.open(f'{A}/fp/{c["key"]}/{kind}/{k:03d}.webp').convert('RGB')
            cv.paste(fit(fp, PANE), (x0, HEAD))
            cv.paste(fit(draw_map(bgs[kind], v['path'], pi), PANE), (x0 + PANE + GAP, HEAD))
            cap(d, 'first-person', x0, HEAD + PANE + 10)
            # 场景写成论文图里那个写法(Scene-00824), 而不是内部 run id
            cap(d, f'Scene-{v["scene"].split("-")[0]}: trajectory',
                x0 + PANE + GAP, HEAD + PANE + 10)
        cv.save(f'{tmp}/{fi:04d}.png')
    mp4 = f'{OUT}/{c["key"]}.mp4'
    if os.path.exists(mp4):
        os.remove(mp4)     # 否则失败时旧文件会被当成新结果
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'warning', '-framerate', str(FPS),
                    '-i', f'{tmp}/%04d.png', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', mp4])
    if not os.path.exists(mp4):
        print('  ★ ffmpeg 失败:', c['key']); continue
    print(f'  {c["key"]:22s} -> {mp4}  {os.path.getsize(mp4)/1e6:.1f} MB')
    for f in os.listdir(tmp): os.remove(f'{tmp}/{f}')
    os.rmdir(tmp)
print('done')
