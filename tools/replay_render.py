#!/usr/bin/env python
"""沿已记录的轨迹回放渲染: 第一视角 + 带贴图的固定视口俯视图 + 轨迹。

不重新推理、不重新决策 —— agent 的落点全部取自各 run 日志里的
`Current position: [x, y, z]`, 这里只是把相机放到那些位置上重新渲一遍。

产出:
    assets/scene/<scene>.webp          每个场景一张正交俯视图(带贴图, 固定视口)
    assets/fp/<case>/<kind>/NNN.webp   逐步第一视角
    assets/cases.json                  问答/步数 + 轨迹像素坐标 + 帧清单

轨迹不再逐帧出图, 而是把像素坐标写进 json 由前端画在 canvas 上 ——
一条路径几百个数, 比几百张图小三个数量级, 动画也更顺。

两点必须说明:
  * 朝向: 日志只记了位置, 没记朝向。第一视角的相机朝向取**行进方向**
    (原地不动的步沿用上一个朝向), 所以这是"沿它走过的路回放", 不等于
    当时它实际看向哪里。
  * 俯视图视口: 按场景包围盒固定, 整段不变 —— run 里存的 topdown 是逐帧
    按已探索范围重新取景的(尺寸都在变), 拿来做滑块回放会飘。
"""
import json
import os
import re

import numpy as np

os.environ.setdefault('MAGNUM_LOG', 'quiet')
os.environ.setdefault('HABITAT_SIM_LOG', 'quiet')
import habitat_sim                                    # noqa: E402
from habitat_sim.utils.common import quat_from_angle_axis   # noqa: E402
from PIL import Image                                 # noqa: E402

VIS = '/egr/research-actionlab/caizhon2/codes/EQA/SPMR/visualization'
HM3D = '/egr/research-actionlab/caizhon2/datasets/HM3D'
OUT = '/egr/research-actionlab/caizhon2/codes/EQA/SPMR/3DSPMR_web/assets'

MAX_FRAMES = 90
FP_RES = 640          # 第一视角渲染分辨率
FP_W = 520            # 落盘宽度
TOP_RES = 900         # 俯视图分辨率
CAM_H, CAM_TILT, HFOV = 1.5, -30.0, 120.0     # 与 eval_sg_snap_2.yaml 一致
Q = 74

GT_FIX = {('00824-Dd4bFSTQ8gi_4', '1'): 'A toilet paper holder'}

# 俯视图逆时针旋转角度(度)。户型本身是正的, 这里只是选一个看着顺眼的朝向。
# 图和轨迹坐标必须转同一个角, 否则线会整体错位。按场景单独调, 默认 90。
ROT_CCW = {}
DEFAULT_ROT_CCW = 90

CASES = [
    dict(key='incorrect_attribute', title='Incorrect Attribute',
         feasible=('incorrect_attribute/sg_snap_qwen_incorrect_attribute_2', '00824-Dd4bFSTQ8gi_4'),
         infeasible=('incorrect_attribute/sg_snap_qwen_incorrect_attribute', '00824-Dd4bFSTQ8gi_3')),
    dict(key='object_not_present', title='Object not Present',
         feasible=('object_not_present/sg_snap_qwen_obj_not_present', '00848-ziup5kvtCCR_3'),
         infeasible=('object_not_present/sg_snap_qwen_obj_not_present', '00255-NGyoyh91xXJ_3')),
    dict(key='room_not_present', title='Room not Present',
         feasible=('room_not_present/sg_snap_qwen_room_not_exist', '00166-RaYrxWt5pR1_2'),
         infeasible=('room_not_present/sg_snap_qwen_room_not_exist_2', '00164-XfUxBGTFQQb_2')),
]


def scene_dir(tag):
    for split in ('val', 'train', 'minival', 'test'):
        p = f'{HM3D}/{split}/{tag}'
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(tag)


def scene_glb(d):
    fs = os.listdir(d)
    for f in fs:
        if f.endswith('.glb') and 'semantic' not in f and 'basis' not in f:
            return os.path.join(d, f)
    for f in fs:
        if f.endswith('.basis.glb') and 'semantic' not in f:
            return os.path.join(d, f)
    raise FileNotFoundError(d)


def positions(run, episode):
    """从 run 日志里按 step 顺序取世界坐标。

    日志形如:  == step: 7 ... Current position: [x y z], <累计路程>
    一个 step 内可能记多条(路径中间点), 只取该 step 的第一条。
    同一个 run 目录可能跑过多个 episode, 用 'Question id <ep> initialization'
    把日志切到对应段。
    """
    txt = open(f'{VIS}/{run}/log_0.00_1.00.log', errors='replace').read()
    marks = [m for m in re.finditer(r'Question id (\S+) initialization successful', txt)]
    if marks:
        seg = None
        for i, m in enumerate(marks):
            if m.group(1) == episode:
                end = marks[i + 1].start() if i + 1 < len(marks) else len(txt)
                seg = txt[m.start():end]
        if seg is not None:
            txt = seg
    out, cur = {}, None
    for a, b in re.findall(r'== step: (\d+)|Current position: \[([^\]]+)\]', txt):
        if a:
            cur = int(a)
        elif cur is not None and cur not in out:
            out[cur] = [float(x) for x in b.split()]
    return [out[k] for k in sorted(out)]


def sample(n, cap):
    if n <= cap:
        return list(range(n))
    return sorted({round(i * (n - 1) / (cap - 1)) for i in range(cap)})


def heading_at(P, i):
    """朝向 = 行进方向; 原地不动就往前/往后找最近的一次位移。"""
    for j in range(i + 1, len(P)):
        d = np.array([P[j][0] - P[i][0], P[j][2] - P[i][2]])
        if np.linalg.norm(d) > 0.05:
            return d / np.linalg.norm(d)
    for j in range(i - 1, -1, -1):
        d = np.array([P[i][0] - P[j][0], P[i][2] - P[j][2]])
        if np.linalg.norm(d) > 0.05:
            return d / np.linalg.norm(d)
    return np.array([0.0, -1.0])


def make_sim(glb, sensors):
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = glb
    cfg.enable_physics = False
    ag = habitat_sim.agent.AgentConfiguration()
    ag.sensor_specifications = sensors
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ag]))


def fp_spec():
    s = habitat_sim.CameraSensorSpec()
    s.uuid = 'rgb'
    s.sensor_type = habitat_sim.SensorType.COLOR
    s.resolution = [FP_RES, FP_RES]
    s.position = [0.0, CAM_H, 0.0]
    s.orientation = [np.deg2rad(CAM_TILT), 0.0, 0.0]
    s.hfov = HFOV
    return s


def top_spec(half):
    """正交俯视相机。half = 想让画面覆盖的半边长(米)。

    ortho_scale 的语义是 **可见world宽度 = 1/ortho_scale**(实测: 0.1->10m, 0.05->20m),
    所以覆盖 ±half 要设 1/(2*half)。

    并且它必须在**建 simulator 之前**写进 spec —— 投影矩阵是构造时算好的,
    事后改 `sim.get_agent(0)._sensors['top'].specification().ortho_scale` 不会重算,
    渲染仍按默认 0.1 (固定 10m 窗口) 出图, 而轨迹却按 half 换算, 于是线和墙对不上。
    """
    s = habitat_sim.CameraSensorSpec()
    s.uuid = 'top'
    s.sensor_type = habitat_sim.SensorType.COLOR
    s.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    s.resolution = [TOP_RES, TOP_RES]
    s.position = [0.0, 0.0, 0.0]
    s.orientation = [np.deg2rad(-90.0), 0.0, 0.0]
    s.ortho_scale = 1.0 / (2.0 * half)
    return s


def scene_extent(glb):
    """只为读包围盒开一次 sim —— ortho_scale 要在建 sim 前就定下来, 而它依赖包围盒。"""
    sim = make_sim(glb, [fp_spec()])
    bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
    lo = np.array([bb.min[0], bb.min[1], bb.min[2]])
    hi = np.array([bb.max[0], bb.max[1], bb.max[2]])
    sim.close()
    return lo, hi


# ---- 按场景分组, 每个场景只开一次 simulator ----
jobs = {}
for c in CASES:
    for kind in ('feasible', 'infeasible'):
        run, ep = c[kind]
        jobs.setdefault(ep.rsplit('_', 1)[0], []).append((c, kind, run, ep))

meta = {'cases': [], 'scenes': {}}
by_key = {}

for tag, items in jobs.items():
    glb = scene_glb(scene_dir(tag))
    lo, hi = scene_extent(glb)
    cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
    half = max(hi[0] - lo[0], hi[2] - lo[2]) / 2 * 1.02
    sim = make_sim(glb, [fp_spec(), top_spec(half)])

    # 相机压到 agent 楼层上方 3 m —— 放在整个包围盒上方的话, 多层场景俯视看到的
    # 是最高那层, 轨迹会被画在一张 agent 根本没走过的平面图上, 看着到处穿墙。
    ys = [p[1] for _c, _k, _r, _e in items for p in positions(_r, _e)]
    floor_y = float(np.median(ys)) if ys else None
    cam_y = hi[1] + 1.0 if floor_y is None else floor_y + 3.0
    st = sim.get_agent(0).get_state()
    st.position = np.array([cx, cam_y, cz], dtype=np.float32)
    st.rotation = quat_from_angle_axis(0.0, np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(st)
    os.makedirs(f'{OUT}/scene', exist_ok=True)
    top = Image.fromarray(sim.get_sensor_observations()['top'][..., :3])
    # 正交渲染四周是没有几何的纯黑, 不裁的话地图只占画面中间一小块。
    # 裁完轨迹坐标要减去同样的偏移, 否则线会整体错位。
    # 阈值太松会被零星杂点(渲染噪声、屋外碎几何)撑满整幅, 等于没裁。
    # 改成: 先抬阈值, 再按行/列的内容像素**占比**定边界, 单像素噪点撑不起一整行。
    arr = np.asarray(top.convert('L'))
    solid = arr > 28
    rows = solid.sum(1); cols = solid.sum(0)
    rt = max(3, int(0.012 * solid.shape[1])); ct = max(3, int(0.012 * solid.shape[0]))
    ys = np.nonzero(rows > rt)[0]; xs = np.nonzero(cols > ct)[0]
    if len(ys) == 0 or len(xs) == 0:
        ys = np.nonzero(rows)[0]; xs = np.nonzero(cols)[0]
    pad = 8
    x0 = max(0, int(xs.min()) - pad); x1 = min(TOP_RES, int(xs.max()) + pad)
    y0 = max(0, int(ys.min()) - pad); y1 = min(TOP_RES, int(ys.max()) + pad)
    top = top.crop((x0, y0, x1, y1))
    cw, ch = top.width, top.height          # 旋转前尺寸, 换算轨迹要用
    rot = ROT_CCW.get(tag, DEFAULT_ROT_CCW) % 360
    for _ in range(rot // 90):
        top = top.transpose(Image.ROTATE_90)     # PIL 的 ROTATE_90 就是逆时针
    top.save(f'{OUT}/scene/{tag}.webp', 'WEBP', quality=88, method=5)
    meta['scenes'][tag] = {'w': top.width, 'h': top.height}
    print(f'   俯视图裁剪 ({x0},{y0})-({x1},{y1}) -> {cw}x{ch}, 逆时针 {rot}° -> {top.width}x{top.height}')

    def rot_pt(a, b):
        """裁剪后坐标 -> 旋转后坐标。逆时针 90: (x,y) -> (y, W-1-x), 尺寸 (W,H)->(H,W)。"""
        w, h = cw, ch
        for _ in range(rot // 90):
            a, b = b, w - 1 - a
            w, h = h, w
        return a, b
    # world(x,z) -> 该图像素:  u = R/2 + (x-cx)/half * R/2   (z 同理 -> v)
    print(f'[{tag}] ortho half={half:.2f}m  center=({cx:.2f},{cz:.2f})  相机 y={cam_y:.2f}')

    for c, kind, run, ep in items:
        P = positions(run, ep)
        res = json.load(open(f'{VIS}/{run}/{ep}/result.json'))
        qid = sorted(res)[0]
        q = dict(res[qid])
        fix = GT_FIX.get((ep, qid))
        if fix:
            q['answer'] = fix
        u = TOP_RES / 2 + (np.array([p[0] for p in P]) - cx) / half * TOP_RES / 2
        v = TOP_RES / 2 + (np.array([p[2] for p in P]) - cz) / half * TOP_RES / 2
        path = []
        for a, b in zip(u, v):
            ra, rb = rot_pt(float(a) - x0, float(b) - y0)
            path.append([round(ra, 1), round(rb, 1)])

        idx = sample(len(P), MAX_FRAMES)
        od = f'{OUT}/fp/{c["key"]}/{kind}'
        os.makedirs(od, exist_ok=True)
        for n, i in enumerate(idx):
            d = heading_at(P, i)
            yaw = float(np.arctan2(-d[0], -d[1]))     # habitat 里 -z 是正前方
            st = sim.get_agent(0).get_state()
            st.position = np.array(P[i], dtype=np.float32)
            st.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
            sim.get_agent(0).set_state(st)
            im = Image.fromarray(sim.get_sensor_observations()['rgb'][..., :3])
            im = im.resize((FP_W, round(im.height * FP_W / im.width)), Image.LANCZOS)
            im.save(f'{od}/{n:03d}.webp', 'WEBP', quality=Q, method=5)

        by_key.setdefault(c['key'], {'key': c['key'], 'title': c['title'],
                                     'variants': {}})['variants'][kind] = {
            'episode': ep, 'scene': tag,
            'question': q['question'], 'answer': q['answer'],
            'prediction': q['prediction'], 'steps': q['step'],
            'question_subtype': q.get('question_subtype'),
            'path': path, 'frames': len(idx), 'frame_step': idx,
            'map_w': top.width, 'map_h': top.height,
        }
        print(f'   {c["key"]:22s} {kind:10s} {ep:24s} {len(P):3d} 步 -> {len(idx):3d} 帧')
    sim.close()

meta['cases'] = [by_key[c['key']] for c in CASES]
json.dump(meta, open(f'{OUT}/cases.json', 'w'), ensure_ascii=False, indent=1)
print(f'\n写到 {OUT}/cases.json')
