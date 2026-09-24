#!/usr/bin/env python
"""按场景包围盒渲染带贴图的正交俯视图, 并给出 world(x,z) -> 图像素 的换算。

ortho_scale 有两个坑, 两个都踩过:
  * 语义是 **可见 world 宽度 = 1 / ortho_scale** (实测 0.1 -> 10 m, 0.05 -> 20 m),
    不是半边长;
  * 必须在**建 simulator 之前**写进 spec —— 投影矩阵是构造时算好的, 事后改
    `sim.get_agent(0)._sensors['top'].specification().ortho_scale` 不会重算,
    渲染仍按默认 0.1 出图(固定 10 m 窗口), 而轨迹按包围盒换算, 线就和墙对不上。
"""
import os

import numpy as np

os.environ.setdefault('MAGNUM_LOG', 'quiet')
os.environ.setdefault('HABITAT_SIM_LOG', 'quiet')
import habitat_sim                                          # noqa: E402
from habitat_sim.utils.common import quat_from_angle_axis   # noqa: E402
from PIL import Image                                       # noqa: E402

HM3D = '/egr/research-actionlab/caizhon2/datasets/HM3D'
TOP_RES = 1000
ROT_CCW = 90            # 户型转正; 图和轨迹必须转同一个角


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


def _sim(glb, sensors):
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = glb
    cfg.enable_physics = False
    ag = habitat_sim.agent.AgentConfiguration()
    ag.sensor_specifications = sensors
    return habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ag]))


def _dummy():
    s = habitat_sim.CameraSensorSpec()
    s.uuid = 'rgb'
    s.sensor_type = habitat_sim.SensorType.COLOR
    s.resolution = [64, 64]
    return s


def _top(half):
    s = habitat_sim.CameraSensorSpec()
    s.uuid = 'top'
    s.sensor_type = habitat_sim.SensorType.COLOR
    s.sensor_subtype = habitat_sim.SensorSubType.ORTHOGRAPHIC
    s.resolution = [TOP_RES, TOP_RES]
    s.position = [0.0, 0.0, 0.0]
    s.orientation = [np.deg2rad(-90.0), 0.0, 0.0]
    s.ortho_scale = 1.0 / (2.0 * half)
    return s


FLOOR_CLIP = 3.0        # 相机放在 agent 楼层上方这么高


def render_scene(tag, floor_y=None, verbose=True):
    """-> (旋转裁剪后的 PIL 图, to_px(list[[x,y,z]]) -> [(u,v), ...])

    floor_y: agent 所在楼层的 y。**多层场景必须传** —— 相机若放在整个包围盒
    上方, 正交俯视看到的是最高那层, 而 agent 可能在下面一层(实测 00202 场景
    agent 在 y=-3.32, 包围盒顶在 +3.22), 轨迹就被画在了一张它根本没走过的
    平面图上, 看起来到处穿墙穿家具。相机压到 agent 楼层上方 FLOOR_CLIP 米,
    上面几层落在相机背后, 自然不渲染。
    """
    glb = scene_glb(scene_dir(tag))

    sim = _sim(glb, [_dummy()])     # 只为读包围盒: ortho_scale 依赖它, 又必须先于建 sim
    bb = sim.get_active_scene_graph().get_root_node().cumulative_bb
    lo = np.array([bb.min[0], bb.min[1], bb.min[2]])
    hi = np.array([bb.max[0], bb.max[1], bb.max[2]])
    sim.close()
    cx, cz = (lo[0] + hi[0]) / 2, (lo[2] + hi[2]) / 2
    half = max(hi[0] - lo[0], hi[2] - lo[2]) / 2 * 1.02

    sim = _sim(glb, [_dummy(), _top(half)])
    cam_y = hi[1] + 1.0 if floor_y is None else float(floor_y) + FLOOR_CLIP
    st = sim.get_agent(0).get_state()
    st.position = np.array([cx, cam_y, cz], dtype=np.float32)
    st.rotation = quat_from_angle_axis(0.0, np.array([0.0, 1.0, 0.0]))
    sim.get_agent(0).set_state(st)
    top = Image.fromarray(sim.get_sensor_observations()['top'][..., :3])
    sim.close()

    # 正交渲染四周是纯黑, 不裁地图只占中间一小块。按行/列的内容像素**占比**定边界,
    # 单像素噪点撑不起一整行(只看"有没有暗像素"会被杂点撑满整幅, 等于没裁)。
    solid = np.asarray(top.convert('L')) > 28
    rows, cols = solid.sum(1), solid.sum(0)
    rt = max(3, int(0.012 * solid.shape[1]))
    ct = max(3, int(0.012 * solid.shape[0]))
    ys = np.nonzero(rows > rt)[0]
    xs = np.nonzero(cols > ct)[0]
    pad = 8
    x0 = max(0, int(xs.min()) - pad); x1 = min(TOP_RES, int(xs.max()) + pad)
    y0 = max(0, int(ys.min()) - pad); y1 = min(TOP_RES, int(ys.max()) + pad)
    top = top.crop((x0, y0, x1, y1))
    cw, ch = top.width, top.height
    for _ in range(ROT_CCW // 90):
        top = top.transpose(Image.ROTATE_90)    # PIL 的 ROTATE_90 就是逆时针

    def to_px(P):
        if len(P) == 0:
            return []
        u = TOP_RES / 2 + (np.array([p[0] for p in P]) - cx) / half * TOP_RES / 2
        v = TOP_RES / 2 + (np.array([p[2] for p in P]) - cz) / half * TOP_RES / 2
        out = []
        for a, b in zip(u, v):
            a, b = float(a) - x0, float(b) - y0
            w = cw
            for _ in range(ROT_CCW // 90):
                a, b = b, w - 1 - a
                w = ch if w == cw else cw
            out.append((a, b))
        return out

    if verbose:
        print(f'   [{tag}] 相机 y={cam_y:.2f} (楼层 {"-" if floor_y is None else f"{floor_y:.2f}"}), '
              f'俯视图 {cw}x{ch} -> {top.width}x{top.height}')
    return top, to_px
