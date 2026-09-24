#!/usr/bin/env python
"""从各 baseline 的 run 日志里抠出同一个 episode 的轨迹(世界坐标)。

日志里唯一的分段锚点是 `Question id <ep> initialization successful!`,
其后到下一个锚点之间的位置行属于该 episode。位置行两种写法:
    Current position: [x y z]     (sg_snap / no_fov / 3d_mem / GraphEQA / Pred-EQA)
    Current pts:      [x y z]     (Explore-EQA)
每个 `== step: N` 只取该 step 的第一条位置(后面的是候选点/回放, 不是落点)。
"""
import json
import os
import re
import sys

LOGS = '/egr/research-actionlab/caizhon2/codes/EQA/SPMR/3DSPMR_code/ExploreVQA/extended_exp/logs'

# 顺序即网页上的展示顺序; ours 放最后, 便于和左侧 baseline 对照
ARMS = [
    ('Explore-EQA', ['explore_eqa_p345.log']),
    ('3D-Mem',      ['3d_mem_p345.log']),
    ('GraphEQA',    ['grapheqa_p345_enriched.log']),
    ('Pred-EQA',    ['predeqa_p345_h1.log', 'predeqa_p345_h2.log',
                     'predeqa_p345.log', 'predeqa_p345_new24.log']),
    ('3DSPMR (ours)', ['sg_snap_p345.log']),
]

INIT = re.compile(r'Question id (\S+) initialization successful')
STEP = re.compile(r'== step:\s*(\d+)')
POS = re.compile(r'Current (?:position|pts):\s*\[([^\]]+)\]')
# 子任务切换 —— step 计数在整个 episode 内连续不重置, 只有这行能划分 5 个子任务
NEWTASK = re.compile(r'Receive New Task')


def paths_for(episode, log_paths):
    """返回 (轨迹, 子任务起始 step)。轨迹按 step 去重, 每 step 只留第一条落点。"""
    out, bounds, cur_step, inside, seen = {}, [], None, False, False
    for lp in log_paths:
        if seen:
            break
        p = os.path.join(LOGS, lp)
        if not os.path.exists(p):
            continue
        with open(p, errors='ignore') as fh:
            for line in fh:
                m = INIT.search(line)
                if m:
                    if inside and m.group(1) != episode:
                        inside, seen = False, True
                        break
                    inside = (m.group(1) == episode)
                    if inside:
                        cur_step = None
                    continue
                if not inside:
                    continue
                if NEWTASK.search(line):
                    bounds.append(cur_step if cur_step is not None else 0)
                    continue
                m = STEP.search(line)
                if m:
                    cur_step = int(m.group(1))
                    continue
                m = POS.search(line)
                if m and cur_step is not None and cur_step not in out:
                    out[cur_step] = [float(x) for x in m.group(1).split()]
    steps = sorted(out)
    return [out[k] for k in steps], steps, bounds


if __name__ == '__main__':
    eps = sys.argv[1:] or ['00170-S3r45BMWy6H_0']
    res = {}
    for ep in eps:
        res[ep] = {}
        print(f'=== {ep} ===')
        for name, logs in ARMS:
            P, steps, bounds = paths_for(ep, logs)
            res[ep][name] = {'path': P, 'steps': steps, 'task_starts': bounds}
            if P:
                xs = [p[0] for p in P]; zs = [p[2] for p in P]
                span = max(max(xs) - min(xs), max(zs) - min(zs))
                print(f'  {name:16s} {len(P):4d} steps  跨度 {span:5.1f} m  '
                      f'子任务起点 {bounds}')
            else:
                print(f'  {name:16s}    - 没抓到')
    json.dump(res, open('/tmp/baseline_paths.json', 'w'))
    print('\n-> /tmp/baseline_paths.json')
