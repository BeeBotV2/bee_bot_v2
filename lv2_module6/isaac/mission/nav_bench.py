#!/usr/bin/env python3
"""Nav2 driving benchmark in Isaac Sim — sends NavigateToPose legs directly (no mission_manager / skills) and scores them
against Isaac ground truth.

  python3 isaac/mission/nav_bench.py                      # default route H → S4 → S5 → S3 → S6 → S2 → S1 → L → H
  python3 isaac/mission/nav_bench.py --legs S4,S5 --dyn popup:1   # pop an obstacle onto the path of leg 1 (0-based)
  python3 isaac/mission/nav_bench.py --dyn cross:3 --tag mppi_a    # obstacle crossing the path of leg 3

Needs the Isaac process started with the truth writer (isaac/standalone/runtime/isaac_startup.py writes
/tmp/beebot_sim_nav.json) and, for --dyn, BEE_DYN_OBS=1. Same ROS_DOMAIN_ID and Fast DDS profile as the Isaac run.

Per leg it records: success, wall time, planning time (one ComputePathToPose before the leg), path length, truth
distance driven, mean / peak truth speed, time standing still, minimum body clearance to the scene walls
(1 cm grid from make_map.py --res 0.01), to every box (truth pose) and to the moving obstacle, box displacement,
localization error (TF map vs truth) and distance of the robot from the current /plan.
Path quality: shortest feasible length (Dijkstra on the wall grid + truth boxes, body half width kept clear) vs planned
and driven length. Smoothness from /cmd_vel: speed reversals >= 5 cm/s per 10 s of driving and share of crawl time.
Results: ~/.ros/nav_bench/<time>_<tag>.json and a table on stdout.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import rclpy
import heapq

from geometry_msgs.msg import PoseStamped, TwistStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import Path
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy
from scipy import ndimage
from tf2_ros import Buffer, TransformListener

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH = '/tmp/beebot_sim_nav.json'
DYN = '/tmp/beebot_dyn_obs.json'
DYN_R = 0.10  # isaac_startup.py DYN_R
FOOT = (0.069, -0.245, 0.151)  # front, rear, half width [m] from base_footprint (nav2_params.yaml footprint)
BOX_HALF = 0.1025  # plan.yaml box_half
GOALS = {  # map frame (x, y, yaw) — 'near' poses the mission used most (~/.ros/mission/*/steps.jsonl)
    'H': (0.0, 0.0, 0.0),
    'S4': (0.000, -3.677, -1.571), 'S5': (-0.778, -4.404, math.pi), 'S3': (-2.002, -3.657, -1.571),
    'S6': (-2.233, -2.759, math.pi), 'S2': (-2.926, -1.662, -1.571), 'S1': (-1.544, 0.0, 0.0),
    'L': (-2.440, -0.015, math.pi), 'L2': (-2.925, -3.669, -1.571),
}
ROUTE = ['S4', 'S5', 'S3', 'S6', 'S2', 'S1', 'L', 'H']


def footprint_pts(x, y, th, step=0.01):
    f, r, w = FOOT
    edges = [((f, w), (f, -w)), ((f, -w), (r, -w)), ((r, -w), (r, w)), ((r, w), (f, w))]
    pts = []
    for (ax, ay), (bx, by) in edges:
        n = max(2, int(math.hypot(bx - ax, by - ay) / step))
        for k in range(n):
            u = k / n
            pts.append((ax + (bx - ax) * u, ay + (by - ay) * u))
    p = np.array(pts)
    c, s = math.cos(th), math.sin(th)
    return np.c_[x + c * p[:, 0] - s * p[:, 1], y + s * p[:, 0] + c * p[:, 1]]


def inside_rect(pts, x, y, th, f, r, w):
    """True if any point lies inside the rectangle (front f, rear r, half width w) at pose (x, y, th)."""
    c, s = math.cos(th), math.sin(th)
    dx, dy = pts[:, 0] - x, pts[:, 1] - y
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return bool(np.any((lx <= f) & (lx >= r) & (np.abs(ly) <= w)))


def rect_clearance(pts, bx, by, bth, h):
    """Distance from points to an axis-aligned square (half side h) at (bx, by, bth); 0 if any point is inside."""
    c, s = math.cos(bth), math.sin(bth)
    dx, dy = pts[:, 0] - bx, pts[:, 1] - by
    lx, ly = np.abs(c * dx + s * dy) - h, np.abs(-s * dx + c * dy) - h
    d = np.hypot(np.maximum(lx, 0), np.maximum(ly, 0))
    return float(d.min())


class Walls:
    """Distance field of the scene walls (make_map.py --res 0.01 grid)."""

    def __init__(self, d):
        import yaml
        from PIL import Image
        m = yaml.safe_load(open(os.path.join(d, 'map.yaml')))
        occ = np.array(Image.open(os.path.join(d, 'map.pgm'))) == 0
        self.res, self.ox, self.oy, self.h = m['resolution'], m['origin'][0], m['origin'][1], occ.shape[0]
        self.dist = ndimage.distance_transform_edt(~occ) * self.res

    def clearance(self, pts):
        i = np.clip((self.h - 1 - (pts[:, 1] - self.oy) / self.res).astype(int), 0, self.dist.shape[0] - 1)
        j = np.clip(((pts[:, 0] - self.ox) / self.res).astype(int), 0, self.dist.shape[1] - 1)
        return float(self.dist[i, j].min()) - self.res / 2

    def shortest(self, a, b, boxes, clear=FOOT[2] + 0.01, step=0.02):
        """Shortest 8-connected path [m] from a to b keeping `clear` from walls and boxes (lower bound for the planner)."""
        k = max(1, int(round(step / self.res)))
        free = self.dist[::k, ::k] > clear
        h, w = free.shape
        jj, ii = np.meshgrid(np.arange(w), np.arange(h))
        X, Y = self.ox + jj * k * self.res, self.oy + (self.h - 1 - ii * k) * self.res
        for bx, by, bth in boxes:
            c, s_ = math.cos(bth), math.sin(bth)
            lx, ly = np.abs(c * (X - bx) + s_ * (Y - by)) - BOX_HALF, np.abs(-s_ * (X - bx) + c * (Y - by)) - BOX_HALF
            free &= np.hypot(np.maximum(lx, 0), np.maximum(ly, 0)) > clear
        cell = lambda p: (int(np.clip(round((self.h - 1 - (p[1] - self.oy) / self.res) / k), 0, h - 1)),
                          int(np.clip(round((p[0] - self.ox) / self.res / k), 0, w - 1)))
        s0, g = cell(a), cell(b)
        free[s0] = free[g] = True
        D = {s0: 0.0}
        q = [(0.0, s0)]
        nb = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0), (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142)]
        while q:
            d, (i, j) = heapq.heappop(q)
            if (i, j) == g:
                return d * k * self.res
            if d > D[(i, j)]:
                continue
            for di, dj, c_ in nb:
                n = (i + di, j + dj)
                if 0 <= n[0] < h and 0 <= n[1] < w and free[n] and d + c_ < D.get(n, 1e18):
                    D[n] = d + c_
                    heapq.heappush(q, (d + c_, n))
        return None


def reversals(v, amp=0.05):
    """Number of speed direction changes of at least `amp` (peak-to-trough), i.e. accel/decel swings."""
    n, ext, d = 0, v[0] if len(v) else 0.0, 0
    for x in v:
        if d >= 0 and x < ext - amp:
            n, d, ext = n + 1, -1, x
        elif d <= 0 and x > ext + amp:
            n, d, ext = n + 1, 1, x
        elif (d >= 0 and x > ext) or (d <= 0 and x < ext):
            ext = x
    return n


def yaw_q(th):
    return math.sin(th / 2), math.cos(th / 2)


def path_len(poses):
    p = np.array([[q.pose.position.x, q.pose.position.y] for q in poses])
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0


def dist_to_poly(px, py, poly):
    if len(poly) < 2:
        return float('nan')
    a, b = poly[:-1], poly[1:]
    ab = b - a
    t = np.clip(((px - a[:, 0]) * ab[:, 0] + (py - a[:, 1]) * ab[:, 1]) / np.maximum((ab ** 2).sum(1), 1e-9), 0, 1)
    q = a + ab * t[:, None]
    return float(np.hypot(q[:, 0] - px, q[:, 1] - py).min())


class Bench:
    def __init__(self, node, walls):
        self.n, self.walls = node, walls
        self.nav = ActionClient(node, NavigateToPose, '/navigate_to_pose')
        self.cp = ActionClient(node, ComputePathToPose, '/compute_path_to_pose')
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, node)
        self.plan = np.zeros((0, 2))
        node.create_subscription(Path, '/plan', self.on_plan, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self.cmd = []
        node.create_subscription(TwistStamped, '/cmd_vel', lambda m: self.cmd.append((m.twist.linear.x, m.twist.angular.z)), 50)

    def on_plan(self, m):
        self.plan = np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses]) if m.poses else np.zeros((0, 2))

    def spin(self, s):
        end = time.time() + s
        while time.time() < end:
            rclpy.spin_once(self.n, timeout_sec=0.01)

    def wait(self, fut, timeout):
        end = time.time() + timeout
        while not fut.done() and time.time() < end:
            rclpy.spin_once(self.n, timeout_sec=0.02)
        return fut.result() if fut.done() else None

    @staticmethod
    def truth():
        try:
            d = json.load(open(TRUTH))
            return d['poses'], d.get('sim_t', 0.0)
        except (OSError, ValueError, KeyError):
            return None, None

    def tf_pose(self):
        try:
            t = self.tf.lookup_transform('map', 'base_footprint', rclpy.time.Time()).transform
        except Exception:  # noqa: BLE001
            return None
        q = t.rotation
        return t.translation.x, t.translation.y, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def goal_msg(self, g):
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x, p.pose.position.y = g[0], g[1]
        p.pose.orientation.z, p.pose.orientation.w = yaw_q(g[2])
        return p

    def compute(self, g):
        if not self.cp.wait_for_server(timeout_sec=10):
            return None
        req = ComputePathToPose.Goal()
        req.goal = self.goal_msg(g)
        req.use_start = False
        h = self.wait(self.cp.send_goal_async(req), 10)
        if not h or not h.accepted:
            return None
        t0 = time.time()
        r = self.wait(h.get_result_async(), 30)
        wall = time.time() - t0
        if not r or not r.result.path.poses:
            return {'ok': False, 'wall_s': wall}
        pt = r.result.planning_time
        return {'ok': True, 'wall_s': wall, 'plan_s': pt.sec + pt.nanosec * 1e-9, 'len_m': path_len(r.result.path.poses),
                'poly': np.array([[q.pose.position.x, q.pose.position.y] for q in r.result.path.poses])}

    def leg(self, name, g, timeout, dyn=None):
        rec = {'leg': name, 'goal': list(g)}
        cp = self.compute(g)
        rec['plan'] = {k: v for k, v in (cp or {}).items() if k != 'poly'}
        poly = cp['poly'] if cp and cp.get('ok') else None
        poses0, _ = self.truth()
        boxes0 = {k: v for k, v in (poses0 or {}).items() if k != 'robot'}
        if poses0 and 'robot' in poses0:
            sp = self.walls.shortest(poses0['robot'][:2], g[:2], boxes0.values())
            rec['shortest_m'] = round(sp, 3) if sp else None
        if not self.nav.wait_for_server(timeout_sec=10):
            rec['ok'], rec['why'] = False, 'no_nav_server'
            return rec
        h = self.wait(self.nav.send_goal_async(NavigateToPose.Goal(pose=self.goal_msg(g))), 10)
        if not h or not h.accepted:
            rec['ok'], rec['why'] = False, 'rejected'
            return rec
        res = h.get_result_async()
        self.cmd = []
        t0 = time.time()
        _, st0 = self.truth()
        last, drove, still, vmax, vs = None, 0.0, 0.0, 0.0, []
        wall_c, box_c, dyn_c, loc_e, dev = 9.0, 9.0, 9.0, [], []
        wall_hit_t, dyn_state = 0.0, None
        traj = []
        while not res.done():
            if time.time() - t0 > timeout:
                h.cancel_goal_async()
                self.spin(1.0)
                rec['ok'], rec['why'] = False, 'timeout'
                break
            self.spin(0.05)
            poses, st = self.truth()
            if not poses or 'robot' not in poses:
                continue
            x, y, th = poses['robot']
            if last is not None and st is not None and st > last[3]:
                dt = st - last[3]
                ds = math.hypot(x - last[0], y - last[1])
                drove += ds
                v = ds / dt
                vs.append(v)
                vmax = max(vmax, v)
                dth = abs(math.remainder(th - last[2], 2 * math.pi)) / dt
                if v < 0.01 and dth < 0.05:
                    still += dt
            if last is None or (st is not None and st > last[3]):
                last = (x, y, th, st if st is not None else 0.0)
            traj.append((round(time.time() - t0, 2), round(x, 3), round(y, 3), round(th, 3)))
            fp = footprint_pts(x, y, th)
            c = self.walls.clearance(fp)
            if c <= 0.005:
                wall_hit_t += 0.05
            if c < wall_c:
                rec['wall_worst'] = {'t': round(time.time() - t0, 2), 'robot': [round(x, 3), round(y, 3), round(th, 3)]}
            wall_c = min(wall_c, c)
            for k, (bx, by, bth) in poses.items():
                if k == 'robot':
                    continue
                bc = rect_clearance(fp, bx, by, bth, BOX_HALF)
                if inside_rect(np.array([[bx, by]]), x, y, th, *FOOT):
                    bc = 0.0
                box_c = min(box_c, bc)
            if dyn:
                dyn_state = self.drive_dyn(dyn, dyn_state, poly, (x, y), time.time() - t0, fp)
                if dyn_state and dyn_state.get('pos') is not None:
                    ox, oy = dyn_state['pos']
                    dc = float(np.hypot(fp[:, 0] - ox, fp[:, 1] - oy).min()) - DYN_R
                    if inside_rect(np.array([[ox, oy]]), x, y, th, *FOOT):
                        dc = 0.0
                    if dc < dyn_c:
                        rec['dyn_worst'] = {'t': round(time.time() - t0, 2), 'robot': [round(x, 3), round(y, 3), round(th, 3)],
                                            'obs': [round(ox, 3), round(oy, 3)]}
                    dyn_c = min(dyn_c, dc)
            tp = self.tf_pose()
            if tp:
                if vs and vs[-1] < 0.02:  # truth file and TF are not time-aligned: compare only when standing
                    loc_e.append(math.hypot(tp[0] - x, tp[1] - y))
                if len(self.plan) > 1:
                    dev.append(dist_to_poly(tp[0], tp[1], self.plan))
        if 'ok' not in rec:
            r = res.result()
            rec['ok'] = r.status == 4  # SUCCEEDED
            rec['why'] = '' if rec['ok'] else f'status {r.status} error {getattr(r.result, "error_code", "?")} {getattr(r.result, "error_msg", "")}'
        if dyn:
            try:
                os.remove(DYN)
            except OSError:
                pass
        poses1, _ = self.truth()
        pushed = {}
        for k, b0 in boxes0.items():
            b1 = (poses1 or {}).get(k)
            if b1:
                d = math.hypot(b1[0] - b0[0], b1[1] - b0[1])
                if d > 0.005 or abs(math.remainder(b1[2] - b0[2], 2 * math.pi)) > math.radians(1.0):
                    pushed[k] = round(d, 3)
        end = (poses1 or {}).get('robot')
        _, st1 = self.truth()
        tp = self.tf_pose()
        rec['tf_goal_err'] = [round(math.hypot(tp[0] - g[0], tp[1] - g[1]), 3),
                              round(math.degrees(abs(math.remainder(tp[2] - g[2], 2 * math.pi))), 1)] if tp else None
        sim_s = (st1 - st0) if st0 is not None and st1 is not None and st1 > st0 else None
        rec['sim_s'] = round(sim_s, 2) if sim_s else None
        rec['v_mean_sim'] = round(drove / sim_s, 3) if sim_s else None
        rec.update({
            'time_s': round(time.time() - t0, 2), 'drove_m': round(drove, 3),
            'v_mean': round(drove / max(time.time() - t0, 1e-3), 3), 'v_max': round(vmax, 3),
            'v_p90': round(float(np.percentile(vs, 90)), 3) if vs else 0.0,
            'still_s': round(still, 2), 'wall_clear_min': round(wall_c, 3), 'wall_contact_s': round(wall_hit_t, 2),
            'box_clear_min': round(box_c, 3), 'pushed': pushed,
            'dyn_clear_min': round(dyn_c, 3) if dyn else None,
            'loc_err_max': round(max(loc_e), 3) if loc_e else None,
            'plan_dev_p95': round(float(np.percentile(dev, 95)), 3) if dev else None,
            'goal_err': [round(math.hypot(end[0] - g[0], end[1] - g[1]), 3),
                         round(math.degrees(abs(math.remainder(end[2] - g[2], 2 * math.pi))), 1)] if end else None,
            'traj': traj[::4],
        })
        sp = rec.get('shortest_m')
        rec['plan_ratio'] = round(rec['plan']['len_m'] / sp, 2) if sp and rec['plan'].get('len_m') else None
        rec['drove_ratio'] = round(drove / sp, 2) if sp else None
        cv = np.array(self.cmd) if self.cmd else np.zeros((0, 2))
        mv = cv[(np.abs(cv[:, 0]) > 0.01) | (np.abs(cv[:, 1]) > 0.05)] if len(cv) else cv
        drive_s = len(mv) / 20.0  # /cmd_vel at controller_frequency 20 Hz
        rec['cmd_swings_10s'] = round(reversals(cv[:, 0]) / drive_s * 10, 1) if drive_s > 1 else None
        rec['cmd'] = [[round(float(v), 3), round(float(w), 3)] for v, w in cv[::2]]  # 10 Hz
        rec['crawl_share'] = round(float(np.mean((np.abs(mv[:, 0]) > 0.01) & (np.abs(mv[:, 0]) < 0.08))), 2) if len(mv) else None
        return rec

    def drive_dyn(self, dyn, st, poly, here, t, fp):
        """popup: once the robot is 0.4 m into the leg, park the obstacle on the path 0.9 m ahead.
        cross: same spot, but the obstacle sweeps across the path ±0.6 m at 0.25 m/s. Like a person it does not walk
        into the robot: a move that would bring it within 0.10 m of the body is held (contacts are then the robot's doing)."""
        if st is None:
            st = {'start': here, 'pos': None}
        if poly is None:
            return st
        if st.get('anchor') is None:
            if math.hypot(here[0] - st['start'][0], here[1] - st['start'][1]) < 0.4:
                return st
            d = np.hypot(poly[:, 0] - here[0], poly[:, 1] - here[1])
            k = int(d.argmin())
            seg = np.hypot(*np.diff(poly[k:], axis=0).T).cumsum() if len(poly) - k > 1 else np.array([0.0])
            j = k + 1 + int(np.searchsorted(seg, dyn['ahead']))
            j = min(j, len(poly) - 2)
            if j <= k:
                return st
            a, b = poly[j], poly[min(j + 1, len(poly) - 1)]
            tx, ty = b - a
            n = math.hypot(tx, ty) or 1.0
            st['anchor'] = (float(a[0]), float(a[1]), -ty / n, tx / n, t)
        ax, ay, nx, ny, t0 = st['anchor']
        if dyn['mode'] == 'popup':
            pos = (ax, ay)
        else:
            s = 0.6 * math.sin(0.25 / 0.6 * (t - t0))
            pos = (ax + nx * s, ay + ny * s)
            if st.get('pos') is not None and float(np.hypot(fp[:, 0] - pos[0], fp[:, 1] - pos[1]).min()) - DYN_R < 0.10:
                pos = st['pos']
        if st.get('pos') is None or math.hypot(pos[0] - st['pos'][0], pos[1] - st['pos'][1]) > 0.01:
            with open(DYN + '.tmp', 'w') as f:
                json.dump({'x': pos[0], 'y': pos[1]}, f)
            os.replace(DYN + '.tmp', DYN)
            st['pos'] = pos
        return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--legs', default=','.join(ROUTE), help='goal names in order (' + ' '.join(GOALS) + ')')
    ap.add_argument('--dyn', action='append', default=[], help='popup:<leg index> or cross:<leg index> (repeatable)')
    ap.add_argument('--ahead', type=float, default=0.9, help='[m] obstacle distance ahead on the path')
    ap.add_argument('--timeout', type=float, default=120.0)
    ap.add_argument('--walls', default=os.path.expanduser('~/.ros/nav_bench/walls'),
                    help='make_map.py --res 0.01 output (built on first use)')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    if not os.path.exists(os.path.join(a.walls, 'map.pgm')):
        subprocess.run([os.path.expanduser('~/pa-course/.venv-isaacsim/bin/python'), os.path.join(HERE, 'make_map.py'),
                        '--res', '0.01', '--out', a.walls], check=True)
    try:
        os.remove(DYN)
    except OSError:
        pass
    dyn = {}
    for d in a.dyn:
        mode, i = d.split(':')
        dyn[int(i)] = {'mode': mode, 'ahead': a.ahead}
    rclpy.init()
    node = rclpy.create_node('nav_bench')
    b = Bench(node, Walls(a.walls))
    b.spin(2.0)
    out = {'tag': a.tag, 'started': time.strftime('%Y%m%d_%H%M%S'), 'legs': []}
    for i, name in enumerate(a.legs.split(',')):
        rec = b.leg(name, GOALS[name], a.timeout, dyn.get(i))
        if i in dyn:
            rec['dyn'] = dyn[i]['mode']
        out['legs'].append(rec)
        print(f"{name:>3} {'ok ' if rec['ok'] else 'NG '} {rec.get('time_s', 0):6.1f}s sim {rec.get('sim_s') or 0:5.1f}s plan {rec['plan'].get('wall_s', float('nan')) * 1000:5.0f}ms "
              f"len {rec['plan'].get('len_m', 0):4.2f} drove {rec.get('drove_m', 0):4.2f} v {rec.get('v_mean_sim') or 0:.3f}/{rec.get('v_p90', 0):.3f} "
              f"still {rec.get('still_s', 0):4.1f} wall {rec.get('wall_clear_min', 9):+.3f} ({rec.get('wall_contact_s', 0):.1f}s) "
              f"box {rec.get('box_clear_min', 9):+.3f} {rec.get('pushed') or ''} "
              f"{'dyn %+.3f ' % rec['dyn_clear_min'] if rec.get('dyn_clear_min') is not None else ''}"
              f"loc {rec.get('loc_err_max')} dev95 {rec.get('plan_dev_p95')} short {rec.get('shortest_m')} "
              f"ratio {rec.get('plan_ratio')}/{rec.get('drove_ratio')} swings {rec.get('cmd_swings_10s')} crawl {rec.get('crawl_share')} {rec.get('why', '')}", flush=True)
    legs = out['legs']
    out['summary'] = {
        'ok': sum(r['ok'] for r in legs), 'n': len(legs), 'time_s': round(sum(r.get('time_s', 0) for r in legs), 1),
        'sim_s': round(sum(r.get('sim_s') or 0 for r in legs), 1),
        'drove_m': round(sum(r.get('drove_m', 0) for r in legs), 2),
        'wall_contacts': sum(r.get('wall_clear_min', 9) <= 0.005 for r in legs),
        'box_pushed': sum(bool(r.get('pushed')) for r in legs),
        'dyn_contacts': sum(r.get('dyn_clear_min') is not None and r['dyn_clear_min'] <= 0.0 for r in legs),
        'shortest_m': round(sum(r.get('shortest_m') or 0 for r in legs), 2),
        'drove_ratio': round(sum(r.get('drove_m', 0) for r in legs) / max(sum(r.get('shortest_m') or 0 for r in legs), 1e-3), 2),
        'swings_10s_med': float(np.median([r['cmd_swings_10s'] for r in legs if r.get('cmd_swings_10s') is not None] or [0])),
        'crawl_med': float(np.median([r['crawl_share'] for r in legs if r.get('crawl_share') is not None] or [0])),
    }
    print('SUMMARY', json.dumps(out['summary']), flush=True)
    d = os.path.expanduser('~/.ros/nav_bench')
    os.makedirs(d, exist_ok=True)
    fn = os.path.join(d, f"{out['started']}_{a.tag or 'run'}.json")
    json.dump(out, open(fn, 'w'))
    print(fn)
    node.destroy_node()
    rclpy.shutdown()
    return 0 if out['summary']['ok'] == len(legs) else 1


if __name__ == '__main__':
    assert inside_rect(np.array([[0.0, 0.0]]), 0, 0, 0, *FOOT) and not inside_rect(np.array([[0.2, 0.0]]), 0, 0, 0, *FOOT)
    assert abs(rect_clearance(np.array([[0.3, 0.0]]), 0, 0, 0, 0.1) - 0.2) < 1e-9
    assert reversals([0, 0.2, 0.1, 0.2, 0.0]) == 4 and reversals([0, 0.1, 0.2, 0.26, 0.26]) == 1
    sys.exit(main())
