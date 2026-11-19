"""maze_map_1711.usda의 경기장 벽 · 기둥(/World/maze)으로 Nav2 지도(map.pgm + map.yaml)를 만든다.

  ~/pa-course/.venv-isaacsim/bin/python isaac/mission/make_map.py            # → isaac/mission/map/maze_map_1711/
  ~/pa-course/.venv-isaacsim/bin/python isaac/mission/make_map.py --res 0.02 --z 0.02 0.45
  ~/pa-course/.venv-isaacsim/bin/python isaac/mission/make_map.py --boxes  # 시작 박스 1~6 · 장소 상자 L1~L3도 (ros2_ws/map/maze_map_1711 · 1911)
  ~/pa-course/.venv-isaacsim/bin/python isaac/mission/make_map.py --walls   # plan.yaml arena_walls 선분 (경기장 좌표) 출력

pxr가 필요해서 Isaac venv python으로 돌린다 (Isaac은 안 띄움 · USD 파일은 읽기만).
지도 좌표 = map 프레임 = 로봇을 켠 자리(beebot prim 자세, base_footprint) — 앞 = +x, cuVSLAM·nvblox의 map과 같은 원점.
장애물 = 아래 ROOTS의 render(default) 메시 삼각형 · 원기둥 중 높이가 --z 띠와 겹치는 것을 바닥에 투영.
  시작 박스 · 장소 받침대(상자)와 퍽은 넣지 않는다 — 미션이 다가가 붙는 대상이라 정적 지도에 있으면 목표 자리가 막힘. 띠 기본 0.02~0.47 m = 바닥 위 ~ 로봇 높이(beebot 0.469 m).
"""
import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from pxr import Gf, Usd, UsdGeom

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, '..', 'usda', 'maze_map_1711.usda')
ROOTS = ['/World/maze']
BOXES = ['/World/start_boxes', '/World/shelf_L1', '/World/shelf_L2', '/World/shelf_L3']  # --boxes
ROBOT = '/World/beebot'
MARGIN = 0.3  # [m] 장애물 둘레 바깥 여백


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--res', type=float, default=0.05, help='[m/칸] — Nav2 costmap 해상도(0.05)와 같게')
    ap.add_argument('--z', type=float, nargs=2, default=[0.02, 0.47], help='[m] 장애물로 볼 높이 띠')
    ap.add_argument('--out', default=os.path.join(HERE, 'map', 'maze_map_1711'))
    ap.add_argument('--boxes', action='store_true', help='시작 박스 1~6 · 장소 상자 L1~L3도 장애물로 넣음')
    ap.add_argument('--walls', action='store_true', help='지도 대신 벽판 선분을 plan.yaml arena_walls 형식으로 출력')
    ap.add_argument('--start-arena', type=float, nargs=3, default=[0.734, 0.950, math.pi],
                    help='plan.yaml start_pose_arena (x y yaw) — map → 경기장 좌표 변환')
    a = ap.parse_args()

    st = Usd.Stage.Open(SCENE)
    to_map = UsdGeom.Xformable(st.GetPrimAtPath(ROBOT)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    z0, z1 = a.z
    polys = []  # map 좌표 [(x, y), …]

    def world_to_map(p):
        q = to_map.Transform(Gf.Vec3d(*p))
        return q[0], q[1], q[2]

    if a.walls:
        sx, sy, sth = a.start_arena
        c, s_ = math.cos(sth), math.sin(sth)
        segs = wall_segments(st, world_to_map, lambda x, y: (sx + c * x - s_ * y, sy + s_ * x + c * y))
        vals = [f'{v:.3f}' for p0, p1 in segs for v in (*p0, *p1)]
        print('    arena_walls:\n      [\n' + ',\n'.join('        ' + v for v in vals) + '\n      ]')
        print(f'# {len(segs)} segments from {os.path.relpath(SCENE, HERE)} */wall/geo', file=sys.stderr)
        return

    for root in ROOTS + (BOXES if a.boxes else []):
        for p in Usd.PrimRange(st.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
            if not p.IsA(UsdGeom.Gprim) or UsdGeom.Imageable(p).ComputePurpose() != UsdGeom.Tokens.default_:
                continue
            xf = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            if p.IsA(UsdGeom.Mesh):
                m = UsdGeom.Mesh(p)
                pts = [world_to_map(xf.Transform(v)) for v in m.GetPointsAttr().Get()]
                counts, idx, k = m.GetFaceVertexCountsAttr().Get(), m.GetFaceVertexIndicesAttr().Get(), 0
                for c in counts:  # 면마다 — 띠와 높이가 겹치면 바닥에 투영 (세로 면은 선이 됨)
                    f = [pts[i] for i in idx[k:k + c]]
                    k += c
                    if min(v[2] for v in f) <= z1 and max(v[2] for v in f) >= z0:
                        polys.append([(v[0], v[1]) for v in f])
            else:  # Cylinder 등 — 국소 bbox 8꼭짓점을 바닥에 투영 (기둥 · 볼트)
                box = UsdGeom.Boundable(p).ComputeLocalBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
                r = box.GetRange()
                if r.IsEmpty():
                    continue
                cs = [world_to_map(box.GetMatrix().Transform(r.GetCorner(i))) for i in range(8)]
                if min(v[2] for v in cs) <= z1 and max(v[2] for v in cs) >= z0:
                    polys.append(convex_hull([(v[0], v[1]) for v in cs]))

    xs = [x for poly in polys for x, _ in poly]
    ys = [y for poly in polys for _, y in poly]
    ox, oy = min(xs) - MARGIN, min(ys) - MARGIN
    w, h = math.ceil((max(xs) + MARGIN - ox) / a.res), math.ceil((max(ys) + MARGIN - oy) / a.res)
    img = Image.new('L', (w, h), 254)  # 254 = 빈 곳 · 0 = 장애물 (map_server trinary)
    d = ImageDraw.Draw(img)
    for poly in polys:  # 영상 행 0 = 지도 맨 위(y 최대)
        px = [((x - ox) / a.res, h - 1 - (y - oy) / a.res) for x, y in poly]
        d.polygon(px, fill=0, outline=0) if len(px) > 2 else d.line(px, fill=0)
    os.makedirs(a.out, exist_ok=True)
    img.save(os.path.join(a.out, 'map.pgm'))
    with open(os.path.join(a.out, 'map.yaml'), 'w') as f:
        f.write(f'# isaac/mission/make_map.py — {os.path.relpath(SCENE, HERE)} 벽 · 기둥{' · 시작 박스 1~6 · 장소 상자 L1~L3' if a.boxes else ''} · map 원점 = 로봇을 켠 자리 · z {z0}~{z1} m\n'
                f'image: map.pgm\nmode: trinary\nresolution: {a.res}\norigin: [{ox:.3f}, {oy:.3f}, 0.0]\n'
                f'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
    occ = int((np.array(img) == 0).sum())
    print(f'{a.out}/map.pgm  {w}x{h} @ {a.res} m · origin ({ox:.3f}, {oy:.3f}) · 면 {len(polys)}개 · 장애물 칸 {occ}')


def wall_segments(st, world_to_map, to_arena, min_len=0.1, pair=0.05):
    """벽판(*/wall/geo) 세로 면 → 경기장 좌표 선분. 같은 판의 두 면(두께 ≤ pair)은 경기장 가운데 쪽 면만 —
    바깥 틀은 안쪽 면, 얇은 안쪽 벽(4 mm)은 사실상 판 가운데. 한 직선 위에서 겹치거나 맞닿은 선분은 하나로."""
    faces = []
    for p in Usd.PrimRange(st.GetPrimAtPath(ROOTS[0]), Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh) or not str(p.GetPath()).endswith('/wall/geo'):
            continue
        m = UsdGeom.Mesh(p)
        xf = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pts = [world_to_map(xf.Transform(v)) for v in m.GetPointsAttr().Get()]
        idx, k = m.GetFaceVertexIndicesAttr().Get(), 0
        for n in m.GetFaceVertexCountsAttr().Get():
            f = [pts[i] for i in idx[k:k + n]]
            k += n
            if max(v[2] for v in f) - min(v[2] for v in f) < 0.1:
                continue  # 윗면 · 밑면
            xy = [to_arena(v[0], v[1]) for v in f]
            a, b = max(((u, w) for u in xy for w in xy), key=lambda t: math.dist(*t))
            if math.dist(a, b) >= min_len:
                faces.append((a, b))
    cx = sum(x for f in faces for x, _ in f) / (2 * len(faces))
    cy = sum(y for f in faces for _, y in f) / (2 * len(faces))

    def line(f):  # 단위 방향 · 법선 거리 · 구간 [t0, t1]
        (ax, ay), (bx, by) = f
        L = math.dist(f[0], f[1])
        ux, uy = (bx - ax) / L, (by - ay) / L
        if ux < -1e-6 or (abs(ux) <= 1e-6 and uy < 0):
            ux, uy = -ux, -uy
        d = ax * -uy + ay * ux
        t0, t1 = sorted((ax * ux + ay * uy, bx * ux + by * uy))
        return ux, uy, d, t0, t1

    L = [line(f) for f in faces]

    def same_dir(p, q):
        return abs(p[0] * q[1] - p[1] * q[0]) < 0.02

    def mid_dist(q):  # 선분 가운데에서 경기장 가운데까지
        t = (q[3] + q[4]) / 2
        return math.hypot(q[0] * t - q[1] * q[2] - cx, q[1] * t + q[0] * q[2] - cy)
    keep = [q for q in L if not any(  # 겹치는 평행 면 중 가운데에서 먼 쪽은 버림
        same_dir(q, o) and 1e-4 < abs(q[2] - o[2]) <= pair and min(q[4], o[4]) - max(q[3], o[3]) > 0.05 and mid_dist(o) < mid_dist(q)
        for o in L)]
    merged = []
    for q in sorted(keep, key=lambda q: (round(q[0], 2), round(q[1], 2), round(q[2], 2), q[3])):
        m = merged[-1] if merged else None
        if m and same_dir(q, m) and abs(q[2] - m[2]) < 0.01 and q[3] <= m[4] + 0.02:
            merged[-1] = (m[0], m[1], (m[2] + q[2]) / 2, m[3], max(m[4], q[4]))
        else:
            merged.append(q)
    return [((ux * t0 - uy * d, uy * t0 + ux * d), (ux * t1 - uy * d, uy * t1 + ux * d)) for ux, uy, d, t0, t1 in merged]


def convex_hull(pts):
    pts = sorted(set(pts))
    if len(pts) < 3:
        return pts

    def half(seq):
        out = []
        for p in seq:
            while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (p[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0]) <= 0:
                out.pop()
            out.append(p)
        return out[:-1]
    return half(pts) + half(reversed(pts))


if __name__ == '__main__':
    assert len(convex_hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])) == 4  # 안쪽 점은 빠짐
    main()
