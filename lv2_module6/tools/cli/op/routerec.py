"""가르친 경로 녹화 (teach & repeat, 11-15) — 수동 조작으로 모는 동안 로봇 자리(map→base_footprint)를 5 cm마다 남겨
arena 좌표로 ~/.ros/routes/<이름>.txt에 저장. AUTO에서 skill_navigate가 시작이 로봇 근처 · 끝이 목표 근처인 경로를
경유점으로 씀 (거꾸로도 — 복귀). 끝 = 다음 navigate 목표 자리(상자 앞 다가설 자리)에서 상자를 보고 멈춘 뒤 Enter."""
from __future__ import annotations

import math
import select
import sys
import time
from pathlib import Path

ROUTES = Path.home() / '.ros' / 'routes'
START = (0.734, 0.950, math.pi)   # plan.yaml · skills.yaml start_pose_arena (map 원점 = 켠 자리)
STEP_M = 0.05


def to_arena(mx: float, my: float, start=START) -> tuple[float, float]:
    """map → arena (skill_navigate의 arena → map 역변환: map = R(−θ0)·(arena − p0))."""
    c, s = math.cos(start[2]), math.sin(start[2])
    return start[0] + c * mx - s * my, start[1] + s * mx + c * my


def record(node, tf_buf, name: str, start=START) -> int:
    import rclpy
    from rclpy.time import Time
    ROUTES.mkdir(parents=True, exist_ok=True)
    out = ROUTES / f'{name}.txt'
    pts: list[tuple[float, float]] = []
    print(f'경로 녹화: {name} — 수동으로 몰고, 목표 자리에서 멈춘 뒤 Enter (취소 Ctrl+C)')
    while True:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            tr = tf_buf.lookup_transform('map', 'base_footprint', Time()).transform.translation
            p = to_arena(tr.x, tr.y, start)
            if not pts or math.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) >= STEP_M:
                pts.append(p)
                print(f'\r  점 {len(pts)}개 · 지금 arena ({p[0]:.2f}, {p[1]:.2f})   ', end='', flush=True)
        except Exception:
            pass
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            break
    print()
    if len(pts) < 2:
        print('점이 2개 미만 — 저장 안 함 (map TF가 없나?)')
        return 1
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    out.write_text(f'# {name} · {time.strftime("%Y-%m-%d %H:%M")} · {len(pts)}점 · {length:.2f} m · arena x y\n'
                   + ''.join(f'{x:.3f} {y:.3f}\n' for x, y in pts))
    print(f'저장: {out} ({len(pts)}점 · {length:.2f} m) — 시작 ({pts[0][0]:.2f}, {pts[0][1]:.2f}) → 끝 ({pts[-1][0]:.2f}, {pts[-1][1]:.2f})')
    return 0


if __name__ == '__main__':   # 자체 확인: 출발 자리(map 원점) = arena 출발 · map +x(로봇 앞) = arena −x
    assert to_arena(0, 0) == START[:2]
    ax, ay = to_arena(1.0, 0.0)
    assert abs(ax - (START[0] - 1.0)) < 1e-9 and abs(ay - START[1]) < 1e-9
    ax, ay = to_arena(-2.202, -3.854)   # 11-15 실기 S3 목표(map) = 상자 3 (2.936, 5.354) −y 앞 0.55 m
    assert abs(ax - 2.936) < 1e-3 and abs(ay - 4.804) < 1e-3
    print('ok routerec')
