#!/usr/bin/env python3
"""실기 전체 미션 한 판 — 관측(바코드 + 신호) → SURVEY → BUILD → VERIFY → RETURN, 실제 인지로.

  python3 isaac/mission/run_full.py                  # 적재함 장부 비우고 시작 · 끝나면 요약
  python3 isaac/mission/run_full.py --keep-basket    # 장부 그대로 (이어 하기)
  python3 isaac/mission/run_full.py --expect R4Y2G6 --code P2-01   # 관측 결과도 비교

run_mission.py(시뮬 — 바코드·신호를 가짜로 넣음)와 달리 아무것도 대신 넣지 않음. mission_manager에 start만 보내고 지켜봄.
진행 중 단계 · 하는 일 · 결과 줄을 찍고, 끝나면 바코드 · 신호 · 놓은 퍽 · 실패 목록을 정리. Ctrl+C = abort.
회차 기록: Jetson ~/.ros/mission/<시각>/ (phases · steps · allocation · build · verify · return · photos/)
"""
import argparse
import os
import re
import sys
import time

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from perception.msg import MissionSignal
from plan.msg import MissionState

IDLE = ('IDLE', 'DONE', 'ERROR')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--keep-basket', action='store_true', help='시작 전 적재함 장부를 비우지 않음')
    ap.add_argument('--expect', default='', help='맞아야 할 신호 (예: R4Y2G6) — 비교만')
    ap.add_argument('--code', default='', help='맞아야 할 바코드 (예: P2-01) — 비교만')
    ap.add_argument('--sim', action='store_true', help='Isaac (use_sim_time · fastdds_local_only)')
    a = ap.parse_args()
    m = re.fullmatch(r'R(\d)Y(\d)G(\d)', a.expect.upper()) if a.expect else None
    if a.expect and not m:
        sys.exit(f'--expect {a.expect}: R<n>Y<n>G<n> 형식')
    want = tuple(int(x) for x in m.groups()) if m else None

    ws = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ros2_ws')
    os.environ.setdefault('FASTDDS_DEFAULT_PROFILES_FILE', os.path.join(ws, 'fastdds_local_only.xml' if a.sim else 'fastdds.xml'))
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'SUBNET'
    rclpy.init()
    n = rclpy.create_node('run_full', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=a.sim)])
    log = n.get_logger()
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    cmd = n.create_publisher(String, '/cmd/mission', 10)
    st = {'phase': None, 'last': None, 'last0': None, 'started': False, 'running': False, 'done': None,
          'msg': None, 'phases': [], 't0': None, 'decoded': None, 'nres': 0}

    if not a.keep_basket:
        c = n.create_client(Trigger, '/coordinator/basket_clear')
        if not c.wait_for_service(timeout_sec=20.0):
            sys.exit('/coordinator/basket_clear 안 보임 — start.sh(coordinator)가 떠 있는지 확인')
        f = c.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(n, f, timeout_sec=10.0)
        log.info(f'적재함 장부 비움: {f.result().message if f.result() else "응답 없음"}')

    def on_state(msg):
        st['phase'], st['msg'] = msg.phase, msg
        if st['running'] and (not st['phases'] or st['phases'][-1][0] != msg.phase):
            st['phases'].append((msg.phase, time.monotonic()))
        line = (msg.phase, msg.note, msg.error, msg.placed)
        if line != st['last']:
            st['last'] = line
            log.info(f'[{msg.phase}] {msg.note}' + (f' · 놓음 {msg.placed}/{msg.total}' if msg.total else '')
                     + (f' · 오류 {msg.error}' if msg.error else ''))
        for r in list(msg.results)[st['nres']:]:
            log.info(f'  결과: {r}')
        st['nres'] = len(msg.results)
        if st['started'] and msg.phase not in IDLE:
            if not st['running']:
                st['running'], st['t0'] = True, time.monotonic()
                st['phases'].append((msg.phase, st['t0']))
        if st['running'] and msg.phase in ('DONE', 'ERROR'):
            st['done'] = msg.phase
        if st['started'] and '시작 거부' in msg.note:
            st['done'] = 'ERROR'

    def on_signal(msg):
        if msg.ok:
            st['decoded'] = (msg.red, msg.yellow, msg.green, msg.raw)

    n.create_subscription(MissionState, '/mission/state', on_state, latched)
    n.create_subscription(MissionSignal, '/mission/signal_status', on_signal, 10)

    def tick():
        if st['started']:
            if not st['running'] and st['last'] == st['last0']:
                cmd.publish(String(data='start'))   # start가 안 닿음 (DDS 연결 직후 유실 등) — 다시
            return
        if st['phase'] in IDLE and cmd.get_subscription_count() > 0:
            cmd.publish(String(data='start'))
            st['started'], st['last0'] = True, st['last']
            log.info('start 보냄 — 전체 미션')
        elif st['phase'] not in (None, *IDLE):
            log.error(f'미션이 이미 {st["phase"]} — 끝나거나 abort한 뒤에 다시')
            st['done'] = 'ERROR'

    n.create_timer(1.0, tick, clock=rclpy.clock.Clock())
    log.info('/mission/state 기다리는 중 (mission_manager)')
    try:
        while rclpy.ok() and st['done'] is None:
            rclpy.spin_once(n, timeout_sec=0.2)
    except KeyboardInterrupt:
        if st['started']:
            cmd.publish(String(data='abort'))
            rclpy.spin_once(n, timeout_sec=0.5)
            print('abort 보냄')
        sys.exit(1)

    s = st['msg']
    end = time.monotonic()
    got = (s.count_r, s.count_y, s.count_g) if s and (s.count_r or s.count_y or s.count_g) else (st['decoded'][:3] if st['decoded'] else None)
    print('\n=== 전체 미션 결과 ===')
    print(f'결과: {"성공" if st["done"] == "DONE" else "실패"} ({st["done"]})' + (f' · 이유 {s.error}' if s and s.error else '')
          + (f' · {end - st["t0"]:.0f} s' if st['t0'] else ''))
    code = s.barcode if s else ''
    print(f'바코드: {code or "(못 읽음)"}' + (f'  · 기대 {a.code} → {"맞음" if code == a.code else "틀림" if code else "못 읽음"}' if a.code else ''))
    print('신호: ' + (f'R{got[0]} Y{got[1]} G{got[2]}' if got else '(확정 못 함)')
          + (f'  · 기대 {a.expect.upper()} → {"맞음" if got == want else "틀림" if got else "못 읽음"}' if want else ''))
    print(f'L1~L3 그룹: {s.order if s and s.order else "-"} · 놓음 {s.placed if s else 0}/{s.total if s else 0}')
    if st['phases']:
        segs = [f'{p} {((st["phases"][i + 1][1] if i + 1 < len(st["phases"]) else end) - t):.0f}s' for i, (p, t) in enumerate(st['phases'])]
        print('단계 시간: ' + ' → '.join(segs))
    fails = [r for r in (s.results if s else []) if not r.endswith(': ok') and 'ok' != r.split(': ')[-1]]
    print(f'실패·기록 {len(fails)}건:')
    for r in fails:
        print(f'  - {r}')
    print('미션 기록: Jetson ~/.ros/mission/ 가장 최근 폴더')
    sys.exit(0 if st['done'] == 'DONE' else 1)


if __name__ == '__main__':
    main()
