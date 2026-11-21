#!/usr/bin/env python3
"""관측만 시험 — 시작 바코드 + 신호등 (SURVEY 직전까지). 실제 인지로 읽음 (run_mission처럼 가짜 값을 안 넣음).

  python3 isaac/mission/run_observe.py                        # 실기 · 신호등 고정 정답 Q1(R4 Y2 G6)과 비교
  python3 isaac/mission/run_observe.py --expect R5Y3G4 --code P2-01
  python3 isaac/mission/run_observe.py --only signal          # 신호만 (start:READ — 바코드 안 봄)
  python3 isaac/mission/run_observe.py --sim                  # Isaac
  python3 isaac/mission/run_observe.py --clear-basket         # 적재함 장부가 남아 시작이 거부될 때 먼저 비움

mission_manager에 start:BARCODE를 보냄 → BARCODE(바코드 · 신호등을 보는 자리 · 손목 엿보기) → READ(신호 해독) → DONE.
SURVEY로는 안 넘어감. 그동안 /mission/qr(바코드 원문) · /mission/signal_status(사이클마다 읽은 신호)를 그대로 찍고,
끝나면 /mission/state의 바코드 · R/Y/G 횟수를 --expect · --code와 비교해 맞으면 0, 틀리거나 실패면 1로 나감.
로봇이 바코드를 찾으며 조금 움직일 수 있음 — 옆에서 지켜볼 것. Ctrl+C = /cmd/mission abort. 회차 기록: ~/.ros/mission/<시각>/signal.json
"""
import argparse
import os
import re
import sys

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from perception.msg import MissionSignal
from plan.msg import MissionState

IDLE = ('IDLE', 'DONE', 'ERROR')   # start를 받는 단계
Q1 = 'R4Y2G6'                      # 신호등 타워 sign_tower_repeat.ino 고정 정답 (발제 공개 연습 Q1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--expect', default=Q1, help='맞아야 할 신호 횟수 (예: R4Y2G6) · 빈 값이면 비교 안 함')
    ap.add_argument('--code', default='', help='맞아야 할 바코드 (예: P2-01) · 빈 값이면 비교 안 함')
    ap.add_argument('--only', choices=('all', 'signal'), default='all', help='signal = 신호만 (start:READ)')
    ap.add_argument('--sim', action='store_true', help='Isaac (use_sim_time · fastdds_local_only)')
    ap.add_argument('--clear-basket', action='store_true', help='시작 전 /coordinator/basket_clear (run_mission과 같음)')
    ap.add_argument('--timeout', type=float, default=240.0, help='[s] 이 안에 안 끝나면 abort')
    a = ap.parse_args()
    m = re.fullmatch(r'R(\d)Y(\d)G(\d)', a.expect.upper()) if a.expect else None
    if a.expect and not m:
        sys.exit(f'--expect {a.expect}: R<n>Y<n>G<n> 형식')
    want = tuple(int(x) for x in m.groups()) if m else None
    start = 'start:READ' if a.only == 'signal' else 'start:BARCODE'

    ws = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ros2_ws')
    os.environ.setdefault('FASTDDS_DEFAULT_PROFILES_FILE', os.path.join(ws, 'fastdds_local_only.xml' if a.sim else 'fastdds.xml'))
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'SUBNET'
    rclpy.init()
    n = rclpy.create_node('run_observe', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=a.sim)])
    log = n.get_logger()
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    cmd = n.create_publisher(String, '/cmd/mission', 10)
    if a.clear_basket:
        c = n.create_client(Trigger, '/coordinator/basket_clear')
        if not c.wait_for_service(timeout_sec=20.0):
            sys.exit('/coordinator/basket_clear 안 보임 — coordinator가 떠 있는지 확인')
        f = c.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(n, f, timeout_sec=10.0)
        log.info(f'적재함 장부: {f.result().message if f.result() else "응답 없음"}')
    st = {'phase': None, 'last': None, 'last0': None, 'started': False, 'running': False, 'done': None, 'msg': None, 'qr': None, 't': 0}

    def on_state(msg):
        st['phase'], st['msg'] = msg.phase, msg
        line = (msg.phase, msg.note, msg.error, msg.barcode, msg.count_r, msg.count_y, msg.count_g)
        if line != st['last']:
            st['last'] = line
            log.info(f'[{msg.phase}] {msg.note}' + (f' · 오류 {msg.error}' if msg.error else ''))
        if st['started'] and msg.phase not in IDLE:
            st['running'] = True
        if st['running'] and msg.phase in ('DONE', 'ERROR'):
            st['done'] = msg.phase
        if st['started'] and '시작 거부' in msg.note:
            st['done'] = 'ERROR'

    def on_qr(msg):
        if msg.data != st['qr']:
            st['qr'] = msg.data
            log.info(f'바코드 인지: {msg.data!r}')

    def on_signal(msg):
        if st['started']:
            log.info(f'신호 사이클: {msg.raw or "-"} · R{msg.red} Y{msg.yellow} G{msg.green} · ok={msg.ok}'
                     + (f' · {msg.error}' if msg.error else '') + f' · 연속 {msg.cycles}')

    n.create_subscription(MissionState, '/mission/state', on_state, latched)
    n.create_subscription(String, '/mission/qr', on_qr, latched)
    n.create_subscription(MissionSignal, '/mission/signal_status', on_signal, 10)

    def tick():
        st['t'] += 1
        if not st['started']:
            if st['phase'] in IDLE and cmd.get_subscription_count() > 0:
                cmd.publish(String(data=start))
                st['started'], st['last0'] = True, st['last']
                log.info(f'{start} 보냄 · 기대 {a.expect or "-"} · 바코드 {a.code or "-"}')
            elif st['phase'] not in (None, *IDLE):
                log.error(f'미션이 이미 {st["phase"]} — 끝나거나 abort한 뒤에 다시')
                st['done'] = 'ERROR'
            elif st['t'] % 5 == 0:
                log.warn('/mission/state 안 옴 — mission_manager · ROS_DOMAIN_ID 확인' if st['phase'] is None
                         else '/cmd/mission 구독자 0 — mission_manager 확인')
            return
        if not st['running'] and st['last'] == st['last0']:   # start가 안 닿음 — 다시
            cmd.publish(String(data=start))
        if st['t'] > a.timeout:
            log.error(f'{a.timeout:.0f} s 안에 안 끝남 — abort')
            cmd.publish(String(data='abort'))
            st['done'] = 'TIMEOUT'

    n.create_timer(1.0, tick, clock=rclpy.clock.Clock())   # 벽시계
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
    got = (s.count_r, s.count_y, s.count_g) if s else (0, 0, 0)
    ok_sig = want is None or got == want
    ok_code = not a.code or a.only == 'signal' or (s and s.barcode == a.code)
    print('\n=== 관측 결과 ===')
    print(f'단계 끝: {st["done"]}' + (f' · 오류 {s.error}' if s and s.error else ''))
    if a.only != 'signal':
        print(f'바코드: {s.barcode if s else ""!r}' + (f'  (기대 {a.code} → {"맞음" if ok_code else "틀림"})' if a.code else ''))
    print(f'신호: R{got[0]} Y{got[1]} G{got[2]}' + (f'  (기대 {a.expect.upper()} → {"맞음" if ok_sig else "틀림"})' if want else ''))
    print(f'L1~L3 그룹: {s.order if s and s.order else "-"}')
    sys.exit(0 if st['done'] == 'DONE' and ok_sig and ok_code else 1)


if __name__ == '__main__':
    main()
