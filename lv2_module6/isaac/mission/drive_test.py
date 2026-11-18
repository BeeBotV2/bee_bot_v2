#!/usr/bin/env python3
"""주행만 시험 — 신호등 · 바코드 · inspect · pick · load · place 없이 상자마다 다가가기(Nav2 → 도킹 → exit)만 하고 출발 구역으로 복귀.

  python3 isaac/mission/drive_test.py                 # 기본 = plan.yaml 순서: survey_order(S) → build_location_order(L) → 그 거꾸로
  python3 isaac/mission/drive_test.py S4,S6,L2        # 고른 상자만 (S1~S6 시작 박스 · L1~L3 장소)
  python3 isaac/mission/drive_test.py --clear-basket  # 적재함 장부가 남아 시작이 거부될 때 먼저 비움
  python3 isaac/mission/drive_test.py --real S4       # 실기 (Jetson start.sh로 띄운 스택) — 시뮬 시계(/clock) 안 씀 · 프로파일 기본 fastdds.xml

mission_manager에 start:GOTO:<상자들>을 보내고 /mission/state를 지켜본다. 끝나면 상자마다 ok/실패 · 걸린 시간 표.
Isaac은 먼저 띄워 둔다 (isaac/standalone/isaac_mission.sh). 환경은 run_mission.py와 같다 (ROS_DOMAIN_ID · Fast DDS 프로파일).
Ctrl+C = /cmd/mission abort. 전부 ok면 0 · 아니면 1로 나간다. 회차 기록: ~/.ros/mission/<시각>/
"""
import argparse
import os
import sys
import time

import rclpy
import yaml
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from plan.msg import MissionState

HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.join(HERE, '..', '..', 'ros2_ws')
IDLE = ('IDLE', 'DONE', 'ERROR')


def default_targets():
    p = yaml.safe_load(open(os.path.join(WS, 'src', 'plan', 'config', 'plan.yaml')))['mission_manager']['ros__parameters']
    s = [f'S{i}' for i in p['survey_order']]
    loc = [f'L{i}' for i in p['build_location_order']]
    return s + loc + loc[-2::-1]  # 마지막 L은 놓은 자리에서 확인이라 되돌아올 때 뺌


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('targets', nargs='?', help='쉼표로: S4,S5,L1 (기본 plan.yaml 순서)')
    ap.add_argument('--clear-basket', action='store_true', help='시작 전 /coordinator/basket_clear')
    ap.add_argument('--real', action='store_true',
                    help='실기: use_sim_time 끔 (/clock이 없으면 타이머가 안 돌아 시작을 못 보냄) · Fast DDS 기본 = start.sh와 같은 fastdds.xml')
    a = ap.parse_args()
    targets = a.targets.upper().split(',') if a.targets else default_targets()
    start = 'start:GOTO:' + ','.join(targets)

    os.environ.setdefault('FASTDDS_DEFAULT_PROFILES_FILE', os.path.join(WS, 'fastdds.xml' if a.real else 'fastdds_local_only.xml'))
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'SUBNET'
    rclpy.init()
    n = rclpy.create_node('drive_test', parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', value=not a.real)])
    log = n.get_logger()
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    cmd = n.create_publisher(String, '/cmd/mission', 10)
    st = {'phase': None, 'note': None, 'started': False, 'running': False, 'exit': None}
    legs = []  # [상자, 시작 wall 시각, 끝 wall 시각, 결과] — 결과 "GOTO S4: ok" · "GOTO S6: nav_failed" (goto_seq)

    if a.clear_basket:
        c = n.create_client(Trigger, '/coordinator/basket_clear')
        if not c.wait_for_service(timeout_sec=20.0):
            sys.exit('/coordinator/basket_clear 안 보임 — coordinator가 떠 있는지 확인')
        f = c.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(n, f, timeout_sec=10.0)
        log.info(f'적재함 장부: {f.result().message if f.result() else "응답 없음"}')

    def on_state(m):
        st['phase'] = m.phase
        if m.note != st['note']:
            st['note'] = m.note
            log.info(f'[{m.phase}] {m.note}' + (f' · 오류 {m.error}' if m.error else ''))
            if m.note.startswith('상자로 가기 ') and m.note.endswith(')'):  # "상자로 가기 S4 (1/9)"
                legs.append([m.note.split()[2], time.monotonic(), None, None])
        goto = [r.split(': ', 1)[1] for r in m.results if r.startswith('GOTO ')]
        for leg, r in zip(legs, goto):
            if leg[3] is None:
                leg[2], leg[3] = time.monotonic(), r
        if st['started'] and m.phase not in IDLE:
            st['running'] = True
        if st['running'] and m.phase in ('DONE', 'ERROR'):
            st['exit'] = m.phase
        if st['started'] and '시작 거부' in m.note:
            st['exit'] = 'ERROR'

    n.create_subscription(MissionState, '/mission/state', on_state, latched)

    def tick():
        # 시작은 상태가 IDLE을 벗어날 때까지 2 s마다 다시 보냄 — 새 발행자는 구독자와 짝이 맞기 전 한 번 보낸 것이 버려짐
        #   (실기 11-18: start 뒤 mission_manager가 IDLE 그대로). 이미 진행 중이면 mission_manager가 무시 (on_command active_)
        if st['running']:
            return
        if st['started']:
            if time.monotonic() - st['sent'] >= 2.0 and st['tries'] < 10:
                cmd.publish(String(data=start))
                st['sent'] = time.monotonic()
                st['tries'] += 1
                log.warning(f'아직 {st["phase"]} — 시작 다시 보냄 ({st["tries"]}/10)')
            elif st['tries'] >= 10 and time.monotonic() - st['sent'] >= 2.0:
                st['exit'] = 'ERROR'
                log.error('시작이 안 받아짐 — mission_manager · ROS_DOMAIN_ID · Fast DDS 프로파일 확인')
            return
        if st['phase'] in IDLE and cmd.get_subscription_count() > 0:
            cmd.publish(String(data=start))
            st['started'] = True
            st['t0'] = st['sent'] = time.monotonic()
            st['tries'] = 0
            log.info(start)
        elif st['phase'] not in (None, *IDLE):
            st['exit'] = 'ERROR'
            log.error(f'미션이 이미 {st["phase"]} — 끝나거나 abort한 뒤에 다시')

    n.create_timer(1.0, tick)
    log.info('/mission/state 기다리는 중 (mission_manager)')
    try:
        while rclpy.ok() and st['exit'] is None:
            rclpy.spin_once(n, timeout_sec=0.2)
    except KeyboardInterrupt:
        if st['started']:
            cmd.publish(String(data='abort'))
            rclpy.spin_once(n, timeout_sec=0.5)
            print('abort 보냄')
        st['exit'] = 'ABORT'

    print(f'\n주행 시험 {st["exit"]} — {start}')
    for box, t0, t1, r in legs:
        print(f'  {box:<3} {r or "끝 못 봄":<24} ' + (f'{t1 - t0:6.1f} s' if t1 else '     —'))
    if st.get('t0'):
        print(f'  전체 {time.monotonic() - st["t0"]:.1f} s (wall · 복귀 포함)')
    ok = st['exit'] == 'DONE' and len(legs) == len(targets) and all(leg[3] == 'ok' for leg in legs)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
