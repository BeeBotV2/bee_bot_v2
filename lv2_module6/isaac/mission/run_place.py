#!/usr/bin/env python3
"""놓기만 시험 — 적재함에 퍽을 다 실어 둔 채 L1 앞(−x 쪽)에서 시작해 approach(도킹) → place부터 (BUILD → VERIFY → RETURN).

  python3 isaac/mission/run_place.py                          # 실기 · L1 · P2-04 · R4Y2G6 · 칸 n = tag n (12개)
  python3 isaac/mission/run_place.py --code P2-01 --counts R5Y3G4
  python3 isaac/mission/run_place.py --slots 0:5,1:2,2:4      # 실제로 실은 칸:tag (나머지 칸은 빔)
  python3 isaac/mission/run_place.py --loc L2                 # 다른 장소부터

준비: 로봇을 L1 상자 −x 쪽 앞(상자를 바라보게, 도킹 전 자리)에 둠 · 적재함 12칸에 퍽을 아무렇게나 다 실음.
      놓기 동작 시험용 — 장부는 '칸 n = tag n'으로 가정하고 칸 순서대로 꺼내 정해진 칸에 놓음. 실제 퍽의 tag·색과 놓인 칸이
      맞는지는 안 봄 (결과의 tag 번호 = 꺼낸 적재함 칸 번호). 맞게 놓는지까지 보려면 --slots로 실제 칸:tag를 줌.
하는 일: ① ~/.ros/beebot_basket.txt에 칸:tag(색·모양은 아래 표)를 적고 /coordinator/basket_reload
        ② /cmd/mission start:PLACE:<장소>:<바코드>:<신호> → mission_manager가 장부 퍽으로 배정 → 그 장소부터 BUILD
           (가까운 면으로 도킹 = location_side_nearest · 장소마다 다 놓으면 물러나서 한 번 봄) → 나머지 장소 → 복귀
Ctrl+C = abort. 기록: ~/.ros/mission/<시각>/ (allocation.json · build.json · photos/)
"""
import argparse
import glob
import json
import os
import re
import sys

import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from plan.msg import MissionState

IDLE = ('IDLE', 'DONE', 'ERROR')
# 실기 퍽 12개 (11-22 allocation.json 기록) — tag → (색, 모양)
PUCKS = {t: ('green', 'box') for t in (0, 1, 2)} | {t: ('red', 'box') for t in (3, 4, 5)} | \
        {t: ('green', 'cyl') for t in (6, 7, 8)} | {t: ('red', 'cyl') for t in (9, 10, 11)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--loc', default='L1', choices=('L1', 'L2', 'L3'), help='먼저 놓을 장소 (로봇이 그 앞에 있음)')
    ap.add_argument('--code', default='P2-04', help='바코드 (P2-01~04)')
    ap.add_argument('--counts', default='R4Y2G6', help='신호 횟수 R<n>Y<n>G<n>')
    ap.add_argument('--slots', default=','.join(f'{i}:{i}' for i in range(12)), help='적재함 칸:tag 목록 (기본 칸 n = tag n)')
    ap.add_argument('--basket-file', default='~/.ros/beebot_basket.txt', help='coordinator basket_file')
    ap.add_argument('--timeout', type=float, default=1200.0, help='[s] 이 안에 안 끝나면 abort')
    a = ap.parse_args()
    if not re.fullmatch(r'R\dY\dG\d', a.counts.upper()):
        sys.exit(f'--counts {a.counts}: R<n>Y<n>G<n> 형식')
    slots = {}
    for kv in a.slots.split(','):
        s, t = (int(x) for x in kv.split(':'))
        if not (0 <= s < 12 and t in PUCKS) or s in slots or t in slots.values():
            sys.exit(f'--slots {kv}: 칸 0~11 · tag 0~11 · 중복 없이')
        slots[s] = t

    path = os.path.expanduser(a.basket_file)
    with open(path, 'w') as f:   # coordinator basket_save와 같은 형식
        for s, t in sorted(slots.items()):
            f.write(f'slot {s} {PUCKS[t][1]} {PUCKS[t][0]} {t} 0\n')
        f.write('order ' + ' '.join(str(s) for s in sorted(slots)) + '\nheld - - -1\n')
    start = f'start:PLACE:{a.loc}:{a.code}:{a.counts.upper()}'
    a_code, a_counts = a.code, a.counts.upper()

    ws = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'ros2_ws')
    os.environ.setdefault('FASTDDS_DEFAULT_PROFILES_FILE', os.path.join(ws, 'fastdds.xml'))
    os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'SUBNET'
    rclpy.init()
    n = rclpy.create_node('run_place')
    log = n.get_logger()
    c = n.create_client(Trigger, '/coordinator/basket_reload')
    if not c.wait_for_service(timeout_sec=20.0):
        sys.exit('/coordinator/basket_reload 안 보임 — coordinator(develop 최신 빌드)가 떠 있는지 확인')
    fut = c.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(n, fut, timeout_sec=10.0)
    log.info(f'적재함 장부 {len(slots)}칸: {fut.result().message if fut.result() else "응답 없음"}')

    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    cmd = n.create_publisher(String, '/cmd/mission', 10)
    st = {'phase': None, 'last': None, 'last0': None, 'started': False, 'running': False, 'done': None, 'msg': None, 't': 0}

    def out(text=''):   # 사람이 읽는 줄 — 로거 머리말([INFO] [시각] [노드]) 없이
        print(text, flush=True)

    def show_plan():   # 미션이 정한 장소별 배정 (회차 폴더 allocation.json) — 같은 컴퓨터(Jetson)에서 실행할 때
        runs = sorted(glob.glob(os.path.expanduser('~/.ros/mission/*/allocation.json')), key=os.path.getmtime)
        if not runs:
            log.warn('allocation.json 없음 — 배정 표를 못 찍음 (Jetson에서 실행했는지 확인)')
            return
        a = json.load(open(runs[-1]))
        slot_of = {t: s for s, t in slots.items()}
        out(f'\n{"─" * 14} 장소별 배정 {"─" * 14}')
        out(f'바코드 {a_code} · 신호 {a_counts} · 회차 {os.path.basename(os.path.dirname(runs[-1]))}')
        for loc, v in a.get('locations', {}).items():
            pk = v.get('pucks', [])
            out(f"\n{loc}  그룹 {v.get('group')} · 주사위 {len(pk)}" + (f"  (빈 칸 {v['empty_cells']})" if v.get('empty_cells') else ''))
            for p in pk:
                color, shape = PUCKS.get(p['tag'], ('?', '?'))
                out(f"    칸 {p['cell']}  ←  tag {p['tag']:>2}   적재함 {slot_of.get(p['tag'], '?'):>2} · {color}/{shape}")
        if not a.get('ok', True):
            out(f"\n배정 실패: {a.get('why')}")
        out('─' * 40 + '\n')

    def on_state(msg):
        if msg.phase == 'BUILD' and not st.get('plan_shown'):
            st['plan_shown'] = True
            show_plan()
        st['phase'], st['msg'] = msg.phase, msg
        line = (msg.phase, msg.note, msg.error, msg.placed)
        if line != st['last']:
            st['last'] = line
            if msg.phase != st.get('shown_phase'):   # 단계가 바뀌면 빈 줄 + 머리줄
                st['shown_phase'] = msg.phase
                out(f'\n[{msg.phase}]')
            t = n.get_clock().now().nanoseconds * 1e-9 - st.setdefault('t0', n.get_clock().now().nanoseconds * 1e-9)
            placed = f'  (놓음 {msg.placed}/{msg.total})' if msg.total else ''
            out(f'  {int(t) // 60:02d}:{int(t) % 60:02d}  {msg.note}{placed}' + (f'\n         ⚠ 오류 {msg.error}' if msg.error else ''))
        if st['started'] and msg.phase not in IDLE:
            st['running'] = True
        if st['running'] and msg.phase in ('DONE', 'ERROR'):
            st['done'] = msg.phase
        if st['started'] and '시작 거부' in msg.note:
            st['done'] = 'ERROR'

    n.create_subscription(MissionState, '/mission/state', on_state, latched)

    def tick():
        st['t'] += 1
        if not st['started']:
            if st['phase'] in IDLE and cmd.get_subscription_count() > 0:
                cmd.publish(String(data=start))
                st['started'], st['last0'] = True, st['last']
                log.info(f'{start} 보냄')
            elif st['phase'] not in (None, *IDLE):
                log.error(f'미션이 이미 {st["phase"]} — 끝나거나 abort한 뒤에 다시')
                st['done'] = 'ERROR'
            return
        if not st['running'] and st['last'] == st['last0']:
            cmd.publish(String(data=start))
        if st['t'] > a.timeout:
            log.error(f'{a.timeout:.0f} s 안에 안 끝남 — abort')
            cmd.publish(String(data='abort'))
            st['done'] = 'TIMEOUT'

    n.create_timer(1.0, tick, clock=rclpy.clock.Clock())
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
    print(f'\n{"─" * 16} 놓기 결과 {"─" * 16}')
    print(f'결과      {st["done"]}' + (f'  (이유 {s.error})' if s and s.error else ''))
    if s:
        print(f'놓음      {s.placed}/{s.total}')
        print(f'장소 그룹  L1 {s.order[:1] or "-"} · L2 {s.order[1:2] or "-"} · L3 {s.order[2:3] or "-"}')
        if s.results:
            print('\n실패·기록')
            for r in s.results:
                print(f'  - {r}')
    print('─' * 42)
    sys.exit(0 if st['done'] == 'DONE' else 1)


if __name__ == '__main__':
    main()
