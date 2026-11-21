#!/usr/bin/env python3
"""Basket state log (sim only, from Isaac ground truth) -> <mission run dir>/basket.jsonl

Why (11-21): pucks end up messy in the basket (leaning on a slot wall, off the slot centre, touching a neighbour) after load
arm_jam / unload bumps. Count which kinds of mess happen and how often before building a tidy-up skill.

Inputs: /tmp/beebot_sim_pucks.json (isaac_startup.py: puck centres in base_link + tilt) and TF base_link -> basket_slot_0..11.
A line is written when the mission note changes and the basket differs from the last line, and always at DONE / ERROR.
Line: {t, phase, note, pucks: [{name, slot, off_mm, dz_mm, tilt, flags}], touching: [[a, b, mm]], counts}
  flags: offset (> OFFSET_MM from the slot centre) · tilted (> TILT_DEG) · lying (> LYING_DEG) · stray (on the robot, not in a slot)
         · shared (another puck sits in the same slot, e.g. one lying on top)
"""
import glob
import json
import math
import os
import time

import rclpy
from plan.msg import MissionState
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener

TRUTH = '/tmp/beebot_sim_pucks.json'
RUNS = os.path.expanduser('~/.ros/mission')
SLOTS = 12
IN_SLOT_M = 0.035   # puck centre within this of a slot centre (xy) = in that slot
ON_ROBOT_Z = 0.09   # [m] base_link z above which a puck rides on the robot (basket floor ~0.09, pucks ~0.12)
OFFSET_MM = 8.0
TILT_DEG = 10.0
LYING_DEG = 60.0
TOUCH_MM = 32.0     # centre distance below which two pucks touch (30 mm pucks + 2 mm)


class BasketLog(Node):
    def __init__(self):
        super().__init__('basket_log')
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.slots = None
        self.last_key = None
        self.note = None
        self.latest = None
        self.end_done = False
        self.create_timer(1.0, self.tick)
        self.create_subscription(MissionState, '/mission/state', self.on_state,  # latched like mission_manager's publisher
                                 QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def slot_xyz(self):
        if self.slots is None:
            try:
                self.slots = []
                for i in range(SLOTS):
                    t = self.tf.lookup_transform('base_link', f'basket_slot_{i}', rclpy.time.Time()).transform.translation
                    self.slots.append((t.x, t.y, t.z))
            except Exception:
                self.slots = None
        return self.slots

    def snapshot(self):
        slots = self.slot_xyz()
        try:
            pucks = json.load(open(TRUTH))['pucks']
        except Exception:
            return None
        if not slots:
            return None
        out, ride = [], {}
        for name, p in sorted(pucks.items()):
            if p['z'] < ON_ROBOT_Z:
                continue
            ride[name] = p
            d, i = min((math.hypot(p['x'] - s[0], p['y'] - s[1]), i) for i, s in enumerate(slots))
            flags = []
            if d > IN_SLOT_M:
                flags.append('stray')
            elif 1000 * d > OFFSET_MM:
                flags.append('offset')
            if p['tilt'] > LYING_DEG:
                flags.append('lying')
            elif p['tilt'] > TILT_DEG:
                flags.append('tilted')
            out.append({'name': name, 'slot': i if d <= IN_SLOT_M else None, 'off_mm': round(1000 * d, 1),
                        'dz_mm': round(1000 * (p['z'] - slots[i][2]), 1), 'tilt': round(p['tilt'], 1), 'flags': flags})
        for q in out:
            if q['slot'] is not None and sum(r['slot'] == q['slot'] for r in out) > 1:
                q['flags'].append('shared')
        names = sorted(ride)
        touching = []
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                pa, pb = ride[names[a]], ride[names[b]]
                mm = 1000 * math.hypot(pa['x'] - pb['x'], pa['y'] - pb['y'])
                if mm < TOUCH_MM:
                    touching.append([names[a], names[b], round(mm, 1)])
        counts = {f: sum(f in q['flags'] for q in out) for f in ('offset', 'tilted', 'lying', 'stray', 'shared')}
        counts.update(pucks=len(out), touching=len(touching), messy=sum(bool(q['flags']) for q in out))
        return {'pucks': out, 'touching': touching, 'counts': counts}

    def on_state(self, m):
        self.latest = m

    def tick(self):  # 1 Hz — TF and the truth file may not be there yet on the first (latched) state
        m = self.latest
        if m is None:
            return
        end = m.phase in ('DONE', 'ERROR')
        if m.note == self.note and not (end and not self.end_done):
            return
        s = self.snapshot()
        if s is None:
            return
        self.note = m.note
        key = json.dumps([[q['name'], q['slot'], q['flags'], round(q['off_mm']), round(q['tilt'])] for q in s['pucks']] + s['touching'])
        if key == self.last_key and not end:
            return
        self.last_key = key
        self.end_done = end
        runs = sorted(glob.glob(os.path.join(RUNS, '*/')), key=os.path.getmtime)
        if not runs:
            return
        line = {'t': round(time.time(), 3), 'phase': m.phase, 'note': m.note, **s}
        with open(os.path.join(runs[-1], 'basket.jsonl'), 'a') as f:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
        if s['counts']['messy'] or s['touching']:
            bad = [f"{q['name']} slot {q['slot']} {'+'.join(q['flags'])} ({q['off_mm']} mm, {q['tilt']} deg)" for q in s['pucks'] if q['flags']]
            self.get_logger().warning(f"basket messy [{m.phase}] {', '.join(bad)}" + (f" | touching {len(s['touching'])} pairs" if s['touching'] else ''))


def main():
    rclpy.init()
    try:
        rclpy.spin(BasketLog())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()
