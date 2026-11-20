#!/usr/bin/env python3
"""바코드 엿보기 시험 (11-20) — 손목만 든 자세 + 손 카메라를 잠깐 1080p로 올려 바코드 읽기, 그 부분만 바로 실행.

로봇(Jetson)에서 bringup을 띄운 뒤, .bashrc가 적용된 터미널에서:
  python3 tools/field/barcode_peek_test.py usb              # 카메라 USB 속도 (5000 이상 = USB 3) · ROS 필요 없음
  python3 tools/field/barcode_peek_test.py cam              # 해상도 전환만: 1080p로 올려 걸린 시간·fps·사진·읽기 → 원래대로
  python3 tools/field/barcode_peek_test.py pose [--down 17] [--yaw 0]
                                                           # 팔을 엿보기 자세로 (MANUAL 모드) + 1080p로 읽기 + pole 사진(신호등 가림 확인) → Enter면 stow
  python3 tools/field/barcode_peek_test.py mission          # 미션 관측 단계만 (start:BARCODE, AUTO 모드) — 진행 기록·시간·사진
사진·기록: ~/barcode_peek/<시각>/ (pole.jpg = 엿보기 중 pole 카메라 · grip.jpg = 1080p 손 카메라)
"""
import argparse
import glob
import os
import sys
import time

CAM = '/gripper_cam'
HI, LO = '1920x1080x15', '640x360x15'
STOW = [0.0, -1.9024, 1.344, 0.9023, 0.0]  # skills/params/basket.yaml pose_stow
PEEK = [0.0, -1.9024, 1.344, 0.5, 0.0]     # skills/params/arm.yaml pose_peek
J4_LEVEL = 0.1954                          # arm.yaml peek_j4_level (손 카메라 수평)


def usb():
    found = False
    for d in sorted(glob.glob('/sys/bus/usb/devices/*')):
        rd = lambda n: open(os.path.join(d, n)).read().strip() if os.path.exists(os.path.join(d, n)) else ''
        if rd('idVendor') != '8086':
            continue
        found = True
        sp = rd('speed')
        print(f'{rd("product")}  serial={rd("serial")}  speed={sp} Mbps  →',
              'USB 3 (1080p 가능)' if sp and int(sp) >= 5000 else 'USB 2 이하 — 케이블·포트 확인')
    if not found:
        print('RealSense(8086)가 안 보임')
    os.system("dpkg -l | grep -i 'realsense2-camera \\|librealsense2 ' | awk '{print $2, $3}'")


def out_dir():
    d = os.path.expanduser(time.strftime('~/barcode_peek/%m%d_%H%M%S'))
    os.makedirs(d, exist_ok=True)
    return d


class Rig:
    """ROS 쪽 공통 — 카메라 파라미터·영상·바코드."""

    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String
        from rcl_interfaces.srv import SetParameters
        self.rclpy = rclpy
        rclpy.init()
        self.n = Node('barcode_peek_test')
        self.info = None
        self.stamps = {'color': [], 'depth': []}
        self.img = {}
        self.qr = []
        sd = qos_profile_sensor_data
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.n.create_subscription(CameraInfo, CAM + '/color/camera_info', lambda m: setattr(self, 'info', m), sd)
        self.n.create_subscription(Image, CAM + '/color/image_raw', lambda m: self._img('grip', m, 'color'), sd)
        self.n.create_subscription(Image, CAM + '/aligned_depth_to_color/image_raw', lambda m: self.stamps['depth'].append(time.time()), sd)
        self.n.create_subscription(Image, '/pole_front_cam/color/image_raw', lambda m: self._img('pole', m, None), sd)
        self.n.create_subscription(String, '/mission/qr', lambda m: self.qr.append((time.time(), m.data)), latched)
        self.mode = ''
        self.n.create_subscription(String, '/state/operation_mode', lambda m: setattr(self, 'mode', m.data), latched)
        self.param_cli = self.n.create_client(SetParameters, CAM + '/set_parameters')

    def _img(self, key, m, stamp_key):
        self.img[key] = m
        if stamp_key:
            self.stamps[stamp_key].append(time.time())

    def spin(self, s):
        end = time.time() + s
        while time.time() < end:
            self.rclpy.spin_once(self.n, timeout_sec=0.05)

    def set_profile(self, prof):
        from rcl_interfaces.srv import SetParameters
        from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
        if not self.param_cli.wait_for_service(timeout_sec=3.0):
            sys.exit(f'{CAM}/set_parameters 서비스 없음 — bringup·카메라 확인')
        req = SetParameters.Request()
        req.parameters = [Parameter(name='rgb_camera.color_profile',
                                    value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=prof))]
        t0 = time.time()
        fut = self.param_cli.call_async(req)
        while not fut.done() and time.time() - t0 < 10:
            self.rclpy.spin_once(self.n, timeout_sec=0.05)
        r = fut.result()
        ok = r is not None and r.results and r.results[0].successful
        print(f'  파라미터 {prof}: {"성공" if ok else "실패 " + (r.results[0].reason if r and r.results else "응답 없음")}'
              f' ({time.time() - t0:.2f} s)')
        want = int(prof.split('x')[0])
        while time.time() - t0 < 10:
            self.rclpy.spin_once(self.n, timeout_sec=0.05)
            if self.info is not None and self.info.width == want and self.img.get('grip') is not None \
                    and self.img['grip'].width == want:
                print(f'  {want} px 영상 들어옴: 요청부터 {time.time() - t0:.2f} s')
                return time.time() - t0
        print(f'  10 s 안에 {want} px 영상이 안 옴 (지금 {self.info.width if self.info else "?"} px)')
        return None

    def rate(self, s):
        for k in self.stamps:
            self.stamps[k].clear()
        self.spin(s)
        return {k: len(v) / s for k, v in self.stamps.items()}

    @staticmethod
    def to_bgr(m):
        import numpy as np
        a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.step)[:, :m.width * 3].reshape(m.height, m.width, 3)
        return a[:, :, ::-1].copy() if m.encoding == 'rgb8' else a.copy()

    def save(self, d, key, name):
        import cv2
        m = self.img.get(key)
        if m is None:
            print(f'  {key} 영상 없음')
            return None
        bgr = self.to_bgr(m)
        cv2.imwrite(os.path.join(d, name), bgr)
        print(f'  저장 {os.path.join(d, name)} ({m.width}x{m.height})')
        return bgr

    @staticmethod
    def decode(bgr):
        """perception과 같은 방식: 1080p면 가운데 절반만 ZXing (Code128)."""
        try:
            import zxingcpp
        except ImportError:
            return 'zxing-cpp 없음 (pip install zxing-cpp)'
        h, w = bgr.shape[:2]
        crop = bgr[h // 4:h * 3 // 4, w // 4:w * 3 // 4] if w >= 1280 else bgr
        t0 = time.time()
        rs = [r.text for r in zxingcpp.read_barcodes(crop) if r.text]
        return f'{rs or "못 읽음"} ({(time.time() - t0) * 1000:.0f} ms · {crop.shape[1]}x{crop.shape[0]})'

    def peek_read(self, d, tries=5):
        """1080p로 올리고 최대 tries장 읽어 보기 → 원래대로."""
        sw = self.set_profile(HI)
        self.spin(0.6)  # 노출 맞추는 첫 장들
        r = self.rate(3.0)
        print(f'  1080p 동안 fps: 컬러 {r["color"]:.1f} · 정렬 depth {r["depth"]:.1f}')
        got = None
        for i in range(tries):
            self.spin(0.2)
            bgr = self.save(d, 'grip', f'grip_{i}.jpg')
            if bgr is None:
                continue
            res = self.decode(bgr)
            print(f'  {i + 1}장째: {res}')
            if 'P2-' in res:
                got = res
                break
        back = self.set_profile(LO)
        print(f'  → 결과: {"읽음 " + got if got else "못 읽음"} · 전환 올림 {sw} s · 내림 {back} s')
        return got


def cam(args):
    rig = Rig()
    d = out_dir()
    rig.spin(1.0)
    print(f'지금 손 카메라 {rig.info.width if rig.info else "?"} px · fps {rig.rate(2.0)}')
    rig.save(d, 'grip', 'grip_before.jpg')
    rig.peek_read(d)
    print(f'원래대로 fps {rig.rate(2.0)}')


def send_arm(rig, q, sec):
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
    pub = rig.n.create_publisher(JointTrajectory, '/arm_controller/joint_trajectory', 10)
    rig.spin(0.5)
    t = JointTrajectory()
    t.joint_names = [f'joint{i}' for i in range(1, 6)]
    p = JointTrajectoryPoint()
    p.positions = [float(v) for v in q]
    p.time_from_start = Duration(sec=int(sec), nanosec=int((sec % 1) * 1e9))
    t.points = [p]
    pub.publish(t)
    rig.spin(sec + 0.8)


def pose(args):
    import math
    rig = Rig()
    d = out_dir()
    rig.spin(1.0)
    if rig.mode and rig.mode != 'MANUAL':
        sys.exit(f'운영 모드 {rig.mode} — TUI에서 MANUAL로 바꾼 뒤 다시 (AUTO면 skill과 팔 명령이 겹침)')
    q = list(PEEK)
    q[0] = math.radians(args.yaw)
    q[3] = min(0.9023, max(0.0, J4_LEVEL + math.radians(args.down)))
    print(f'엿보기 자세로: joint1 {q[0]:+.3f} · joint4 {q[3]:.3f} (손 카메라 아래 {args.down:.0f}°)')
    send_arm(rig, q, 2.0)
    rig.save(d, 'pole', 'pole.jpg')  # 신호등이 팔에 가리는지 이 사진으로 확인
    rig.peek_read(d)
    input('사진 확인 뒤 Enter → stow로 접음 ')
    send_arm(rig, STOW, 2.0)
    print(f'끝 — {d}')


def mission(args):
    from std_msgs.msg import String
    from plan.msg import MissionState
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    rig = Rig()
    d = out_dir()
    log = open(os.path.join(d, 'log.txt'), 'w')
    st = {'phase': '', 'note': ''}
    t0 = time.time()

    def say(s):
        line = f'[{time.time() - t0:6.1f}s] {s}'
        print(line)
        log.write(line + '\n')
        log.flush()

    def on_state(m):
        if (m.phase, m.note) != (st['phase'], st['note']):
            st.update(phase=m.phase, note=m.note)
            say(f'{m.phase}: {m.note}' + (f' · 바코드 {m.barcode}' if m.barcode else '') + (f' · 오류 {m.error}' if m.error else ''))
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    rig.n.create_subscription(MissionState, '/mission/state', on_state, latched)
    pub = rig.n.create_publisher(String, '/cmd/mission', QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
    rig.spin(1.5)
    if rig.mode and rig.mode != 'AUTO':
        sys.exit(f'운영 모드 {rig.mode} — TUI에서 AUTO로 바꾼 뒤 다시')
    t0 = time.time()
    say('start:BARCODE 보냄 (관측 단계만 → READ)')
    pub.publish(String(data='start:BARCODE'))
    width, peek_t, pole_saved = 0, None, False
    while time.time() - t0 < args.timeout:
        rig.spin(0.1)
        w = rig.info.width if rig.info else 0
        if w != width:
            width = w
            say(f'손 카메라 컬러 {w} px')
            if w >= 1280:
                peek_t = time.time()
        if peek_t and not pole_saved and time.time() - peek_t > 0.5:
            pole_saved = True
            rig.save(d, 'pole', 'pole.jpg')
            rig.save(d, 'grip', 'grip.jpg')
        if st['phase'] in ('DONE', 'ERROR', 'IDLE') and time.time() - t0 > 3:
            break
    for t, q in rig.qr:
        if t >= t0:
            say(f'/mission/qr {q} (시작 뒤 {t - t0:.1f} s)')
            break
    say(f'끝 — {st["phase"]} · 기록 {d}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('usb')
    sp.add_parser('cam')
    p = sp.add_parser('pose')
    p.add_argument('--down', type=float, default=17.0, help='손 카메라 아래쪽 각도 [deg] (기본 17 = 바코드 0.15 m 높이 · 0.73 m 앞)')
    p.add_argument('--yaw', type=float, default=0.0, help='joint1 방위 [deg] (+ = 왼쪽)')
    m = sp.add_parser('mission')
    m.add_argument('--timeout', type=float, default=200.0)
    a = ap.parse_args()
    {'usb': lambda: usb(), 'cam': lambda: cam(a), 'pose': lambda: pose(a), 'mission': lambda: mission(a)}[a.cmd]()


if __name__ == '__main__':
    main()
