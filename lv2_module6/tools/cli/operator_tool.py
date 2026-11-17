#!/usr/bin/env python3
"""Read-only plain-text robot monitor; bringup and actuation run separately."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import CameraInfo, Imu, JointState, PointCloud2
from std_msgs.msg import Bool, Float32MultiArray, String

from op.console import ConsolePrompt
from op.monitor import monitor_lines, mission_status_lines
from op.state import StateStore
from op.terminal import Terminal

LATCHED = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)
LIVE = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)
# 생존·Hz만 보는 빠른 토픽(odom·imu 100 Hz …) — 최신 1개만 쌓아 파이썬이 밀린 메시지를 다 꺼내지 않게 (Hz는 발행 순번으로)
PULSE = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
STATE_TOPICS = {
    'operation_mode': '/state/operation_mode',
    'map_mode': '/state/map_mode',
}


class OperatorTool(Node):
    """배선 전용 노드 — 상태·화면·버스는 op/ 패키지가 맡는다."""

    def __init__(self):
        super().__init__('operator_tool')
        self.term = Terminal(color=False)
        self.state = StateStore()
        self.prompt = ConsolePrompt()
        self.bus = None
        if os.environ.get('CLI_SOURCE', 'real') == 'fake':
            # The in-process simulator uses Bus as its test actuator. Real and
            # Docker monitoring stay read-only and do not load operator commands.
            from op.bus import Bus
            self.bus = Bus(self, self.state, self.prompt, self.term)
        self._monitor_param_clients = {}
        self._monitor_param_poll_at = 0.0
        # 전역 executor에 노드를 붙여 둔다 — rclpy.spin_once(node)가 매번 add/remove_node로 wait set을 다시 만드는 일을 줄인다.
        if hasattr(rclpy, 'get_global_executor'):   # fake/docker 대역엔 없음
            rclpy.get_global_executor().add_node(self)
        self._wire()

    # ------------------------------------------------------------- ROS 배선
    def poll(self, seconds: float = 0.2):
        """콜백을 잠깐 돌려 상태를 갱신한다."""
        end = time.monotonic() + max(0.0, seconds)
        while time.monotonic() < end and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=min(0.05, max(0.01, end - time.monotonic())))

    def cmd_monitor(self, args) -> int:
        """Append a compact status sample until Ctrl-C. Never starts bringup."""
        interval = max(0.1, float(args.interval))
        next_sample = time.monotonic()
        print('BeeBot monitor · Ctrl-C to stop · bringup is managed separately', flush=True)
        while rclpy.ok():
            now = time.monotonic()
            self._poll_monitor_params()
            if now >= next_sample:
                for line in monitor_lines(self.state, latency_window_s=args.latency_window):
                    print(line, flush=True)
                print('', flush=True)
                next_sample = now + interval
            self.poll(min(0.05, max(0.0, next_sample - time.monotonic())))
            if self.bus is not None:  # In-process fake-world harness
                self.bus.nav.tick()
        return 0

    def cmd_status(self, args) -> int:
        self.poll(0.3)
        self._poll_monitor_params()
        for line in monitor_lines(self.state, latency_window_s=8.0):
            print(line)
        if getattr(args, 'diagnostics', None) not in (None, False):
            for name, (level, message) in sorted(self.state.diagnostics.items()):
                if not args.diagnostics or args.diagnostics.lower() in name.lower():
                    print(f'DIAG {name}: {message}')
        return 0

    def cmd_mission(self, args) -> int:
        if args.action != 'status':
            print('run-cli is read-only; mission execution is managed by bringup/ROS commands', file=sys.stderr)
            return 2
        self.poll(0.5)
        for line in mission_status_lines(self.state):
            print(line)
        return 0

    def _wire(self):
        for key, topic in STATE_TOPICS.items():
            self.create_subscription(
                String, topic,
                lambda msg, key=key: self.state.on_state_value(key, msg.data),
                LATCHED)
        self.create_subscription(Bool, '/state/emergency',
                                 lambda m: self.state.on_emergency(m.data), LATCHED)
        self.create_subscription(JointState, '/joint_states', self._on_joints, PULSE)
        # Keep large PointCloud2 payloads raw; other small messages expose header stamps.
        for key, topic, typ in (('odom', '/odom', Odometry), ('filtered_odom', '/odometry/filtered', Odometry),
                                ('imu', '/imu', Imu),
                                ('front_cam', '/pole_front_cam/color/camera_info', CameraInfo),
                                ('gripper_cam', '/gripper_cam/color/camera_info', CameraInfo),
                                ('cuvslam', '/cuvslam/odom', Odometry)):
            self.create_subscription(typ, topic, self._pulse(key), PULSE)
        try:
            self.create_subscription(PointCloud2, '/pole_front_cam/depth/color/points_filtered',
                                     self._pulse('points', raw=True), PULSE, raw=True)
        except TypeError:  # fake/docker RCLPY shims may not accept raw subscriptions
            self.create_subscription(PointCloud2, '/pole_front_cam/depth/color/points_filtered',
                                     self._pulse('points'), PULSE)
        try:
            self.create_subscription(PointCloud2, '/nvblox_node/combined_esdf_pointcloud',
                                     self._pulse('nav_points', raw=True), PULSE, raw=True)
        except TypeError:
            self.create_subscription(PointCloud2, '/nvblox_node/combined_esdf_pointcloud',
                                     self._pulse('nav_points'), PULSE)
        for key, topic in (('cmd_vel_nav', '/cmd_vel_nav'), ('cmd_vel_mux', '/cmd_vel_mux'),
                           ('cmd_vel_smoothed', '/cmd_vel_smoothed'), ('cmd_vel', '/cmd_vel')):
            self.create_subscription(TwistStamped, topic, self._pulse(key), PULSE)
        self.create_subscription(Float32MultiArray, '/controller_server/FollowPath/status',
                                 lambda m: self.state.on_rl_status(m.data), LIVE)
        self.create_subscription(DiagnosticArray, '/diagnostics_agg',
                                 lambda m: self.state.on_diagnostics(m.status), LIVE)
        # 커스텀 msg가 빌드 전이면 해당 줄만 비활성 (cli 동작에는 지장 없음)
        try:
            from perception.msg import TrackedObjectArray  # type: ignore
            self.create_subscription(
                TrackedObjectArray, '/tracked_objects', self._on_objects, LIVE)
        except Exception:
            pass
        try:
            from plan.msg import TaskStatus  # type: ignore
            self.create_subscription(
                TaskStatus, '/task/status',
                lambda m: self.state.on_task(m.plan_id, m.step, m.state, m.reason),
                LATCHED)
        except Exception:
            pass
        try:
            from plan.msg import MissionState  # type: ignore
            self.create_subscription(MissionState, '/mission/state', self.state.on_mission, LATCHED)
        except Exception:
            pass
        self._wire_mission()
        # Parameter snapshots are read asynchronously by the live monitor.
        self._wire_monitor_params()

    def _wire_monitor_params(self):
        """Poll only the active planner/controller/mission policy values for the watch output."""
        try:
            from rclpy.parameter_client import AsyncParameterClient
        except Exception:
            return
        self._monitor_param_names = {
            'controller_server': [
                'controller_frequency', 'failure_tolerance', 'FollowPath.enable',
                'FollowPath.model_path', 'FollowPath.fallback_controller',
                'FollowPath.max_lin', 'FollowPath.max_ang',
                'FollowPath.inner_dwb.max_vel_x', 'FollowPath.inner_dwb.max_vel_theta',
            ],
            'planner_server': ['planner_plugins', 'GridBased.plugin', 'GridBased.cost_travel_multiplier'],
            'mission_manager': ['planner'],
        }
        for name in self._monitor_param_names:
            try:
                self._monitor_param_clients[name] = AsyncParameterClient(self, f'/{name}')
            except Exception:
                pass

    @staticmethod
    def _parameter_value(value):
        kind = int(getattr(value, 'type', 0))
        if kind == 1:
            return bool(value.bool_value)
        if kind == 2:
            return int(value.integer_value)
        if kind == 3:
            return float(value.double_value)
        if kind == 4:
            return str(value.string_value)
        if kind == 7:
            return [int(v) for v in value.integer_array_value]
        if kind == 8:
            return [float(v) for v in value.double_array_value]
        if kind == 9:
            return list(value.string_array_value)
        return None

    def _poll_monitor_params(self):
        now = time.monotonic()
        if now - self._monitor_param_poll_at < 5.0:
            return
        self._monitor_param_poll_at = now
        for name, client in self._monitor_param_clients.items():
            try:
                if not client.wait_for_services(timeout_sec=0.0):
                    continue
                names = self._monitor_param_names[name]
                future = client.get_parameters(names)
                if hasattr(future, 'add_done_callback'):
                    future.add_done_callback(lambda f, node=name, keys=names: self._on_monitor_params(node, keys, f))
                elif future.done():
                    self._on_monitor_params(name, names, future)
            except Exception:
                continue

    def _on_monitor_params(self, node: str, names: list[str], future):
        try:
            response = future.result()
            values = {key: self._parameter_value(value) for key, value in zip(names, response.values)}
            self.state.set_runtime_params(node, {key: value for key, value in values.items() if value is not None})
        except Exception:
            pass

    def _ros_now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    @staticmethod
    def _seq(info):
        return info.get('publication_sequence_number') if info is not None else None

    def _pulse(self, key: str, raw: bool = False):
        """Track frequency and DDS transit latency without retaining message payloads."""
        def on(msg, info=None):
            seq = self._seq(info)
            if raw:
                if info is not None:
                    self.state.on_transport(key, info, seq)
                else:
                    self.state.tick(key, seq)
                return
            self.state.on_stream(key, msg, seq, self._ros_now_s(), info)
        return on

    def _on_joints(self, msg, info=None):
        self.state.on_joints(msg.name, msg.position, msg.velocity, msg.effort,
                             self._seq(info), msg.header.stamp, self._ros_now_s(), info)

    def _on_objects(self, msg, info=None):
        self.state.on_objects(msg.objects, msg.header.frame_id, msg.header.stamp,
                              self._ros_now_s(), self._seq(info), info)

    def _wire_mission(self):
        """미션 화면 디버그 줄·상단 미션 칸용 구독 — msg가 없는 환경이면 그 줄만 빠진다."""
        subs = (  # (패키지.msg, 클래스, 토픽, QoS, 콜백) — QoS는 발행 쪽과 맞춤 (mission_manager.cpp · coordinator.cpp)
            ('plan.msg', 'SubGoalArray', '/task/current_plan', LATCHED, self.state.on_sub_goals),
            # 구형 coordinator/Isaac/fake가 아직 legacy ingress만 발행할 때의 화면 호환
            ('plan.msg', 'SubGoalArray', '/task/sub_goals', LATCHED, self.state.on_sub_goals),
            ('plan.msg', 'BasketState', '/coordinator/basket', LATCHED, self.state.on_basket),
            ('perception.msg', 'MissionSignal', '/mission/signal_status', LIVE, self.state.on_signal_status),
            ('ml.msg', 'SkillContext', '/skill_context', LIVE, self.state.on_skill_context),
            # nav2 collision_monitor — 동작이 바뀔 때만 발행
            ('nav2_msgs.msg', 'CollisionMonitorState', '/collision_monitor_state', LIVE, self.state.on_collision_state),
        )
        import importlib
        for mod, cls, topic, qos, cb in subs:
            try:
                typ = getattr(importlib.import_module(mod), cls)
            except Exception:
                continue
            self.create_subscription(typ, topic, cb, qos)
        try:  # Nav2 주행 진행 (navigate skill이 보낸 goal)
            from nav2_msgs.action import NavigateToPose  # type: ignore
            self.create_subscription(NavigateToPose.Impl.FeedbackMessage, '/navigate_to_pose/_action/feedback',
                                     lambda m: self.state.on_nav_feedback(m.feedback), LIVE)
        except Exception:
            pass
        try:  # 각 skill의 실제 진행 단계 (stage feedback) — feedback은 skill이 바뀔 때만 저빈도로 발행
            from plan.action import Skill  # type: ignore
            for skill in ('navigate', 'approach', 'pick', 'pick_far', 'place', 'load', 'unload', 'stand', 'pose', 'inspect'):
                self.create_subscription(
                    Skill.Impl.FeedbackMessage, f'/skills/{skill}/_action/feedback',
                    lambda m, name=skill: self.state.on_skill_feedback(name, m.feedback.stage), LIVE)
        except Exception:
            pass

    def __getattr__(self, name: str):
        # The fake world test harness can reach its in-process test actuator.
        if self.bus is not None and (name.startswith('cmd_') or name in ('map_args', 'status_args')):
            try:
                bus = object.__getattribute__(self, 'bus')
            except AttributeError:
                raise AttributeError(name) from None
            return getattr(bus, name)
        raise AttributeError(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='run-cli',
        description='BeeBot read-only monitor. Bringup is managed separately with ros2_ws/start.sh and stop.sh.')
    sub = parser.add_subparsers(dest='command', required=True)

    monitor = sub.add_parser('monitor', help='live plain-text monitor (default, Ctrl-C to stop)')
    monitor.add_argument('--interval', type=float, default=1.0, help='sample interval in seconds (default 1.0)')
    monitor.add_argument('--latency-window', type=float, default=8.0,
                         help='rolling command/odom response window in seconds (default 8.0)')

    status = sub.add_parser('status', help='one-shot compact robot status')
    status.add_argument('--diagnostics', nargs='?', const='', default=None, metavar='NAME',
                        help='also print diagnostics, optionally filtered by name')

    mission = sub.add_parser('mission', help='read-only mission status')
    mission.add_argument('action', choices=('status',))
    return parser

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ['monitor']
    parser = build_parser()
    args = parser.parse_args(argv)
    rclpy.init(args=[])
    node = OperatorTool()
    try:
        method = getattr(node, f'cmd_{args.command}')
        return int(method(args) or 0)
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
