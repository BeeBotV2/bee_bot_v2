"""Plain-text robot monitor and latency estimates used by run-cli."""
from __future__ import annotations

import math
import time

INNER = {1: 'DWB', 2: 'MPPI', 3: 'RPP', 4: 'RotationShim'}
REASONS = {
    0: '', 1: 'model missing', 2: 'policy disabled', 3: 'point cloud/scan stale', 4: 'inference failed',
}

# Nominal rates from the robot configuration. These are display references, not
# safety limits. The measured age and rate remain visible when a stream is slow.
EXPECTED_HZ = {
    'front_cam': 15, 'gripper_cam': 15, 'points': 10, 'nav_points': 2, 'tracked_objects': 10,
    'cuvslam': 15, 'odom': 50, 'filtered_odom': 50, 'imu': 100, 'joint_states': 50,
    'cmd_vel_nav': 20, 'cmd_vel_mux': 20, 'cmd_vel_smoothed': 20, 'cmd_vel': 20,
}

STREAMS = {
    'SENSE': ('front_cam', 'gripper_cam', 'points', 'nav_points', 'tracked_objects'),
    'POSE': ('cuvslam', 'odom', 'filtered_odom', 'imu'),
    'CMD': ('cmd_vel_nav', 'cmd_vel_mux', 'cmd_vel_smoothed', 'cmd_vel'),
}


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 12 or len(xs) != len(ys):
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    vx, vy = sum(x * x for x in dx), sum(y * y for y in dy)
    if vx < 1e-5 or vy < 1e-5:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / math.sqrt(vx * vy)


def _nearest(samples, when: float, tolerance: float = 0.075):
    if not samples:
        return None
    item = min(samples, key=lambda s: abs(s[0] - when))
    return item if abs(item[0] - when) <= tolerance else None


def estimate_response_lag(state, window_s: float = 8.0, max_lag_s: float = 0.8):
    """Estimate final-command to wheel-odometry lag by rolling cross-correlation.

    This is intentionally reported as an estimate with correlation confidence;
    constant commands or insufficient motion produce no value.
    """
    now = time.monotonic()
    commands = [s for s in state.command_samples if now - s[0] <= window_s]
    odom = [s for s in state.odom_samples if now - s[0] <= window_s + max_lag_s]
    if len(commands) < 12 or len(odom) < 12:
        return None

    result = {}
    steps = max(1, int(max_lag_s / 0.05))
    for label, index in (('v', 1), ('w', 2)):
        best = None
        for n in range(steps + 1):
            lag = n * 0.05
            pairs = [(cmd[index], actual[index]) for cmd in commands
                     if (actual := _nearest(odom, cmd[0] + lag)) is not None]
            corr = _pearson([p[0] for p in pairs], [p[1] for p in pairs])
            if corr is not None and (best is None or corr > best[1]):
                best = (lag, corr, len(pairs))
        result[label] = best if best is not None and best[1] >= 0.35 else None
    return result


def _stream_text(state, key: str, now: float) -> str:
    rate = state.rates.get(key)
    if rate is None or rate.age() is None:
        return f'{key}=n/a'
    age = state.stream_age(key, now)
    transport = state.transport_latency_ms(key, now)
    hz = rate.hz()
    expected = EXPECTED_HZ.get(key, 0)
    stale_s = max(0.5, 3.0 / expected) if expected else 1.0
    mark = '!' if ((age is not None and age > stale_s) or
                   (expected and hz < expected * 0.5 and rate.age() > 1.0)) else ''
    age_text = f'{age * 1000:.0f}ms' if age is not None else 'n/a'
    dds_text = f'{transport:.1f}ms' if transport is not None else 'n/a'
    return f'{key}[age={age_text},dds={dds_text},hz={hz:.0f}]{mark}'


def _diag_values(state, contains: str) -> dict[str, str]:
    for name, values in state.diag_values.items():
        if contains in name:
            return dict(values)
    return {}


def _param(state, node: str, name: str, default=None):
    item = state.runtime_params.get(f'{node}.{name}')
    if item is None or time.monotonic() - item[1] > 15.0:
        return default
    return item[0]


def controller_line(state) -> str:
    values = _diag_values(state, 'controller_server: FollowPath')
    status = state.rl_status
    recent = status and time.monotonic() - state.rl_status_time <= 2.0
    active_inner = INNER.get(int(status[9]), 'unknown') if recent and len(status) > 9 else 'unknown'
    if recent and status[0] >= 0.5:
        mode = 'RL'
    elif recent:
        mode = f'{active_inner} fallback'
        reason = REASONS.get(int(status[1]), '') if len(status) > 1 else ''
        if reason:
            mode += f' ({reason})'
    else:
        mode = 'no controller status'
    model = values.get('model', 'n/a')
    device = values.get('device', 'n/a')
    configured = _param(state, 'controller_server', 'FollowPath.fallback_controller', 'n/a')
    planner_plugins = _param(state, 'planner_server', 'planner_plugins', ['GridBased'])
    planner_plugin = _param(state, 'planner_server', 'GridBased.plugin', 'nav2_smac_planner::SmacPlanner2D')
    if isinstance(planner_plugins, (list, tuple)):
        planner_id = ','.join(str(x) for x in planner_plugins)
    else:
        planner_id = str(planner_plugins)
    max_v = _param(state, 'controller_server', 'FollowPath.max_lin',
                   _param(state, 'controller_server', 'FollowPath.inner_dwb.max_vel_x'))
    max_w = _param(state, 'controller_server', 'FollowPath.max_ang',
                   _param(state, 'controller_server', 'FollowPath.inner_dwb.max_vel_theta'))
    limits = f' limits={max_v}/{max_w}' if max_v is not None or max_w is not None else ''
    command = f' cmd={status[7]:+.2f}/{status[8]:+.2f}' if recent and len(status) > 8 else ''
    mission_planner = _param(state, 'mission_manager', 'planner', 'n/a')
    fallback = active_inner if recent else str(configured)
    policy_enabled = values.get('enable', 'n/a')
    deviation = values.get('path_deviation_m', 'n/a')
    deviation_limit = values.get('path_deviation_tolerance_m', 'n/a')
    deviation_state = values.get('path_deviation_state', 'n/a')
    sensor_age = values.get('sensor_age', 'n/a')
    return (f'ACTIVE controller={mode} policy={policy_enabled}/{model}@{device} fallback={fallback} '
            f'planner={planner_id}:{planner_plugin} mission-policy={mission_planner}{limits}{command} '
            f'path_dev={deviation}/{deviation_limit}m:{deviation_state} sensor_age={sensor_age}s')


def mission_lines(state) -> list[str]:
    now = time.monotonic()
    lines = []
    mission = state.mission
    if mission is None:
        lines.append('MISSION offline: /mission/state not received')
    else:
        elapsed = max(0.0, now - state.mission_since)
        lines.append(f'MISSION {mission.phase} {elapsed:.0f}s · {mission.note or "-"}')
        if mission.error:
            lines.append(f'  error={mission.error}')
        if getattr(mission, 'barcode', ''):
            lines.append(f'  barcode={mission.barcode}')
    if state.signal is not None:
        signal, received = state.signal
        status = 'confirmed' if signal.ok else f'pending/{signal.error or "unconfirmed"}'
        lines.append(f'SIGNAL {status} cycles={signal.cycles} raw={signal.raw or "-"} '
                     f'age={max(0.0, now - received):.1f}s')
    if state.task is None:
        lines.append('SKILL none')
    else:
        plan_id, step, status, reason = state.task
        current = state.current_skill()
        name, target = current if current else ('?', '?')
        label = {0: 'RUNNING', 1: 'SUCCEEDED', 2: 'FAILED'}.get(status, str(status))
        elapsed = max(0.0, now - state.task_since)
        lines.append(f'SKILL plan={plan_id} step={step} {name} target={target} {label} {elapsed:.1f}s'
                     + (f' reason={reason}' if reason else ''))
    if state.skill_ctx is not None:
        ctx, received = state.skill_ctx
        age = now - received
        if age < 5.0 and getattr(ctx, 'skill', ''):
            lines.append(f'ARM_POLICY skill={ctx.skill} policy={ctx.policy or "none"} '
                         f'residual={"live" if ctx.residual_live else "fixed"} age={age:.1f}s')
    if state.barcode_cam is not None and state.barcode_target is not None:
        camera, content = state.barcode_cam
        frame, x, y, z, received = state.barcode_target
        age = max(0.0, now - received)
        kind = f'decoded={content}' if content else 'candidate=undecoded'
        lines.append(f'BARCODE {kind} camera={camera} target[{frame}]={x:.2f},{y:.2f},{z:.2f}m age={age:.1f}s')
    if state.nav is not None:
        (distance, recoveries, elapsed), received = state.nav
        lines.append(f'NAV remaining={distance:.2f}m recoveries={recoveries} elapsed={elapsed:.1f}s '
                     f'feedback_age={max(0.0, now - received):.2f}s')
    return lines


def monitor_lines(state, latency_window_s: float = 8.0) -> list[str]:
    now = time.monotonic()
    lines = [time.strftime('%H:%M:%S') + ' BeeBot live monitor']
    state_age = now - state.state_last_update if state.state_last_update else None
    if state_age is None:
        lines.append('ROS waiting for bringup/state_manager (run-cli does not start bringup)')
    else:
        lines.append(f'ROS connected mode={state.operation or "-"} map={state.map_mode or "-"} '
                     f'estop={"ON" if state.emergency else "OFF"} state_age={state_age:.1f}s')
    lines.extend(mission_lines(state))
    lines.append(controller_line(state))
    for group, keys in STREAMS.items():
        lines.append(group + ' ' + ' '.join(_stream_text(state, key, now) for key in keys))
    wheels = ' '.join(f'{name}={state.joint_vel[name]:+.2f}rad/s'
                      for name in ('wheel_left_joint', 'wheel_right_joint') if name in state.joint_vel)
    lines.append('ACT ' + _stream_text(state, 'joint_states', now) + ' ' + _stream_text(state, 'odom', now)
                 + (f' wheels[{wheels}]' if wheels else ' wheels=n/a'))
    lag = estimate_response_lag(state, latency_window_s)
    if lag is None or not any(lag.values()):
        lines.append(f'RESPONSE lag=n/a (need varied cmd/odom data; window={latency_window_s:.0f}s)')
    else:
        parts = []
        for label in ('v', 'w'):
            value = lag.get(label)
            parts.append(f'{label}={value[0] * 1000:.0f}ms r={value[1]:.2f}' if value else f'{label}=n/a')
        lines.append('RESPONSE cmd_vel→odom ' + ' '.join(parts))
    bad = sorted(((level, name, message) for name, (level, message) in state.diagnostics.items() if level > 0),
                 reverse=True)
    if bad:
        lines.append('WARN ' + ' | '.join(f'{name.split(": ", 1)[-1]}: {message}' for _, name, message in bad[:3]))
    return lines


def mission_status_lines(state) -> list[str]:
    return mission_lines(state)
