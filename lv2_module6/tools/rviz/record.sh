#!/usr/bin/env bash
# 주행 · 매니퓰레이터 디버깅용 rosbag 녹화 (PC) — 가벼운 토픽만 (영상·점구름 제외: Wi-Fi로 받으면 RViz·Jetson 송신이 막힘)
#   켜 두면 운행 시작을 기다렸다가 녹화: 새 /task/sub_goals(미션·approach 계획) 또는 첫 /plan이 오는 순간부터.
#   그 전(RViz만 켜 둔 시간)은 버림. Ctrl-C(또는 start.sh가 RViz 끌 때)로 끝내면 파일을 닫음.
#   ./record.sh                         # → ~/bee_bags/nav_<월일_시분초>/
#   RVIZ_BAG_DIR=/data/bags ./record.sh
#   start.sh가 RViz와 같이 띄움 (PC에서만 · RVIZ_RECORD=0이면 안 띄움)
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../ros2_ws" 2>/dev/null && pwd || true)"
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO_NAME:-lyrical}/setup.bash"
# shellcheck disable=SC1091
[ -n "$WS" ] && [ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"

TOPICS=(/tf /tf_static /odom /odometry/filtered /cuvslam/odom /imu /joint_states
        /cmd_vel /cmd_vel_nav /cmd_vel_smoothed /plan /unsmoothed_plan /transformed_global_plan
        /local_costmap/costmap /global_costmap/costmap /local_costmap/published_footprint
        /collision_monitor_state /collision_monitor/triggering_points /polygon_approach
        /navigate_to_pose/_action/feedback /skills/navigate/_action/feedback
        /task/status /task/sub_goals /state/collision /state/operation_mode /arena/correction /tracked_objects /rosout
        /mission/state /arena/walls_cloud /lookahead_point /lookahead_collision_arc  # 11-18: 미션 단계 · 도면 벽 점(collision_monitor 입력) · RPP carrot·충돌 검사 호
        # 11-21 매니퓰레이터: 관절 전류는 /joint_states effort · 막힘·과부하 이벤트 · 손 카메라 상대 위치 · skill 단계 · 팔 궤적 · 보정 정책
        /arm_safety_event /hand_relative /hand_relative/target /skill_context /rl_arm_correction/debug
        /arm_controller/controller_state /arm_controller/joint_trajectory /display_planned_path /trajectory_execution_event
        /execute_trajectory/_action/feedback /gripper_controller/gripper_cmd/_action/feedback /attached_collision_object
        /state/devices/arm /state/devices/gripper /arena/correction /localization/wall_fit /localization/map_to_odom
        /skills/pick/_action/feedback /skills/pick_far/_action/feedback /skills/load/_action/feedback /skills/unload/_action/feedback
        /skills/place/_action/feedback /skills/inspect/_action/feedback /skills/pose/_action/feedback /skills/stand/_action/feedback
        /skills/stand_pick/_action/feedback /skills/approach/_action/feedback /task/current_plan
        /perception/gripper_cam/detections)  # 11-22: 손 카메라 검출 — hand_relative valid가 끊기는 이유 (퍽 못 봄 · tag 못 읽음)
# RVIZ_RECORD_IMAGES=1 (odom_nvblox.sh image_raw): 컬러 영상도 — 퍽 축·파지 확인용. Wi-Fi 부하 큼 (depth는 뺌)
[ "${RVIZ_RECORD_IMAGES:-0}" = "1" ] && TOPICS+=(/pole_front_cam/color/image_raw /gripper_cam/color/image_raw)
OUT="${RVIZ_BAG_DIR:-$HOME/bee_bags}/nav_$(date +%m%d_%H%M%S)"
mkdir -p "$(dirname "$OUT")"

echo "[record] 운행 시작 기다림 (새 /task/sub_goals 또는 첫 /plan) → $OUT"
# 기다림: /task/sub_goals는 latched라 구독 직후 오는 옛 계획은 건너뜀 (2 s 안에 온 것)
python3 - <<'EOF' || exit 0
import time, rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import Path
from plan.msg import SubGoalArray
rclpy.init()
n = rclpy.create_node('rviz_bag_trigger')
t0, hit = time.time(), []
latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
n.create_subscription(SubGoalArray, '/task/sub_goals', lambda m: hit.append('sub_goals') if time.time() - t0 > 2.0 and m.goals else None, latched)
n.create_subscription(Path, '/plan', lambda m: hit.append('plan'), 10)
try:
    while not hit:
        rclpy.spin_once(n, timeout_sec=0.2)
except KeyboardInterrupt:
    raise SystemExit(1)
print(f'[record] {hit[0]} 받음 — 녹화 시작', flush=True)
EOF

# 녹화 — 끝낼 때 SIGINT를 받아야 mcap이 닫힘 (exec로 신호를 그대로 받음) · hidden: 액션 feedback (_action/feedback)
exec ros2 bag record --disable-keyboard-controls --include-hidden-topics -s mcap -o "$OUT" --topics "${TOPICS[@]}"
