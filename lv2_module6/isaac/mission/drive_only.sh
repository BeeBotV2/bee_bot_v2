#!/usr/bin/env bash
# 주행만 — 신호등 · 바코드 · inspect · pick · load · unload · place 없이 미션 순서대로 상자에 다가가기만 하고 출발 구역으로 복귀.
#   mission_manager의 start:GOTO:<상자들> (Nav2 → 상자 도킹 → 다음 상자 · 끝나면 RETURN)을 보낸다.
#
#   isaac/standalone/isaac_mission.sh        # 터미널 1: Isaac + bringup (저장 지도 ros2_ws/map/scene_sim)
#   isaac/mission/drive_only.sh              # 터미널 2: 기본 = plan.yaml 순서 (지금 S4 S5 S3 S6 S2 S1 → L1 L2 L3 → L2 L1 → 복귀)
#   isaac/mission/drive_only.sh S1,L2        # 고른 상자만 (S1~S6 시작 박스 · L1~L3 장소)
#   CLEAR_BASKET=1 isaac/mission/drive_only.sh   # 적재함 장부가 남아 시작이 거부될 때 먼저 비움
#
# Ctrl+C = /cmd/mission abort. Fast DDS 프로파일은 Isaac 쪽과 같아야 함 (터미널 값 · 없으면 fastdds_local_only.xml).
HERE=$(dirname "$(realpath "$0")")
WS=$(realpath "$HERE/../../ros2_ws")
# 기본 경로 = plan.yaml mission_manager 순서: survey_order(S) → build_location_order(L) → 그 거꾸로(마지막 L은 놓은 자리에서 확인이라 뺌)
TARGETS=${1:-$(python3 - "$WS/src/plan/config/plan.yaml" <<'EOF'
import sys, yaml
p = yaml.safe_load(open(sys.argv[1]))['mission_manager']['ros__parameters']
s = [f'S{i}' for i in p['survey_order']]
l = [f'L{i}' for i in p['build_location_order']]
print(','.join(s + l + l[-2::-1]))
EOF
)}

export FASTDDS_DEFAULT_PROFILES_FILE="${FASTDDS_DEFAULT_PROFILES_FILE:-$WS/fastdds_local_only.xml}" ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
source /opt/ros/lyrical/setup.bash
source "$WS/install/setup.bash"

[ "${CLEAR_BASKET:-0}" = 1 ] && ros2 service call /coordinator/basket_clear std_srvs/srv/Trigger
trap 'ros2 topic pub -t 3 -w 2 /cmd/mission std_msgs/msg/String "{data: abort}" >/dev/null; echo "abort 보냄"; exit 1' INT
ros2 topic pub -t 3 -w 2 /cmd/mission std_msgs/msg/String "{data: 'start:GOTO:$TARGETS'}" >/dev/null
echo "[drive_only] start:GOTO:$TARGETS — 진행 (Ctrl+C = abort)"
ros2 topic echo --qos-durability transient_local --qos-reliability reliable /mission/state plan/msg/MissionState --field note
