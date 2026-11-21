#!/usr/bin/env bash
# 실패 순간 손 카메라 영상 (Jetson에서 실행) — 영상은 Wi-Fi로 안 보냄
#   gripper_cam 컬러를 메모리에 최근 RECORD_FAIL_S초(기본 10)만 들고 있다가, pick · pick_far · load · stand · stand_pick · place · approach
#   step이 실패하면(/task/status FAILED) 그 10초를 bag으로 씀 → ~/bee_bags/fail_<월일_시분초>/ · 끝나면 pa2로 scp
#   ./record_fail.sh
#   RECORD_FAIL_S=15 ./record_fail.sh
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$HERE/../../ros2_ws" 2>/dev/null && pwd || true)"
# shellcheck disable=SC1090
source "/opt/ros/${ROS_DISTRO_NAME:-lyrical}/setup.bash"
# shellcheck disable=SC1091
[ -n "$WS" ] && [ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"

OUT="${RVIZ_BAG_DIR:-$HOME/bee_bags}/fail_$(date +%m%d_%H%M%S)"
mkdir -p "$(dirname "$OUT")"
ros2 bag record --disable-keyboard-controls -s mcap -o "$OUT" --snapshot-mode \
  --max-cache-duration "${RECORD_FAIL_S:-10}" --max-cache-size 0 \
  --topics /gripper_cam/color/image_raw /perception/gripper_cam/detections /task/status /task/current_plan &
REC=$!
trap 'kill -INT $REC 2>/dev/null; wait $REC' EXIT INT TERM
echo "[record_fail] 실패 때만 씀 (최근 ${RECORD_FAIL_S:-10} s) → $OUT"
python3 "$HERE/record_fail_trigger.py"
