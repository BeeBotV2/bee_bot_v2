#!/usr/bin/env bash
# Isaac odom/nvblox RViz entry point, through tools/rviz/start.sh (including optional bag recording).
#   ./odom_nvblox.sh                 # simulation time on by default; may start before Isaac Play
#   ./odom_nvblox.sh 42              # first arg = ROS_DOMAIN_ID; remaining args go to rviz2
#   (no domain arg: copy the ROS env of the running Isaac — isaac_sim.sh before Play, or its launch)
#   RVIZ_USE_SIM_TIME=false ./odom_nvblox.sh  # wall time for a real-robot view
#   RVIZ_RECORD=0 ./odom_nvblox.sh   # skip bag recording
# When an Isaac launch in the selected ROS domain is running, borrow its DDS settings so both join the same graph.
# Do not use the Isaac PID file to decide simulation time: it is created only after Play and ROS launch startup.
HERE="$(cd "$(dirname "$0")" && pwd)"

read_env() {
  tr '\0' '\n' < "/proc/$1/environ" 2>/dev/null | sed -n "s/^$2=//p" | head -n 1
}
DDS_KEYS="ROS_DOMAIN_ID FASTDDS_DEFAULT_PROFILES_FILE ROS_AUTOMATIC_DISCOVERY_RANGE ROS_STATIC_PEERS ROS_DISCOVERY_SERVER ROS_LOCALHOST_ONLY FASTRTPS_DEFAULT_PROFILES_FILE RMW_IMPLEMENTATION"

if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  export ROS_DOMAIN_ID="$1"
  shift
else
  # 도메인 인자가 없으면 지금 띄운 Isaac(isaac_sim.sh — Play 전에도 있음 · 또는 그 ROS launch)의 ROS 환경을 그대로 씀.
  #   Isaac 터미널에서 export ROS_DOMAIN_ID=30 등을 하고 RViz는 새 터미널(bashrc 88)에서 띄우면 다른 도메인이라 아무것도 안 보였음 (11-20)
  SRC=$(pgrep -f "bin/ros2 launch .*isaac_system.launch.py" | sort -rn | head -n 1)
  [[ -n "$SRC" ]] || SRC=$(pgrep -f "isaac/standalone/isaac_sim.sh" | sort -rn | head -n 1)
  if [[ -n "$SRC" ]]; then
    for key in $DDS_KEYS; do
      value="$(read_env "$SRC" "$key")"
      if [[ -n "$value" ]]; then export "$key=$value"; else unset "$key"; fi
    done
    echo "Isaac(pid $SRC) ROS 환경을 따름 — ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}"
  fi
fi
TARGET_DOMAIN="${ROS_DOMAIN_ID:-0}"
ISAAC_FOUND=false
MATCHING_LAUNCH=false
LAUNCH_PID=""

read_launch_env() {
  local pid="$1" key="$2"
  tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | sed -n "s/^${key}=//p" | head -n 1
}

# Newest matching launch wins if a stale Isaac launch from another domain is also present.
for pid in $(pgrep -f "bin/ros2 launch .*isaac_system.launch.py" 2>/dev/null | sort -rn || true); do
  ISAAC_FOUND=true
  launch_domain="$(read_launch_env "$pid" ROS_DOMAIN_ID)"
  launch_domain="${launch_domain:-0}"
  if [[ "$launch_domain" == "$TARGET_DOMAIN" ]]; then
    MATCHING_LAUNCH=true
    LAUNCH_PID="$pid"
    for key in FASTDDS_DEFAULT_PROFILES_FILE ROS_AUTOMATIC_DISCOVERY_RANGE \
               ROS_STATIC_PEERS ROS_DISCOVERY_SERVER ROS_LOCALHOST_ONLY \
               FASTRTPS_DEFAULT_PROFILES_FILE RMW_IMPLEMENTATION; do
      value="$(read_launch_env "$pid" "$key")"
      if [[ -n "$value" ]]; then export "$key=$value"; else unset "$key"; fi
    done
    break
  fi
done

# This script is for Isaac visualization, so pre-Play startup must already use sim time.
# If an Isaac launch is present only on another domain, default to wall time for the selected graph.
SIM="${RVIZ_USE_SIM_TIME:-true}"
if [[ "$SIM" != true && "$SIM" != false ]]; then
  echo "RVIZ_USE_SIM_TIME must be true or false (got: $SIM)" >&2
  exit 2
fi
if [[ -z "${RVIZ_USE_SIM_TIME+x}" && "$ISAAC_FOUND" == true && "$MATCHING_LAUNCH" == false ]]; then
  SIM=false
fi

echo "Isaac launch pid ${LAUNCH_PID:-none}, ROS_DOMAIN_ID=$TARGET_DOMAIN, DDS profile=${FASTDDS_DEFAULT_PROFILES_FILE:-unset}, use_sim_time=$SIM"
RVIZ_CONFIG=../../isaac/loops/vis/odom_nvblox \
  exec "$HERE/../../../tools/rviz/start.sh" "$@" --ros-args -p "use_sim_time:=$SIM"
