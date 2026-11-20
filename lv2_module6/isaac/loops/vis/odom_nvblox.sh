#!/usr/bin/env bash
# Isaac odom/nvblox RViz entry point, through tools/rviz/start.sh (including optional bag recording).
#   ./odom_nvblox.sh                 # simulation time on by default; may start before Isaac Play
#   ./odom_nvblox.sh 42              # first arg = ROS_DOMAIN_ID; remaining args go to rviz2 (default: domain of the running Isaac launch)
#   RVIZ_USE_SIM_TIME=false ./odom_nvblox.sh  # wall time for a real-robot view
#   RVIZ_RECORD=0 ./odom_nvblox.sh   # skip bag recording
# When an Isaac launch in the selected ROS domain is running, borrow its DDS settings so both join the same graph.
# Do not use the Isaac PID file to decide simulation time: it is created only after Play and ROS launch startup.
HERE="$(cd "$(dirname "$0")" && pwd)"

DOMAIN_ARG=false
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  export ROS_DOMAIN_ID="$1"
  DOMAIN_ARG=true
  shift
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
# No domain argument: follow the newest Isaac launch, whatever this terminal's ROS_DOMAIN_ID is
#   (11-20: Isaac on 77 via bee_sim_tools, terminal on the bashrc domain -> RViz joined an empty graph with wall time).
PIDS=$(pgrep -f "bin/ros2 launch .*isaac_system.launch.py" 2>/dev/null | sort -rn || true)
NEWEST=$(head -n 1 <<<"$PIDS")
if [[ "$DOMAIN_ARG" == false && -n "$NEWEST" ]]; then
  TARGET_DOMAIN="$(read_launch_env "$NEWEST" ROS_DOMAIN_ID)"
  TARGET_DOMAIN="${TARGET_DOMAIN:-0}"
  export ROS_DOMAIN_ID="$TARGET_DOMAIN"
fi
for pid in $PIDS; do
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

[[ "$ISAAC_FOUND" == false ]] && echo "WARN: Isaac launch(isaac_system.launch.py) 안 보임 — Play 전이면 Play 뒤 RViz가 토픽을 받음 · 다른 도메인이면 ./odom_nvblox.sh <도메인>" >&2
echo "Isaac launch pid ${LAUNCH_PID:-none}, ROS_DOMAIN_ID=$TARGET_DOMAIN, DDS profile=${FASTDDS_DEFAULT_PROFILES_FILE:-unset}, use_sim_time=$SIM"
RVIZ_CONFIG=../../isaac/loops/vis/odom_nvblox \
  exec "$HERE/../../../tools/rviz/start.sh" "$@" --ros-args -p "use_sim_time:=$SIM"
