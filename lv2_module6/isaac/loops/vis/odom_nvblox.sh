#!/usr/bin/env bash
# odom_nvblox.rviz via tools/rviz/start.sh — same ROS env checks + record.sh bag (~/bee_bags/nav_*) as the operator RViz.
#   ./odom_nvblox.sh                 # ROS env = this terminal (~/.bashrc: domain, Fast DDS profile, static peer)
#   ./odom_nvblox.sh 42              # first arg = ROS_DOMAIN_ID (numeric) — e.g. a headless Isaac run; rest goes to rviz2
#   RVIZ_RECORD=0 ./odom_nvblox.sh   # no bag
# use_sim_time is on only when a running isaac_system launch uses the same ROS_DOMAIN_ID (an Isaac run on another
#   domain must not make RViz wait for /clock while viewing the real robot). Its Fast DDS profile is borrowed then too.
HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then export ROS_DOMAIN_ID="$1"; shift; fi
SIM=false
for p in $(pgrep -f "bin/ros2 launch .*isaac_system.launch.py" || true); do
  env_of() { tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n "s/^$1=//p"; }
  if [ "$(env_of ROS_DOMAIN_ID)" = "${ROS_DOMAIN_ID:-0}" ]; then
    SIM=true
    prof=$(env_of FASTDDS_DEFAULT_PROFILES_FILE)
    [ -n "$prof" ] && export FASTDDS_DEFAULT_PROFILES_FILE="$prof"
  fi
done
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}, FASTDDS_DEFAULT_PROFILES_FILE=${FASTDDS_DEFAULT_PROFILES_FILE:-unset}, ROS_STATIC_PEERS=${ROS_STATIC_PEERS:-unset}, use_sim_time=$SIM"
RVIZ_CONFIG=../../isaac/loops/vis/odom_nvblox exec "$HERE/../../../tools/rviz/start.sh" "$@" --ros-args -p use_sim_time:=$SIM
