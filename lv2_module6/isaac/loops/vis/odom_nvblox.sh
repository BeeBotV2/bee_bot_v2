#!/usr/bin/env bash
# odom_nvblox.rviz via tools/rviz/start.sh — same ROS env checks + record.sh bag (~/bee_bags/nav_*) as the operator RViz.
#   ./odom_nvblox.sh                 # RVIZ_RECORD=0 ./odom_nvblox.sh to skip the bag
#   ./odom_nvblox.sh 8               # first arg = ROS_DOMAIN_ID (numeric); rest goes to rviz2
# ROS env (domain, Fast DDS profile) is copied from the running isaac_system launch so RViz joins the same graph
# (isaac_sim.sh falls back to fastdds_local_only.xml when the terminal has no profile). use_sim_time follows the Isaac pid file.
HERE="$(cd "$(dirname "$0")" && pwd)"
LAUNCH_PID=$(pgrep -f -n "bin/ros2 launch .*isaac_system.launch.py" || true)
if [ -n "$LAUNCH_PID" ]; then
  for v in ROS_DOMAIN_ID FASTDDS_DEFAULT_PROFILES_FILE ROS_AUTOMATIC_DISCOVERY_RANGE; do
    val=$(tr '\0' '\n' < "/proc/$LAUNCH_PID/environ" | sed -n "s/^$v=//p")
    if [ -n "$val" ]; then export "$v=$val"; else unset "$v"; fi
  done
fi
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then export ROS_DOMAIN_ID="$1"; shift; fi
SIM=false
SIM_PID=$(cat /tmp/beebot_isaac_sim.pid 2>/dev/null || true)
[ -n "$SIM_PID" ] && kill -0 "$SIM_PID" 2>/dev/null && SIM=true
echo "isaac_system launch pid ${LAUNCH_PID:-none}, FASTDDS_DEFAULT_PROFILES_FILE=${FASTDDS_DEFAULT_PROFILES_FILE:-unset}, use_sim_time=$SIM"
RVIZ_CONFIG=../../isaac/loops/vis/odom_nvblox exec "$HERE/../../../tools/rviz/start.sh" "$@" --ros-args -p use_sim_time:=$SIM
