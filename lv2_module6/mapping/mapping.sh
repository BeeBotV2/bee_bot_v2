#!/usr/bin/env bash
# 매핑(map:=slam) 시험 — Isaac(scene_sim, 퍽 · 신호등 없음) 또는 실기. 텔레옵은 teleop_twist_keyboard, RViz는 mapping.rviz.
#
# Isaac (터미널 3개 — 순서대로):
#   mapping/mapping.sh              # 1. Isaac + bringup map:=slam · 지도 폴더 ros2_ws/map/scene_sim/ (Isaac 창을 닫거나 Ctrl+C로 끄면 저장)
#   mapping/mapping.sh rviz         # 2. 매핑 RViz
#   mapping/mapping.sh teleop       # 3. 키보드 주행 (i 앞 · , 뒤 · j/l 회전 · k 정지)
#   BEE_MAP_NAME=scene_sim_2 mapping/mapping.sh   # 다른 폴더 이름으로 (같은 이름이 있으면 거절)
#   mapping/mapping.sh saved [지도]  # 저장 지도로 다시 띄움 (map:=saved · 기본 scene_sim) — cuVSLAM이 그 지도에서 자리를 찾음
#
# 실기 (PC에서 — Jetson에 ./ros2_ws/start.sh로 스택이 떠 있어야 함 · drive_test.py --real과 같은 fastdds.xml):
#   mapping/mapping.sh --real        # /cmd/map_mode MAPPING → state_manager가 스택을 map:=slam으로 다시 띄움 (새 폴더 map/<날짜_시각>/)
#   mapping/mapping.sh rviz --real
#   mapping/mapping.sh teleop --real
#   mapping/mapping.sh save --real   # /cmd/map_mode LOCALIZATION → 다시 띄우며 nvblox가 끌 때 저장 · 그 지도로 위치추정
#   (Jetson에서 직접: ./ros2_ws/start.sh map:=slam  → 끝나면 ./ros2_ws/stop.sh가 저장)
#
# bag으로 (실기에서 하는 것처럼 — 녹화는 SLAM 없이, 매핑 · 위치추정은 나중에 PC에서 재생):
#   mapping/mapping.sh drive                 # Isaac + bringup map:=none (SLAM · Nav2 없이 주행만 — 실기 녹화 때와 같은 상태)
#   mapping/mapping.sh record [이름]         # 다른 터미널: SLAM 입력 토픽 녹화 → $BAG_DIR/<이름>/ (Ctrl+C로 끝)
#   mapping/mapping.sh rviz drive            # 녹화 중 보기 — 기준 odom (map 프레임 없음): 점구름 · 궤적 · 로봇
#   mapping/mapping.sh record --real [이름]  # 실기: PC에서 Jetson 토픽을 녹화 (Jetson은 ./ros2_ws/start.sh map:=none operation:=MANUAL)
#   mapping/mapping.sh replay <bag>          # bag 재생 → 매핑 · 지도 폴더 ros2_ws/map/<bag 이름>/ (BEE_MAP_NAME으로 바꿈) · 끝나면 저장
#   mapping/mapping.sh localize <bag> <지도> # bag 재생 → 저장 지도 <지도>에서 위치 찾기 (로그 '저장 지도 안에서 위치 찾음')
#   bag 위치 BEE_BAG_DIR (기본 ~/.ros/beebot_bags) · 재생 중 RViz는 mapping.sh rviz (시뮬 시계 = bag 시계)
#
#   BEE_TELEOP_SPEED=0.15 BEE_TELEOP_TURN=0.6   # 텔레옵 시작 속도 [m/s · rad/s] (키 q/z로 바꿈) · 매핑은 천천히 돌아야 cuVSLAM이 안 놓침
HERE=$(dirname "$(realpath "$0")")
WS=$(realpath -m "$HERE/../ros2_ws")
REAL=false
ARGS=()
for a in "$@"; do [ "$a" = --real ] && REAL=true || ARGS+=("$a"); done
CMD=${ARGS[0]:-map}
BAG_DIR=$(realpath -m "${BEE_BAG_DIR:-$HOME/.ros/beebot_bags}")
# SLAM 입력 (cuVSLAM · nvblox · EKF · 카메라 TF) — /tf는 안 넣음: 재생할 때 EKF · robot_state_publisher가 다시 냄 (replay.launch.py)
BAG_TOPICS="/pole_front_cam/color/image_raw /pole_front_cam/color/camera_info
  /pole_front_cam/aligned_depth_to_color/image_raw /pole_front_cam/aligned_depth_to_color/camera_info /odom /imu /joint_states"

# 다른 터미널(RViz · 텔레옵 · CLI)은 Isaac · 스택과 같은 Fast DDS 프로파일이어야 토픽이 보인다 (isaac_sim.sh · drive_test.py와 같은 기본)
ros_env() {
  if $REAL; then PROFILE=$WS/fastdds.xml; SIM=false; else PROFILE=$WS/fastdds_local_only.xml; SIM=true; fi
  export FASTDDS_DEFAULT_PROFILES_FILE="${FASTDDS_DEFAULT_PROFILES_FILE:-$PROFILE}" ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
  source /opt/ros/lyrical/setup.bash
  # 워크스페이스도 — RViz RobotModel이 package://ros_system/urdf/.../meshes를 찾음 (없으면 메시마다 "Could not load resource")
  [ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash"
}

map_mode() {  # state_manager에 지도 모드 요청 (실기) — 결과는 /state/map_mode
  ros_env
  echo "[mapping] /cmd/map_mode $1 → state_manager가 bringup을 다시 띄움 (수십 초)"
  ros2 topic pub --once -w 1 --qos-reliability reliable /cmd/map_mode std_msgs/msg/String "{data: $1}"
  echo "[mapping] 확인: ros2 topic echo --once --qos-durability transient_local /state/map_mode"
}

need_build() {
  # isaac_sim.sh는 install이 없어도 Isaac을 띄우고 ROS launch만 조용히 죽는다 ("package 'bringup' not found") — 먼저 막음
  #   setup.bash는 빌드가 중간에 실패해도 생기므로 bringup 패키지로 확인
  [ -d "$WS/install/bringup" ] || { echo "ros2_ws 빌드 안 됨 (install/bringup 없음) — 먼저 $WS/rebuild.sh"; exit 1; }
}

isaac_up() {  # $1 = 지도 인자 (map:=slam … · map:=none) — 나머지는 isaac_mission.sh map과 같음
  need_build
  # operation MANUAL(텔레옵) · arena_map:=none = /map 벽은 실기처럼 /arena/walls(plan.yaml)
  export ISAAC_ROS_ARGS="$1 use_pole_back_cam:=false operation:=MANUAL use_perception:=auto arena_map:=none ${ISAAC_EXTRA_ROS_ARGS:-}"
  export BEE_BACK_CAM=false  # camera_adapter도 앞 · 손 카메라만 (실기 start.sh 기본과 같게)
  export ISAAC_AUTOPLAY="${ISAAC_AUTOPLAY:-1}"
  echo "[mapping] 장면 scene_sim(퍽 · 신호등 없음) · ROS 인자: $ISAAC_ROS_ARGS"
  # cm_kick: controller_manager가 /robot_description을 못 받고 멈추면 다시 보냄 (isaac_mission.sh와 같음)
  STANDALONE=$(realpath "$HERE/../isaac/standalone")
  (ros_env; /usr/bin/python3 "$STANDALONE/runtime/cm_kick.py") &
  EXTRA=$!
  cleanup() { trap - EXIT INT TERM HUP; kill $EXTRA 2>/dev/null; pkill -f "$STANDALONE/runtime/cm_kick.py"; }
  trap cleanup EXIT INT TERM HUP
  "$STANDALONE/isaac_sim.sh" "$HERE/scene_sim_nopuck.usda"
}

replay() {  # $1 = bag · $2 = 지도 이름 · $3 = localization true|false
  need_build
  [ -e "$1/metadata.yaml" ] || { echo "bag 아님: $1 (metadata.yaml 없음)"; exit 1; }
  ros_env
  source "$WS/install/setup.bash"
  # 재생 SLAM은 Isaac · 실기 스택과 같은 도메인에 있으면 TF가 섞인다 — 혼자 있을 때만
  OTHERS=$(ros2 node list --no-daemon --spin-time 3 2>/dev/null | grep -vE '^/(rviz|transform_listener_impl_|_ros2cli_)')
  [ -n "$OTHERS" ] && { echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}에 다른 노드가 떠 있음 — Isaac · 스택을 끄고 다시:"; echo "$OTHERS" | sed 's/^/  /'; exit 1; }
  # Fast DDS 공유 메모리 찌꺼기 비우기 (isaac_sim.sh와 같음) — local_only 프로파일은 SHM뿐이라, 꺼진 실행이 남긴 포트 잠금이 있으면
  #   새 노드 일부가 서로 못 찾는다 (11-19 replay: EKF가 그래프에 안 보여 odom TF 없음 → map→odom 0). 쓰는 프로세스가 없을 때만 전부 지움
  SHM_USERS=$(for p in /proc/[0-9]*; do grep -qs "/dev/shm/fastdds_" "$p/maps" && echo "${p#/proc/}"; done | paste -sd,)
  if [ -z "$SHM_USERS" ]; then
    rm -f /dev/shm/fastdds_* /dev/shm/sem.fastdds_*
  else
    echo "[mapping] Fast DDS 공유 메모리를 쓰는 프로세스가 있어 다 비우지 않음 (RViz 등을 끄고 재생하면 확실):"
    ps -o pid=,args= -p "$SHM_USERS" | cut -c1-100 | sed 's/^/  /'
    fastdds shm clean >/dev/null 2>&1
  fi
  ros2 daemon stop >/dev/null 2>&1
  setsid ros2 launch "$HERE/replay.launch.py" map_dir:="$WS/map" map_name:="$2" localization:="$3" &
  LAUNCH=$!
  stop() { trap - EXIT INT TERM HUP; kill -INT -- -$LAUNCH 2>/dev/null; wait $LAUNCH 2>/dev/null; }  # SIGINT → cuVSLAM · nvblox 저장
  trap stop EXIT INT TERM HUP
  sleep "${BEE_REPLAY_WAIT:-8}"  # 노드가 뜨고 구독할 때까지 — 앞 프레임을 놓치면 출발 자리가 달라짐
  # EKF가 그래프에 보이고 /odom을 구독하는지 — 안 보이면 odom TF가 없어 지도가 안 생기므로 재생 전에 멈춤
  ros2 topic info -v /odom --no-daemon --spin-time 3 2>/dev/null | grep -q "Node name: ekf_filter_node" \
    || { echo "[mapping] EKF가 /odom을 구독하지 않음 — DDS가 노드를 못 찾음. 모든 ROS 창(RViz 포함)을 끄고 다시"; exit 1; }
  echo "[mapping] 재생: $1 → 지도 $WS/map/$2 ($( [ "$3" = true ] && echo 위치추정 || echo 매핑))"
  ros2 bag play --clock 100 "$1"
  sleep 3  # 마지막 프레임 처리
  echo "[mapping] 재생 끝 — SLAM 끔$( [ "$3" = true ] || echo ' (끌 때 저장)')"
}

case "$CMD" in
  map)
    if $REAL; then map_mode MAPPING; exit; fi
    NAME=${BEE_MAP_NAME:-scene_sim}
    [ -e "$WS/map/$NAME/map.yaml" ] || [ -e "$WS/map/$NAME/cuvslam" ] && { echo "이미 있는 지도: $WS/map/$NAME — 다시 만들려면 그 폴더를 지우거나 BEE_MAP_NAME을 바꿈"; exit 1; }
    isaac_up "map:=slam map_dir:=$WS/map map_name:=$NAME"
    ;;
  saved)
    $REAL && { echo "실기는 Jetson에서 ./ros2_ws/start.sh map:=saved map_name:=<지도>"; exit 1; }
    NAME=${ARGS[1]:-scene_sim}
    [ -f "$WS/map/$NAME/map.yaml" ] || { echo "저장 지도 없음: $WS/map/$NAME/map.yaml — 먼저 mapping.sh로 매핑"; exit 1; }
    isaac_up "map:=saved map_dir:=$WS/map map_name:=$NAME"
    ;;
  drive)
    $REAL && { echo "실기는 Jetson에서 ./ros2_ws/start.sh map:=none"; exit 1; }
    isaac_up "map:=none"
    ;;
  record)
    ros_env
    NAME=${ARGS[1]:-$(date +%Y%m%d_%H%M%S)}
    [ -e "$BAG_DIR/$NAME" ] && { echo "이미 있는 bag: $BAG_DIR/$NAME"; exit 1; }
    mkdir -p "$BAG_DIR"
    echo "[mapping] 녹화 → $BAG_DIR/$NAME (Ctrl+C로 끝) · 출발 자리에서 몇 초 서 있다가 출발"
    # shellcheck disable=SC2086 — 토픽은 공백으로 나눔
    # Isaac: 메시지 시각이 시뮬 시계라 bag 시각도 /clock으로 (아니면 재생 --clock이 녹화 때 PC 시각을 내 EKF · TF가 어긋남 — 11-19 lap1)
    $REAL && SIMT=() || SIMT=(--use-sim-time)
    exec ros2 bag record "${SIMT[@]}" -s mcap --compression-mode file --compression-format zstd -o "$BAG_DIR/$NAME" --topics $BAG_TOPICS  # Lyrical: 토픽은 --topics 뒤 (위치 인자 안 받음)
    ;;
  replay)
    BAG=$(realpath -m "${ARGS[1]:?bag 경로}")
    NAME=${BEE_MAP_NAME:-$(basename "$BAG")}
    replay "$BAG" "$NAME" false
    ;;
  localize)
    BAG=$(realpath -m "${ARGS[1]:?bag 경로}")
    replay "$BAG" "${ARGS[2]:?지도 이름 (ros2_ws/map/<이름>)}" true
    ;;
  save)
    $REAL || { echo "Isaac은 Isaac 창을 닫거나 mapping.sh 터미널에서 Ctrl+C — 끌 때 $WS/map/<이름>/ 에 map.yaml · cuvslam/ · nvblox 저장"; exit 1; }
    map_mode LOCALIZATION
    ;;
  rviz)
    ros_env
    # 녹화(drive · map:=none)에는 map 프레임이 없어 기준을 odom으로 — 그대로 map이면 아무것도 안 보임
    case "${ARGS[1]:-}" in drive|record|odom) FRAME=(-f odom) ;; *) FRAME=() ;; esac
    exec rviz2 -d "$HERE/mapping.rviz" "${FRAME[@]}" --ros-args -p use_sim_time:=$SIM
    ;;
  teleop)
    ros_env
    # manual_control이 TwistStamped를 /cmd/manual/cmd_vel로 받는다 (control/manual_control.hpp) · MANUAL 모드에서만 바퀴로 감
    exec ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args \
      -r cmd_vel:=/cmd/manual/cmd_vel -p stamped:=true -p use_sim_time:=$SIM \
      -p speed:="${BEE_TELEOP_SPEED:-0.15}" -p turn:="${BEE_TELEOP_TURN:-0.6}"
    ;;
  *) sed -n '2,27p' "$0"; exit 1 ;;
esac
