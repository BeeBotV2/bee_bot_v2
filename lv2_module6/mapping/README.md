# mapping — 매핑(map:=slam) 시험

| 파일 | 내용 |
| --- | --- |
| `mapping.sh` | Isaac 매핑 · 저장 지도 주행 · bag 녹화/재생. `--real`이면 실기 Jetson 스택에 붙음 (`sed -n 2,26p mapping/mapping.sh`로 하위 명령 보기) |
| `scene_sim_nopuck.usda` | `isaac/usda/scene_sim.usda`를 sublayer로 불러와 `/World/pucks` · `/World/sign`(신호등)을 끈 장면 |
| `bag_retime.py` | 옛 Isaac bag의 시각을 메시지 header 시각으로 다시 씀 (`--use-sim-time` 없이 녹화한 것) |
| `replay.launch.py` | bag 재생용 SLAM만 (robot_state_publisher · EKF · cuVSLAM · nvblox, 시뮬 시계) — `mapping.sh replay`가 씀 |
| `mapping.rviz` | 매핑 전용 RViz: `/map` · nvblox 원본 격자 · EKF 궤적 · 앞 카메라 점구름 · TF(map·odom·base_footprint) |

## 설치

### PC (x86 · Isaac Sim · Ubuntu 24.04 · ROS 2 Lyrical)

ROS 패키지 전체 목록은 팀 Dockerfile(`docker/lv2_module6/Dockerfile`)과 같다. 아래는 그중 호스트에 따로 깔아야 했던 것과 매핑 · bag에 쓰는 것이다.

1. ROS 패키지 (빌드 · Nav2 플러그인 · MoveIt · 텔레옵 · RViz · bag).
   ```bash
   sudo apt install ros-lyrical-onnxruntime-vendor ros-lyrical-nav2-common ros-lyrical-nav2-dwb-controller ros-lyrical-nav2-mppi-controller ros-lyrical-nav2-regulated-pure-pursuit-controller ros-lyrical-nav2-rotation-shim-controller ros-lyrical-nav2-smac-planner ros-lyrical-moveit-ros-perception ros-lyrical-moveit-ros-visualization ros-lyrical-teleop-twist-keyboard ros-lyrical-rviz2 ros-lyrical-depth-image-proc ros-lyrical-rosbag2-storage-mcap ros-lyrical-rosbag2-compression-zstd
   ```
   `trying to overwrite '/usr/bin/markdown-it'`로 apt가 깨지면 Isaac ROS 저장소의 `python3-markdown-it-py`가 Ubuntu의 `python3-markdown-it`과 부딪힌 것이다. 옛 Ubuntu 패키지만 지우고 이어서 설치한다.
   ```bash
   sudo dpkg -r --force-depends python3-markdown-it && sudo apt --fix-broken install
   ```
2. NVIDIA 저장소 세 개. Isaac ROS(nvblox), CUDA(nvblox가 쓰는 CUDA 13.2), Jetson x86(Isaac ROS가 요구하는 VPI 4 — Isaac ROS 저장소에는 없음).
   ```bash
   curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-isaac-ros.gpg
   echo "deb [signed-by=/usr/share/keyrings/nvidia-isaac-ros.gpg] https://isaac.download.nvidia.com/isaac-ros/release-5.0 noble main" | sudo tee /etc/apt/sources.list.d/nvidia-isaac-ros.list
   cd /tmp && wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && sudo dpkg -i cuda-keyring_1.1-1_all.deb
   curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-jetson.gpg
   echo "deb [arch=amd64 signed-by=/usr/share/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/x86_64/noble r39.2 main" | sudo tee /etc/apt/sources.list.d/nvidia-jetson-x86.list
   sudo apt update
   ```
3. CUDA 13.2 툴킷과 nvblox. 툴킷에는 드라이버가 들어 있지 않으므로 이미 깔린 NVIDIA 드라이버(CUDA 13 지원, 595 이상)는 그대로 둔다. nvblox는 `--no-install-recommends`로 nvblox만 깐다 (메타패키지 `isaac-ros-nvblox`는 detectnet · triton까지 40개 가까이 딸려 옴).
   ```bash
   sudo apt install -y cuda-toolkit-13-2
   sudo apt install -y --no-install-recommends ros-lyrical-nvblox-ros
   ```
4. PyCuVSLAM 17.0.1. x86은 GitHub 릴리스 휠(PyPI에 없음)을 시스템 python3에 깐다. `ros2 launch`가 `/usr/bin/python3`로 돌아 venv에 깔면 launch의 SLAM 검사(`check_slam`)가 못 찾는다.
   ```bash
   gh release download v17.0.1 -R NVlabs/PyCuVSLAM -p "cuvslam-17.0.1+cu13-cp312-abi3-manylinux_2_39_x86_64.whl" -D /tmp
   /usr/bin/python3 -m pip install --user --break-system-packages /tmp/cuvslam-17.0.1+cu13-cp312-abi3-manylinux_2_39_x86_64.whl
   ```
5. cuVSLAM C++ 노드용 헤더. `ros_system` CMake가 `~/cuvslam/src/libs/cuvslam/cuvslam2.h`를 찾는다. 없으면 C++ 노드는 빌드에서 빠지고 파이썬판이 돈다.
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch v17.0.1 https://github.com/NVlabs/PyCuVSLAM.git ~/cuvslam/src
   ```
6. 다시 빌드하고 확인한다.
   ```bash
   ros2_ws/rebuild.sh
   /usr/bin/python3 -c "import cuvslam; print(cuvslam.get_version()[0])"
   ros2 pkg prefix nvblox_ros
   ls ros2_ws/install/ros_system/lib/ros_system/cuvslam_slam_cpp
   ```

### Jetson (Orin Nano · JetPack 7 / L4T R39)

Jetson은 x86용 휠 · apt 패키지를 쓰지 못해 방법이 다르다. 설치 순서는 이미 있는 문서를 따른다.

| 대상 | 방법 | 문서 |
| --- | --- | --- |
| ROS 패키지 | apt (`ros-lyrical-*` 목록) | [SETTING.md](../../documents/jetson-orin-nano/SETTING.md) |
| cuVSLAM | 배포 휠이 `Illegal instruction`으로 죽어 PyCuVSLAM v17.0.1 **소스 빌드** (`--cuda_arch=87`) | [CUVSLAM.md](../../documents/jetson-orin-nano/CUVSLAM.md) 2절 |
| nvblox | Isaac ROS apt의 Lyrical nvblox는 amd64만 있음. `~/isaac_ros_ws` 소스 빌드 또는 Isaac ROS 컨테이너 | CUVSLAM.md 2-1절 |
| bag 녹화 | `sudo apt install ros-lyrical-rosbag2-storage-mcap ros-lyrical-rosbag2-compression-zstd` | — |

cuVSLAM 지도 저장 · 불러오기는 같은 v17.0.1 라이브러리에 있어 새로 깔 것은 없다. 코드를 받은 뒤 Jetson 위에서 `ros2_ws/rebuild.sh`만 다시 한다 (C++ 노드의 CUDA 아키텍처는 `native` = Orin 87). nvblox가 없으면 Jetson에서는 `map:=slam` 매핑이 안 되므로, 녹화만 하고(`mapping.sh record --real`) 매핑은 PC에서 재생으로 한다. 저장 지도 주행(`map:=saved`, 기본 `static_2d`)은 nvblox 없이 된다.

## 결과물

매핑을 끝내면(끌 때) 지도 폴더 `ros2_ws/map/<이름>/` 하나에 세 가지가 저장된다.

| 파일 | 쓰는 노드 | 내용 · 주행에서 쓰는 곳 |
| --- | --- | --- |
| `map.pgm` · `map.yaml` | nvblox_map | 끌 때의 `/map` (nvblox + plan.yaml 벽). 저장 지도 주행 기본 `map_backend:=static_2d`가 Nav2 `/map`으로 읽는 길찾기 지도 |
| `cuvslam/` | cuvslam_slam | 랜드마크 · pose graph. 주행 때 이 지도 안에서 로봇 자리를 찾아 map→odom을 맞춤 |
| `nvblox.nvblx.png` · `.yaml` | nvblox_node | nvblox 5.0(apt)이 끌 때 쓰는 2D 점유 격자 (`after_shutdown_map_save_path`). 3D `nvblox.nvblx`는 생기지 않는다 (11-19 확인) — 그래서 이 폴더는 state_manager 지도 목록(`/state/map_list`, `nvblox.nvblx`가 있는 폴더)에 안 뜨고 `map_backend:=nvblox` 위치추정도 못 쓴다. 기본 `static_2d`(`map.yaml`)로 주행한다 |

세 개는 원점(map 프레임)이 같다 — 매핑을 시작한 자리. 저장 지도로 띄우면(`map:=saved`) cuVSLAM이 `cuvslam/` 안에서 지금 자리를 찾고, 찾으면 로그에 `저장 지도 안에서 위치 찾음`이 뜬다. 그때부터 map 원점은 매핑 출발 자리라서 출발 자리가 조금 달라도 된다. 찾는 범위는 추측 자세 둘레 1.5 m(`localize_horizontal_radius`), 못 찾는 동안은 2초(`localize_retry_s`)마다 다시 시도하며 켠 자리를 원점으로 쓴다. `cuvslam/`이 없는 옛 지도 폴더(예: `plan_to_map.py`로 그린 것)는 예전처럼 켠 자리 = 원점이다.

끄지 않고 cuVSLAM 지도만 먼저 저장하려면 `ros2 service call /cuvslam_slam/save_map std_srvs/srv/Trigger`. 같은 이름의 지도가 있으면 매핑 시작을 거절하므로 폴더를 지우거나 다른 이름을 준다.

## 실행 — Sim (PC · Isaac)

모든 명령은 `lv2_module6/`에서 실행한다. 터미널마다 Isaac과 같은 Fast DDS 프로파일(`ros2_ws/fastdds_local_only.xml`)을 `mapping.sh`가 알아서 맞춘다.

### 1. 매핑

터미널 3개를 순서대로 연다.

```bash
mapping/mapping.sh          # 터미널 1: Isaac + bringup map:=slam → ros2_ws/map/scene_sim/
mapping/mapping.sh rviz     # 터미널 2: nvblox raw 표시를 켜면 SLAM이 본 것만
mapping/mapping.sh teleop   # 터미널 3: i 앞 · , 뒤 · j/l 회전 · k 정지 · q/z 속도
```

천천히 한 바퀴 돌아 출발 자리로 돌아온 뒤(루프 클로저) 터미널 1에서 Ctrl+C. 로그에 `cuVSLAM 지도 저장` · `map_server 지도 저장`이 뜨면 끝이다. 다른 이름은 `BEE_MAP_NAME=scene_sim_2 mapping/mapping.sh`.

### 2. 저장 지도로 주행 (위치추정)

```bash
mapping/mapping.sh saved scene_sim   # 터미널 1: map:=saved
mapping/mapping.sh rviz              # 터미널 2
mapping/mapping.sh teleop            # 터미널 3
```

`저장 지도 안에서 위치 찾음`이 뜨고 RViz에서 로봇이 `/map` 벽과 맞으면 성공. `위치 못 찾음`만 반복되면 실패다.

### 3. bag으로 (실기 순서 연습)

실기와 같은 순서 — 녹화는 SLAM 없이, 매핑 · 위치추정은 나중에 재생으로.

```bash
mapping/mapping.sh drive                      # 터미널 1: Isaac + map:=none (주행만)
mapping/mapping.sh record lap1                # 터미널 2: 녹화 → ~/.ros/beebot_bags/lap1 (Ctrl+C로 끝)
mapping/mapping.sh teleop                     # 터미널 3: 한 바퀴
```

위치추정 시험용으로 다른 출발 자리에서 `lap2`를 짧게 하나 더 녹화한다 (`isaac/standalone/runtime/reset.sh` 뒤 텔레옵으로 옮긴 다음). 그다음 Isaac을 끄고 재생한다.

```bash
mapping/mapping.sh replay ~/.ros/beebot_bags/lap1          # 매핑 → ros2_ws/map/lap1/
mapping/mapping.sh localize ~/.ros/beebot_bags/lap2 lap1   # lap1 지도에서 위치 찾기
```

재생 중 RViz는 `mapping/mapping.sh rviz` (bag 시계). 재생은 같은 ROS 도메인에 다른 노드가 있으면 거절한다 (TF가 섞임).

Isaac 녹화는 `--use-sim-time`으로 bag 시각도 시뮬 시계로 찍는다 (`mapping.sh record`가 붙임). 이게 없던 옛 bag은 재생하면 `/clock`(녹화 때 PC 시각)과 메시지 header(시뮬 시계)가 어긋나 odom이 튀고 `map` 프레임이 안 생긴다. 그런 bag은 header 시각으로 다시 써서 쓴다 (원본은 그대로, 압축을 디스크에 풀지 않음).

```bash
python3 mapping/bag_retime.py ~/.ros/beebot_bags/lap1 ~/.ros/beebot_bags/lap1_sim
```

## 실행 — 실기 (Jetson + PC)

Jetson은 센서 · 바퀴 스택만 띄우고, 녹화 · 매핑은 PC에서 한다. Jetson에 nvblox가 없어도 되고 Jetson 부하도 그대로다. PC 명령은 `--real`을 붙이면 Jetson과 같은 `ros2_ws/fastdds.xml`(SHM + UDP)로 붙는다 (`drive_test.py --real`과 같음).

### 1. 녹화

1. 경기장에서 퍽과 신호등을 치우고 출발 자리에 로봇을 둔다.
2. Jetson: SLAM · Nav2 없이 띄운다. 텔레옵을 받으려면 MANUAL이어야 한다 (bringup 기본은 AUTO).
   ```bash
   ./ros2_ws/start.sh map:=none operation:=MANUAL
   ```
3. PC: 터미널 3개.
   ```bash
   mapping/mapping.sh rviz --real
   mapping/mapping.sh record --real lap1   # ~/.ros/beebot_bags/lap1
   mapping/mapping.sh teleop --real
   ```
   출발 자리에서 몇 초 서 있다가 천천히 한 바퀴 돌아 출발 자리로 돌아온다. 녹화 터미널에서 Ctrl+C.
4. 위치추정 시험용으로 다른 출발 자리에서 `lap2`를 짧게 하나 더 녹화한다.
5. 영상이 빠지지 않았는지 본다. 15 Hz × 녹화 초 수에 가까워야 한다.
   ```bash
   ros2 bag info ~/.ros/beebot_bags/lap1
   ```

PC 녹화는 Jetson → PC 네트워크로 영상을 받는다. pole_front_cam 640×360 15 Hz면 컬러 · depth 합쳐 초당 약 17 MB(약 140 Mbps)다. Wi-Fi에서는 프레임이 빠질 수 있으니 유선을 권한다. 5번 메시지 수가 크게 모자라면 Jetson에서 직접 녹화하고(Jetson에서도 `mapping/mapping.sh record --real lap1` — `--real` = start.sh와 같은 `fastdds.xml`) bag 폴더를 PC로 옮긴다.

### 2. 매핑 (PC에서 재생)

```bash
mapping/mapping.sh replay ~/.ros/beebot_bags/lap1           # → ros2_ws/map/lap1/
mapping/mapping.sh localize ~/.ros/beebot_bags/lap2 lap1    # 그 지도에서 위치 찾기 시험
```

### 3. 지도를 Jetson으로 옮겨 주행

```bash
scp -r ros2_ws/map/lap1 <jetson>:<레포>/lv2_module6/ros2_ws/map/
```

Jetson:

```bash
./ros2_ws/start.sh map:=saved map_name:=lap1
```

Jetson도 이 저장/불러오기 코드로 다시 빌드돼 있어야 `cuvslam/`을 읽는다. PC(x86)에서 만든 `cuvslam/`을 Jetson(aarch64)이 그대로 읽는지는 아직 확인 전이다 — 같은 cuVSLAM 17.0.1이지만 시험해 보고, 안 되면 `map.yaml`만 쓰고 출발 자리를 매핑 때와 맞춘다 (예전 방식).

### (Jetson에 nvblox가 있을 때만) 실기에서 바로 매핑

```bash
mapping/mapping.sh --real          # /cmd/map_mode MAPPING → state_manager가 map:=slam으로 다시 띄움
mapping/mapping.sh rviz --real
mapping/mapping.sh teleop --real
mapping/mapping.sh save --real     # /cmd/map_mode LOCALIZATION → 다시 띄우며 저장 · 그 지도로 위치추정
```

### 녹화 토픽

SLAM 입력만: pole_front_cam 컬러 · 정렬 depth · camera_info 둘, `/odom`, `/imu`, `/joint_states`. `/tf`는 넣지 않는다 — 녹화 때의 odom→base가 재생 때 다시 돌리는 EKF(`replay.launch.py`)와 부딪힌다.

## 퍽 · 신호등을 빼는 이유

퍽은 미션 중에 옮겨진다. 저장 지도에 굳으면 집어 간 뒤에도 없는 장애물로 남아 Nav2 경로와 목표 자리를 막는다 (`mission/make_map.py`가 장면 지도에서 퍽을 빼는 것과 같은 이유). 실기 매핑도 경기장에서 퍽을 치우고 한다.

신호등도 뺀다. 실기 매핑 때 경기장에서 치우는 것과 맞추고, 깜빡이는 등은 켜질 때마다 모양이 바뀌어 cuVSLAM 랜드마크로도 나쁘다. 지도에 없으므로 미션 중 신호등 앞 주행은 local costmap(카메라 점구름)이 피한다.

시작 박스 · 장소 상자는 장면에 남긴다. 미션 동안 고정이라 지도에 들어가도 된다. 다만 `make_map.py`는 다가가 붙는 대상이라 정적 지도에서 뺀다. 저장 지도로 주행할 때 상자 앞 목표 자리가 막히면 이 차이를 먼저 의심한다.

## 볼 것

- `/map`은 nvblox_map이 plan.yaml 벽(`/arena/walls`) 위에 nvblox를 겹친 것이다. SLAM이 실제로 본 것만 보려면 RViz의 `nvblox raw` 표시를 켠다.
- 지도 원점(map 프레임)은 매핑을 시작한 자리다. `cuvslam/`이 있는 지도는 출발 자리가 1.5 m 안에서 달라도 위치를 찾지만, 없는 지도는 매핑 때와 같은 자리 · 방향에서 켜야 한다.
- 매핑은 천천히 돈다. 텔레옵 시작 속도는 0.15 m/s · 0.6 rad/s (`BEE_TELEOP_SPEED` · `BEE_TELEOP_TURN`).
