# `run-cli` 명령 및 로봇 실행 절차

## 모니터 명령

```bash
cd lv2_module6/tools
./run-cli                         # 1초 간격 상태 누적 출력, Ctrl-C 종료
./run-cli monitor --interval 2    # 출력 간격 변경
./run-cli monitor --latency-window 12
./run-cli status                  # 한 번 출력
./run-cli status --diagnostics    # 진단 추가
./run-cli mission status          # 미션 상태만 출력
```

`run-cli`는 ROS bringup을 시작하거나 로봇에 명령을 보내지 않습니다. 모니터는
센서·위치추정·명령 스트림의 나이/수신률/DDS 수송 지연, `/cmd_vel`과 `/odom`의
상관 기반 응답 지연 추정, 미션/skill 상태 및 활성 Nav2 controller/planner 정책을
표시합니다. 응답 지연은 실제 바퀴 신호의 직접 계측값이 아니라 명령과 odometry의
시간 상관 추정치입니다.

## Bringup 개별 실행

```bash
cd lv2_module6/ros2_ws
./start.sh                         # 실기 기본 설정
./start.sh mock_hardware:=true     # 하드웨어 없이 launch
./start.sh --build                 # 빌드 후 launch
./stop.sh                          # 이 호스트의 ROS 종료
./rebuild.sh [--packages-select PACKAGE]
```

저수준 실행이 필요할 때는 환경을 source한 뒤 직접 launch합니다.

```bash
cd lv2_module6/ros2_ws
source /opt/ros/lyrical/setup.bash
[ -f "$HOME/isaac_ros_ws/install/setup.bash" ] && source "$HOME/isaac_ros_ws/install/setup.bash"
source install/setup.bash
export ROS_DOMAIN_ID=88
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTDDS_DEFAULT_PROFILES_FILE="$PWD/fastdds.xml"
ros2 launch bringup bringup.launch.py map:=saved map_name:=scene_sim
```

`start.sh`는 위 launch 외에 장치 확인, serial autodetect, DDS 기본값 설정을
수행합니다. 실기 외부에서 실행할 때는 장치 경로와 ROS domain을 확인하세요.

## 시뮬레이션 및 반복 시험

```bash
cd lv2_module6/tools
./run-cli --source fake status
./run-cli --source docker --debug-dir ../debug status
```

fake/docker 경로는 테스트·시뮬레이션용입니다. `--sim`은 Docker SIL viewer를
명시적으로 띄울 때만 사용합니다. viewer와 fake-world 인프라는 반복 시험에
사용되므로 유지합니다.

## 제거된 명령 인벤토리

아래는 TUI/직접 조작 CLI에서 제거한 예전 명령입니다. 실행 가능한 인터페이스가
아니며, 로봇 조작은 별도 ROS 명령 또는 전용 도구에서 수행합니다.

| 구 명령 | 예전 용도 |
| --- | --- |
| `menu` 및 TUI 화면 | 상태, 장치, 제어, 지도, 노드, RViz, calibration, topic/bag, 미션 화면 |
| `mode`, `estop`, `collision` | 조작 모드/비상정지/collision monitor 변경 |
| `drive`, `teleop` | 바퀴 직접 조작 |
| `arm`, `gripper`, `beep`, `recover` | 팔·그리퍼·부저·모터 복구 조작 |
| `mission start/abort/skill/stop` | 미션/개별 skill 직접 실행 및 중지 |
| `map mode/switch/save/delete/rename` | 지도 관리 |
| `node restart/activate`, `restart` | 노드 수명주기 및 bringup 관리 |
| `bag start/stop`, `route`, `calibrate wheel` | 데이터 녹화·수동 경로/구동 캘리브레이션 |

기존 상세 구현 이력은 Git 기록에서 확인할 수 있습니다. 위 동작이 필요한 경우
기존 `run-cli`를 되살리지 말고, 해당 작업에 맞는 별도 ROS/시험 명령을 사용하세요.
