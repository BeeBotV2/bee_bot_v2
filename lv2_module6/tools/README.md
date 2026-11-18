# BeeBot tools

이 디렉터리에는 실기 bringup 도구, 읽기 전용 주행 모니터, 시뮬레이션 및 반복
시험 도구가 있습니다. TUI와 `run-cli`를 통한 직접 로봇 조작은 제거했습니다.

## 실기 주행

```bash
cd lv2_module6/ros2_ws
./start.sh
```

다른 터미널에서 모니터를 실행합니다.

```bash
cd lv2_module6/tools
./run-cli
```

`run-cli`는 상태만 읽습니다. bringup을 시작하거나 정지 명령을 내리지 않습니다.
모니터 지표 의미, 단발 출력 및 파라미터, 별도 bringup 명령은
[RUN-CLI-COMMANDS.md](RUN-CLI-COMMANDS.md)에 정리했습니다.

## 반복 시험

```bash
cd lv2_module6/tools
./run-cli --source fake status
./run-cli --source docker --debug-dir ../debug status
```

fake-world와 Docker SIL 경로는 시뮬레이션 및 지속적 시험을 위해 유지합니다.
Docker viewer는 명시적으로 `--sim`을 지정할 때 실행됩니다.

## 디렉터리 안내

- `cli/op/monitor.py`, `state.py`: 읽기 전용 상태 수집과 텍스트 모니터
- `cli/ros/fake`, `cli/ros/docker`: 시뮬레이션·test-faker 시험 계층
- `cli/tests`: 모니터, fake-world 및 CLI 상태 시험
- `rviz/`, `foxglove/`: 개발·현장 시각화 설정
- `plan/`: 미션 계획 도구

로봇과 관련된 ROS launch, 파라미터, 하드웨어 코드는 `../ros2_ws`에 있습니다.
