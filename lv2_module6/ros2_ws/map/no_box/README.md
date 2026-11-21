# no_box — 상자를 뺀 지도 (11-21)

`maze_map_1711`과 같은 장면·원점·크기(104×142 @ 0.05 m, origin −4.080, −5.759)에서 시작 상자 1~6 · 장소 상자 L1~L3만 뺀 것.
상자는 카메라(costmap 장애물 층)로 본 실제 자리에만 장애물로 남음 — 도면과 다르게 놓인 상자가 지도 상자와 겹쳐 길이 좁아지던 것(11-21 실기 S1 12 cm)을 없앰.

    ~/pa-course/.venv-isaacsim/bin/python isaac/mission/make_map.py --out ros2_ws/map/no_box   # --boxes 없이

- 실기: `./start.sh map_name:=no_box`
- 시뮬: `ISAAC_MAP_NAME=no_box isaac/standalone/isaac_mission.sh`
- 주의: 상자 높이 약 7 cm — local costmap `mark_threshold 2`(높이 칸 3개 = 9 cm)면 빈 상자를 장애물로 못 칠할 수 있음. 시뮬에서 먼저 확인
