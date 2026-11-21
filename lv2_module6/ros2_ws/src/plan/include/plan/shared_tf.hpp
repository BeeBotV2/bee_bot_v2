// 같은 프로세스의 plan 노드(coordinator · mission_manager)가 TF 버퍼·리스너 하나를 같이 씀 (11-16 메모리)
//   리스너 하나 = /tf·/tf_static 구독 둘 + 10 s 버퍼 + 내부 노드·스레드 — mission_nodes 컨테이너(bringup core_composition)에
//   둘을 올리면 하나로. 따로 프로세스(실행 파일)면 지금처럼 노드마다 하나.
//   열쇠 = (context, 시계 종류) — 시뮬 시간 노드와 벽시계 노드가 버퍼를 섞지 않게. 마지막 노드가 내려놓으면 함께 사라짐
//   (skills ArmStateCache와 같은 방식)
#pragma once

#include <memory>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace plan {

std::pair<std::shared_ptr<tf2_ros::Buffer>, std::shared_ptr<tf2_ros::TransformListener>> shared_tf(rclcpp::Node & node);

// 기억한 자리 (11-21): map 점과 같이 그때 odom 점 · 로봇 odom 자리를 적어 둠. 로봇이 그 자리에 서 있는 동안(도킹 중)은 odom 점을
//   지금 map→odom으로 옮겨 씀 — 서 있는데 벽 정합이 map→odom을 흔들면(11-21 실기 S1 10 s에 3~4 cm · S4 yaw 2°) map 점은 로봇에서
//   그만큼 어긋남. 로봇이 움직인 뒤(BUILD 때 다시 옴)는 odom이 바퀴 미끄러짐만큼 틀어지므로 적은 map 점 그대로
struct Spot {
  double x = 0, y = 0;    // map
  double ox = 0, oy = 0;  // odom
  double rx = 0, ry = 0;  // 그때 로봇 (odom)
  bool odom = false;      // TF가 있어 odom 점을 적었음
};
Spot make_spot(const tf2_ros::Buffer & tf, double x, double y);
// 지금 쓸 map 점 — 로봇이 적은 자리에서 still_m 안이면 odom 점을 옮기고, 아니면(또는 TF 없음) 적은 map 점
void spot_map(const tf2_ros::Buffer & tf, const Spot & s, double & x, double & y, double still_m = 0.05);

}  // namespace plan
