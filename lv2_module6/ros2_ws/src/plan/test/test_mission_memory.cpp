// puck_for — 집기 직전에 퍽 트랙 다시 찾기 (인지 long_memory: false로 안 보이던 퍽이 다른 id로 다시 나타나도 tag·자리로 찾음)
#include <gtest/gtest.h>

#include <cmath>

#include "plan/mission_manager.hpp"

using perception::msg::TrackedObject;
using perception::msg::TrackedObjectArray;

namespace {
TrackedObject puck(uint32_t id, double x, double y, const std::string & tag, int sec = 1) {
  TrackedObject o;
  o.id = id;
  o.class_name = "puck_standing";
  o.pose.pose.position.x = x;
  o.pose.pose.position.y = y;
  o.last_seen.sec = sec;
  if (!tag.empty()) { o.details.push_back(diagnostic_msgs::msg::KeyValue().set__key("tag").set__value(tag)); }
  return o;
}
}  // namespace

TEST(PuckFor, FindsByTagEvenWithNewTrackId) {
  TrackedObjectArray a;
  a.objects = {puck(40, 1.0, 1.0, "3"), puck(41, 1.5, 1.0, "7", 2), puck(42, 1.52, 1.0, "7", 5)};
  const auto * o = plan::puck_for(&a, 7, nullptr, 0.04);
  ASSERT_NE(o, nullptr);
  EXPECT_EQ(o->id, 42u);  // 같은 tag면 가장 최근에 본 트랙
}

TEST(PuckFor, FallsBackToSpotWhenTagUnread) {
  TrackedObjectArray a;
  a.objects = {puck(50, 2.00, 1.00, "5"), puck(51, 2.02, 1.01, "")};
  const double spot[2] = {2.0, 1.0};
  const auto * o = plan::puck_for(&a, 9, spot, 0.04);
  ASSERT_NE(o, nullptr);
  EXPECT_EQ(o->id, 51u);  // 다른 tag(5)를 읽은 퍽은 고르지 않음 — tag 못 읽은 근처 퍽
  EXPECT_EQ(plan::puck_for(&a, 9, nullptr, 0.04), nullptr);  // 자리를 모르면 tag로만
  const double far[2] = {3.0, 1.0};
  EXPECT_EQ(plan::puck_for(&a, 9, far, 0.04), nullptr);
  EXPECT_EQ(plan::puck_for(nullptr, 9, spot, 0.04), nullptr);
}

// Spot — 기억한 자리: 로봇이 서 있는 동안 map→odom이 흔들려도 odom 기준으로 다시 꺼냄 · 움직인 뒤는 적은 map 그대로 (11-21 실기 S1)
namespace {
geometry_msgs::msg::TransformStamped tf_xy(const std::string & parent, const std::string & child, double x, double y, double yaw) {
  geometry_msgs::msg::TransformStamped t;
  t.header.frame_id = parent;
  t.child_frame_id = child;
  t.transform.translation.x = x;
  t.transform.translation.y = y;
  t.transform.rotation.z = std::sin(yaw / 2);
  t.transform.rotation.w = std::cos(yaw / 2);
  return t;
}
}  // namespace

TEST(Spot, FollowsMapOdomWhileRobotStill) {
  auto clock = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);
  tf2_ros::Buffer tf(clock);
  tf.setTransform(tf_xy("map", "odom", 0.0, 0.0, 0.0), "test", true);
  tf.setTransform(tf_xy("odom", "base_footprint", 1.0, 0.0, 0.0), "test", true);
  const auto s = plan::make_spot(tf, 1.2, 0.1);  // 로봇 앞 0.2 m 퍽
  EXPECT_TRUE(s.odom);
  // 서 있는데 벽 정합이 map→odom을 4 cm · 2° 옮김 → 퍽도 로봇과 같이 옮겨 꺼냄
  tf.setTransform(tf_xy("map", "odom", 0.0, -0.04, 0.035), "test", true);
  double x, y;
  plan::spot_map(tf, s, x, y);
  EXPECT_NEAR(x, 1.2 * std::cos(0.035) - 0.1 * std::sin(0.035), 1e-6);
  EXPECT_NEAR(y, -0.04 + 1.2 * std::sin(0.035) + 0.1 * std::cos(0.035), 1e-6);
  // 로봇이 움직임 (BUILD 때 다시 옴) → 적은 map 그대로
  tf.setTransform(tf_xy("odom", "base_footprint", 2.0, 0.0, 0.0), "test", true);
  plan::spot_map(tf, s, x, y);
  EXPECT_DOUBLE_EQ(x, 1.2);
  EXPECT_DOUBLE_EQ(y, 0.1);
}

TEST(Spot, KeepsMapWithoutTf) {
  auto clock = std::make_shared<rclcpp::Clock>(RCL_SYSTEM_TIME);
  tf2_ros::Buffer tf(clock);
  const auto s = plan::make_spot(tf, 0.5, -0.3);
  EXPECT_FALSE(s.odom);
  double x, y;
  plan::spot_map(tf, s, x, y);
  EXPECT_DOUBLE_EQ(x, 0.5);
  EXPECT_DOUBLE_EQ(y, -0.3);
}

// 상자 조사 — 보인 tag 퍽이 할 일인지 (11-21 실기 S6: S4에서 떨어뜨린 퍽 5를 '아는 tag'라 건너뜀)
TEST(BoxTodo, PicksNewAndMovedPucksOnly) {
  EXPECT_FALSE(plan::box_todo(true, true, false));  // 적재함에 있음
  EXPECT_TRUE(plan::box_todo(false, false, false));  // 처음 봄
  EXPECT_FALSE(plan::box_todo(false, true, true));  // 이 상자 퍽으로 이미 적힘
  EXPECT_TRUE(plan::box_todo(false, true, false));  // 다른 상자로 알던 퍽 (S6의 퍽 5)
}

TEST(FreeSpot, NearRobotOnBoxAwayFromOthers) {
  // 로봇 (0, 0) 앞(+x) · 상자 가운데 (0.25, 0.02) 반지름 한도 0.0625 · 바라는 앞 0.21
  const double lim = 0.0625;
  auto p = plan::free_spot(0, 0, 0, 0.25, 0.02, lim, 0.21, {}, 0.05, 0.065);
  EXPECT_NEAR(p.first, 0.21, 1e-9);  // 다른 퍽 없음 → 로봇 앞 0.21 · 옆은 상자 가운데 줄
  EXPECT_NEAR(p.second, 0.02, 1e-9);
  // 그 자리 옆에 퍽 → 옆으로 옮긴 빈 곳 · 상자 윗면 안 · 로봇 쪽(앞 0.21 아래)으로는 안 감
  const std::vector<std::pair<double, double>> one = {{0.215, 0.0}};
  p = plan::free_spot(0, 0, 0, 0.25, 0.02, lim, 0.21, one, 0.05, 0.065);
  EXPECT_GE(std::hypot(p.first - 0.215, p.second - 0.0), 0.065);
  EXPECT_LE(std::hypot(p.first - 0.25, p.second - 0.02), lim + 1e-9);
  EXPECT_GE(p.first, 0.21 - 1e-9);
  // 바라는 점이 상자 밖(상자가 멀리) → 윗면 안으로 당김
  p = plan::free_spot(0, 0, 0, 0.40, 0.0, lim, 0.21, {}, 0.05, 0.065);
  EXPECT_NEAR(p.first, 0.40 - lim, 1e-9);
  // 빈 곳이 없으면 다른 퍽에서 가장 먼 후보 (상자 윗면 안)
  const std::vector<std::pair<double, double>> crowd = {{0.21, 0.02}, {0.23, 0.07}, {0.23, -0.03}, {0.27, 0.02}};
  p = plan::free_spot(0, 0, 0, 0.25, 0.02, lim, 0.21, crowd, 0.05, 0.065);
  EXPECT_LE(std::hypot(p.first - 0.25, p.second - 0.02), lim + 1e-9);
}
