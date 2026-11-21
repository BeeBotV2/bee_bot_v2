#include <chrono>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "rclcpp/rclcpp.hpp"
#include "ml/msg/arm_correction_debug.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "skills/arm_state_cache.hpp"

TEST(ArmStateCache, SharesJointStateReaderWithinContext) {
  rclcpp::init(0, nullptr);

  auto first = std::make_shared<rclcpp::Node>("arm_cache_first");
  auto second = std::make_shared<rclcpp::Node>("arm_cache_second");
  auto first_cache = skills::ArmStateCache::for_node(*first);
  auto second_cache = skills::ArmStateCache::for_node(*second);
  EXPECT_EQ(first_cache.get(), second_cache.get());

  auto publisher = second->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(first);

  sensor_msgs::msg::JointState state;
  state.name = {"joint1", "joint2"};
  state.position = {0.25, -0.5};
  state.effort = {0.1, 0.2};
  bool received = false;
  double observed = 0.0;
  size_t reader_count = 0;
  for (int attempt = 0; attempt < 30 && !received; ++attempt) {
    publisher->publish(state);
    executor.spin_some();
    received = first_cache->position("joint1", observed);
    if (!received) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  reader_count = publisher->get_subscription_count();
  EXPECT_EQ(reader_count, 1u);
  double position = 0.0;
  ASSERT_TRUE(first_cache->position("joint1", position));
  EXPECT_DOUBLE_EQ(position, 0.25);

  // 추종 오차 (load 막힘 판정 jam_lag_rad): JTC 목표 + 정책 Δ − 실제
  auto debug_pub = second->create_publisher<ml::msg::ArmCorrectionDebug>("/rl_arm_correction/debug", 10);
  ml::msg::ArmCorrectionDebug dbg;
  dbg.ref = {0.0, 0.70, -0.95, 1.8, 0.0};
  dbg.q = {0.0, 0.69, -0.92, 1.8, 0.0};
  dbg.delta = {0.0, 0.0, 0.005, 0.0, 0.0};
  std::array<double, 5> lag{};
  bool got = false;
  for (int attempt = 0; attempt < 30 && !got; ++attempt) {
    debug_pub->publish(dbg);
    executor.spin_some();
    got = first_cache->lag(first->now().seconds(), 0.5, lag);
    if (!got) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  ASSERT_TRUE(got);
  EXPECT_NEAR(lag[1], 0.01, 1e-9);
  EXPECT_NEAR(lag[2], -0.025, 1e-9);
  EXPECT_FALSE(first_cache->lag(first->now().seconds() + 1.0, 0.5, lag));  // 오래됨

  executor.remove_node(first);
  rclcpp::shutdown();
}
