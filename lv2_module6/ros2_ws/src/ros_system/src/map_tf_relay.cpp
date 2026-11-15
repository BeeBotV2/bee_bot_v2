// map_tf_relay — cuvslam_slam이 계산한 map→odom(/cuvslam/map_odom)을 받아 rate_hz로 계속 TF로 냄 (11-16)
//   cuvslam_slam(파이썬)은 카메라 프레임을 처리할 때만 map→odom을 냈음 → Jetson CPU 몰림에 노드가 1 s 멈추면 TF도 끊겨
//   controller "Failed to transform end pose"(102)로 출발 실패. map→odom은 천천히 바뀌는 보정이라 마지막 값을 계속 내면 됨
//   (AMCL · slam_toolbox와 같은 방식: 계산과 TF 내기를 나눔). 시각 = 지금 + future_s (받는 쪽 transform_tolerance 안에서 늘 최신)
// 실시간 우선순위 (11-16): 시작할 때 SCHED_FIFO MAP_TF_RELAY_RT_PRIO(기본 50 · 0 = 안 함) — CPU가 몰려도 먼저 돎.
//   rclcpp::init 전에 올려 DDS 스레드도 물려받음. 계정 rtprio 한도 필요 (/etc/security/limits.d/90-bee-rtprio.conf, 다시 로그인)
//   — 없으면 경고만 하고 보통 우선순위로 계속
#include <sched.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"

class MapTfRelay : public rclcpp::Node {
public:
  MapTfRelay() : Node("map_tf_relay"), tf_(this) {
    const double hz = declare_parameter<double>("rate_hz", 30.0);
    future_ = rclcpp::Duration::from_seconds(declare_parameter<double>("future_s", 0.2));
    stale_s_ = declare_parameter<double>("stale_warn_s", 2.0);  // 보정이 이만큼 안 오면 경고 (값은 계속 냄)
    sub_ = create_subscription<geometry_msgs::msg::TransformStamped>(
      declare_parameter<std::string>("topic", "/cuvslam/map_odom"), rclcpp::QoS(1).reliable().transient_local(),
      [this](geometry_msgs::msg::TransformStamped::ConstSharedPtr m) { last_ = *m; got_ = now(); have_ = true; });
    timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / hz), [this]() {
      if (!have_) return;
      auto t = last_;
      t.header.stamp = now() + future_;
      tf_.sendTransform(t);
      if ((now() - got_).seconds() > stale_s_)
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "map→odom 보정이 %.1f s째 안 옴 — 마지막 값을 계속 냄",
                             (now() - got_).seconds());
    });
  }

private:
  tf2_ros::TransformBroadcaster tf_;
  rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  geometry_msgs::msg::TransformStamped last_;
  rclcpp::Time got_{0, 0, RCL_ROS_TIME};
  rclcpp::Duration future_{0, 0};
  double stale_s_ = 2.0;
  bool have_ = false;
};

int main(int argc, char ** argv) {
  const char * env = std::getenv("MAP_TF_RELAY_RT_PRIO");
  const int prio = env ? std::atoi(env) : 50;
  std::string rt = "보통 우선순위 (MAP_TF_RELAY_RT_PRIO=0)";
  bool ok = false;
  if (prio > 0) {
    sched_param sp{};
    sp.sched_priority = prio;
    ok = sched_setscheduler(0, SCHED_FIFO, &sp) == 0;
    rt = ok ? "SCHED_FIFO " + std::to_string(prio)
            : std::string("실시간 우선순위 못 올림 (") + std::strerror(errno) + ") — 보통 우선순위로 (계정 rtprio 한도 확인)";
  }
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MapTfRelay>();
  if (ok) RCLCPP_INFO(node->get_logger(), "%s", rt.c_str());
  else RCLCPP_WARN(node->get_logger(), "%s", rt.c_str());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
