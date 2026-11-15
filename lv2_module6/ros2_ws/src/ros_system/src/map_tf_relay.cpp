// map_tf_relay — cuvslam_slam이 계산한 map→odom(/cuvslam/map_odom)을 받아 rate_hz로 계속 TF로 냄 (11-16)
//   cuvslam_slam(파이썬)은 카메라 프레임을 처리할 때만 map→odom을 냈음 → Jetson CPU 몰림에 노드가 1 s 멈추면 TF도 끊겨
//   controller "Failed to transform end pose"(102)로 출발 실패. map→odom은 천천히 바뀌는 보정이라 마지막 값을 계속 내면 됨
//   (AMCL · slam_toolbox와 같은 방식: 계산과 TF 내기를 나눔). 시각 = 지금 + future_s (받는 쪽 transform_tolerance 안에서 늘 최신)
#include <memory>

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
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MapTfRelay>());
  rclcpp::shutdown();
  return 0;
}
