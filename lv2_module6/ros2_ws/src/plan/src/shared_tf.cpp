#include "plan/shared_tf.hpp"

#include <cmath>
#include <map>
#include <mutex>

namespace plan {
namespace {
struct Holder {
  std::shared_ptr<tf2_ros::Buffer> buffer;
  std::shared_ptr<tf2_ros::TransformListener> listener;  // buffer보다 먼저 사라짐 (선언 반대 순서)
};
std::mutex registry_mutex;
std::map<std::pair<const rclcpp::Context *, int>, std::weak_ptr<Holder>> registry;
}  // namespace

std::pair<std::shared_ptr<tf2_ros::Buffer>, std::shared_ptr<tf2_ros::TransformListener>> shared_tf(rclcpp::Node & node) {
  const auto context = node.get_node_base_interface()->get_context();
  const std::pair<const rclcpp::Context *, int> key{context.get(), static_cast<int>(node.get_clock()->get_clock_type())};
  std::lock_guard<std::mutex> lock(registry_mutex);
  auto holder = registry[key].lock();
  if (!holder) {
    holder = std::make_shared<Holder>();
    holder->buffer = std::make_shared<tf2_ros::Buffer>(node.get_clock());
    holder->listener = std::make_shared<tf2_ros::TransformListener>(*holder->buffer);
    registry[key] = holder;
  }
  // aliasing: 노드가 쥔 포인터 둘이 holder 하나를 살려 둠
  return {std::shared_ptr<tf2_ros::Buffer>(holder, holder->buffer.get()),
          std::shared_ptr<tf2_ros::TransformListener>(holder, holder->listener.get())};
}

namespace {
// to ← from 변환으로 (x, y) 옮김 · TF 없으면 false
bool move_xy(const tf2_ros::Buffer & tf, const std::string & to, const std::string & from, double & x, double & y) {
  try {
    const auto t = tf.lookupTransform(to, from, tf2::TimePointZero).transform;
    const double yaw = std::atan2(2.0 * (t.rotation.w * t.rotation.z + t.rotation.x * t.rotation.y),
                                  1.0 - 2.0 * (t.rotation.y * t.rotation.y + t.rotation.z * t.rotation.z));
    const double c = std::cos(yaw), s = std::sin(yaw), x0 = x;
    x = t.translation.x + c * x0 - s * y;
    y = t.translation.y + s * x0 + c * y;
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool robot_odom(const tf2_ros::Buffer & tf, double & x, double & y) {
  x = y = 0.0;
  return move_xy(tf, "odom", "base_footprint", x, y);
}
}  // namespace

Spot make_spot(const tf2_ros::Buffer & tf, double x, double y) {
  Spot s;
  s.x = s.ox = x;
  s.y = s.oy = y;
  s.odom = move_xy(tf, "odom", "map", s.ox, s.oy) && robot_odom(tf, s.rx, s.ry);
  return s;
}

void spot_map(const tf2_ros::Buffer & tf, const Spot & s, double & x, double & y, double still_m) {
  x = s.x;
  y = s.y;
  double rx, ry, ox = s.ox, oy = s.oy;
  if (!s.odom || !robot_odom(tf, rx, ry) || std::hypot(rx - s.rx, ry - s.ry) > still_m) return;
  if (move_xy(tf, "map", "odom", ox, oy)) { x = ox; y = oy; }
}

}  // namespace plan
