// wall_localization — 도면 벽(plan.yaml arena_walls)에 카메라가 본 벽을 맞춰 map→odom을 정함 (미션과 따로)
//   11-20 절대값 방식: 앞 카메라 depth 점(/pole_front_cam/depth/color/points_filtered)을 odom 좌표로 최근 arena_corr_wall_keep_s만
//   5 cm 칸에 모아 벽에 맞춤 → 그 결과가 곧 map→odom. odom 점은 map→odom과 무관해 측정이 보정에 끌려가지 않고,
//   받는 쪽(cuvslam_slam)에는 조각(증분)이 아니라 목표값을 /localization/map_to_odom으로 냄 → 같은 차이가 쌓이지 않음.
//   예전(nvblox map 격자 → /localization/map_correction 증분)은 MapGate가 잠긴 동안 같은 −1°를 2 s마다 더해 98번 쌓였음 (11-20 시뮬).
//   높이 띠 기본 0.10~0.30 m — 상자(7 cm)·퍽은 빠지고 벽만. /localization/wall_fit은 mission_manager용 그대로 (남은 보정 · 낸 조각)
#include "plan/mission_solver.hpp"
#include "plan/shared_tf.hpp"

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <tf2/LinearMath/Transform.hpp>

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

namespace plan {
class WallLocalization : public rclcpp::Node {
public:
  WallLocalization() : Node("wall_localization") {
    const auto start = declare_parameter<std::vector<double>>("start_pose_arena", {0.734, 0.950, 3.14159});
    walls_ = declare_parameter<std::vector<double>>("arena_walls", std::vector<double>{});
    if (start.size() != 3 || walls_.size() % 4 != 0)
      throw std::invalid_argument("invalid start_pose_arena or arena_walls");
    const Pose2 st{start[0], start[1], start[2]};
    for (size_t i = 0; i + 3 < walls_.size(); i += 4) {
      const auto a = arena_to_map({walls_[i], walls_[i + 1], 0}, st), b = arena_to_map({walls_[i + 2], walls_[i + 3], 0}, st);
      segs_.push_back({a.x, a.y, b.x, b.y});
    }
    const bool enabled = declare_parameter<bool>("arena_corr_wall_fit", true);
    apply_ = declare_parameter<bool>("arena_corr_to_localization", true);
    max_shift_ = declare_parameter<double>("arena_corr_max_shift", 0.8);
    max_rot_ = declare_parameter<double>("arena_corr_max_rot", 0.35);
    min_m_ = declare_parameter<double>("arena_corr_loc_min_m", 0.02);
    min_rad_ = declare_parameter<double>("arena_corr_loc_min_rad", 0.0035);
    step_m_ = declare_parameter<double>("arena_corr_loc_step_m", 0.05);
    step_rad_ = declare_parameter<double>("arena_corr_loc_step_rad", 0.0175);
    wall_gate_ = declare_parameter<double>("arena_corr_wall_gate", wall_gate_);
    wall_min_cells_ = static_cast<int>(declare_parameter<int64_t>("arena_corr_wall_min_cells", wall_min_cells_));
    wall_min_eig_ = declare_parameter<double>("arena_corr_wall_min_eig", wall_min_eig_);
    keep_s_ = declare_parameter<double>("arena_corr_wall_keep_s", keep_s_);
    res_ = declare_parameter<double>("arena_corr_wall_res", res_);
    z_min_ = declare_parameter<double>("arena_corr_wall_z_min", z_min_);
    z_max_ = declare_parameter<double>("arena_corr_wall_z_max", z_max_);
    range_ = declare_parameter<double>("arena_corr_wall_range", range_);
    stride_ = static_cast<size_t>(std::max<int64_t>(1, declare_parameter<int64_t>("arena_corr_wall_stride", 4)));
    cloud_period_ = declare_parameter<double>("arena_corr_wall_cloud_period", cloud_period_);
    const double wall_period = declare_parameter<double>("arena_corr_wall_period", 2.0);
    const auto cloud = declare_parameter<std::string>("arena_corr_wall_cloud", "/pole_front_cam/depth/color/points_filtered");
    if (!enabled || segs_.empty()) return;
    if (wall_period <= 0 || !(res_ > 0) || keep_s_ <= 0) throw std::invalid_argument("wall period, res and keep must be positive");
    std::tie(tf_buffer_, tf_listener_) = shared_tf(*this);
    target_ = create_publisher<Message>("/localization/map_to_odom", rclcpp::QoS(1).transient_local());
    fit_ = create_publisher<Message>("/localization/wall_fit", rclcpp::QoS(1).transient_local());
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud, rclcpp::SensorDataQoS().keep_last(1),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr m) { on_cloud(*m); });
    wall_timer_ = create_wall_timer(std::chrono::duration<double>(wall_period), [this] { tick(); });
  }

private:
  using Message = std_msgs::msg::Float64MultiArray;
  struct Cell { double x, y, t; };

  static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  // depth 점 → odom (그 영상 시각의 TF) → 높이 띠 안만 5 cm 칸에 (칸마다 마지막 본 시각)
  void on_cloud(const sensor_msgs::msg::PointCloud2 & m) {
    const double t = now_s();
    if (t - last_cloud_ < cloud_period_) return;
    geometry_msgs::msg::TransformStamped o;
    try {
      o = tf_buffer_->lookupTransform("odom", m.header.frame_id, m.header.stamp);
    } catch (const std::exception &) {
      return;
    }
    last_cloud_ = t;
    const auto & q = o.transform.rotation;
    const auto & p = o.transform.translation;
    const tf2::Transform tf(tf2::Quaternion(q.x, q.y, q.z, q.w), tf2::Vector3(p.x, p.y, p.z));
    sensor_msgs::PointCloud2ConstIterator<float> x(m, "x"), y(m, "y"), z(m, "z");
    for (size_t i = 0; x != x.end(); ++x, ++y, ++z, ++i) {
      if (i % stride_ || !std::isfinite(*z) || *z > range_) continue;  // 광학 frame z = 깊이
      const tf2::Vector3 w = tf * tf2::Vector3(*x, *y, *z);
      if (w.z() < z_min_ || w.z() > z_max_) continue;
      const long long key = std::lround(w.x() / res_) * 1000003LL + std::lround(w.y() / res_);
      cells_[key] = {std::round(w.x() / res_) * res_, std::round(w.y() / res_) * res_, t};
    }
  }

  void tick() {
    const double t = now_s();
    std::vector<std::pair<double, double>> pts;
    for (auto it = cells_.begin(); it != cells_.end();) {
      if (t - it->second.t > keep_s_) { it = cells_.erase(it); continue; }  // odom이 서서히 틀어지니 최근 것만
      pts.push_back({it->second.x, it->second.y});
      ++it;
    }
    if (!sent_init_) {  // 첫 목표 = 지금 map→odom (켠 자리 ≈ 항등) — 도면 보정 꼴로 (= 역)
      try {
        const auto mo = tf_buffer_->lookupTransform("map", "odom", tf2::TimePointZero).transform;
        const double yaw = 2 * std::atan2(mo.rotation.z, mo.rotation.w);
        sent_ = corr_after_step({}, {mo.translation.x, mo.translation.y, yaw});
        last_fit_ = sent_;
        sent_init_ = true;
      } catch (const std::exception &) {
        return;
      }
    }
    // odom 점을 도면 벽에 — 결과 c는 odom → 도면 보정 (c⁻¹ = map→odom). 시작값만 지난 결과, 측정은 map→odom과 무관
    const auto f = fit_walls(segs_, pts, last_fit_, wall_gate_, wall_min_cells_, wall_min_eig_, max_shift_, max_rot_);
    if (!f.ok) {
      if (f.why != wall_why_) RCLCPP_INFO(get_logger(), "Wall fit unavailable: %s (%d cells)", f.why.c_str(), f.inliers);
      wall_why_ = f.why;
      return;
    }
    wall_why_.clear();
    last_fit_ = f.c;
    ArenaCorr residual = corr_after_step(f.c, sent_), step;  // 낸 목표 기준 남은 보정 (mission_manager가 목표 계산에 씀)
    residual.n = f.inliers;
    if (apply_ && (std::hypot(residual.dx, residual.dy) >= min_m_ || std::abs(residual.dth) >= min_rad_)) {
      step = wall_target_step(f.c, sent_, step_m_, step_rad_);
      residual = corr_after_step(residual, step);
      const auto mo = corr_after_step({}, sent_);  // map→odom 목표 = sent⁻¹
      Message target;
      target.data = {mo.dx, mo.dy, mo.dth};
      target_->publish(target);
      RCLCPP_INFO(get_logger(), "Wall map->odom target: %.3f m, %.3f m, %.2f deg (step %.3f, %.3f, %.2f deg · RMS %.3f m · %d cells)",
                  mo.dx, mo.dy, mo.dth * 180 / M_PI, step.dx, step.dy, step.dth * 180 / M_PI, f.rms, f.inliers);
    }
    // 남은 보정 · 맞은 칸 · RMS · 이번에 옮긴 조각 (mission_manager: 조각이 오면 loc_last_ — marks는 안 옮김)
    Message result;
    result.data = {residual.dx, residual.dy, residual.dth, static_cast<double>(residual.n),
                   f.rms, step.dx, step.dy, step.dth};
    fit_->publish(result);
  }

  std::vector<double> walls_;
  std::vector<std::array<double, 4>> segs_;
  bool apply_, sent_init_ = false;
  double max_shift_, max_rot_, min_m_, min_rad_, step_m_, step_rad_;
  double wall_gate_ = 0.3, wall_min_eig_ = 0.03;
  int wall_min_cells_ = 60;
  double keep_s_ = 25.0, res_ = 0.05, z_min_ = 0.10, z_max_ = 0.30, range_ = 2.5, cloud_period_ = 0.5, last_cloud_ = 0;
  size_t stride_ = 4;
  ArenaCorr sent_, last_fit_;
  std::string wall_why_;  // 마지막 실패 이유 (바뀔 때만 로그)
  std::unordered_map<long long, Cell> cells_;  // odom 칸 → 마지막 본 시각
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Publisher<Message>::SharedPtr target_, fit_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::TimerBase::SharedPtr wall_timer_;
};
}  // namespace plan

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<plan::WallLocalization>());
  rclcpp::shutdown();
  return 0;
}
