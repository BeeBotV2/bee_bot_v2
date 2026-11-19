#include "plan/mission_solver.hpp"
#include "plan/shared_tf.hpp"

#include <nav_msgs/msg/occupancy_grid.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <stdexcept>
#include <tuple>

namespace plan {
class WallLocalization : public rclcpp::Node {
public:
  WallLocalization() : Node("wall_localization") {
    const auto start = declare_parameter<std::vector<double>>("start_pose_arena", {0.734, 0.950, 3.14159});
    walls_ = declare_parameter<std::vector<double>>("arena_walls", std::vector<double>{});
    if (start.size() != 3 || walls_.size() % 4 != 0)
      throw std::invalid_argument("invalid start_pose_arena or arena_walls");
    start_ = {start[0], start[1], start[2]};
    const bool enabled = declare_parameter<bool>("arena_corr_wall_fit", true);
    apply_ = declare_parameter<bool>("arena_corr_to_localization", true);
    const double loc_period = declare_parameter<double>("arena_corr_loc_period", 2.0);
    settle_ = declare_parameter<double>("arena_corr_loc_settle_s", 1.5);
    max_shift_ = declare_parameter<double>("arena_corr_max_shift", 0.8);
    max_rot_ = declare_parameter<double>("arena_corr_max_rot", 0.35);
    min_m_ = declare_parameter<double>("arena_corr_loc_min_m", 0.02);
    min_rad_ = declare_parameter<double>("arena_corr_loc_min_rad", 0.0035);
    step_m_ = declare_parameter<double>("arena_corr_loc_step_m", 0.05);
    step_rad_ = declare_parameter<double>("arena_corr_loc_step_rad", 0.0175);
    wall_gate_ = declare_parameter<double>("arena_corr_wall_gate", wall_gate_);
    wall_min_cells_ = static_cast<int>(declare_parameter<int64_t>("arena_corr_wall_min_cells", wall_min_cells_));
    wall_min_eig_ = declare_parameter<double>("arena_corr_wall_min_eig", wall_min_eig_);
    wall_keep_s_ = declare_parameter<double>("arena_corr_wall_keep_s", wall_keep_s_);
    const double wall_period = declare_parameter<double>("arena_corr_wall_period", 2.0);
    const auto wall_grid = declare_parameter<std::string>("arena_corr_wall_grid", "/nvblox_node/static_occupancy_grid");
    if (!enabled) return;
    if (wall_period <= 0 || loc_period <= 0)
      throw std::invalid_argument("wall and localization periods must be positive");
    std::tie(tf_buffer_, tf_listener_) = shared_tf(*this);
    correction_ = create_publisher<Message>("/localization/map_correction", 10);
    fit_ = create_publisher<Message>("/localization/wall_fit", rclcpp::QoS(1).transient_local());
    grid_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      wall_grid, rclcpp::SensorDataQoS().keep_last(1),
      [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr msg) { grid_ = msg; });
    wall_timer_ = create_wall_timer(std::chrono::duration<double>(
      apply_ ? std::max({wall_period, loc_period, settle_}) : wall_period), [this] { tick(); });
  }

private:
  using Message = std_msgs::msg::Float64MultiArray;

  void tick() {
    auto grid = std::move(grid_);
    if (!grid || grid->header.frame_id != "map" || walls_.empty()) return;
    const auto & g = *grid;
    std::vector<std::pair<double, double>> cells;
    const double res = g.info.resolution;
    const int w = static_cast<int>(g.info.width), h = static_cast<int>(g.info.height);
    const auto at = [&g, w, h](int r, int k) { return r < 0 || k < 0 || r >= h || k >= w ? -1 : g.data[static_cast<size_t>(r) * w + k]; };
    // 벽 겉면만 = 빈 칸(0)과 맞닿은 점유 칸 — nvblox 벽은 ~20 cm 두께로 뒤로 번져, 다 쓰면 한쪽 벽만 보일 때 그쪽으로 끌림
    //   (11-15 지도 20261115_094059: 위 벽을 못 봐서 아래 벽 띠 가운데로 dy −0.35 m)
    geometry_msgs::msg::TransformStamped mo;
    try {
      mo = tf_buffer_->lookupTransform("map", "odom", tf2::TimePointZero);
    } catch (const std::exception &) {
      return;
    }
    const auto & q = mo.transform.rotation;
    const double myaw = std::atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
    const double mx = mo.transform.translation.x, my = mo.transform.translation.y, mc = std::cos(myaw), ms = std::sin(myaw);
    // 새로 생긴 겉면 칸 → odom (받은 지금의 map→odom ≈ 쌓을 때의 것). 이미 있던 칸은 옛 map 좌표라 다시 안 넣음
    if (!(res > 0.0) || g.data.size() != static_cast<size_t>(w) * h) return;
    const double t_now = std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
    std::set<long long> cur;
    for (int r = 0; r < h; ++r)
      for (int k = 0; k < w; ++k)
        if (at(r, k) >= 50 && (at(r - 1, k) == 0 || at(r + 1, k) == 0 || at(r, k - 1) == 0 || at(r, k + 1) == 0)) {
          const double px = g.info.origin.position.x + (k + 0.5) * res, py = g.info.origin.position.y + (r + 0.5) * res;
          const long long key = static_cast<long long>(std::lround(px / res)) * 1000003LL + std::lround(py / res);
          cur.insert(key);
          if (wall_prev_.count(key)) continue;
          const double dx = px - mx, dy = py - my;
          wall_pts_.push_back({mc * dx + ms * dy, -ms * dx + mc * dy, t_now});
        }
    wall_prev_ = std::move(cur);
    wall_pts_.erase(std::remove_if(wall_pts_.begin(), wall_pts_.end(), [&](const WallPt & p) { return t_now - p.t > wall_keep_s_; }),
                    wall_pts_.end());
    // 같은 odom 칸은 하나만 (오래 서 있으면 같은 벽이 쌓이지 않게 — 새 칸만 넣으니 대개 이미 하나)
    std::set<long long> seen;
    for (auto it = wall_pts_.rbegin(); it != wall_pts_.rend(); ++it) {
      if (!seen.insert(static_cast<long long>(std::lround(it->x / res)) * 1000003LL + std::lround(it->y / res)).second) continue;
      cells.push_back({mc * it->x - ms * it->y + mx, ms * it->x + mc * it->y + my});
    }
    std::vector<std::array<double, 4>> segs;
    for (size_t i = 0; i + 3 < walls_.size(); i += 4) {
      const auto a = arena_to_map({walls_[i], walls_[i + 1], 0}, start_), b = arena_to_map({walls_[i + 2], walls_[i + 3], 0}, start_);
      segs.push_back({a.x, a.y, b.x, b.y});
    }
    const auto f = fit_walls(segs, cells, residual_, wall_gate_, wall_min_cells_,
                             wall_min_eig_, max_shift_, max_rot_);
    if (!f.ok) {
      if (f.why != wall_why_)
        RCLCPP_INFO(get_logger(), "Wall fit unavailable: %s (%d cells)", f.why.c_str(), f.inliers);
      wall_why_ = f.why;
      return;
    }
    wall_why_.clear();
    residual_ = f.c;
    ArenaCorr step;
    if (apply_ && correction_->get_subscription_count() > 0 &&
        (std::hypot(residual_.dx, residual_.dy) >= min_m_ ||
                   std::abs(residual_.dth) >= min_rad_)) {
      step = corr_step(residual_, step_m_, step_rad_);
      Message request;
      request.data = {step.dx, step.dy, step.dth};
      correction_->publish(request);
      residual_ = corr_after_step(residual_, step);
      RCLCPP_INFO(get_logger(), "Wall correction: %.3f m, %.3f m, %.3f rad (RMS %.3f m)",
                  step.dx, step.dy, step.dth, f.rms);
    }
    // Residual pose, support count, RMS, and the step sent to localization.
    Message result;
    result.data = {residual_.dx, residual_.dy, residual_.dth, static_cast<double>(residual_.n),
                   f.rms, step.dx, step.dy, step.dth};
    fit_->publish(result);
  }

  Pose2 start_{};
  std::vector<double> walls_;
  ArenaCorr residual_;
  bool apply_;
  double max_shift_, max_rot_, min_m_, min_rad_, step_m_, step_rad_, settle_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Publisher<Message>::SharedPtr correction_, fit_;
  double wall_gate_ = 0.3, wall_min_eig_ = 0.03;
  int wall_min_cells_ = 60;
  std::string wall_why_;              // 마지막 실패 이유 (바뀔 때만 로그)
  nav_msgs::msg::OccupancyGrid::ConstSharedPtr grid_;
  // 벽 정합 점 (odom) — 격자에 새로 생긴 겉면 칸만, 받은 때의 map→odom으로 odom에 둠. 맞출 때 지금 map→odom으로 map에 옮김:
  //   로봇 위치를 옮겨도(map→odom 바뀜) 점이 같이 옮겨져 같은 보정을 또 내지 않음 (nvblox 격자는 map에 그대로 남아 되먹임 — 11-15)
  struct WallPt { double x, y, t; };
  std::vector<WallPt> wall_pts_;
  std::set<long long> wall_prev_;     // 지난 격자의 겉면 칸 (map 칸 번호) — 새로 생긴 칸만 고름
  double wall_keep_s_ = 120.0;        // [s] 이보다 오래된 점은 버림 (arena_corr_wall_keep_s)
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_sub_;
  rclcpp::TimerBase::SharedPtr wall_timer_;
};
}  // namespace plan

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<plan::WallLocalization>());
  rclcpp::shutdown();
  return 0;
}
