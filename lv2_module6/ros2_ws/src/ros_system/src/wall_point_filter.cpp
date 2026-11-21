// wall_point_filter — 카메라 점구름에서 저장 지도(/map) 벽 near_m(0.15) 안의 점을 뺌 → 전역 costmap 칠하기 입력 (11-21)
//   "벽은 지도에서, 상자는 카메라에서": 위치 추정이 몇 cm 틀리면 카메라가 지도 벽을 옆에 한 번 더 칠하고, 광선으로는 벽에 붙은
//   칸이 안 지워져 벽이 두꺼워짐 → 좁은 통로가 닫힘 (실기 nav_1121_144601 S1: 2.2 s 열림 → 10.9 s 닫힘 → 우회 · 11-15 문틀 7 m 우회).
//   지도에 이미 있는 벽은 지도로만 칠하고, 지도에 없는 것(no_box의 상자 · 떨어진 퍽)만 카메라로 칠함.
//   지우기(광선)는 nav2_params에서 원래 점구름을 따로 씀 — 여기서 뺀 점도 그 앞 칸은 계속 지움.
//   점은 바꾸지 않고 고르기만 함 (frame · 필드 그대로). 지도 밖 점 · TF 없을 때는 그대로 통과
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

class WallPointFilter : public rclcpp::Node {
public:
  WallPointFilter() : Node("wall_point_filter"), buf_(get_clock()), tf_(buf_) {
    near_ = declare_parameter<double>("near_m", 0.15);   // 지도 벽에서 이 안의 점은 뺌 — 11-21 실기 S1 통로를 닫은 칸이 벽에서 15 cm · 벽에 붙은 상자(가장자리 6~11 cm)는 앞쪽 12 cm 이상 남음
    occ_ = declare_parameter<int>("occupied_min", 65);
    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      declare_parameter<std::string>("output_topic", "/pole_front_cam/depth/color/points_offwall"), rclcpp::SensorDataQoS().keep_last(1));
    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
      declare_parameter<std::string>("map_topic", "/map"), rclcpp::QoS(1).reliable().transient_local(),
      [this](nav_msgs::msg::OccupancyGrid::ConstSharedPtr m) { on_map(*m); });
    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      declare_parameter<std::string>("input_topic", "/pole_front_cam/depth/color/points_filtered"), rclcpp::SensorDataQoS().keep_last(1),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr m) { on_cloud(*m); });
  }

  // 벽 칸에서 near_m 안(원)을 미리 표시 — 지도는 latched 한 번이라 점마다 거리를 안 잼
  void on_map(const nav_msgs::msg::OccupancyGrid & m) {
    info_ = m.info;
    frame_ = m.header.frame_id;
    const int w = static_cast<int>(m.info.width), h = static_cast<int>(m.info.height);
    const int r = static_cast<int>(std::ceil(near_ / m.info.resolution));
    near_mask_.assign(static_cast<size_t>(w) * h, 0);
    for (int j = 0; j < h; ++j)
      for (int i = 0; i < w; ++i) {
        if (m.data[static_cast<size_t>(j) * w + i] < occ_) continue;
        for (int dj = -r; dj <= r; ++dj)
          for (int di = -r; di <= r; ++di) {
            const int x = i + di, y = j + dj;
            if (x < 0 || y < 0 || x >= w || y >= h || (di * di + dj * dj) * m.info.resolution * m.info.resolution > near_ * near_ + 1e-9) continue;
            near_mask_[static_cast<size_t>(y) * w + x] = 1;
          }
      }
    RCLCPP_INFO(get_logger(), "지도 %dx%d — 벽 %.2f m 안 칸 표시", w, h, near_);
  }

  void on_cloud(const sensor_msgs::msg::PointCloud2 & in) {
    if (near_mask_.empty()) { pub_->publish(in); return; }  // 지도 전: 그대로
    geometry_msgs::msg::TransformStamped t;
    try {
      t = buf_.lookupTransform(frame_, in.header.frame_id, in.header.stamp, rclcpp::Duration::from_seconds(0.05));
    } catch (const tf2::TransformException &) {
      try {
        t = buf_.lookupTransform(frame_, in.header.frame_id, tf2::TimePointZero);
      } catch (const tf2::TransformException &) {
        pub_->publish(in);
        return;
      }
    }
    const auto & q = t.transform.rotation;
    const double tx = t.transform.translation.x, ty = t.transform.translation.y;
    // 회전 행렬의 위 두 줄 (지도 x · y만 씀)
    const double r00 = 1 - 2 * (q.y * q.y + q.z * q.z), r01 = 2 * (q.x * q.y - q.z * q.w), r02 = 2 * (q.x * q.z + q.y * q.w);
    const double r10 = 2 * (q.x * q.y + q.z * q.w), r11 = 1 - 2 * (q.x * q.x + q.z * q.z), r12 = 2 * (q.y * q.z - q.x * q.w);
    const double res = info_.resolution, ox = info_.origin.position.x, oy = info_.origin.position.y;
    const int w = static_cast<int>(info_.width), h = static_cast<int>(info_.height);

    sensor_msgs::msg::PointCloud2 out;
    out.header = in.header;
    out.fields = in.fields;
    out.is_bigendian = in.is_bigendian;
    out.point_step = in.point_step;
    out.is_dense = in.is_dense;
    out.height = 1;
    out.data.reserve(in.data.size());
    const size_t n = static_cast<size_t>(in.width) * in.height;
    sensor_msgs::PointCloud2ConstIterator<float> x(in, "x"), y(in, "y"), z(in, "z");
    size_t kept = 0;
    for (size_t k = 0; k < n; ++k, ++x, ++y, ++z) {
      bool drop = false;
      if (std::isfinite(*x) && std::isfinite(*y) && std::isfinite(*z)) {
        const double mx = r00 * *x + r01 * *y + r02 * *z + tx, my = r10 * *x + r11 * *y + r12 * *z + ty;
        const int i = static_cast<int>(std::floor((mx - ox) / res)), j = static_cast<int>(std::floor((my - oy) / res));
        drop = i >= 0 && j >= 0 && i < w && j < h && near_mask_[static_cast<size_t>(j) * w + i];
      }
      if (drop) continue;
      const auto * p = &in.data[k * in.point_step];
      out.data.insert(out.data.end(), p, p + in.point_step);
      ++kept;
    }
    out.width = static_cast<uint32_t>(kept);
    out.row_step = out.point_step * out.width;
    pub_->publish(out);
  }

private:
  tf2_ros::Buffer buf_;
  tf2_ros::TransformListener tf_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  nav_msgs::msg::MapMetaData info_;
  std::string frame_ = "map";
  std::vector<uint8_t> near_mask_;
  double near_ = 0.15;
  int occ_ = 65;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<WallPointFilter>());
  rclcpp::shutdown();
  return 0;
}
