// cuvslam_slam C++판 (11-16) — scripts/cuvslam_slam.py와 같은 일 · 같은 노드 이름 · 같은 파라미터 · 같은 토픽
//   cuVSLAM C++ API(cuvslam2.h)를 바로 부르고, 영상은 메시지에서 GPU 장치 메모리로 바로 올려(cudaMemcpy2D) 가까운 depth
//   마스크도 GPU 커널로 만든 뒤 GPU 포인터로 넘김 (is_gpu_mem) — 버퍼는 두 벌을 번갈아 씀 (앞 프레임을 cuVSLAM이 아직 읽어도 안전)
//   파이썬판은 프레임마다 numpy 변환 · 마스크 · map→odom 계산이 인터프리터에서 돌아 CPU 몰림에 1 s씩 멈췄음
// 설명(MapGate · initialpose · 도면 보정 · BEE_CUVSLAM_LEVEL · BEE_MAP_FREEZE · BEE_CUVSLAM_VIS)은 파이썬판 머리말·주석과 같음
#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cuvslam2.h"
#include "cuvslam_gpu.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "message_filters/subscriber.hpp"
#include "message_filters/sync_policies/approximate_time.hpp"
#include "message_filters/synchronizer.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "rclcpp/rclcpp.hpp"
#include "ros_system/slam_math.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/transform_listener.h"
#include "visualization_msgs/msg/marker_array.hpp"

using ros_system::Pose2;
using Image = sensor_msgs::msg::Image;

namespace {

bool env_on(const char * k) {
  const char * v = std::getenv(k);
  return v && std::string(v) == "1";
}

Eigen::Matrix4d mat(const cuvslam::Pose & p) {
  return ros_system::mat(p.rotation[0], p.rotation[1], p.rotation[2], p.rotation[3], p.translation[0], p.translation[1],
                         p.translation[2]);
}

Eigen::Matrix4d mat(const geometry_msgs::msg::Transform & t) {
  return ros_system::mat(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w, t.translation.x, t.translation.y,
                         t.translation.z);
}

// 영상 버퍼 하나 — gpu면 장치 메모리(cudaMallocPitch, 메시지에서 cudaMemcpy2D로 바로 올림), 아니면 보통 메모리
//   (mapped pinned는 cuVSLAM이 GPU 포인터로 안 봄 — cudaPointerGetAttributes가 Host · Orin은 concurrentManagedAccess=0이라 managed도 안 씀)
struct Buf {
  void * p = nullptr;
  size_t pitch = 0;  // 행 바이트 (여백 포함)
  uint32_t w = 0, h = 0;
  size_t bpp = 0;
  bool gpu = false;
  std::vector<uint8_t> host;

  bool fits(uint32_t W, uint32_t H, size_t B, bool G) const { return p && w == W && h == H && bpp == B && gpu == G; }
  bool alloc(uint32_t W, uint32_t H, size_t B, bool G) {
    release();
    w = W;
    h = H;
    bpp = B;
    if (G && cudaMallocPitch(&p, &pitch, W * B, H) == cudaSuccess) {
      gpu = true;
      return true;
    }
    p = nullptr;
    gpu = false;
    pitch = W * B;
    host.assign(pitch * H, 0);
    p = host.data();
    return !G;
  }
  void release() {
    if (gpu && p) cudaFree(p);
    p = nullptr;
    gpu = false;
    host.clear();
  }
  ~Buf() { release(); }
};

}  // namespace

class CuvslamSlam : public rclcpp::Node {
public:
  CuvslamSlam() : Node("cuvslam_slam") {
    base_ = declare_parameter<std::string>("base_frame", "base_footprint");
    odom_ = declare_parameter<std::string>("odom_frame", "odom");
    map_ = declare_parameter<std::string>("map_frame", "map");
    publish_map_tf_ = declare_parameter<bool>("publish_map_tf", true);
    if (env_on("BEE_CUVSLAM_NO_MAP_TF")) publish_map_tf_ = false;
    map_tf_tolerance_ = rclcpp::Duration::from_seconds(declare_parameter<double>("map_tf_tolerance", 1.0));
    depth_scale_ = declare_parameter<double>("depth_scale_factor", 1000.0);
    min_depth_ = declare_parameter<double>("min_depth", 0.5);
    frame_skip_ = std::max<int64_t>(1, declare_parameter<int64_t>("frame_skip", 1));
    near_raw_ = ros_system::near_limit(min_depth_, depth_scale_);
    max_map_size_ = static_cast<uint32_t>(declare_parameter<int64_t>("max_map_size", 150));
    max_lm_dist_ = static_cast<float>(declare_parameter<double>("max_landmarks_distance", 100.0));
    throttle_ms_ = static_cast<uint32_t>(declare_parameter<int64_t>("throttling_time_ms", 0));
    lin_var_ = declare_parameter<double>("linear_variance", 0.0025);
    ang_var_ = declare_parameter<double>("angular_variance", 0.0025);
    // 영상을 GPU 장치 메모리로 넘길지 (메시지 → cudaMemcpy2D · 마스크 커널) — false면 CPU 버퍼 (cuVSLAM이 안에서 복사)
    gpu_input_ = declare_parameter<bool>("gpu_input", true);
    if (gpu_input_ && cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) != cudaSuccess) {
      RCLCPP_WARN(get_logger(), "CUDA 스트림 못 만듦 — CPU 입력으로");
      gpu_input_ = false;
    }
    const double jm = declare_parameter<double>("gate_jump_m", 0.30), jd = declare_parameter<double>("gate_jump_deg", 15.0);
    const double rm = declare_parameter<double>("gate_rate_m", 0.05), rd = declare_parameter<double>("gate_rate_deg", 5.0);
    const double ss = declare_parameter<double>("gate_settle_s", 15.0);
    const bool lock = declare_parameter<bool>("gate_lock", true);
    if (declare_parameter<bool>("map_gate", true)) gate_.emplace(jm, jd, rm, rd, ss, lock);
    // [rad/s] odom 회전 속도가 이보다 크면 map→odom을 그대로 둠 (회전 중 보정 멈춤, 11-16) · 0 = 안 멈춤
    hold_wz_ = declare_parameter<double>("gate_hold_wz", 0.2);
    // map 기준 (11-18): cuvslam(기본) = map→odom을 cuVSLAM 자세로 맞춤 · odom = EKF odom을 그대로 map에 두고 도면 보정(/localization/map_correction)만 얹음
    //   Isaac nav_bench: EKF odom 오차 ~2 cm인데 cuVSLAM map→odom이 27~38 cm(보정 켜면 10~28 cm) 끌고 감 → 시뮬은 odom 기준이 정확.
    //   실기에서는 바퀴 미끄러짐을 cuVSLAM이 잡는지 확인 전이라 기본은 그대로. 환경 변수 BEE_MAP_ANCHOR=odom이 덮음
    odom_anchor_ = declare_parameter<std::string>("map_anchor", "cuvslam") == "odom";
    odom_tol_ = declare_parameter<double>("odom_lookup_tolerance", 0.1);  // [s] 영상 시각의 odom TF가 아직 없을 때 최근 TF를 쓰는 한도
    max_twist_lin_ = declare_parameter<double>("max_twist_lin", 0.6);  // [m/s] 넘으면 /cuvslam/odom 안 냄 (자세 튐)
    max_twist_ang_ = declare_parameter<double>("max_twist_ang", 4.0);  // [rad/s]
    // 11-22 실기 nav_1122_091419: 600 s 뒤 추적이 15 → 9 Hz로 느려지며 영상 시각보다 최대 1.7 s 늦게 냄 — EKF history_length(0.6 s)보다
    //   늦은 회전 속도가 들어가 yaw가 −10°씩 세 번 튐 → 벽 정합 over_limit로 잠김. 이보다 늦은 측정은 안 냄 (바퀴·IMU가 받침)
    max_odom_delay_ = declare_parameter<double>("max_odom_delay", 0.5);  // [s]
    if (const char * a = std::getenv("BEE_MAP_ANCHOR")) odom_anchor_ = std::string(a) == "odom";
    if (odom_anchor_) RCLCPP_WARN(get_logger(), "map 기준 = EKF odom (+도면 보정) — cuVSLAM 자세는 map→odom에 안 씀");
    if (env_on("BEE_MAP_FREEZE")) {
      const double inf = std::numeric_limits<double>::infinity();
      gate_.emplace(inf, inf, 0.0, 0.0, 0.0);
      RCLCPP_WARN(get_logger(), "BEE_MAP_FREEZE=1 — map→odom 보정 끔 (출발 값에 고정)");
    }
    level_ = env_on("BEE_CUVSLAM_LEVEL");
    vis_ = env_on("BEE_CUVSLAM_VIS");
    // cuVSLAM 지도(랜드마크 · pose graph) 저장 · 불러오기 — 지도 폴더 <map_dir>/<map_name>/cuvslam/ (bringup이 정함)
    //   매핑(map_save_dir): 끌 때 저장 (+ ~/save_map 서비스) · 위치추정(map_load_dir): 그 지도 안에서 지금 자리를 찾음.
    //   찾으면 SLAM이 저장 지도로 바뀌어 map = 그 지도를 만들 때의 출발 자리 (nvblox.nvblx · map.yaml과 같은 원점) — 출발 자리가 달라도 됨.
    //   못 찾는 동안은 예전처럼 켠 자리가 map 원점이다 (localize_retry_s마다 다시 시도)
    save_dir_ = declare_parameter<std::string>("map_save_dir", "");
    load_dir_ = declare_parameter<std::string>("map_load_dir", "");
    // 찾는 범위 = 추측 자세(지금 SLAM 자세) 둘레 — 기본값은 Isaac ROS visual_slam localizer_* 와 같음
    loc_.horizontal_search_radius = static_cast<float>(declare_parameter<double>("localize_horizontal_radius", 1.5));
    loc_.vertical_search_radius = static_cast<float>(declare_parameter<double>("localize_vertical_radius", 0.5));
    loc_.horizontal_step = static_cast<float>(declare_parameter<double>("localize_horizontal_step", 0.5));
    loc_.vertical_step = static_cast<float>(declare_parameter<double>("localize_vertical_step", 0.25));
    loc_.angular_step_rads = static_cast<float>(declare_parameter<double>("localize_angular_step", 0.1745));
    loc_retry_s_ = declare_parameter<double>("localize_retry_s", 2.0);
    if (!load_dir_.empty() && !std::filesystem::is_directory(load_dir_)) {
      RCLCPP_WARN(get_logger(), "cuVSLAM 저장 지도 없음: %s — 켠 자리를 원점으로 새로 시작 (nvblox 지도만 있는 옛 폴더)", load_dir_.c_str());
      load_dir_.clear();
    }
    if (!save_dir_.empty()) {
      save_srv_ = create_service<std_srvs::srv::Trigger>(
        "~/save_map", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                             std::shared_ptr<std_srvs::srv::Trigger::Response> res) { res->success = save_map(10.0, res->message); });
      RCLCPP_INFO(get_logger(), "cuVSLAM 지도 저장 위치: %s (끌 때 · ~/save_map)", save_dir_.c_str());
    }
    if (!load_dir_.empty()) RCLCPP_INFO(get_logger(), "cuVSLAM 저장 지도에서 위치 찾기: %s", load_dir_.c_str());

    tf_buf_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buf_, this, true);  // 따로 스레드 — 콜백 안에서 TF를 읽어도 안 막힘
    tf_pub_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    // map_tf_relay (11-16): true면 map→odom을 TF 대신 토픽으로만 — map_tf_relay(C++)가 30 Hz로 계속 TF로 냄
    //   토픽은 파이썬판·map_tf_relay와 같은 /cuvslam/map_to_odom (예전 /cuvslam/map_odom이면 relay가 못 받아 map→odom TF가 없었음)
    relay_ = declare_parameter<bool>("map_tf_relay", true);
    map_odom_pub_ = create_publisher<geometry_msgs::msg::TransformStamped>(
      declare_parameter<std::string>("map_to_odom_topic", "/cuvslam/map_to_odom"), rclcpp::QoS(1).reliable().transient_local());
    odom_pub_ = create_publisher<nav_msgs::msg::Odometry>(declare_parameter<std::string>("odom_topic", "/cuvslam/odom"), 10);
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/cuvslam/path", 1);
    path_.header.frame_id = map_;

    init_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      declare_parameter<std::string>("initialpose_topic", "/initialpose"), 10,
      [this](geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr m) { on_initialpose(*m); });
    corr_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      declare_parameter<std::string>("map_correction_topic", "/localization/map_correction"), 10,
      [this](std_msgs::msg::Float64MultiArray::ConstSharedPtr m) {
        if (m->data.size() < 3) return;
        // 매핑(map_save_dir)이면 도면 보정을 안 받음 — 도면(plan.yaml) 벽 정합이 쌓는 중인 지도를 도면 쪽으로 당겨
        //   지도에 도면이 섞였음 (실기 지도 20261119_140649: 85번 · 한 번에 최대 0.8°). 매핑 흐름은 cuVSLAM으로만 (사용자 2026-11-19)
        if (!save_dir_.empty()) {
          if (n_corr_ignored_++ == 0) RCLCPP_WARN(get_logger(), "매핑 중 — 도면 보정(/localization/map_correction) 무시");
          return;
        }
        map_offset_ = ros_system::apply_map_correction(map_offset_, {m->data[0], m->data[1], m->data[2]});
        RCLCPP_INFO(get_logger(), "도면 보정 받음 → map 기준 옮김 (%+.3f, %+.3f, %+.2f°)", m->data[0], m->data[1],
                    m->data[2] * 180.0 / M_PI);
      });
    // 벽 위치 보정 목표 (wall_localization 11-20 절대값): map→odom을 이 값으로 덮어씀 — 증분이 아니라 MapGate가 잠겨도 안 쌓임.
    //   map 기준 odom일 때만 (map→odom = map_offset_). cuVSLAM 기준이면 map_offset_가 cuVSLAM 자세에 얹히므로 안 맞음 → 무시
    target_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      declare_parameter<std::string>("map_to_odom_topic_in", "/localization/map_to_odom"), rclcpp::QoS(1).transient_local(),
      [this](std_msgs::msg::Float64MultiArray::ConstSharedPtr m) {
        if (m->data.size() < 3) return;
        if (!save_dir_.empty() || !odom_anchor_) {
          if (n_target_ignored_++ == 0)
            RCLCPP_WARN(get_logger(), "벽 위치 목표(/localization/map_to_odom) 무시 — %s",
                        save_dir_.empty() ? "map 기준이 odom이 아님 (BEE_MAP_ANCHOR=odom)" : "매핑 중");
          return;
        }
        map_offset_ = {m->data[0], m->data[1], m->data[2]};
        RCLCPP_INFO(get_logger(), "벽 위치 목표 받음 → map→odom = (%+.3f, %+.3f, %+.2f°)", m->data[0], m->data[1],
                    m->data[2] * 180.0 / M_PI);
      });
    const auto sensor = rclcpp::SensorDataQoS();
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      declare_parameter<std::string>("camera_info", "/pole_front_cam/color/camera_info"), sensor,
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) { info_ = m; });
    color_sub_ = std::make_shared<message_filters::Subscriber<Image>>(
      this, declare_parameter<std::string>("rgb", "/pole_front_cam/color/image_raw"), sensor);
    depth_sub_ = std::make_shared<message_filters::Subscriber<Image>>(
      this, declare_parameter<std::string>("depth", "/pole_front_cam/aligned_depth_to_color/image_raw"), sensor);
    sync_ = std::make_shared<Sync>(Policy(10), *color_sub_, *depth_sub_);
    sync_->getPolicy()->setMaxIntervalDuration(rclcpp::Duration::from_seconds(0.02));
    sync_->registerCallback(
      [this](const Image::ConstSharedPtr & c, const Image::ConstSharedPtr & d) { on_rgbd(*c, *d); });

    stats_ = create_wall_timer(std::chrono::seconds(10), [this]() {
      std::string s = "추적 " + std::to_string(ok_) + " · 잃음 " + std::to_string(lost_);
      if (publish_map_tf_) s += " · 경로 " + std::to_string(path_.poses.size());
      if (gate_)
        s += " · 보정 거름 " + std::to_string(gate_->rejected) + " · 회전 중 멈춤 " + std::to_string(gate_->held_rot) +
             (gate_->locked ? " · 잠금" : "");
      s += gpu_used_ ? " · GPU 입력" : " · CPU 입력";
      RCLCPP_INFO(get_logger(), "%s", s.c_str());
    });
    if (vis_) {
      vis_obs_ = create_publisher<Image>("/cuvslam/vis/observations_image", 1);
      vis_lm_ = create_publisher<sensor_msgs::msg::PointCloud2>("/cuvslam/vis/landmarks_cloud", 1);
      vis_map_ = create_publisher<sensor_msgs::msg::PointCloud2>("/cuvslam/vis/map_landmarks_cloud", 1);
      vis_loop_ = create_publisher<sensor_msgs::msg::PointCloud2>("/cuvslam/vis/loop_closure_cloud", 1);
      vis_graph_ = create_publisher<visualization_msgs::msg::MarkerArray>("/cuvslam/vis/pose_graph", 1);
      vis_timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / declare_parameter<double>("vis_rate_hz", 2.0)),
                                     [this]() { publish_vis(); });
      RCLCPP_INFO(get_logger(), "BEE_CUVSLAM_VIS=1 — /cuvslam/vis/* 냄 (특징점 영상 · 랜드마크 · 포즈 그래프)");
    }
  }

  // spin이 끝난 뒤 (SIGINT) — 매핑이면 지도 저장. launch가 sigterm_timeout 안에 기다려 줌 (ros_system.launch.py)
  void on_exit() {
    if (save_dir_.empty()) return;
    if (!slam_ || ok_ == 0) {
      RCLCPP_WARN(get_logger(), "cuVSLAM 지도 저장 안 함 — 추적한 프레임 없음");
      return;
    }
    std::string msg;
    const bool ok = save_map(15.0, msg);
    if (ok) RCLCPP_INFO(get_logger(), "%s", msg.c_str());
    else RCLCPP_ERROR(get_logger(), "%s", msg.c_str());
  }

private:
  using Policy = message_filters::sync_policies::ApproximateTime<Image, Image>;
  using Sync = message_filters::Synchronizer<Policy>;

  // RViz 2D Pose Estimate — cuVSLAM은 리셋하지 않고 map 기준에 offset (추적은 계속)
  void on_initialpose(const geometry_msgs::msg::PoseWithCovarianceStamped & m) {
    if (!m.header.frame_id.empty() && m.header.frame_id != map_) {
      RCLCPP_WARN(get_logger(), "initialpose frame '%s' 무시 — %s만 지원", m.header.frame_id.c_str(), map_.c_str());
      return;
    }
    const auto & q = m.pose.pose.orientation;
    const Pose2 desired{m.pose.pose.position.x, m.pose.pose.position.y,
                        std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))};
    if (!last_base_) {
      pending_initialpose_ = desired;
      RCLCPP_INFO(get_logger(), "initialpose 대기 — 첫 추적 프레임 뒤 적용");
      return;
    }
    apply_initialpose(desired);
  }

  void apply_initialpose(const Pose2 & desired) {
    if (gate_) gate_->reset();  // 사람이 위치를 다시 맞춤 — 잠금 풀고 새 보정부터
    map_offset_ = ros_system::compose(desired, ros_system::inverse(*last_base_));
    path_.poses.clear();
    RCLCPP_WARN(get_logger(), "initialpose 적용: map (%+.2f, %+.2f, %+.1f°)", desired.x, desired.y, desired.yaw * 180.0 / M_PI);
  }

  // 첫 프레임: camera_info로 rig, URDF TF로 base_footprint←카메라. 단독 SLAM이면 루프 클로저(평면 제약) 켬
  void start(const std::string & frame) {
    geometry_msgs::msg::TransformStamped t;
    try {
      t = tf_buf_->lookupTransform(base_, frame, tf2::TimePointZero);
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "TF %s←%s 대기: %s", base_.c_str(), frame.c_str(), e.what());
      return;
    }
    base_from_cam_ = mat(t.transform);
    const auto & k = info_->k;
    cuvslam::Camera cam;
    cam.size = {static_cast<int32_t>(info_->width), static_cast<int32_t>(info_->height)};
    cam.principal = {static_cast<float>(k[2]), static_cast<float>(k[5])};
    cam.focal = {static_cast<float>(k[0]), static_cast<float>(k[4])};
    cam.distortion.model = cuvslam::Distortion::Model::Pinhole;
    // BEE_CUVSLAM_LEVEL=1: rig를 바닥과 수평(OpenCV 축)으로 두고 카메라는 그 안에서 숙인 것으로 — 이후 base_from_cam_ = base_from_rig
    if (level_) {
      Eigen::Matrix4d base_from_rig = Eigen::Matrix4d::Identity();
      base_from_rig.block<3, 3>(0, 0) << 0, 0, 1, -1, 0, 0, 0, -1, 0;
      base_from_rig.block<3, 1>(0, 3) = base_from_cam_.block<3, 1>(0, 3);
      const Eigen::Matrix4d rig_from_cam = base_from_rig.inverse() * base_from_cam_;
      const Eigen::Quaterniond q(Eigen::Matrix3d(rig_from_cam.block<3, 3>(0, 0)));
      cam.rig_from_camera.rotation = {static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z()),
                                      static_cast<float>(q.w())};
      cam.rig_from_camera.translation = {static_cast<float>(rig_from_cam(0, 3)), static_cast<float>(rig_from_cam(1, 3)),
                                         static_cast<float>(rig_from_cam(2, 3))};
      base_from_cam_ = base_from_rig;
      RCLCPP_WARN(get_logger(), "BEE_CUVSLAM_LEVEL=1 — rig 수평, 카메라는 rig 안에서 숙임 (rig_from_camera)");
    }
    cuvslam::Rig rig;
    rig.cameras.push_back(cam);
    cuvslam::Odometry::Config oc;
    oc.odometry_mode = cuvslam::Odometry::OdometryMode::RGBD;
    oc.async_sba = true;
    oc.rgbd_settings.depth_scale_factor = static_cast<float>(depth_scale_);
    oc.rgbd_settings.depth_camera_id = 0;
    oc.rgbd_settings.enable_depth_stereo_tracking = false;
    // SLAM은 Odometry 상태(관측·랜드마크)가 있어야 함 — 파이썬 Tracker도 SLAM이면 켬
    oc.enable_observations_export = oc.enable_landmarks_export = publish_map_tf_ || vis_;
    try {
      odom_trk_ = std::make_unique<cuvslam::Odometry>(rig, oc);
      if (publish_map_tf_) {
        cuvslam::Slam::Config sc;
        sc.planar_constraints = true;
        sc.max_map_size = max_map_size_;
        sc.max_landmarks_distance = max_lm_dist_;
        sc.throttling_time_ms = throttle_ms_;
        sc.enable_reading_internals = vis_;  // get_slam_landmarks · get_pose_graph
        slam_ = std::make_unique<cuvslam::Slam>(rig, odom_trk_->GetPrimaryCameras(), sc);
      }
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "cuVSLAM 시작 실패: %s", e.what());
      odom_trk_.reset();
      slam_.reset();
      return;
    }
    const auto ver = cuvslam::GetVersion(nullptr, nullptr, nullptr);
    RCLCPP_INFO(get_logger(), "cuVSLAM %.*s 시작 (%s) · %ux%u · 카메라 %s%s", static_cast<int>(ver.size()), ver.data(),
                slam_ ? "단독 SLAM" : "시각 오도메트리", info_->width, info_->height, frame.c_str(),
                frame_skip_ > 1 ? (" · " + std::to_string(frame_skip_) + "장에 1장 추적").c_str() : "");
  }

  // 메시지 한 장 → 버퍼 (GPU면 cudaMemcpy2D가 행 여백까지 처리). bpp = 픽셀당 바이트
  bool copy_in(const Image & m, Buf & b, size_t bpp) {
    const size_t row = static_cast<size_t>(m.width) * bpp;
    if (m.height == 0 || m.width == 0 || m.step < row || m.data.size() < static_cast<size_t>(m.height) * m.step) return false;
    if (m.is_bigendian && bpp > 1) return false;
    if (!b.fits(m.width, m.height, bpp, gpu_input_) && !b.alloc(m.width, m.height, bpp, gpu_input_))
      RCLCPP_WARN(get_logger(), "GPU 메모리 못 잡음 — CPU 버퍼로 (cuVSLAM이 안에서 복사)");
    if (b.gpu) return cudaMemcpy2DAsync(b.p, b.pitch, m.data.data(), m.step, row, m.height, cudaMemcpyHostToDevice, stream_) ==
                      cudaSuccess;
    auto * dst = static_cast<uint8_t *>(b.p);
    for (uint32_t r = 0; r < m.height; ++r) std::memcpy(dst + r * b.pitch, m.data.data() + r * m.step, row);
    return true;
  }

  static cuvslam::Image as_image(const Buf & b, int ch, cuvslam::ImageData::DataType dt, int64_t ns) {
    cuvslam::Image im;
    im.pixels = b.p;
    im.width = static_cast<int32_t>(b.w);
    im.height = static_cast<int32_t>(b.h);
    im.pitch = static_cast<int32_t>(b.pitch);
    im.encoding = ch == 3 ? cuvslam::ImageData::Encoding::RGB : cuvslam::ImageData::Encoding::MONO;
    im.data_type = dt;
    im.is_gpu_mem = b.gpu;
    im.timestamp_ns = ns;
    im.camera_index = 0;
    return im;
  }

  void on_rgbd(const Image & c, const Image & d) {
    if (!info_) return;
    if (frame_skip_ > 1 && (++frame_n_ % frame_skip_)) return;
    if (!odom_trk_) {
      start(c.header.frame_id);
      if (!odom_trk_) return;
    }
    if (c.height != d.height || c.width != d.width) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "컬러 %ux%u ≠ depth %ux%u", c.width, c.height, d.width, d.height);
      return;
    }
    const int64_t ns = rclcpp::Time(c.header.stamp).nanoseconds();
    if (ns <= last_ns_) return;  // cuVSLAM은 시각이 늘 늘어나야 함
    last_ns_ = ns;
    const int ch = (c.encoding == "rgb8" || c.encoding == "bgr8") ? 3 : 1;
    if (ch == 1 && c.encoding != "mono8") {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "컬러 인코딩 %s 지원 안 함 (rgb8 · bgr8 · mono8)", c.encoding.c_str());
      return;
    }
    if (d.encoding != "16UC1" && d.encoding != "mono16") {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "depth 인코딩 %s 지원 안 함 (16UC1)", d.encoding.c_str());
      return;
    }
    auto & B = bufs_[flip_ ^= 1];  // 두 벌을 번갈아 — 앞 프레임 버퍼는 건드리지 않음
    if (!copy_in(c, B.rgb, ch) || !copy_in(d, B.depth, 2)) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "영상 크기·step이 맞지 않음 (잘린 버퍼 · big endian · GPU 복사 실패)");
      return;
    }
    // 마스크도 depth와 같은 쪽에서 — GPU면 커널, 아니면 CPU (cuVSLAM은 한 번에 같은 종류만 받음)
    const bool gpu = B.rgb.gpu && B.depth.gpu;
    if (!B.mask.fits(c.width, c.height, 1, gpu)) B.mask.alloc(c.width, c.height, 1, gpu);
    if (gpu && B.mask.gpu) {
      if (near_mask_gpu(B.depth.p, B.depth.pitch, B.mask.p, B.mask.pitch, c.width, c.height, near_raw_, stream_) != cudaSuccess ||
          cudaStreamSynchronize(stream_) != cudaSuccess) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "GPU 마스크 실패: %s", cudaGetErrorString(cudaGetLastError()));
        return;
      }
    } else {
      for (uint32_t r = 0; r < c.height; ++r)
        ros_system::near_mask(reinterpret_cast<const uint16_t *>(static_cast<const uint8_t *>(B.depth.p) + r * B.depth.pitch),
                              static_cast<uint8_t *>(B.mask.p) + r * B.mask.pitch, c.width, near_raw_);
    }
    gpu_used_ = gpu && B.mask.gpu;
    if (B.rgb.gpu != gpu_used_ || B.depth.gpu != gpu_used_) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "영상·depth·마스크 버퍼 종류가 섞임 (GPU/CPU) — 이 프레임 건너뜀");
      return;
    }
    using DT = cuvslam::ImageData::DataType;
    const cuvslam::Odometry::ImageSet imgs{as_image(B.rgb, ch, DT::UINT8, ns)};
    const cuvslam::Odometry::ImageSet masks{as_image(B.mask, 1, DT::UINT8, ns)};
    const cuvslam::Odometry::ImageSet depths{as_image(B.depth, 1, DT::UINT16, ns)};
    cuvslam::PoseEstimate est;
    std::optional<cuvslam::Pose> slam_pose;
    try {
      est = odom_trk_->Track(imgs, masks, depths);
      if (slam_ && est.world_from_rig) {
        cuvslam::Odometry::State st;
        odom_trk_->GetState(st);
        slam_->Track(st);
        slam_pose = slam_->GetPose();
      }
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "cuVSLAM Track 실패: %s", e.what());
      return;
    }
    if (vis_) {
      last_vis_ = LastVis{c.header.stamp, ch, c.width, c.height, {}, {}};
      copy_gray(c, ch, last_vis_->gray);
      if (ch == 3) copy_rgb(c, last_vis_->rgb);
    }
    if (!est.world_from_rig) {
      ++lost_;
      prev_.reset();
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "추적 잃음");
      return;
    }
    ++ok_;
    const auto & p = est.world_from_rig->pose;
    const Pose2 cur = ros_system::base_pose(base_from_cam_, mat(p));
    if (prev_) {
      // 11-18: 자세가 튀거나 dt가 아주 짧으면 차분 속도가 수 m/s로 나와 EKF가 적분해 odom이 순간이동 (sim M10: 0.1 s에 1.78 m · −108°,
      //   벽 보정 없는 map 기준이라 2 m 틀린 채 남음). 몸이 낼 수 없는 속도(최고 0.26 m/s · 약 1.8 rad/s의 2배 넘게)는 버림
      const Pose2 v = ros_system::body_twist(prev_->second, cur, (ns - prev_->first) * 1e-9);
      if (std::hypot(v.x, v.y) <= max_twist_lin_ && std::abs(v.yaw) <= max_twist_ang_) publish_odom(c.header.stamp, v);
      else RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "속도 튐 버림 (%.2f m/s, %.2f rad/s)", std::hypot(v.x, v.y), v.yaw);
    }
    prev_ = std::make_pair(ns, cur);
    last_base_ = cur;
    if (pending_initialpose_) {
      apply_initialpose(*pending_initialpose_);
      pending_initialpose_.reset();
    }
    if (slam_ && !load_dir_.empty()) localize_step(c, ch, ns);
    if (publish_map_tf_) publish_map(c.header.stamp, slam_pose ? *slam_pose : p);  // 루프 클로저 반영된 SLAM 위치 우선
  }

  // SaveMap은 비동기 — 콜백을 기다림 (다른 스레드에서 불릴 수 있어 promise는 shared_ptr로 콜백이 늦어도 살아 있게)
  bool save_map(double timeout_s, std::string & msg) {
    if (!slam_) {
      msg = "cuVSLAM이 아직 시작 안 함 (첫 프레임 전)";
      return false;
    }
    auto done = std::make_shared<std::promise<bool>>();
    auto f = done->get_future();
    try {
      slam_->SaveMap(save_dir_, [done](bool ok) { done->set_value(ok); });
    } catch (const std::exception & e) {
      msg = std::string("cuVSLAM 지도 저장 실패: ") + e.what();
      return false;
    }
    if (f.wait_for(std::chrono::duration<double>(timeout_s)) != std::future_status::ready) {
      msg = "cuVSLAM 지도 저장 시간 초과 (" + std::to_string(timeout_s) + " s): " + save_dir_;
      return false;
    }
    const bool ok = f.get();
    msg = (ok ? "cuVSLAM 지도 저장: " : "cuVSLAM 지도 저장 실패: ") + save_dir_;
    return ok;
  }

  // 위치추정: 저장 지도 안에서 지금 자리 찾기 — 요청은 하나씩, 결과는 다음 프레임에서 받음 (콜백은 cuVSLAM 스레드)
  void localize_step(const Image & c, int ch, int64_t ns) {
    if (loc_done_.exchange(false)) {
      std::lock_guard<std::mutex> lk(loc_mu_);
      if (loc_ok_) {
        localized_ = true;
        if (gate_) gate_->reset();  // map 원점이 저장 지도 쪽으로 옮겨 감 — 튐으로 잠그지 않게
        map_offset_ = Pose2{};      // 켠 자리 기준으로 받은 initialpose · 도면 보정은 저장 지도 원점에서는 안 맞음
        path_.poses.clear();
        RCLCPP_WARN(get_logger(), "저장 지도 안에서 위치 찾음 (%d번째 시도) — map 원점 = 그 지도의 출발 자리", loc_tries_);
        return;
      }
      RCLCPP_WARN(get_logger(), "저장 지도에서 위치 못 찾음 (%d번째): %s — %.0f s 뒤 다시", loc_tries_, loc_err_.c_str(), loc_retry_s_);
    }
    if (localized_ || loc_busy_) return;
    const double now = ns * 1e-9;
    if (now - loc_last_try_ < loc_retry_s_) return;
    loc_last_try_ = now;
    // 영상은 요청이 끝날 때까지 쓰이므로 따로 CPU 사본 (추적용 두 벌 버퍼는 다음 프레임이 덮어씀)
    const size_t row = static_cast<size_t>(c.width) * ch;
    if (!loc_buf_.fits(c.width, c.height, ch, false)) loc_buf_.alloc(c.width, c.height, ch, false);
    auto * dst = static_cast<uint8_t *>(loc_buf_.p);
    for (uint32_t r = 0; r < c.height; ++r) std::memcpy(dst + r * loc_buf_.pitch, c.data.data() + r * c.step, row);
    const cuvslam::Slam::ImageSet imgs{as_image(loc_buf_, ch, cuvslam::ImageData::DataType::UINT8, ns)};
    ++loc_tries_;
    loc_busy_ = true;
    try {
      slam_->LocalizeInMap(load_dir_, ns, slam_->GetPose(), imgs, loc_, [] {},
                           [this](const cuvslam::Result<cuvslam::Pose> & r) {
                             std::lock_guard<std::mutex> lk(loc_mu_);
                             loc_ok_ = r.data.has_value();
                             loc_err_ = std::string(r.error_message);  // string_view는 콜백 밖에서 못 씀
                             loc_busy_ = false;
                             loc_done_ = true;
                           });
    } catch (const std::exception & e) {
      loc_busy_ = false;
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "LocalizeInMap 실패: %s", e.what());
    }
  }

  // 속도만 (자세는 안 씀 — cuVSLAM 좌표는 추적을 잃으면 새로 시작해 자세를 EKF에 그대로 넣으면 튐)
  void publish_odom(const builtin_interfaces::msg::Time & stamp, const Pose2 & v) {
    const double late = (now() - rclcpp::Time(stamp, RCL_ROS_TIME)).seconds();
    if (late > max_odom_delay_) {
      ++stale_;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "늦은 측정 버림 (%.2f s > %.2f s · 누적 %zu)", late, max_odom_delay_, stale_);
      return;
    }
    nav_msgs::msg::Odometry o;
    o.header.stamp = stamp;
    o.header.frame_id = odom_;
    o.child_frame_id = base_;
    o.twist.twist.linear.x = v.x;
    o.twist.twist.linear.y = v.y;
    o.twist.twist.angular.z = v.yaw;
    o.twist.covariance[0] = o.twist.covariance[7] = lin_var_;
    o.twist.covariance[35] = ang_var_;
    o.twist.covariance[14] = o.twist.covariance[21] = o.twist.covariance[28] = 1e6;  // 안 쓰는 축
    odom_pub_->publish(o);
  }

  void publish_map(const builtin_interfaces::msg::Time & stamp, const cuvslam::Pose & p) {
    geometry_msgs::msg::TransformStamped o;
    try {
      o = tf_buf_->lookupTransform(odom_, base_, rclcpp::Time(stamp));
    } catch (const tf2::ExtrapolationException &) {
      // 영상이 EKF의 마지막 odom TF보다 먼저 오면 (bag 재생에서 15~35 ms — 11-19) 그 시각이 아직 없음 → 거의 모든 프레임을 건너뛰어
      //   map→odom이 처음 한 번만 나왔다. odom_lookup_tolerance 안이면 가장 최근 TF로 (0.26 m/s에서 0.1 s = 2.6 cm 이하)
      try {
        o = tf_buf_->lookupTransform(odom_, base_, tf2::TimePointZero);
      } catch (const std::exception &) {
        return;
      }
      if (std::abs((rclcpp::Time(stamp) - rclcpp::Time(o.header.stamp)).seconds()) > odom_tol_) return;
    } catch (const std::exception &) {  // 그 시각 odom이 없으면 이 프레임은 TF만 건너뜀 (추적은 계속)
      return;
    }
    const Pose2 odom = ros_system::planar(mat(o.transform));
    const Pose2 base_raw = odom_anchor_ ? odom : ros_system::base_pose(base_from_cam_, mat(p));
    last_base_ = base_raw;
    Pose2 base = ros_system::compose(map_offset_, base_raw);
    Pose2 mo = ros_system::map_to_odom(base, odom);
    const double ts = rclcpp::Time(stamp).seconds();
    bool rotating = false;
    if (prev_odom_ && ts > prev_odom_->first) {
      const double wz = std::remainder(odom.yaw - prev_odom_->second.yaw, 2 * M_PI) / (ts - prev_odom_->first);
      rotating = hold_wz_ > 0.0 && std::abs(wz) > hold_wz_;
    }
    prev_odom_ = std::make_pair(ts, odom);
    if (gate_) {
      bool accepted = false;
      mo = gate_->update(mo, ts, odom, accepted, rotating);
      if (gate_->resettled) {
        RCLCPP_WARN(get_logger(), "map→odom 출발 정착 중 다시 맞춤 — 보정이 %.2f m · %.1f° 바뀜 (gate_settle_s 안이라 잠그지 않고 받음 · "
                                  "map = cuVSLAM 출발 자리 기준)", gate_->resettled->x, gate_->resettled->y);
        gate_->resettled.reset();
      }
      if (!accepted) {
        if (gate_->locked)
          RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
                                "map→odom 보정 잠금 — cuVSLAM 위치가 크게 튐, 바퀴+IMU로만 이어 감 (/initialpose로 다시 맞추면 풀림)");
        base = ros_system::compose(mo, odom);  // 낸 보정 기준 로봇 자세 (경로 표시)
      }
    }
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = rclcpp::Time(stamp) + map_tf_tolerance_;
    tf.header.frame_id = map_;
    tf.child_frame_id = odom_;
    tf.transform.translation.x = mo.x;
    tf.transform.translation.y = mo.y;
    tf.transform.rotation.z = std::sin(mo.yaw / 2);
    tf.transform.rotation.w = std::cos(mo.yaw / 2);
    if (relay_) map_odom_pub_->publish(tf);
    else tf_pub_->sendTransform(tf);
    geometry_msgs::msg::PoseStamped ps;
    ps.header = tf.header;
    ps.pose.position.x = base.x;
    ps.pose.position.y = base.y;
    ps.pose.orientation.z = std::sin(base.yaw / 2);
    ps.pose.orientation.w = std::cos(base.yaw / 2);
    path_.header.stamp = stamp;
    path_.poses.push_back(ps);
    if (path_.poses.size() > 300) path_.poses.erase(path_.poses.begin(), path_.poses.end() - 300);  // 최근 300개만 (보기용)
    if (ok_ % 5 == 0 && path_pub_->get_subscription_count() > 0) path_pub_->publish(path_);
  }

  // BEE_CUVSLAM_VIS: 특징점(영상 위 점) · 그 프레임 랜드마크 · SLAM 지도 랜드마크 · 루프 클로저 랜드마크 · 포즈 그래프
  void publish_vis() {
    if (!odom_trk_ || !last_vis_) return;
    const auto & lv = *last_vis_;
    // 회색 영상(컬러면 G 채널) → bgr8 + 특징점 초록 점 (3x3)
    Image img;
    img.header.stamp = lv.stamp;
    img.header.frame_id = "pole_front_cam_color_optical_frame";
    img.width = lv.w;
    img.height = lv.h;
    img.encoding = "bgr8";
    img.step = lv.w * 3;
    img.data.resize(static_cast<size_t>(img.step) * lv.h);
    for (size_t i = 0; i < lv.gray.size(); ++i) img.data[i * 3] = img.data[i * 3 + 1] = img.data[i * 3 + 2] = lv.gray[i];
    try {
      const auto obs = odom_trk_->GetLastObservations(0);
      if (!lv.rgb.empty()) {
        if (lm_rgb_.size() > 500000) lm_rgb_.clear();  // 메모리 한도 (보기용)
        for (const auto & o : obs) {
          const int u = static_cast<int>(o.u), v = static_cast<int>(o.v);
          if (u < 0 || v < 0 || u >= static_cast<int>(lv.w) || v >= static_cast<int>(lv.h)) continue;
          const uint8_t * px = &lv.rgb[(static_cast<size_t>(v) * lv.w + u) * 3];
          lm_rgb_[o.id] = (uint32_t{px[0]} << 16) | (uint32_t{px[1]} << 8) | px[2];
        }
      }
      for (const auto & o : obs)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            const int u = static_cast<int>(o.u) + dx, v = static_cast<int>(o.v) + dy;
            if (u < 0 || v < 0 || u >= static_cast<int>(lv.w) || v >= static_cast<int>(lv.h)) continue;
            auto * px = &img.data[(static_cast<size_t>(v) * lv.w + u) * 3];
            px[0] = 0;
            px[1] = 255;
            px[2] = 0;
          }
    } catch (const std::exception &) {
    }
    vis_obs_->publish(img);
    std_msgs::msg::Header hdr;
    hdr.stamp = lv.stamp;
    hdr.frame_id = map_;
    // xyz + rgb (RViz Color Transformer RGB8) — 색 모르는 점은 회색
    auto cloud = [&](const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr & pub, const std::vector<Eigen::Vector3d> & pts,
                     const std::vector<uint64_t> & ids) {
      if (pts.empty()) return;
      sensor_msgs::msg::PointCloud2 pc;
      pc.header = hdr;
      sensor_msgs::PointCloud2Modifier mod(pc);
      mod.setPointCloud2FieldsByString(2, "xyz", "rgb");
      mod.resize(pts.size());
      sensor_msgs::PointCloud2Iterator<float> x(pc, "x"), y(pc, "y"), z(pc, "z");
      sensor_msgs::PointCloud2Iterator<uint8_t> cr(pc, "r"), cg(pc, "g"), cb(pc, "b");
      for (size_t i = 0; i < pts.size(); ++i, ++cr, ++cg, ++cb) {
        const auto it = lm_rgb_.find(ids[i]);
        const uint32_t col = it == lm_rgb_.end() ? 0x808080 : it->second;
        *cr = static_cast<uint8_t>(col >> 16);
        *cg = static_cast<uint8_t>(col >> 8);
        *cb = static_cast<uint8_t>(col);
        const auto & p = pts[i];
        const Eigen::Vector3d q = base_from_cam_.block<3, 3>(0, 0) * p + base_from_cam_.block<3, 1>(0, 3);
        *x = static_cast<float>(q.x());
        *y = static_cast<float>(q.y());
        *z = static_cast<float>(q.z());
        ++x;
        ++y;
        ++z;
      }
      pub->publish(pc);
    };
    try {
      std::vector<Eigen::Vector3d> lm;
      std::vector<uint64_t> ids;
      for (const auto & l : odom_trk_->GetLastLandmarks()) {
        lm.emplace_back(l.coords[0], l.coords[1], l.coords[2]);
        ids.push_back(l.id);
      }
      cloud(vis_lm_, lm, ids);
    } catch (const std::exception &) {
    }
    if (!slam_) return;
    using L = cuvslam::Slam::DataLayer;
    for (auto [layer, pub] : {std::pair{L::Map, vis_map_}, std::pair{L::LoopClosure, vis_loop_}}) {
      slam_->EnableReadingData(layer, 100000);  // 파이썬 바인딩과 같게 — 읽을 때마다 켬
      if (const auto ls = slam_->ReadLandmarks(layer)) {
        std::vector<Eigen::Vector3d> pts;
        std::vector<uint64_t> ids;
        for (const auto & l : ls->landmarks) {
          pts.emplace_back(l.coords[0], l.coords[1], l.coords[2]);
          ids.push_back(l.id);
        }
        cloud(pub, pts, ids);
      }
    }
    slam_->EnableReadingData(L::PoseGraph, 1000);
    const auto g = slam_->ReadPoseGraph();
    if (!g) return;
    std::map<uint64_t, Pose2> xy;
    for (const auto & n : g->nodes) xy[n.id] = ros_system::base_pose(base_from_cam_, mat(n.node_pose));
    visualization_msgs::msg::Marker nodes, edges;
    for (auto [mk, ns, typ, sc, r, gg, b] : {std::tuple{&nodes, "nodes", visualization_msgs::msg::Marker::SPHERE_LIST, 0.04, 1.0, 0.6, 0.0},
                                             std::tuple{&edges, "edges", visualization_msgs::msg::Marker::LINE_LIST, 0.01, 0.2, 0.8, 1.0}}) {
      mk->header = hdr;
      mk->ns = ns;
      mk->type = typ;
      mk->action = visualization_msgs::msg::Marker::ADD;
      mk->scale.x = mk->scale.y = mk->scale.z = sc;
      mk->color.r = static_cast<float>(r);
      mk->color.g = static_cast<float>(gg);
      mk->color.b = static_cast<float>(b);
      mk->color.a = 1.0f;
      mk->pose.orientation.w = 1.0;
    }
    auto pt = [](const Pose2 & p) {
      geometry_msgs::msg::Point q;
      q.x = p.x;
      q.y = p.y;
      q.z = 0.05;
      return q;
    };
    for (const auto & [id, p] : xy) nodes.points.push_back(pt(p));
    for (const auto & e : g->edges) {
      const auto a = xy.find(e.node_from), b = xy.find(e.node_to);
      if (a == xy.end() || b == xy.end()) continue;
      edges.points.push_back(pt(a->second));
      edges.points.push_back(pt(b->second));
    }
    visualization_msgs::msg::MarkerArray ma;
    ma.markers = {nodes, edges};
    vis_graph_->publish(ma);
  }

  std::string base_, odom_, map_;
  bool publish_map_tf_ = true, relay_ = true, gpu_input_ = true, gpu_used_ = false, level_ = false, vis_ = false;
  rclcpp::Duration map_tf_tolerance_{0, 0};
  double hold_wz_ = 0.2;
  std::optional<std::pair<double, Pose2>> prev_odom_;  // 직전 프레임 odom 자세 (회전 속도)
  double depth_scale_ = 1000.0, min_depth_ = 0.5, lin_var_ = 0.0025, ang_var_ = 0.0025;
  int64_t frame_skip_ = 1, frame_n_ = 0, last_ns_ = 0;
  uint16_t near_raw_ = 0;
  uint32_t max_map_size_ = 150;
  float max_lm_dist_ = 100.f;
  uint32_t throttle_ms_ = 0;
  long ok_ = 0, lost_ = 0;
  std::optional<ros_system::MapGate> gate_;
  Eigen::Matrix4d base_from_cam_ = Eigen::Matrix4d::Identity();
  Pose2 map_offset_;
  double max_twist_lin_ = 0.6, max_twist_ang_ = 4.0, odom_tol_ = 0.1, max_odom_delay_ = 0.5;
  size_t stale_ = 0;
  bool odom_anchor_ = false;
  uint64_t n_corr_ignored_ = 0, n_target_ignored_ = 0;
  std::optional<Pose2> last_base_, pending_initialpose_;
  std::optional<std::pair<int64_t, Pose2>> prev_;  // 직전 오도메트리 자세 — 추적을 잃으면 비움
  // 위치추정 콜백이 쓰는 것들은 slam_보다 먼저 선언 — 소멸 때 slam_(작업 스레드)이 먼저 없어지게
  std::string save_dir_, load_dir_;
  cuvslam::Slam::LocalizationSettings loc_{};
  double loc_retry_s_ = 2.0, loc_last_try_ = -1e18;
  std::atomic<bool> loc_busy_{false}, loc_done_{false};
  bool loc_ok_ = false, localized_ = false;
  int loc_tries_ = 0;
  std::string loc_err_;
  std::mutex loc_mu_;
  Buf loc_buf_;  // LocalizeInMap에 넘긴 영상 (요청이 끝날 때까지)
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_srv_;
  std::unique_ptr<cuvslam::Odometry> odom_trk_;
  std::unique_ptr<cuvslam::Slam> slam_;
  struct Bufs {
    Buf rgb, depth, mask;
  };
  cudaStream_t stream_ = nullptr;
  std::array<Bufs, 2> bufs_;
  int flip_ = 0;
  struct LastVis {
    builtin_interfaces::msg::Time stamp;
    int ch;
    uint32_t w, h;
    std::vector<uint8_t> gray;  // 보기용 회색 (컬러면 G 채널) — GPU 버퍼는 CPU에서 못 읽어 메시지에서 따로 둠
    std::vector<uint8_t> rgb;   // 랜드마크 색 (r g b 순서) — 컬러 입력일 때만
  };
  static void copy_rgb(const Image & c, std::vector<uint8_t> & o) {
    const bool bgr = c.encoding == "bgr8";
    o.resize(static_cast<size_t>(c.width) * c.height * 3);
    for (uint32_t y = 0; y < c.height; ++y)
      for (uint32_t x = 0; x < c.width; ++x) {
        const uint8_t * s = &c.data[y * c.step + x * 3];
        uint8_t * d = &o[(static_cast<size_t>(y) * c.width + x) * 3];
        d[0] = s[bgr ? 2 : 0];
        d[1] = s[1];
        d[2] = s[bgr ? 0 : 2];
      }
  }
  // 랜드마크 id → 색 (0xRRGGBB) — 관측된 프레임의 그 픽셀. SLAM 지도 랜드마크도 같은 id면 이 색, 없으면 회색
  std::unordered_map<uint64_t, uint32_t> lm_rgb_;
  static void copy_gray(const Image & c, int ch, std::vector<uint8_t> & g) {
    g.resize(static_cast<size_t>(c.width) * c.height);
    for (uint32_t y = 0; y < c.height; ++y)
      for (uint32_t x = 0; x < c.width; ++x) g[y * c.width + x] = c.data[y * c.step + x * ch + (ch == 3 ? 1 : 0)];
  }
  std::optional<LastVis> last_vis_;
  nav_msgs::msg::Path path_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr info_;
  std::shared_ptr<tf2_ros::Buffer> tf_buf_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr map_odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr init_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr corr_sub_, target_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  std::shared_ptr<message_filters::Subscriber<Image>> color_sub_, depth_sub_;
  std::shared_ptr<Sync> sync_;
  rclcpp::TimerBase::SharedPtr stats_, vis_timer_;
  rclcpp::Publisher<Image>::SharedPtr vis_obs_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr vis_lm_, vis_map_, vis_loop_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr vis_graph_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CuvslamSlam>();
  rclcpp::spin(node);
  node->on_exit();  // 매핑이면 cuVSLAM 지도 저장 (ROS 통신 없이 — 이미 shutdown)
  rclcpp::shutdown();
  return 0;
}
