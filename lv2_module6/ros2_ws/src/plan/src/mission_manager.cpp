// mission_manager 구현 — 노드 설명·입출력은 include/plan/mission_manager.hpp, main은 mission_manager_main.cpp
#include "plan/mission_manager.hpp"
#include "plan/shared_tf.hpp"

#include "ament_index_cpp/get_package_share_directory.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <limits>
#include <sstream>
#include <fstream>
#include <set>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace plan {

namespace {
constexpr uint32_t PLAN_ID_BASE = 1000000;

// 물체의 details 값 (없으면 "")
std::string detail(const perception::msg::TrackedObject & o, const std::string & key) {
  for (const auto & kv : o.details) if (kv.key == key) return kv.value;
  return "";
}

std::vector<Box> boxes(const std::vector<double> & flat, double half) {  // [x, y, …] 또는 half < 0이면 [x, y, half, …]
  std::vector<Box> out;
  const size_t n = half < 0 ? 3 : 2;
  for (size_t i = 0; i + n <= flat.size(); i += n) out.push_back({flat[i], flat[i + 1], half < 0 ? flat[i + 2] : half});
  return out;
}

std::string fmt(const char * f, double a, double b = 0, double c = 0) {
  char buf[96];
  std::snprintf(buf, sizeof(buf), f, a, b, c);
  return buf;
}

std::string q(const std::string & s) {  // JSON 문자열 (따옴표·역슬래시·줄바꿈만 — 이유·target은 그 밖 제어 문자가 없음)
  std::string out = "\"";
  for (const char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c == '\n' ? ' ' : c;
  }
  return out + "\"";
}
double wall_s() {  // 기록용 벽시계 [s, epoch] — 시뮬 /clock과 상관없이 실제 걸린 시간
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}
std::string str(int v) { return std::to_string(v); }
std::string join(const std::vector<std::string> & v) {
  std::string out;
  for (const auto & s : v) out += (out.empty() ? "" : ", ") + s;
  return out;
}
bool is_puck(const perception::msg::TrackedObject & o) { return o.class_name.rfind("puck", 0) == 0; }
bool is_lying(const perception::msg::TrackedObject & o) { return o.class_name == "puck_lying"; }
// coordinator가 사람·안전 때문에 멈춘 실패 — 다시 시도하지 않고 미션을 멈춤 (estop · 수동/탐사 전환 · 다른 계획이 끼어듦)
bool interrupted(const std::string & why) {
  for (const char * p : {"estop", "manual", "explore", "cancelled", "preempted"})
    if (why.rfind(p, 0) == 0) return true;
  return false;
}
// 주행 전 팔을 접음 (05-arm.md — 주행 동안 팔은 stow. 놓기·보기 뒤 팔이 박스 위에 나가 있음)
SubGoal stow() { return SubGoal().set__skill("pose").set__target("stow"); }
}  // namespace

const perception::msg::TrackedObject * puck_for(const perception::msg::TrackedObjectArray * objs, int tag,
                                                const double * spot_xy, double radius) {
  if (!objs) return nullptr;
  const perception::msg::TrackedObject * best = nullptr;
  for (const auto & o : objs->objects)
    if (is_puck(o) && detail(o, "tag") == str(tag) &&
        (!best || rclcpp::Time(o.last_seen, RCL_ROS_TIME) > rclcpp::Time(best->last_seen, RCL_ROS_TIME))) best = &o;
  if (best || !spot_xy) return best;
  double best_d = radius;
  for (const auto & o : objs->objects) {
    if (!is_puck(o)) continue;
    const std::string t = detail(o, "tag");
    if (!t.empty() && t != str(tag)) continue;
    const double d = std::hypot(o.pose.pose.position.x - spot_xy[0], o.pose.pose.position.y - spot_xy[1]);
    if (d < best_d) { best_d = d; best = &o; }
  }
  return best;
}

MissionManager::MissionManager(const rclcpp::NodeOptions & options)
: rclcpp::Node("mission_manager", options) {
  phase_beeps_ = declare_parameter<bool>("phase_beeps", true);
  const auto sp = declare_parameter<std::vector<double>>("start_pose_arena", {0.734, 0.950, M_PI});
  start_ = {sp.at(0), sp.at(1), sp.at(2)};
  start_zone_ = declare_parameter<std::vector<double>>("start_zone", {0.434, 0.650, 1.034, 1.250});
  footprint_ = declare_parameter<std::vector<double>>("footprint", {0.069, -0.233, 0.153});
  const double box_half = declare_parameter<double>("box_half", 0.1025);
  start_boxes_ = boxes(declare_parameter<std::vector<double>>("start_boxes",
    {1.468, 0.950, 3.670, 3.152, 2.936, 5.354, 0.734, 5.354, 2.202, 5.354, 3.670, 3.886}), box_half);
  for (const auto & b : boxes(declare_parameter<std::vector<double>>("locations",
         {3.670, 0.950, 2.202, 3.700, 3.670, 5.354}), 0.0)) locations_.push_back({b.x, b.y});
  loc_boxes_ = boxes(declare_parameter<std::vector<double>>("location_boxes",
    {3.670, 0.950, 0.1025, 2.202, 3.700, 0.1025, 3.670, 5.354, 0.1025}), -1.0);
  for (const auto v : declare_parameter<std::vector<int64_t>>("location_tags", {12, 13, 14})) loc_tags_.push_back(static_cast<int>(v));
  // 박스마다 다가갈 변 (11-13 사용자 지정) — 한 줄에 "+x -y"처럼, 비거나 개수가 안 맞으면 네 변 다 (벽 거르기만)
  const auto set_sides = [this](std::vector<Box> & bs, const char * name) {
    const auto v = declare_parameter<std::vector<std::string>>(name, std::vector<std::string>{});
    if (v.empty()) return;
    if (v.size() != bs.size()) {
      RCLCPP_WARN(get_logger(), "%s %zu개 ≠ 박스 %zu개 — 무시 (네 변 다)", name, v.size(), bs.size());
      return;
    }
    for (size_t i = 0; i < bs.size(); ++i) {
      std::istringstream in(v[i]);
      for (std::string t; in >> t;) bs[i].sides.push_back(t);
    }
  };
  set_sides(start_boxes_, "start_box_sides");
  set_sides(loc_boxes_, "location_box_sides");
  // 11-18 사용자: 장소(L1~L3)는 적은 순서 말고 로봇에서 가까운 면부터 — 네 면 다 들어갈 수 있는데 첫 면 고정이면 반대편으로 돌아감 (sim L3→L2)
  //   시작 박스는 적은 순서 유지 (첫 면 = 붙은 뒤 돌 공간까지 확인한 진입 면)
  if (declare_parameter<bool>("location_side_nearest", true))
    for (auto & b : loc_boxes_) b.nearest = true;
  // 박스마다 더 물러날 거리 — 칸 안 박스(B6)는 다가설 자리가 아직 칸 입구 안이라 돌 공간이 없어 다음 주행이 85 s 갇혔음 (11-15 시뮬)
  const auto retreat = declare_parameter<std::vector<double>>("start_box_retreat", std::vector<double>{});
  if (!retreat.empty() && retreat.size() != start_boxes_.size())
    RCLCPP_WARN(get_logger(), "start_box_retreat %zu개 ≠ 박스 %zu개 — 무시", retreat.size(), start_boxes_.size());
  else
    for (size_t i = 0; i < retreat.size(); ++i) start_boxes_[i].retreat = retreat[i];
  // 장소 상자도 같은 방식 (11-22 사용자) — exit_turn_clear_m(회전 여유) 위에 상자마다 더 물러날 거리 [L1, L2, L3]
  const auto loc_retreat = declare_parameter<std::vector<double>>("location_box_retreat", std::vector<double>{});
  if (!loc_retreat.empty() && loc_retreat.size() != loc_boxes_.size())
    RCLCPP_WARN(get_logger(), "location_box_retreat %zu개 ≠ 장소 %zu개 — 무시", loc_retreat.size(), loc_boxes_.size());
  else
    for (size_t i = 0; i < loc_retreat.size(); ++i) loc_boxes_[i].retreat = loc_retreat[i];
  puck_dx_ = declare_parameter<std::vector<double>>("puck_offsets_x", {-0.040, 0.040});
  box_top_z_ = declare_parameter<double>("box_top_z", 0.070);
  puck_h_ = declare_parameter<double>("puck_height", 0.060);
  grid_step_ = declare_parameter<double>("grid_step", 0.060);
  standoff_ = declare_parameter<double>("standoff", 0.25);
  approach_gap_ = declare_parameter<double>("approach_gap", 0.085);
  // 장소 상자(L1~L3)만 더 바짝 · 앞면 거리로 멈춤 (11-21 사용자: place 때 4~5 cm) — 시작 상자는 approach_gap · odom 목표 그대로
  {
    const double g = declare_parameter<double>("place_approach_gap", -1.0), f = declare_parameter<double>("place_face_stop_m", 0.0);
    for (auto & b : loc_boxes_) { b.gap = g; b.face_stop = f; }
  }
  pick_reach_m_ = declare_parameter<double>("pick_reach_m", pick_reach_m_);
  creep_min_gap_ = declare_parameter<double>("creep_min_gap", creep_min_gap_);
  back_min_x_ = declare_parameter<double>("inspect_back_min_x", back_min_x_);
  back_want_x_ = declare_parameter<double>("inspect_back_want_x", back_want_x_);
  back_max_m_ = declare_parameter<double>("inspect_back_max_m", back_max_m_);
  back_recent_s_ = declare_parameter<double>("inspect_back_recent_s", back_recent_s_);
  nav_turn_m_ = declare_parameter<double>("nav_turn_m", 0.10);
  approach_tries_ = static_cast<int>(declare_parameter<int64_t>("approach_tries", 3));
  retries_ = static_cast<int>(declare_parameter<int64_t>("retries", 2));
  load_tries_ = static_cast<int>(declare_parameter<int64_t>("survey_load_tries", load_tries_));
  put_back_ring_m_ = declare_parameter<double>("put_back_ring_m", put_back_ring_m_);
  put_back_clear_m_ = declare_parameter<double>("put_back_clear_m", put_back_clear_m_);
  put_back_x_m_ = declare_parameter<double>("put_back_x_m", put_back_x_m_);
  match_radius_ = declare_parameter<double>("match_radius", 0.04);
  stand_check_s_ = declare_parameter<double>("stand_check_s", 1.0);
  stand_find_m_ = declare_parameter<double>("stand_find_m", 0.10);  // [m] stand 뒤 선 퍽을 누운 가운데에서 이 안에서 찾음 → pick
  verify_radius_ = declare_parameter<double>("verify_radius", 0.15);
  verify_in_place_ = declare_parameter<bool>("verify_in_place", true);
  declare_parameter<std::string>("start_stop", "start");  // 미션 시작 때 로봇이 선 곳 (leg 이름) — 시뮬 구간 시험용 (ros2 param set)
  // 방문 순서 (1부터) — 경기장 고정이라 오프라인 A*(tools/plan/route_order.py, 도면 벽)로 정한 값. 비거나 틀리면 1, 2, 3, …
  survey_order_ = visit_order(declare_parameter<std::vector<int64_t>>("survey_order", std::vector<int64_t>{}), start_boxes_.size());
  build_loc_order_ = visit_order(declare_parameter<std::vector<int64_t>>("build_location_order", std::vector<int64_t>{}), 3);
  build_nearest_ = declare_parameter<bool>("build_location_nearest", build_nearest_);
  walls_ = declare_parameter<std::vector<double>>("arena_walls", std::vector<double>{});
  keepout_ = declare_parameter<std::vector<double>>("keepout_zones", std::vector<double>{});
  wall_clearance_ = declare_parameter<double>("wall_clearance", 0.22);
  place_tol_ = declare_parameter<double>("place_tol", 0.010);
  place_search_ = declare_parameter<double>("place_search", 0.05);
  loc_keep_s_ = declare_parameter<double>("loc_box_keep_s", loc_keep_s_);
  place_reach_m_ = declare_parameter<double>("place_reach_m", place_reach_m_);
  realign_m_ = declare_parameter<double>("place_realign_m", realign_m_);
  realign_wait_s_ = declare_parameter<double>("place_realign_wait_s", realign_wait_s_);
  realign_tries_ = static_cast<int>(declare_parameter<int64_t>("place_realign_tries", 2));
  exit_turn_clear_ = declare_parameter<double>("exit_turn_clear_m", exit_turn_clear_);
  view_m_ = declare_parameter<double>("place_view_m", 0.80);
  loc_any_.assign(loc_boxes_.size(), std::nullopt);
  corr_max_ = declare_parameter<double>("tag_correction_max", 0.6);
  acorr_on_ = declare_parameter<bool>("arena_correction", acorr_on_);
  acorr_idle_ = declare_parameter<bool>("arena_corr_idle", acorr_idle_);
  wall_fit_ = declare_parameter<bool>("arena_corr_wall_fit", wall_fit_);
  acorr_boxes_ = declare_parameter<bool>("arena_corr_boxes", acorr_boxes_);
  acorr_gate_ = declare_parameter<double>("arena_corr_gate", acorr_gate_);
  acorr_max_shift_ = declare_parameter<double>("arena_corr_max_shift", acorr_max_shift_);
  acorr_max_rot_ = declare_parameter<double>("arena_corr_max_rot", acorr_max_rot_);
  acorr_min_sep_ = declare_parameter<double>("arena_corr_min_sep", acorr_min_sep_);
  acorr_alpha_ = declare_parameter<double>("arena_corr_alpha", acorr_alpha_);
  acorr_range_ = declare_parameter<double>("arena_corr_range", acorr_range_);
  acorr_outlier_ = declare_parameter<double>("arena_corr_outlier", acorr_outlier_);
  acorr_step_m_ = declare_parameter<double>("arena_corr_step_m", acorr_step_m_);
  acorr_step_rad_ = declare_parameter<double>("arena_corr_step_rad", acorr_step_rad_);
  acorr_walls_m_ = declare_parameter<double>("arena_corr_walls_m", acorr_walls_m_);
  acorr_walls_rad_ = declare_parameter<double>("arena_corr_walls_rad", acorr_walls_rad_);
  acorr_min_hits_ = static_cast<int>(declare_parameter<int64_t>("arena_corr_min_hits", acorr_min_hits_));
  near_perceived_ = declare_parameter<bool>("approach_perceived", near_perceived_);
  loc_on_ = declare_parameter<bool>("arena_corr_to_localization", loc_on_);
  loc_min_n_ = static_cast<int>(declare_parameter<int64_t>("arena_corr_loc_min_n", loc_min_n_));
  loc_step_m_ = declare_parameter<double>("arena_corr_loc_step_m", loc_step_m_);
  loc_step_rad_ = declare_parameter<double>("arena_corr_loc_step_rad", loc_step_rad_);
  loc_min_m_ = declare_parameter<double>("arena_corr_loc_min_m", loc_min_m_);
  loc_min_rad_ = declare_parameter<double>("arena_corr_loc_min_rad", loc_min_rad_);
  loc_settle_s_ = declare_parameter<double>("arena_corr_loc_settle_s", loc_settle_s_);
  const double loc_period = declare_parameter<double>("arena_corr_loc_period", 2.0);
  signal_timeout_s_ = declare_parameter<double>("signal_timeout_s", 150.0);
  signal_preroll_s_ = declare_parameter<double>("signal_preroll_s", signal_preroll_s_);
  occlude_rad_ = declare_parameter<double>("barcode_occlude_rad", occlude_rad_);
  occlude_lost_s_ = declare_parameter<double>("barcode_occlude_lost_s", occlude_lost_s_);
  signal_error_cycles_ = static_cast<int>(declare_parameter<int64_t>("signal_error_cycles", 2));
  barcode_timeout_s_ = declare_parameter<double>("barcode_timeout_s", 20.0);
  barcode_arm_pose_ = declare_parameter<std::string>("barcode_arm_pose", "gaze");
  barcode_pole_s_ = declare_parameter<double>("barcode_pole_s", 3.0);
  light_check_s_ = declare_parameter<double>("light_check_s", light_check_s_);
  signal_read_min_s_ = declare_parameter<double>("signal_read_min_s", signal_read_min_s_);
  barcode_gaze_wait_s_ = declare_parameter<double>("barcode_gaze_wait_s", 3.0);
  barcode_advance_step_m_ = declare_parameter<double>("barcode_advance_step_m", 0.06);
  barcode_advance_max_m_ = declare_parameter<double>("barcode_advance_max_m", 0.30);
  barcode_min_gap_m_ = declare_parameter<double>("barcode_min_gap_m", 0.13);
  barcode_gaze_after_signal_ = declare_parameter<bool>("barcode_gaze_after_signal", barcode_gaze_after_signal_);
  arm_sag_rad_ = declare_parameter<double>("arm_sag_rad", arm_sag_rad_);
  barcode_peek_ = declare_parameter<bool>("barcode_peek", barcode_peek_);
  scan_from_light_ = declare_parameter<bool>("barcode_scan_from_light", scan_from_light_);
  scan_offsets_ = declare_parameter<std::vector<double>>("barcode_scan_offsets", scan_offsets_);
  scan_range_m_ = declare_parameter<double>("barcode_scan_range_m", scan_range_m_);
  cand_z_lo_ = declare_parameter<double>("barcode_cand_z_min", cand_z_lo_);
  cand_z_hi_ = declare_parameter<double>("barcode_cand_z_max", cand_z_hi_);
  cand_offsets_ = declare_parameter<std::vector<double>>("barcode_cand_offsets", cand_offsets_);
  if (cand_offsets_.empty()) cand_offsets_ = {0.0};
  peek_dirs_ = static_cast<size_t>(std::max<int64_t>(1, declare_parameter<int64_t>("barcode_peek_dirs", 2)));
  hires_settle_s_ = declare_parameter<double>("barcode_hires_settle_s", hires_settle_s_);
  barcode_peek_s_ = declare_parameter<double>("barcode_peek_s", barcode_peek_s_);
  barcode_peek_switch_s_ = declare_parameter<double>("barcode_peek_switch_s", barcode_peek_switch_s_);
  barcode_height_m_ = declare_parameter<double>("barcode_height_m", barcode_height_m_);
  peek_profile_ = declare_parameter<std::string>("barcode_peek_profile", peek_profile_);
  cam_profile_ = declare_parameter<std::string>("barcode_cam_profile", cam_profile_);
  {
    const auto cam = declare_parameter<std::string>("barcode_peek_cam", "/gripper_cam");  // realsense2_camera 노드 (이름 = 토픽 앞부분)
    cam_params_ = std::make_shared<rclcpp::AsyncParametersClient>(this, cam);
    cam_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(cam + "/color/camera_info", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr m) { cam_width_ = m->width; cam_info_at_ = now(); });
  }
  motor_error_stop_ = declare_parameter<bool>("motor_error_stop", motor_error_stop_);
  joint_sub_ = create_subscription<sensor_msgs::msg::JointState>("/joint_states", rclcpp::SensorDataQoS(),
    [this](sensor_msgs::msg::JointState::ConstSharedPtr m) {
      for (size_t i = 0; i < m->name.size() && i < m->position.size(); ++i)
        if (m->name[i].rfind("joint", 0) == 0) joints_now_[m->name[i]] = m->position[i];  // 팔 joint1~5만
    });
  // OpenCR 모터 에러 — ros2_control 하드웨어 진단 "모터 에러: joint2 과부하 → ~/reboot_motors …" (ERROR)
  diag_sub_ = create_subscription<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", rclcpp::QoS(50),
    [this](diagnostic_msgs::msg::DiagnosticArray::ConstSharedPtr m) {
      for (const auto & st : m->status) {
        if (st.message.find("모터 에러") == std::string::npos) continue;
        if (st.level < diagnostic_msgs::msg::DiagnosticStatus::ERROR) continue;
        const auto cut = st.message.find(" →");
        const std::string what = st.message.substr(0, cut);
        if (what != motor_error_) RCLCPP_ERROR(get_logger(), "%s (%s)", what.c_str(), st.name.c_str());
        motor_error_ = what;
        if (motor_error_stop_ && active_) stop("motor_error: " + what + " — ~/reboot_motors 뒤 다시");
      }
    });
  // 손 카메라 정면 거리 — 바코드 단계에서만 계산 (화면 가운데 60×60 px depth 하위 10 %, 0 = 측정 없음은 뺌)
  grip_depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
    declare_parameter<std::string>("gripper_depth_topic", "/gripper_cam/aligned_depth_to_color/image_raw"), rclcpp::SensorDataQoS(),
    [this](sensor_msgs::msg::Image::ConstSharedPtr m) {
      if (phase_ != Phase::BARCODE || m->encoding != "16UC1" || m->width < 60 || m->height < 60) return;
      std::vector<uint16_t> v;
      v.reserve(3600);
      for (uint32_t y = m->height / 2 - 30; y < m->height / 2 + 30; ++y) {
        const auto * row = reinterpret_cast<const uint16_t *>(&m->data[y * m->step]);
        for (uint32_t x = m->width / 2 - 30; x < m->width / 2 + 30; ++x) if (row[x]) v.push_back(row[x]);
      }
      grip_front_at_ = now();
      if (v.size() < 300) { grip_front_m_ = -1.0; return; }  // 너무 가까워 depth가 비거나 측정 없음
      std::nth_element(v.begin(), v.begin() + v.size() / 10, v.end());  // 가까운 쪽 10 % — 반투명 판 너머를 재도 가장 가까운 면 기준
      grip_front_m_ = v[v.size() / 10] * 0.001;
      grip_front_at_ = now();
    });
  // 비우면 출발(관측) 자리에서 그대로 바코드를 봄 — 이동 없음 (실기 11-12: 출발 자리에서 신호등·바코드가 다 보이고,
  //   26 cm 뒤 자리로 가려면 Nav2가 전진만 해서 돌아섰다 다시 돌아 한 바퀴를 돎)
  const auto bp = declare_parameter<std::vector<double>>("barcode_pose_arena", std::vector<double>{});
  observe_move_ = bp.size() >= 3;
  if (observe_move_) barcode_pose_ = {bp.at(0), bp.at(1), bp.at(2)};
  home_hold_s_ = declare_parameter<double>("home_hold_s", 3.0);
  // 현장 진단 실행 — 실패해도 기록하고 계속 (치명 이유만 정지 · mission_solver diag_fatal). "start:DIAG…"는 그 회차만 켬
  continue_on_fail_ = declare_parameter<bool>("continue_on_fail", false);
  stuck_limit_ = static_cast<int>(declare_parameter<int64_t>("diag_stuck_limit", 3));
  log_dir_ = declare_parameter<std::string>("log_dir", "~/.ros/mission");
  photo_topic_ = declare_parameter<std::string>("photo_topic", "/gripper_cam/color/image_raw");
  // 학습된 플래너 (learning/plan/mission_sim.py에서 학습 · models/plan/plan_next.onnx) — 기본은 지금 규칙 순서
  planner_ = declare_parameter<std::string>("planner", "rule");
  std::string pn_path = declare_parameter<std::string>("plan_model", "");
  if (pn_path.empty()) pn_path = ament_index_cpp::get_package_share_directory("ml") + "/model/plan/plan_next.onnx";
  if (planner_ == "plan_next") {
    std::string why;
    plan_next_ok_ = plan_next_.load(pn_path, why);
    if (plan_next_ok_) RCLCPP_INFO(get_logger(), "planner plan_next: %s", pn_path.c_str());
    else RCLCPP_WARN(get_logger(), "plan_next를 못 씀 (%s) — 규칙 순서로", why.c_str());
  }
  // plan_id를 노드가 뜰 때마다 다른 구간에서 시작 — /task/status가 latched라 재시작하면 지난 실행의 같은 id 결과가 바로 들어왔음
  plan_seq_ = static_cast<uint32_t>(std::time(nullptr) % 1000) * 1000;
  if (log_dir_.rfind("~/", 0) == 0 && std::getenv("HOME")) log_dir_ = std::string(std::getenv("HOME")) + log_dir_.substr(1);

  std::tie(tf_buffer_, tf_listener_) = shared_tf(*this);  // 같은 프로세스 plan 노드와 TF 하나 (plan/shared_tf.hpp)
  const auto latched = rclcpp::QoS(1).reliable().transient_local();
  execute_client_ = rclcpp_action::create_client<ExecutePlan>(this, "/task/execute_plan");
  state_pub_ = create_publisher<MissionState>("/mission/state", latched);
  beep_pub_ = create_publisher<std_msgs::msg::Int32MultiArray>("/cmd/beep", 10);
  markers_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/mission/markers", latched);
  walls_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/arena/walls", latched);
  walls_only_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/arena/walls_only", latched);
  publish_walls();
  corr_pub_ = create_publisher<std_msgs::msg::String>("/arena/correction", latched);
  if (wall_fit_) {
    wall_fit_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      "/localization/wall_fit", latched,
      [this](const std_msgs::msg::Float64MultiArray & msg) {
        const auto & d = msg.data;
        if (d.size() != 8) return;
        wall_ok_ = true;
        wall_rms_ = d[4];
        // 상자 인지 위치(marks, map)는 옮기지 않음 — map→odom 보정은 map에서 벽·상자가 제자리에 있게 하는 것이라 perception의
        //   map 좌표는 그대로 (11-21 실기 nav_1121_223313: 재방문 S2 인지 상자는 첫 방문과 같은 map 자리). 보정마다 marks를 따라 옮기면
        //   상자를 다시 볼 때까지 148번 쌓여 S2가 50 cm 반대쪽 (도면 −29 cm · 실제 +21 cm) → 진입점이 상자 위 (lethal) · 다른 변으로
        if (d[5] != 0.0 || d[6] != 0.0 || d[7] != 0.0) loc_last_ = wall_s();
        const ArenaCorr residual{d[0], d[1], d[2], static_cast<int>(d[3])};
        if (acorr_src_ != "walls" || corr_moved(residual, acorr_, loc_min_m_, loc_min_rad_) ||
            d[5] != 0.0 || d[6] != 0.0 || d[7] != 0.0)
          commit_corr(residual, "walls", "wall_localization");
      });
  }
  loc_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/localization/map_correction", 10);
  if (loc_on_ && !wall_fit_) loc_timer_ = create_wall_timer(std::chrono::duration<double>(loc_period), [this]() { loc_tick(); });
  marks_.assign(start_boxes_.size() + loc_boxes_.size(), Mark{});  // 미션 전(arena_corr_idle)에도 보정
  publish_corr("");
  cmd_sub_ = create_subscription<std_msgs::msg::String>("/cmd/mission", rclcpp::QoS(10).reliable(),
    [this](std_msgs::msg::String::ConstSharedPtr m) { on_command(m->data); });
  barcode_sub_ = create_subscription<std_msgs::msg::String>("/mission/qr", latched,
    [this](std_msgs::msg::String::ConstSharedPtr m) { barcode_msg_ = m->data; barcode_rx_ = now(); });
  // 신호 — 이번 READ 시작 뒤에 끝난 사이클만 (지난 회차의 latched 답은 버림).
  //   /mission/signal은 새 답일 때만 나와서 지난 회차와 같은 신호면 다시 안 옴 → 사이클마다 오는 /mission/signal_status의 ok도 확정으로 봄
  //   BARCODE(관측 — 손 카메라 바코드 + pole_front_cam 신호 동시) 동안에도 받음
  //   오류 사이클은 팔·바퀴가 멈춰 있을 때만 셈 (움직이는 동안은 팔이 가리거나 흐려서 생긴 오류일 수 있음 — 다음 사이클을 봄)
  const auto on_signal = [this](const perception::msg::MissionSignal & m) {
    if (m.ok) last_ok_signal_ = m;  // 관측 밖에서도 기억 — 해독기는 같은 답을 다시 안 내서, 시작 전 확정을 여기서 안 잡으면 사라졌음 (11-22 실기)
    const bool barcode = phase_ == Phase::BARCODE;
    if ((phase_ != Phase::READ && !barcode) || signal_ready_ || !signal_error_.empty()) return;
    // 시작 직전(signal_preroll_s 안)에 끝난 사이클도 받음 — 해독기는 미션 전부터 읽고 있어 방금 확정된 답을 버리면 한 사이클(약 27 s) 더 기다렸음
    //   (11-21 실기 21:02: 시작 5 s 전 확정 → 버림). 그보다 오래된 것(지난 회차 latched 답)은 버림
    if (rclcpp::Time(m.header.stamp, RCL_ROS_TIME) < read_started_ - rclcpp::Duration::from_seconds(signal_preroll_s_)) return;
    if (barcode && !m.ok && !observe_still()) {
      RCLCPP_WARN(get_logger(), "신호 사이클 %s [%s] (팔·바퀴 움직이는 중 — 안 셈)", m.error.c_str(), m.raw.c_str());
      return;
    }
    const auto a = signal_action(m.ok, m.error, signal_streak_, signal_error_cycles_);
    if (a == SignalAction::ACCEPT) { signal_ = m; signal_ready_ = true; }
    else if (a == SignalAction::STOP) { signal_ = m; signal_error_ = m.error; }
    else if (!m.ok) RCLCPP_WARN(get_logger(), "신호 사이클 %s [%s] — 다음 사이클을 봄", m.error.c_str(), m.raw.c_str());
  };
  signal_sub_ = create_subscription<perception::msg::MissionSignal>("/mission/signal", latched,
    [on_signal](perception::msg::MissionSignal::ConstSharedPtr m) { on_signal(*m); });
  signal_status_sub_ = create_subscription<perception::msg::MissionSignal>("/mission/signal_status", rclcpp::QoS(10).reliable(),
    [on_signal](perception::msg::MissionSignal::ConstSharedPtr m) { on_signal(*m); });
  objects_sub_ = create_subscription<perception::msg::TrackedObjectArray>("/tracked_objects", rclcpp::QoS(5).reliable(),
    [this](perception::msg::TrackedObjectArray::ConstSharedPtr m) {
      objects_ = m;
      // 근래 본 퍽 (near_body) — 차체에 붙은 퍽은 보이다 사라짐 (실기 nav_1121_214753 S3: stand 뒤 퍽 5가 0.14 m에 2 s 보이고 빠짐 → 물러나지 않고 puck_lost)
      for (const auto & o : m->objects) if (is_puck(o)) recent_pucks_[o.id] = o;
      for (const auto & o : m->objects)
        {
          // 장소 상자 가운데 = 윗면 정중앙 태그 (perception) — 태그를 읽은 관측이면 잘려도 됨
          // perception은 장소 태그를 읽은 상자만 location_box · 못 읽은 상자는 start_box로 냄 (perception_node 윗면 상자) →
          //   진입점 잡기용으로는 도면 장소 0.35 m 안의 잘리지 않은 start_box도 그 장소 상자로 (11-22 실기 L3: 0.6 m에서 태그 못 읽어 '못 봄')
          if (o.class_name != "location_box" && o.class_name != "start_box") continue;
          const std::string tag = o.class_name == "location_box" ? detail(o, "tag") : std::string();
          if (tag.empty()) {  // 태그 못 읽음 — 윗면이 잘리지 않았으면 도면 장소 0.35 m 안의 것을 그 장소 상자로 (진입점 다시 잡기용)
            if (detail(o, "cut") == "true") continue;
            const auto a = to_arena(o.pose.pose.position.x, o.pose.pose.position.y);
            for (size_t li = 0; li < loc_boxes_.size() && li < loc_any_.size(); ++li)
              if (std::hypot(a.x - loc_boxes_[li].x, a.y - loc_boxes_[li].y) < 0.35)
                loc_any_[li] = LocSeen{make_spot(*tf_buffer_, o.pose.pose.position.x, o.pose.pose.position.y), rclcpp::Time(o.last_seen, RCL_ROS_TIME)};
            continue;
          }
          loc_whole_[tag] = LocSeen{make_spot(*tf_buffer_, o.pose.pose.position.x, o.pose.pose.position.y), rclcpp::Time(o.last_seen, RCL_ROS_TIME)};
        }
      for (auto it = recent_pucks_.begin(); it != recent_pucks_.end();)
        it = (now() - rclcpp::Time(it->second.last_seen, RCL_ROS_TIME)).seconds() > back_recent_s_ ? recent_pucks_.erase(it) : std::next(it);
      // 상자 인지 위치(marks_)는 arena_correction을 꺼도 approach_perceived면 셈 (보정 자체는 update_corr 안에서 acorr_on_만)
      if ((acorr_on_ || near_perceived_) && (active_ || acorr_idle_)) update_corr(*m);
      // 태그를 읽은 퍽의 마지막 모습 — 안 보이게 돼도 그 자리로 바로 pick (locate) · 되돌림 확인 (put_back).
      //   inspect 단계 동안만 적음 — pick·load·place 중(손이 움직이며 가까이서 봄)에 읽은 태그는 틀리기 쉬움 (11-21 사용자 ·
      //   실기 165205 S2: 되돌린 퍽 9를 place 뒤 tag 0으로 읽음)
      if (!inspecting()) return;
      for (const auto & o : m->objects)
        if (is_puck(o) && !detail(o, "tag").empty()) {
          const int tag = std::stoi(detail(o, "tag"));
          last_seen_[tag] = o;
          last_spot_[tag] = make_spot(*tf_buffer_, o.pose.pose.position.x, o.pose.pose.position.y);
        }
    });
  basket_sub_ = create_subscription<plan::msg::BasketState>("/coordinator/basket", rclcpp::QoS(1).reliable().transient_local(),
    [this](plan::msg::BasketState::ConstSharedPtr m) { basket_ = m; if (active_) publish_state(); });
  markers_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() { publish_markers(); });
  diag_.setHardwareID("beebot");
  diag_.add("상태", [this](diagnostic_updater::DiagnosticStatusWrapper & s) {
    using diagnostic_msgs::msg::DiagnosticStatus;
    s.summary(phase_ == Phase::ERROR ? DiagnosticStatus::ERROR : DiagnosticStatus::OK, phase_name(phase_));
    s.add("배정 L1·L2·L3", order_.empty() ? "-" : order_);
    s.add("조사한 퍽", pucks_.size());
    int n = 0;
    if (basket_) for (const auto & b : basket_->slots) n += b.filled;
    s.add("적재함", n);
    s.add("놓음", str(placed_) + "/" + str(static_cast<int>(jobs_.size())));
    s.add("바코드", barcode_.empty() ? "-" : barcode_);
  });
  note("/cmd/mission start 대기");
}

// ---------- 단계
void MissionManager::on_command(const std::string & cmd) {
  if (cmd == "abort") { if (active_) stop("aborted"); return; }
  // 시작 명령 문법은 mission_solver parse_start — "start" · "start:<단계>"(BARCODE·READ·SURVEY·RETURN만, TUI 단계별 시험)
  //   · "start:DIAG[:<바코드>:<신호>]" (실패해도 계속) · "start:[DIAG_]FROM_SURVEY:<바코드>:<신호>" (관측 없이 준 값으로 SURVEY부터)
  StartCmd sc;
  std::string bad;
  if (!parse_start(cmd, sc, bad)) {
    if (bad != "not_start") note("시작 거부: " + cmd + " (" + bad + ")");
    return;
  }
  if (active_) return;
  int left = 0;
  if (basket_) {
    for (const auto & b : basket_->slots) left += b.filled;
    left += basket_->held.filled;
  }
  if (left && sc.place_loc < 0) {  // 지난 미션 장부(파일)가 남음 — 적재함이 정말 비었는지 사람이 보고 비워야 함
    RCLCPP_WARN(get_logger(), "적재함 장부에 퍽 %d개가 남아 있음 — 시작 거부", left);
    note("시작 거부: 적재함 장부에 퍽 " + str(left) + "개 — 확인 뒤 ros2 service call /coordinator/basket_clear std_srvs/srv/Trigger");
    return;
  }
  active_ = true;
  pn_action_ = -1;
  pucks_.clear();
  puck_at_.clear();
  puck_seen_.clear();
  box_seen_.clear();
  on_box_.clear();
  leftover_.clear();
  second_pass_ = false;
  alloc_.clear();
  jobs_.clear();
  order_.clear();
  results_.clear();
  build_json_.clear();
  verify_json_.clear();
  verified_.clear();
  last_place_loc_ = -1;
  stop_ = get_parameter("start_stop").as_string();  // 미션은 출발 자리에서 시작 (운영 규칙) · 시뮬 구간 시험은 그 상자 앞 (S4 등)
  seen_tags_.clear();
  lying_.clear();
  loc_pose_.assign(locations_.size(), std::nullopt);
  loc_fix_.assign(locations_.size(), std::nullopt);
  loc_rot_.assign(locations_.size(), std::nullopt);
  marks_.assign(start_boxes_.size() + loc_boxes_.size(), Mark{});
  acorr_seen_.clear();
  acorr_used_.clear();
  acorr_sent_.clear();
  if (!wall_ok_) {
    acorr_src_ = "none";
    if (acorr_.n) { acorr_ = {}; publish_walls(); }
  }
  publish_corr("");  // active true
  acorr_walls_ = {};
  barcode_.clear();
  error_.clear();
  back_.clear();
  placed_ = 0;
  photo_n_ = 0;
  char stamp[32];
  const std::time_t t = std::time(nullptr);
  std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&t));
  run_dir_ = log_dir_ + "/" + stamp;
  std::filesystem::create_directories(run_dir_ + "/photos");
  only_ = sc.only;
  signal_window_ = false;
  mission_t0_ = wall_s();
  diag_run_ = sc.diag || continue_on_fail_;
  stuck_ = 0;
  counts_ = {};
  given_ = sc.given;
  given_barcode_ = sc.barcode;
  given_counts_ = sc.counts;
  if (diag_run_) note("현장 진단 실행 — 실패해도 기록하고 계속 (치명 이유만 정지)");
  if (sc.only != Phase::IDLE) note(std::string("단계만 실행: ") + phase_name(sc.only));
  goto_ = sc.go_to;
  if (!goto_.empty()) {  // 실기 이동 시험 — 관측·조사 없이 적은 상자들에 미션처럼 다가가기만
    std::string list;
    for (const auto & g : goto_) list += (list.empty() ? "" : ",") + g;
    note("상자로 가기 " + list + " — SURVEY처럼 (Nav2 → 상자 도킹) · 끝나면 복귀");
    phase_ = Phase::SURVEY;
    beep_phase(phase_);
    phase_t0_ = wall_s();
    ctx_.clear();
    goto_seq(0);
    return;
  }
  build_loc_order_ = visit_order(get_parameter("build_location_order").as_integer_array(), 3);
  use_given_order_ = false;
  if (sc.place_loc >= 0) {  // 적재함 장부(이미 다 실음)의 퍽으로 배정 → 그 장소부터 BUILD → VERIFY → RETURN
    const size_t li = static_cast<size_t>(sc.place_loc);
    if (basket_) for (const auto & b : basket_->slots)
      if (b.filled && b.tag >= 0) pucks_.push_back({b.tag, b.color, b.shape});
    note("놓기 시험 — 적재함 퍽 " + str(static_cast<int>(pucks_.size())) + "개 · L" + str(sc.place_loc + 1) + "부터 BUILD");
    build_loc_order_.erase(std::remove(build_loc_order_.begin(), build_loc_order_.end(), li), build_loc_order_.end());
    build_loc_order_.insert(build_loc_order_.begin(), li);
    use_given_order_ = true;
    use_given(false);
    phase_ = Phase::SURVEY;  // 배정이 끝나면 BUILD로 (next_phase)
    phase_t0_ = wall_s();
    solve_allocation();
    return;
  }
  if (sc.from_survey) {  // 관측 건너뜀 — 준 바코드·신호를 본 것처럼 (SURVEY → BUILD → VERIFY → RETURN)
    note("관측 건너뜀 — 바코드 " + sc.barcode + " · 신호 " + cmd.substr(cmd.rfind(':') + 1) + "로 SURVEY부터");
    use_given(false);
    enter(Phase::SURVEY);
    return;
  }
  enter(sc.only != Phase::IDLE ? sc.only : next_phase(Phase::IDLE, true));
}

void MissionManager::finish(bool ok, const std::string & why) {
  if (!active_) return;
  // 단계만 실행이면 여기서 끝 — start:BARCODE는 관측(바코드 + 신호)이라 READ까지
  const Phase n = only_ == Phase::IDLE ? next_phase(phase_, ok, diag_run_)
                : !ok ? Phase::ERROR
                : (only_ == Phase::BARCODE && phase_ == Phase::BARCODE) ? Phase::READ : Phase::DONE;
  if (!ok) result(std::string(phase_name(phase_)) + ": " + why);
  phase_line(ok, why, n);
  enter(n, why);
}

void MissionManager::enter(Phase p, const std::string & why) {
  phase_ = p;
  phase_t0_ = wall_s();
  ctx_.clear();
  if (p != Phase::ERROR) beep_phase(p);
  switch (p) {
    case Phase::GO_OBSERVE: go_observe(); break;
    case Phase::BARCODE: barcode_phase(); break;
    case Phase::READ: read_phase(); break;
    case Phase::SURVEY: use_given(true); survey(0); break;
    case Phase::BUILD: build(0); break;
    case Phase::VERIFY: verify(0); break;
    case Phase::RETURN: home_tries_ = 0; go_home(); break;
    case Phase::DONE:
      mission_line(true, "");
      active_ = false;
      note("기록: " + run_dir_ + (results_.empty() ? "" : " · 실패 " + str(static_cast<int>(results_.size())) + "건 (결과 목록)"));
      break;
    case Phase::ERROR: beep_phase(p); stop(why); break;
    default: break;
  }
}

void MissionManager::go_observe() {
  // 팔은 stow 그대로 — 예전 pose scan(왼쪽 뒤 150°)은 팔을 뻗은 채 회전해 관성으로 밀리고 무게중심이 쏠림 (11-09 시뮬). stow에서도 pole_front_cam에 신호등이 보임
  note("바코드 보는 자세로 (출발 구역 안 · 팔 stow)");
  if (!observe_move_) {  // 출발 자리에서 그대로 봄
    note("바코드 · 신호등을 출발 자리에서 봄 (이동 없음 · 팔 stow)");
    finish(true, "");
    return;
  }
  const auto p = to_map(barcode_pose_.x, barcode_pose_.y, barcode_pose_.yaw);
  stop_.clear();  // 관측 자리로 옮김 — 출발 자리 표(start_*)는 안 맞음
  run({SubGoal().set__skill("navigate").set__target(fmt("%.3f,%.3f,%.3f", p.x, p.y, p.yaw))},
    [this](bool ok, const std::string & why) { finish(ok, "observe_nav_failed:" + why); });
}

void MissionManager::barcode_phase() {
  // ① pole_front_cam은 신호등을 계속 관측하고, barcode_pole_s 뒤에는 막대 패턴 후보만 저비용으로 검출
  // ② 후보의 map 위치로부터 상대 방위를 계산해 그때만 barcode_arm_pose(gaze)를 정렬 · 신호 확정과 병렬 실행 가능
  //    (coordinator는 새 계획이 오면 진행 중 goal을 취소함 · 취소된 gaze 결과는 waiting_plan_이 바뀌어 무시됨)
  // ③ gaze에 도착하고 barcode_gaze_wait_s 안에 못 읽으면 팔은 그대로 barcode_advance_step_m씩 앞으로 (최대 barcode_advance_max_m)
  //    — 가는 도중에 읽혀도 바로 접음. 끝나면 팔 접고 다가간 만큼 똑바로 후진 (approach 후진)
  //    다가가기는 손 카메라 정면 depth가 barcode_min_gap_m보다 멀 때만 (모르거나 1 s 넘게 안 오면 안 감) · 최대 barcode_advance_max_m
  // ④ 후보·gaze가 없거나 마지막 자리에서 barcode_timeout_s 동안 안 읽히면 stow·후진 후 미션 오류
  // 이번에 실제로 본 바코드만 — /tracked_objects의 barcode(last_seen ≥ 시작) 또는 시작 뒤 받은 /mission/qr
  note("관측 — 바코드 후보 정렬·손 카메라 gaze + 신호(pole_front_cam) 병렬");
  phase_started_ = now();
  signal_ = perception::msg::MissionSignal();  // 신호 관측 창을 바코드 시작부터 엶 (READ와 병렬)
  signal_ready_ = false;
  signal_streak_ = 0;
  signal_error_.clear();
  read_started_ = phase_started_;
  signal_window_ = true;
  take_recent_signal();
  gaze_sent_ = false;
  peek_sent_ = false;
  peek_done_ = !barcode_peek_;
  cand_.reset();
  light_.reset();
  scan_ = false;
  scan_only_ = false;
  occl_wait_ = false;
  light_seen_at_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  scan_k_ = 0;
  cand_k_ = 0;
  gaze_seen_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  peek_at_ = hires_at_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  returning_ = false;
  light_warned_ = false;
  hold_noted_ = false;
  bad_barcode_.clear();
  gaze_at_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  gaze_free_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  gaze_wait_noted_ = false;
  gaze_joints_.clear();
  advanced_m_ = 0.0;
  moving_ = false;
  grip_front_m_ = -1.0;  // 지난 실행 값을 믿지 않음 — 이번 단계의 손 카메라 depth로만
  if (!robot_arena(barcode_origin_)) barcode_origin_ = start_;  // 다가갔다가 돌아올 자리
  timer_ = create_wall_timer(std::chrono::milliseconds(250), [this]() {
    if (phase_ != Phase::BARCODE) { timer_->cancel(); return; }
    if (!signal_error_.empty() && !diag_run_) {  // 멈춰 있는데도 신호 오류 사이클이 이어짐 — 추정하지 않고 팔 접고 정지 (diag: 바코드는 계속 · READ가 기록)
      timer_->cancel();
      write_log("signal.json", "{\"ok\": false, \"error\": " + q(signal_error_) + ", \"raw\": " + q(signal_.raw) + "}\n");
      barcode_done(false, "signal_" + signal_error_);
      return;
    }
    if (!light_warned_ && (now() - phase_started_).seconds() > light_check_s_) {  // 신호등이 pole_front_cam에 안 보임 — 방향·가림 확인
      light_warned_ = true;
      bool seen = false;
      if (objects_) for (const auto & o : objects_->objects)
        if (o.class_name == "traffic_light" && rclcpp::Time(o.last_seen, RCL_ROS_TIME) >= phase_started_) seen = true;
      if (!seen) {
        RCLCPP_WARN(get_logger(), "관측 %.0f s 동안 신호등이 안 보임 — 로봇 방향·팔 가림 확인 (바코드는 계속)", light_check_s_);
        note("신호등 안 보임 — 방향·팔 가림 확인");
      }
    }
    // 미등록 바코드(P2-01~04 아님)는 잘못 읽은 것일 수 있어 버리고 계속 찾음 — 끝까지 그것만 보이면 unknown_barcode로 정지 (발제: 미등록 → 안전 정지)
    const auto take = [this](const std::string & t) {
      bool asc = false;
      if (t.empty() || !barcode_.empty()) return;
      if (parse_barcode(t, asc)) { barcode_ = t; return; }
      if (t != bad_barcode_) RCLCPP_WARN(get_logger(), "미등록 바코드 '%s' — 버리고 계속 찾음", t.c_str());
      bad_barcode_ = t;
    };
    if (objects_) for (const auto & o : objects_->objects)
      if (o.class_name == "barcode" && rclcpp::Time(o.last_seen, RCL_ROS_TIME) >= phase_started_) take(detail(o, "content"));
    if (barcode_rx_ >= phase_started_) take(barcode_msg_);
    const double t = (now() - phase_started_).seconds();
    if (barcode_.empty()) {
      // 후보를 찾으면 목표 방향으로 gaze. 후보가 없으면 팔을 임의로 뻗지 않는다.
      const bool signal_wait = signal_window_ && !signal_ready_ && signal_error_.empty() &&
                               (now() - read_started_).seconds() < signal_timeout_s_;
      const bool arm_wait = barcode_gaze_after_signal_ && signal_wait;
      if (!signal_wait && t > barcode_pole_s_ && gaze_free_.nanoseconds() == 0) gaze_free_ = now();
      if (arm_wait && !gaze_sent_ && t > barcode_pole_s_ && !gaze_wait_noted_) {
        gaze_wait_noted_ = true;
        note("팔 접은 채 신호 먼저 확정 (팔 과부하 방지) — 확정 뒤 " + barcode_arm_pose_ + "로 바코드");
      }
      // gaze 유지 중 관절이 처지면(과부하·토크 꺼짐) 손 카메라가 바닥을 봄 — 기다리지 않고 정지
      if (arm_sag_rad_ > 0 && gaze_at_.nanoseconds() > 0 && !gaze_joints_.empty()) {
        for (const auto & [name, q0] : gaze_joints_) {
          const auto it = joints_now_.find(name);
          if (it == joints_now_.end() || std::abs(it->second - q0) <= arm_sag_rad_) continue;
          RCLCPP_ERROR(get_logger(), "gaze 유지 중 %s가 %.2f rad 벗어남 — 팔 처짐 (과부하?)", name.c_str(), it->second - q0);
          timer_->cancel();
          barcode_done(false, "arm_sag:" + name);
          return;
        }
      }
      std::optional<double> candidate_yaw, light_yaw;
      double candidate_range = 0.0, candidate_z = barcode_height_m_;
      Pose2 base_arena;
      if (objects_ && robot_arena(base_arena)) {
        // 신호등 방위 — 바코드는 신호등 왼쪽에 있음 (11-21 사용자). 오른쪽·먼 곳의 잘못 본 '바코드'(실기 20:43·21:02: 왼쪽 +0.61 rad ·
        //   0.51 m를 바코드로 봐 그쪽을 엿보기·gaze)는 거르고, 바코드가 안 보이면 신호등에서 왼쪽으로 차례로 gaze (scan)
        for (const auto & o : objects_->objects)
          if (o.class_name == "traffic_light" && rclcpp::Time(o.last_seen, RCL_ROS_TIME) >= phase_started_) {
            const auto l = to_arena(o.pose.pose.position.x, o.pose.pose.position.y, 0.0);
            light_ = std::array<double, 2>{l.x, l.y};
            light_seen_at_ = std::max(light_seen_at_, rclcpp::Time(o.last_seen, RCL_ROS_TIME));
          }
        double light_range = 1.0;
        if (scan_from_light_ && light_ && !scan_offsets_.empty()) {
          const double dx = (*light_)[0] - base_arena.x, dy = (*light_)[1] - base_arena.y, yaw = std::atan2(dy, dx) - base_arena.yaw;
          light_yaw = std::atan2(std::sin(yaw), std::cos(yaw));
          light_range = std::hypot(dx, dy);
        }
        for (const auto & o : objects_->objects) {
          if (o.class_name != "barcode" || rclcpp::Time(o.last_seen, RCL_ROS_TIME) < phase_started_) continue;
          const auto target = to_arena(o.pose.pose.position.x, o.pose.pose.position.y, 0.0);
          const double dx = target.x - base_arena.x, dy = target.y - base_arena.y;
          const double range = std::hypot(dx, dy);
          // 11-22: perception이 작은 흰 라벨도 후보로 냄(막대 확인 없이) → 바닥 밝은 점·벽 아래 띠(높이 ≈ 0)는 높이로 거름
          if (o.pose.pose.position.z < cand_z_lo_ || o.pose.pose.position.z > cand_z_hi_) continue;
          const double yaw = std::atan2(dy, dx) - base_arena.yaw;
          const double relative = std::atan2(std::sin(yaw), std::cos(yaw));
          // 신호등을 봤으면 그 왼쪽 부채꼴(신호등 −0.05 ~ 마지막 scan + 0.15 rad)의 바코드만
          //   11-21 실기 23:51: 신호등(−0.30) 바로 옆 −0.20을 바코드 후보로 봐 첫 엿보기를 신호등에 씀 → 가장 작은 scan 방향보다 안쪽은 버림
          const double lo = *std::min_element(scan_offsets_.begin(), scan_offsets_.end()) - 0.03;
          const double hi = *std::max_element(scan_offsets_.begin(), scan_offsets_.end()) + 0.15;
          const bool left_of_light = !light_yaw || (relative > *light_yaw + lo && relative < *light_yaw + hi);
          if (!scan_only_ && range >= 0.20 && range <= 4.0 && std::abs(relative) <= 0.65 && left_of_light) {
            scan_ = false;
            candidate_yaw = relative;
            candidate_range = range;
            const double z = o.pose.pose.position.z;  // map 높이 (바닥 = 0) — 이상하면 barcode_height_m
            if (z > 0.03 && z < 0.8) candidate_z = z;
            cand_ = std::array<double, 3>{target.x, target.y, candidate_z};
            break;
          }
        }
        // 지금 안 보여도 이번 관측에서 본 후보 자리로 — 엿보기·접기 동안 카메라가 돌아가 인지 목록에서 빠지면 gaze를 못 보내고
        //   barcode_candidate_missing으로 끝났음 (11-21 실기 20:43: +0.60 rad · 0.53 m에서 보고 엿보기 뒤 29 s 동안 gaze 없음)
        if (!candidate_yaw && cand_ && !scan_only_) {
          const double dx = (*cand_)[0] - base_arena.x, dy = (*cand_)[1] - base_arena.y;
          const double yaw = std::atan2(dy, dx) - base_arena.yaw;
          candidate_yaw = std::atan2(std::sin(yaw), std::cos(yaw));
          candidate_range = std::hypot(dx, dy);
          candidate_z = (*cand_)[2];
        }
        if (!candidate_yaw && light_yaw) {  // 바코드를 못 봄 — 신호등에서 왼쪽으로 scan_k_번째 방향
          scan_ = true;
          const double yaw = *light_yaw + scan_offsets_[std::min(scan_k_, scan_offsets_.size() - 1)];
          candidate_yaw = std::atan2(std::sin(yaw), std::cos(yaw));
          // 아래 각도는 고정 거리로 — 신호등 거리로 쓰면 순간 0.38 m로 잡혀 아래 38°(바닥)를 봤음 (11-21 실기 23:51 3번째 방향)
          candidate_range = scan_range_m_;
          (void)light_range;
        }
      }
      // ⓪ 엿보기 — 팔은 접은 채 손목만 바코드 쪽으로(joint1 방위 · joint4 높이) + 손 카메라를 잠깐 고해상도로
      //    뻗지 않으니 신호를 기다리지 않음(과부하 없음) · 다가가지 않으니 신호등이 pole 시야에 남음
      // 팔이 신호등을 가릴 것 같으면 신호 먼저 (11-21 사용자): ① 보낼 방향이 신호등 방위 barcode_occlude_rad 안 ·
      //   ② 팔을 둔 동안 pole_front_cam이 신호등을 barcode_occlude_lost_s 넘게 놓침 → 팔을 접고 신호 확정 뒤에 바코드
      //   (11-21 실기 21:23: 신호등 −0.31 · 엿보기 −0.16 · gaze −0.11 rad로 손이 신호등을 가려 확정이 3분)
      const bool signal_open = !signal_ready_ && signal_error_.empty() && (now() - read_started_).seconds() < signal_timeout_s_;
      const bool arm_posed = (peek_sent_ && peek_at_.nanoseconds() > 0) || gaze_at_.nanoseconds() > 0;
      if (signal_open && arm_posed && !occl_wait_ && light_seen_at_.nanoseconds() > 0 &&
          (now() - light_seen_at_).seconds() > occlude_lost_s_) {
        occl_wait_ = true;
        RCLCPP_WARN(get_logger(), "팔을 둔 동안 신호등이 %.1f s 안 보임 — 팔 접고 신호 먼저", (now() - light_seen_at_).seconds());
        note("팔이 신호등을 가림 — 접고 신호 먼저 확정한 뒤 바코드");
        peek_sent_ = gaze_sent_ = false;
        peek_at_ = hires_at_ = gaze_at_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        run({stow()}, [](bool, const std::string &) {});
        return;
      }
      const bool near_light = light_yaw && candidate_yaw &&
                              std::abs(std::remainder(*candidate_yaw - *light_yaw, 2 * M_PI)) < occlude_rad_;
      // 엿보기(팔 접고 손목만)는 신호와 병렬 — 가리지 않는 자세라 신호등 가까운 방향도 봄 (11-21 사용자 우선순위). 실제로 가리면 위 감시가 접음
      //   팔을 뻗는 gaze는 신호등 가까운 방향이면 신호 확정 뒤에만
      const bool peek_hold = signal_open && occl_wait_;
      const bool arm_hold = signal_open && (occl_wait_ || near_light);
      if (peek_hold && !arm_posed && !hold_noted_) {
        hold_noted_ = true;
        note("신호 확정 기다림 (팔이 신호등을 가렸음) — 확정 뒤 바코드");
      }
      if (!peek_done_ && !peek_sent_ && candidate_yaw && t > barcode_pole_s_ && !peek_hold) {
        peek_sent_ = true;
        peek_yaw_ = *candidate_yaw;
        hires(true);  // 팔이 움직이는 동안 카메라 전환(1~2 s)을 같이
        // 손 카메라 ≈ 앞 0.16 m · 높이 0.32 m (pose_peek, URDF) → 바코드 중심을 보는 아래쪽 각도
        const double down = std::atan2(0.32 - candidate_z, std::max(0.10, candidate_range - 0.16));
        note("바코드 엿보기 " + fmt("%+.2f rad · 아래 %.0f° · %.2f m", *candidate_yaw, down * 180.0 / M_PI, candidate_range) +
             " — 팔 접은 채 손목만 · 손 카메라 " + peek_profile_);  // fmt 버퍼 96 B — 한글은 밖에서 붙임
        run({SubGoal().set__skill("pose").set__target(fmt("peek@%.3f,%.3f", *candidate_yaw, down))},
          [this](bool ok, const std::string & why) {
            if (phase_ != Phase::BARCODE || !barcode_.empty() || peek_done_) return;
            if (ok) { peek_at_ = now(); return; }
            peek_end("자세 실패 " + why);
          });
      }
      if (peek_sent_ && !peek_done_ && peek_at_.nanoseconds() > 0) {
        const bool hi = cam_width_ >= 1280 && cam_info_at_ > peek_at_;  // 도착 뒤 고해상도 영상이 실제로 들어옴
        // 스트림을 다시 켠 직후 몇 장은 자동 노출이 안 잡혀 어두움 (11-22 실기 첫 방향 사진) → barcode_hires_settle_s 뒤부터 셈
        if (hi && hires_live_at_.nanoseconds() == 0) hires_live_at_ = now();
        if (hi && hires_at_.nanoseconds() == 0 && (now() - hires_live_at_).seconds() >= hires_settle_s_) {
          hires_at_ = now();
          snap(fmt("barcode_peek_%+.2f", peek_yaw_));  // 고해상도 첫 장 (미션 기록 photos/)
        }
        // 엿보기로 못 읽음 — 신호등 왼쪽 다음 방향으로 손목만 (고해상도 그대로 · 팔은 접은 채라 신호등을 덜 가림, 11-21 사용자: 엿보기로 찾기)
        //   본 후보로 실패했으면 그 후보는 버리고 신호등 왼쪽을 처음부터 훑음
        // 11-22: 0.7 m 엿보기는 바코드가 흐려 거의 안 읽힘(오프라인 재시험 0/13) — 위치 확인용으로 barcode_peek_dirs 방향만 보고 gaze로
        const bool next_dir = light_ && scan_from_light_ && (!scan_ || scan_k_ + 1 < std::min(peek_dirs_, scan_offsets_.size()));
        if (hires_at_.nanoseconds() > 0 && (now() - hires_at_).seconds() > barcode_peek_s_ && next_dir) {
          if (scan_) ++scan_k_;
          else { scan_only_ = true; scan_k_ = 0; }
          peek_sent_ = false;
          peek_at_ = hires_at_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
          note(fmt("엿보기 안 읽힘 — 신호등 왼쪽 %.2f rad로 손목만 (%.0f/%.0f)", scan_offsets_[scan_k_], scan_k_ + 1.0,
                   static_cast<double>(scan_offsets_.size())));
          return;
        }
        if (hires_at_.nanoseconds() > 0 && (now() - hires_at_).seconds() > barcode_peek_s_)
          peek_end(fmt("%.0f s 동안 안 읽힘", barcode_peek_s_));
        else if (hires_at_.nanoseconds() == 0 && (now() - peek_at_).seconds() > barcode_peek_switch_s_)
          peek_end(fmt("고해상도 영상이 안 옴 (지금 폭 %.0f px)", static_cast<double>(cam_width_)));
      }
      if (!gaze_sent_ && peek_done_ && candidate_yaw && !barcode_arm_pose_.empty() && t > barcode_pole_s_ && !arm_wait && !arm_hold) {
        gaze_sent_ = true;
        // 후보를 봤으면 그 방향과 둘레(barcode_cand_offsets) — 11-22 실기 시험: gaze에서 읽히는 폭이 약 0.1 rad
        const double yaw = *candidate_yaw + (scan_ ? 0.0 : cand_offsets_[std::min(cand_k_, cand_offsets_.size() - 1)]);
        const std::string target = barcode_arm_pose_ == "gaze" ? "gaze@" + fmt("%.3f", yaw) : barcode_arm_pose_;
        hires(true);  // 11-22: 0.7 m 엿보기는 흐려 못 읽고 gaze + 1080p는 바로 읽힘 → 손 카메라 1080p는 gaze에서
        note("바코드 후보 방향 " + fmt("%+.2f rad", yaw) + "으로 gaze — 손 카메라 " + peek_profile_ + " · 신호등 관측 계속");
        run({SubGoal().set__skill("pose").set__target(target)}, [this](bool ok, const std::string & why) {
          if (phase_ != Phase::BARCODE || !barcode_.empty()) return;
          if (ok) { gaze_at_ = gaze_seen_ = now(); gaze_joints_ = joints_now_; snap("barcode_gaze"); return; }  // 도착 — 그대로 계속 봄 (관절 값 기억: 처짐 감시)
          timer_->cancel();
          barcode_done(false, "barcode_pose_failed:" + why);
        });
      }
      // gaze에서 안 읽히면 팔은 그대로 한 걸음 앞으로 (후보가 없으면 팔·바퀴 모두 움직이지 않음)
      // 신호가 확정되기 전에는 다가가지 않음 — 움직이면 오른쪽 신호등이 pole_front_cam 시야 밖으로 나감 (11-12 실기: 다가가다 신호 못 읽음)
      //   확정되면 그때부터 다가감 · 신호 시간(signal_timeout_s)이 지나면 기다리지 않음 (READ에서 signal_timeout)
      const bool hold = signal_window_ && !signal_ready_ && signal_error_.empty() && (now() - read_started_).seconds() < signal_timeout_s_;
      if (hold && gaze_at_.nanoseconds() > 0) gaze_at_ = now();  // 기다리는 동안은 바코드 대기·포기 시간을 안 셈
      if (hold && !hold_noted_ && gaze_at_.nanoseconds() > 0) { hold_noted_ = true; note("gaze 제자리에서 바코드·신호 — 신호 확정 뒤에 다가감"); }
      const bool waited = !hold && gaze_at_.nanoseconds() > 0 && (now() - gaze_at_).seconds() > barcode_gaze_wait_s_;
      // 읽힐 때까지 다가감 — 손 카메라 정면이 barcode_min_gap_m보다 가깝거나(또는 depth를 모르면) · 최대 거리면 그만
      const bool fresh = (now() - grip_front_at_).seconds() < 1.0;  // 1 s 넘게 depth가 안 오면 모름 → 안 다가감
      const bool room = fresh && grip_front_m_ > barcode_min_gap_m_ + barcode_advance_step_m_ * 0.5 &&
                        advanced_m_ + barcode_advance_step_m_ <= barcode_advance_max_m_ + 1e-6;
      // 신호등 왼쪽 scan 중이면 다가가기 전에 다음 방향으로 (팔만 돌림 — 신호등은 pole 시야에 남음)
      // 후보 둘레 다음 방향 (팔만 돌림 — 신호를 기다리는 중에도)
      const bool looked = gaze_seen_.nanoseconds() > 0 && gaze_at_.nanoseconds() > 0 &&
                          (now() - gaze_seen_).seconds() > barcode_gaze_wait_s_;
      if (looked && !moving_ && !scan_ && advanced_m_ == 0.0 && cand_k_ + 1 < cand_offsets_.size()) {
        ++cand_k_;
        gaze_sent_ = false;
        gaze_at_ = gaze_seen_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        note(fmt("바코드 안 읽힘 — 후보 둘레 %+.2f rad로 (%.0f/%.0f)", cand_offsets_[cand_k_], cand_k_ + 1.0, static_cast<double>(cand_offsets_.size())));
        return;
      }
      if (looked && !moving_ && scan_ && scan_k_ + 1 < scan_offsets_.size()) {
        ++scan_k_;
        gaze_sent_ = false;
        gaze_at_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        note(fmt("바코드 안 읽힘 — 신호등 왼쪽 %.2f rad로 (scan %.0f/%.0f)", scan_offsets_[scan_k_], scan_k_ + 1.0, static_cast<double>(scan_offsets_.size())));
        return;
      }
      if (waited && !moving_ && room) {
        {
          moving_ = true;
          const double d = std::min(barcode_advance_step_m_, grip_front_m_ - barcode_min_gap_m_);  // 마지막 걸음은 남은 만큼만
          // 목표 = 출발 자리에서 출발 방향으로 (지금까지 + d) — 지금 자세 기준으로 잡으면 걸음마다 방향 오차가 쌓였음 (11-12: 0 → −9 → −14°)
          const Pose2 & o = barcode_origin_;
          const double k = advanced_m_ + d;
          const auto g = to_map(o.x + k * std::cos(o.yaw), o.y + k * std::sin(o.yaw), o.yaw);
          note(fmt("바코드가 안 읽혀 다가감 (지금까지 %.2f m · 손 카메라 정면 %.2f m · 팔 그대로)", advanced_m_ + d, grip_front_m_));
          run({SubGoal().set__skill("approach").set__target(fmt("%.4f,%.4f,%.4f", g.x, g.y, g.yaw)).set__relation("drive_straight")},
            [this, d](bool ok, const std::string & why) {
              if (phase_ != Phase::BARCODE || !barcode_.empty()) return;
              moving_ = false;
              advanced_m_ += d;  // 실패해도 간 만큼으로 보고 돌아올 때 원래 자리로 감
              gaze_at_ = now();  // 새 자리에서 다시 barcode_gaze_wait_s 동안 봄
              if (!ok) RCLCPP_WARN(get_logger(), "바코드 다가가기 멈춤 (%s) — 이 자리에서 봄", why.c_str());
            });
        }
      }
      // 포기는 더 다가갈 수 없는 마지막 자리에서 barcode_timeout_s를 기다린 뒤에만 (다가가는 시간은 안 셈 — 실기 11-12: 23 s 전체
      //   제한에 걸려 1 s 뒤 읽힌 바코드를 놓침) · gaze에 아예 못 가면 pole_s + timeout_s
      const bool final_wait = gaze_at_.nanoseconds() > 0 && !moving_ && !room && (now() - gaze_at_).seconds() > barcode_timeout_s_;
      const bool no_gaze = !signal_wait && gaze_at_.nanoseconds() == 0 && gaze_free_.nanoseconds() > 0 &&
                           (now() - gaze_free_).seconds() > barcode_timeout_s_;  // gaze를 보낼 수 있게 된 뒤 timeout 안에 도착 못 함
      if (final_wait || no_gaze) {
        timer_->cancel();
        const std::string failure = !bad_barcode_.empty() ? "unknown_barcode:" + bad_barcode_
          : (gaze_sent_ ? "barcode_missing" : "barcode_candidate_missing");
        barcode_done(false, failure);
      }
      return;
    }
    timer_->cancel();
    RCLCPP_INFO(get_logger(), "바코드 %s (%.1f s · %s)", barcode_.c_str(), t,
                gaze_sent_ ? "손 카메라 gaze" : peek_sent_ ? "손 카메라 엿보기" : "팔 안 움직임");
    snap("barcode");
    if (!observe_move_) { barcode_done(true, ""); return; }  // 이미 관측 자리 — 팔을 뻗었으면 접음
    hires(false);
    const auto p = to_map(start_.x, start_.y, M_PI);  // 관측 = 출발 자리에서 −X(π)를 봄 — 팔부터 접고 이동
    run({stow(), SubGoal().set__skill("navigate").set__target(fmt("%.3f,%.3f,%.3f", p.x, p.y, p.yaw))},
      [this](bool ok, const std::string & why) { finish(ok, "observe_nav_failed:" + why); });
  });
}

bool MissionManager::observe_still() const {
  // 관측 중 팔·바퀴가 멈춰 있음 — pole 3초(팔 stow) 또는 gaze 도착 뒤 · 다가가기·복귀 중 아님
  return !moving_ && !returning_ && (!gaze_sent_ || gaze_at_.nanoseconds() > 0) &&
         (!peek_sent_ || gaze_sent_ || peek_at_.nanoseconds() > 0);
}

// 관측 시작 — 시작 signal_preroll_s 안에 확정된 신호가 있으면 바로 씀 (11-22 실기 00:12:48: 시작 20 s 전 확정을 못 써 27 s 더 기다림)
void MissionManager::take_recent_signal() {
  if (!last_ok_signal_ || signal_ready_) return;
  const rclcpp::Time t(last_ok_signal_->header.stamp, RCL_ROS_TIME);
  if (t < read_started_ - rclcpp::Duration::from_seconds(signal_preroll_s_)) return;
  signal_ = *last_ok_signal_;
  signal_ready_ = true;
  RCLCPP_INFO(get_logger(), "신호 — 관측 시작 %.0f s 전에 확정된 것 씀 (%s)", (read_started_ - t).seconds(), signal_.raw.c_str());
}

void MissionManager::hires(bool on) {
  if (on == hires_on_) return;
  hires_on_ = on;
  hires_live_at_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
  const auto & prof = on ? peek_profile_ : cam_profile_;
  if (!cam_params_->service_is_ready()) {
    RCLCPP_WARN(get_logger(), "손 카메라 파라미터 서비스 없음 — 해상도 %s 못 바꿈", prof.c_str());
    return;
  }
  // realsense-ros는 켜진 스트림의 프로필만 바꾸면 "re-enable the stream for the change to take effect"만 내고 그대로 —
  //   껐다 켜야 적용 (11-21 실기: 640 px 그대로 → 엿보기 5 s 뒤 "고해상도 영상이 안 옴").
  //   응답을 기다려 다음을 보내던 방식은 응답 콜백이 안 와 멈췄음(같은 실기 로그에 결과 줄이 없음) → 세 요청을 순서대로 바로 보냄
  //   (realsense 노드가 한 줄로 처리 · 결과는 cam_info 폭으로 확인 — peek 쪽 hires_at_)
  cam_params_->set_parameters({rclcpp::Parameter("rgb_camera.color_profile", prof)});
  cam_params_->set_parameters({rclcpp::Parameter("enable_color", false)});
  // 켜기는 끄기가 끝난 뒤에 — 바로 이어 보내면 realsense가 끄는 중에 받은 켜기를 버려 컬러가 꺼진 채 남음
  //   (11-21 실기 21:43: "Stopping Sensor: RGB Camera" 뒤 다시 안 켜짐 → 손 카메라 영상 없음 · 바코드 못 읽음 · 사진 0장).
  //   컬러 cam_info가 다시 올 때까지 0.7 s마다 켜기를 다시 보냄 (최대 12번)
  cam_req_at_ = now();
  cam_tries_ = 0;
  cam_timer_ = create_wall_timer(std::chrono::milliseconds(700), [this, prof]() {
    if (cam_info_at_ > cam_req_at_ + rclcpp::Duration::from_seconds(0.3) && (now() - cam_info_at_).seconds() < 1.0) {
      RCLCPP_INFO(get_logger(), "손 카메라 컬러 %s 켜짐 (폭 %u px · %.1f s)", prof.c_str(), cam_width_, (now() - cam_req_at_).seconds());
      cam_timer_->cancel();
      return;
    }
    if (++cam_tries_ > 12) {
      RCLCPP_ERROR(get_logger(), "손 카메라 컬러가 다시 안 켜짐 (%s) — enable_color 확인", prof.c_str());
      cam_timer_->cancel();
      return;
    }
    cam_params_->set_parameters({rclcpp::Parameter("enable_color", true)});
  });
  RCLCPP_INFO(get_logger(), "손 카메라 컬러 %s 요청 (프로필 → 스트림 끔 → 켜질 때까지 켬)", prof.c_str());
}

void MissionManager::peek_end(const std::string & why) {
  peek_done_ = true;
  hires(false);
  note("엿보기로 못 읽음 (" + why + ") — gaze로");
}

void MissionManager::barcode_done(bool ok, const std::string & why) {
  if (!ok) snap("barcode_fail");  // 손을 접기 전 손 카메라가 본 것
  hires(false);
  if (!gaze_sent_ && !peek_sent_) { ok ? finish(true, "") : finish(false, why); return; }  // 팔을 안 움직였음
  // 뻗던·다가가던 중이면 이 계획이 진행 중 goal을 취소함 · 팔 접고 → 다가갔으면 원래 자리로 똑바로 후진
  std::vector<SubGoal> back{stow()};
  if (advanced_m_ > 0.0 || moving_) {
    const auto o = to_map(barcode_origin_.x, barcode_origin_.y, barcode_origin_.yaw);
    back.push_back(SubGoal().set__skill("approach").set__target(fmt("%.4f,%.4f,%.4f", o.x, o.y, o.yaw)).set__relation("drive_straight"));
  }
  moving_ = false;
  returning_ = true;
  run(back, [this, ok, why](bool sok, const std::string & swhy) {  // 실패해도 접고(·돌아오고) 끝냄
    returning_ = false;
    gaze_sent_ = peek_sent_ = false;  // 팔 접음 (abort 때 다시 접을 필요 없음)
    if (!ok) { finish(false, why); return; }
    // 바코드는 읽었음 — 후진 복귀가 조금 덜 맞아도(approach_blocked 등) 미션은 계속 (다음 이동은 Nav2) · 팔 접기 실패만 멈춤
    if (!sok) RCLCPP_WARN(get_logger(), "바코드 뒤 복귀 덜 됨 (%s) — 계속", swhy.c_str());
    finish(sok || swhy.rfind("approach", 0) == 0, "barcode_return_failed:" + swhy);
  });
}

void MissionManager::read_phase() {
  // 신호는 BARCODE 시작부터 받고 있음 (barcode_phase) — 이미 확정됐으면 바로 다음 · READ만 실행이면 여기서 창을 엶
  if (!signal_window_) {
    signal_ = perception::msg::MissionSignal();
    signal_ready_ = false;
    signal_streak_ = 0;
    signal_error_.clear();
    read_started_ = now();
    signal_window_ = true;
    take_recent_signal();
  }
  read_phase_at_ = now();
  note(signal_ready_ ? "신호 — 관측(바코드) 중에 이미 확정" : "신호 12회 관측 중 (바코드 뒤 팔 접고 계속)");
  timer_ = create_wall_timer(std::chrono::milliseconds(500), [this]() {
    if (phase_ != Phase::READ) { timer_->cancel(); return; }
    if (!signal_error_.empty()) {  // 오류 사이클이 이어짐 — 추정하지 않고 정지
      timer_->cancel();
      write_log("signal.json", "{\"ok\": false, \"error\": " + q(signal_error_) + ", \"raw\": " + q(signal_.raw) + "}\n");
      finish(false, "signal_" + signal_error_);
      return;
    }
    if (!signal_ready_) {
      // 신호 시간 = 관측 시작부터 signal_timeout_s — 바코드에 시간을 다 써도 READ에서 최소 signal_read_min_s는 봄
      const double waited = (now() - read_started_).seconds(), in_read = (now() - read_phase_at_).seconds();
      if (waited > signal_timeout_s_ && in_read > signal_read_min_s_) { timer_->cancel(); finish(false, "signal_timeout"); }
      return;
    }
    timer_->cancel();
    counts_ = {signal_.red, signal_.yellow, signal_.green};
    std::string why, seq;
    for (size_t i = 0; i < signal_.sequence.size(); ++i) seq += (i ? "," : "") + q(signal_.sequence[i]);
    bool asc = false;
    const bool code_ok = parse_barcode(barcode_, asc);
    if (code_ok) order_ = assign_locations(counts_, asc);
    write_log("signal.json", "{\"barcode\": " + q(barcode_) + ", \"ascending\": " + (asc ? "true" : "false") +
      ", \"sequence\": [" + seq + "], \"raw\": " + q(signal_.raw) + ", \"cycles\": " + str(signal_.cycles) + ", \"counts\": {\"R\": " + str(counts_.r) +
      ", \"Y\": " + str(counts_.y) + ", \"G\": " + str(counts_.g) + "}, \"order\": " + q(order_) + "}\n");
    if (!valid_counts(counts_, why)) { finish(false, why); return; }
    if (only_ == Phase::READ && barcode_.empty()) {  // READ만 실행 — 바코드 없이 신호만 확인
      RCLCPP_INFO(get_logger(), "신호 R%d Y%d G%d (READ만 — 바코드 없어 장소 배정은 안 함)", counts_.r, counts_.y, counts_.g);
      finish(true);
      return;
    }
    if (!code_ok) { finish(false, "unknown_barcode:" + barcode_); return; }
    RCLCPP_INFO(get_logger(), "신호 R%d Y%d G%d · %s → L1 %c · L2 %c · L3 %c", counts_.r, counts_.y, counts_.g,
                barcode_.c_str(), order_[0], order_[1], order_[2]);
    finish(true);
  });
}

// 시작 명령으로 받은 바코드·신호 — from_survey면 처음부터 쓰고, diag면 관측이 실패한 것만 바꿔 씀 (fallback)
void MissionManager::use_given(bool fallback) {
  if (!given_) return;
  bool asc = false;
  std::string why;
  const bool code_ok = parse_barcode(barcode_, asc), count_ok = valid_counts(counts_, why);
  if (code_ok && count_ok) return;  // 관측이 다 됨
  std::string used;
  if (!code_ok) { barcode_ = given_barcode_; used += "바코드 " + barcode_; }
  if (!count_ok) {
    counts_ = given_counts_;
    used += std::string(used.empty() ? "" : " · ") + "신호 R" + str(counts_.r) + "Y" + str(counts_.y) + "G" + str(counts_.g);
  }
  parse_barcode(barcode_, asc);
  order_ = assign_locations(counts_, asc);
  if (fallback) result("diag: 관측 실패 → 시작 명령 값 " + used);
  write_log(fallback ? "signal_given.json" : "signal.json", "{\"given\": true, \"barcode\": " + q(barcode_) +
    ", \"ascending\": " + (asc ? "true" : "false") + ", \"counts\": {\"R\": " + str(counts_.r) + ", \"Y\": " +
    str(counts_.y) + ", \"G\": " + str(counts_.g) + "}, \"order\": " + q(order_) + "}\n");
  publish_state();
}

// ---------- SURVEY — 시작 박스 6곳 전부 조사하면서 적재함에 싣기
void MissionManager::goto_seq(size_t k) {
  if (!active_) return;
  if (k >= goto_.size()) { note("상자로 가기 끝 — 출발 구역으로 복귀"); only_ = Phase::RETURN; enter(Phase::RETURN); return; }
  const std::string id = goto_[k];
  const size_t i = static_cast<size_t>(id[1] - '1');
  const Box box = id[0] == 'L' ? loc_boxes_.at(i) : start_boxes_.at(i);
  ctx_ = "\"goto\": " + q(id);
  note("상자로 가기 " + id + " (" + str(static_cast<int>(k) + 1) + "/" + str(static_cast<int>(goto_.size())) + ")");
  go_near(box, box.x, box.y, [this, k, id](bool ok, const std::string & why) {
    result("GOTO " + id + (ok ? ": ok" : ": " + why));
    if (!ok) RCLCPP_WARN(get_logger(), "상자로 가기 %s 실패 (%s) — 다음", id.c_str(), why.c_str());
    else snap("goto_" + id);
    goto_seq(k + 1);
  });
}

void MissionManager::survey(size_t k) {  // k = 방문 순서 인덱스 (survey_order_) · b = 박스 번호(0부터)
  if (k >= survey_order_.size()) {
    // 못 실은 퍽이 남은 상자는 BUILD(place) 전에 한 번 더 (11-21 사용자: 시작 박스의 퍽은 다 싣고 · 들고 다니지 않음)
    std::vector<size_t> again;
    for (const size_t b : survey_order_) if (leftover_.count(b)) again.push_back(b);
    if (!second_pass_ && !again.empty()) { second_pass_ = true; revisit(again, 0); return; }
    solve_allocation();
    return;
  }
  survey_k_ = k;
  visit_box(survey_order_[k], false, [this, k]() { survey(k + 1); });
}

void MissionManager::revisit(std::vector<size_t> boxes, size_t k) {
  if (k >= boxes.size()) { solve_allocation(); return; }
  visit_box(boxes[k], true, [this, boxes, k]() { revisit(boxes, k + 1); });
}

// 상자 보기 inspect — 그 상자에 남은 퍽 수(on_box_ · 조사한 상자)를 4번째 값으로 (11-22 실기: 재방문에 1개 남았는데 2개를 기대해 inspect_partial 3번)
//   모르면(아직 조사 전 · 퍽을 못 봄) 2
SubGoal MissionManager::box_inspect(size_t b, double x, double y) const {
  int n = 0;
  for (const int t : on_box_) if (const auto it = puck_at_.find(t); it != puck_at_.end() && it->second.first == b) ++n;
  return SubGoal().set__skill("inspect").set__target(fmt("%.4f,%.4f,%.4f", x, y, box_top_z_ + puck_h_) + "," + str(n > 0 ? n : 2)).set__relation("box");
}

void MissionManager::visit_box(size_t b, bool again, std::function<void()> then) {
  const Box box = start_boxes_[b];
  ctx_ = "\"box\": " + str(static_cast<int>(b) + 1);
  const std::vector<int> left = again ? leftover_[b] : std::vector<int>{};
  leftover_.erase(b);
  note("시작 박스 " + str(static_cast<int>(b) + 1) + (again ? " 다시 (못 실은 퍽)" : " 조사"));
  go_near(box, box.x, box.y, [this, b, box, again, left, then](bool ok, const std::string & why) {
    if (!ok) {
      result("box " + str(static_cast<int>(b) + 1) + ": approach_failed " + why);
      if (!again) leftover_[b];  // 퍽을 모름 — place 전에 다시 와서 봄
      for (const int tag : left) result("puck " + str(tag) + ": not_loaded approach_failed");
      then();
      return;
    }
    // 기준 = 붙은 상자: 도면 자리에서 0.5 m 안 인지 start_box 중 가장 가까운 것 (없으면 도면).
    //   11-20 sim S4: 인지 상자가 도면에서 31 cm (map) → 도면 기준으로 세면 inspect는 0.35 원 안이라 ok, 조사는 윗면 밖이라 0개 → pick 없이 지나감
    const auto m0 = to_map(box.x, box.y);
    double cx = m0.x, cy = m0.y, best = 0.5;
    if (objects_) for (const auto & o : objects_->objects) {
      if (o.class_name != "start_box") continue;
      const double d = std::hypot(o.pose.pose.position.x - m0.x, o.pose.pose.position.y - m0.y);
      if (d < best) { best = d; cx = o.pose.pose.position.x; cy = o.pose.pose.position.y; }
    }
    if (best < 0.5) RCLCPP_INFO(get_logger(), "시작 박스 %zu 기준 = 인지 상자 (도면에서 %.0f cm)", b + 1, 100 * std::hypot(cx - m0.x, cy - m0.y));
    box_seen_[b] = make_spot(*tf_buffer_, cx, cy);
    // 상자 한 번 보기 (look 자세 → 퍽이 모자라거나 태그를 못 읽으면 본 퍽을 담는 자세로 다시) — 퍽 자리마다 따로 보던 것 (11/07)
    const std::vector<SubGoal> goals = {
      box_inspect(b, cx, cy)};
    run_retry(goals, [this, b, cx, cy, goals, again, left, then](bool ok2, const std::string & why2) {
      if (!ok2) RCLCPP_WARN(get_logger(), "시작 박스 %zu inspect 실패 (%s) — 본 것만", b + 1, why2.c_str());
      // 차체에 붙은 퍽(tag 못 읽음 · 팔이 접히는 자리) — look 자세로 물러나 다시 본 뒤 셈
      back_off(b, cx, cy, 0, [this, b, cx, cy, goals, again, left, then]() {
        snap("box" + str(static_cast<int>(b) + 1));
        auto found = collect(b, cx, cy);
        for (const int tag : left) if (!std::count(found.begin(), found.end(), tag) && basket_slot_of(tag) < 0) found.push_back(tag);
        const auto start = [this, b, then](const std::vector<int> & f, bool counted) {
          if (counted)
            for (int missing = static_cast<int>(puck_dx_.size()) - static_cast<int>(f.size()); missing > 0; --missing)
              result("box " + str(static_cast<int>(b) + 1) + ": tag_unread");
          load(b, f, 0, then);
        };
        // tag를 못 읽어 할 일이 박스 퍽 수보다 적음 — 한 번 더 봄 (1차만 · 다시 온 상자는 이미 실은 퍽이 있어 수를 모름)
        if (again || found.size() >= puck_dx_.size()) { start(found, !again); return; }
        run(goals, [this, b, cx, cy, found, start](bool, const std::string &) {
          auto all = found;
          for (const int tag : collect(b, cx, cy)) if (!std::count(all.begin(), all.end(), tag)) all.push_back(tag);
          std::stable_partition(all.begin(), all.end(), [this](int tag) { return !lying_.count(tag); });
          start(all, true);
        });
      });
    });
  });
}

// 박스 윗면 위에서 tag를 읽은 퍽 전부 (자리 제한 없음) — 예전엔 발제 자리(X±40, Y=0)에서 4 cm(match_radius) 안만 봐서
//   퍽이 비켜 있거나 위치 오차가 있으면 하나도 못 찾고 pick 없이 지나감 (실기 11-20 box4·box5). 집기는 원래 tag로 찾음.
//   이미 아는 tag도 적재함에 없고 이 상자에 있다고 적혀 있지 않으면 할 일로 다시 적음 — 11-21 실기 S6: S4에서 되돌리지 못하고 들고 온
//   퍽 5가 S6에 떨어져 퍽이 3개였는데 '아는 tag'라 건너뜀
std::vector<int> MissionManager::collect(size_t b, double cx, double cy) {
  const Box & box = start_boxes_[b];
  std::vector<int> found;
  if (objects_) for (const auto & o : objects_->objects) {
    if (!is_puck(o) || detail(o, "tag").empty()) continue;
    // 붙은 상자 윗면 안 (모서리까지 half·√2 + 3 cm) — 이웃 시작 박스는 0.734 m
    if (std::hypot(o.pose.pose.position.x - cx, o.pose.pose.position.y - cy) > box.half * M_SQRT2 + 0.03) continue;
    const int tag = std::stoi(detail(o, "tag"));
    if (std::count(found.begin(), found.end(), tag)) continue;  // 같은 tag 트랙 둘
    const auto at = puck_at_.find(tag);
    const bool known = at != puck_at_.end();
    if (!box_todo(basket_slot_of(tag) >= 0, known, known && at->second.first == b)) continue;
    const auto a = to_arena(o.pose.pose.position.x, o.pose.pose.position.y, 0.0);
    const double dx = a.x - box.x, dy = a.y - box.y;
    if (!known) pucks_.push_back({tag, o.color, detail(o, "shape")});
    puck_at_[tag] = {b, dx};
    puck_seen_[tag] = make_spot(*tf_buffer_, o.pose.pose.position.x, o.pose.pose.position.y);
    on_box_.insert(tag);
    if (is_lying(o)) lying_.insert(tag);  // 실을 때 세워서 집음 (stand → stand_pick)
    else lying_.erase(tag);
    found.push_back(tag);
    RCLCPP_INFO(get_logger(), "퍽 %d: %s %s (박스 %zu, %+.3f %+.3f)%s", tag, o.color.c_str(), detail(o, "shape").c_str(), b + 1, dx, dy,
                !known ? "" : " — 다른 자리로 알던 퍽 (떨어뜨렸거나 옮겨짐)");
  }
  publish_state();
  // 선 퍽 먼저 싣고 누운 퍽은 나중에 (세우다 옆 퍽을 칠 수 있음 · 11-15 사용자)
  std::stable_partition(found.begin(), found.end(), [this](int tag) { return !lying_.count(tag); });
  return found;
}

// 상자 윗면 위 퍽 중 차체에 붙은 것 — 로봇 앞 inspect_back_min_x 안이거나 tag를 못 읽음 (적재함에 실린 tag는 뺌).
//   need = 가장 붙은 퍽이 inspect_back_want_x에 오게 물러날 거리 (tag만 못 읽은 퍽은 그 자리 기준)
bool MissionManager::near_body(size_t b, double & need) const {
  Pose2 r;
  if (!robot_map(r) || !box_seen_.count(b)) return false;
  const auto [bx, by] = here(box_seen_.at(b));
  const double c = std::cos(r.yaw), sn = std::sin(r.yaw);
  bool any = false;
  need = 0.0;
  for (const auto & [id, o] : recent_pucks_) {  // 지금 목록 + 근래 보였다 빠진 퍽 (map 자리 그대로 · 로봇은 지금 자세)
    const auto & p = o.pose.pose.position;
    if (p.z > box_top_z_ + puck_h_ + 0.04) continue;  // 손에 든 퍽은 윗면보다 위
    if (std::hypot(p.x - bx, p.y - by) > start_boxes_[b].half * M_SQRT2 + 0.03) continue;
    const std::string t = detail(o, "tag");
    if (!t.empty() && basket_slot_of(std::stoi(t)) >= 0) continue;
    const double fwd = (p.x - r.x) * c + (p.y - r.y) * sn;
    // tag를 못 읽은 트랙은 붙어 있을 때만 (inspect_back_want_x 안) — 11-22 실기 nav_1122_095737: 물러남 8번이 모두 0.21~0.34 m 앞의
    //   tag 없는 트랙 (같은 퍽의 tag 읽기 전 트랙 · 근래 목록) 때문이었음. 실제로 붙은 퍽(0.13~0.15 m)만 tag를 못 읽음
    if (fwd >= (t.empty() ? back_want_x_ : back_min_x_)) continue;
    any = true;
    need = std::max(need, back_want_x_ - fwd);
  }
  return any;
}

// 차체에 붙은 퍽이 있으면 look 자세로 곧게 물러나 다시 inspect (두 번까지 · 이번 도킹에서 inspect_back_max_m까지) → then.
//   11-21 사용자: 차체 쪽 퍽이 안 보이면 look 자세로 후진 (퍽 IK 안정권 + tag·모양·색) · 실기 nav_1121_183927: 로봇 앞 x 0.13~0.15 m 퍽은
//   tag를 못 읽거나 안 보임(S2 퍽 0 · S5 퍽 6 → puck_lost), 0.20~0.25 m는 다 읽음. 물러난 만큼 pick_near가 다시 다가갈 수 있음 (creep_used_)
void MissionManager::back_off(size_t b, double cx, double cy, int tries, std::function<void()> then) {
  double need = 0.0;
  const double room = back_max_m_ - backed_;
  Pose2 r;
  if (tries >= 2 || back_.empty() || room < 0.01 || !near_body(b, need) || !robot_map(r)) { then(); return; }
  const double step = std::clamp(need, 0.02, room);
  note(fmt("상자 위 퍽이 차체에 붙음 — look 자세로 %.0f mm 물러남", step * 1000));
  const double c = std::cos(r.yaw), sn = std::sin(r.yaw);
  run({SubGoal().set__skill("pose").set__target("look"),
       SubGoal().set__skill("approach").set__target(fmt("%.4f,%.4f,%.4f", r.x - step * c, r.y - step * sn, r.yaw)).set__relation("drive_straight")},
      [this, b, cx, cy, tries, step, then](bool ok, const std::string & why) {
        // 실패해도 일부 움직였을 수 있음 — 물러날 거리는 덜 빼는 쪽 (더 물러나는 쪽이 안전) · 다가갈 여유는 늘림
        backed_ += step;
        creep_used_ -= step;
        if (ok && !back_.empty()) back_ = fmt("%.3f", std::max(0.0, std::stod(back_) - step));
        if (!ok) RCLCPP_WARN(get_logger(), "물러나기 실패 (%s) — 그 자리에서 inspect", why.c_str());
        run({box_inspect(b, cx, cy)},
            [this, b, cx, cy, tries, then](bool, const std::string &) { back_off(b, cx, cy, tries + 1, then); });
      });
}

void MissionManager::load(size_t b, std::vector<int> found, size_t k, std::function<void()> then) {
  if (k >= found.size()) { then(); return; }
  load_one(b, found[k], 0, [this, b, found, k, then]() { load(b, found, k + 1, then); });
}

void MissionManager::load_one(size_t b, int tag, int load_try, std::function<void()> next) {
  if (basket_slot_of(tag) >= 0) { next(); return; }  // 이미 실림
  if (!basket_has_room()) { result("puck " + str(tag) + ": not_loaded slot_full"); next(); return; }
  ctx_ = "\"box\": " + str(static_cast<int>(b) + 1) + ", \"tag\": " + str(tag);
  note("퍽 " + str(tag) + " → 적재함 (load next)" + (load_try ? " · 다시 " + str(load_try + 1) + "/" + str(load_tries_) : ""));
  // 칸은 coordinator가 고르고 장부에 tag까지 적음 (pick 때 그 퍽의 tag·색·모양을 기억해 둠)
  grab(tag, [this, b, tag, load_try, next](bool ok, const std::string & why) {
    if (!ok) { give_up(b, tag, "load_pick_failed " + why); next(); return; }
    on_box_.erase(tag);  // 손에 듦
    run_retry({SubGoal().set__skill("load").set__target("next")}, [this, b, tag, load_try, next](bool ok2, const std::string & why2) {
      if (ok2) { next(); return; }
      // 놓은 뒤 실패 — 퍽은 칸에 있음 (coordinator 장부 '확인 필요')
      if (why2.rfind("load_released", 0) == 0) { result("puck " + str(tag) + ": " + why2 + " (칸에 둔 것으로 봄)"); next(); return; }
      // 적재함에 못 실음 — 상자에 되돌려 놓고 다시 집어 실음 (들고 다니지 않음)
      put_back(b, tag, why2, [this, b, tag, load_try, next, why2](bool placed) {
        if (!placed) { give_up(b, tag, "load_tray_failed " + why2 + " (되돌리기 실패)"); next(); return; }
        if (load_try + 1 >= load_tries_) { give_up(b, tag, "load_tray_failed " + why2); next(); return; }
        load_one(b, tag, load_try + 1, next);
      });
    });
  });
}

void MissionManager::give_up(size_t b, int tag, const std::string & why) {
  if (second_pass_) { result("puck " + str(tag) + ": not_loaded " + why); return; }
  leftover_[b].push_back(tag);
  result("puck " + str(tag) + ": " + why + " (place 전에 다시 옴)");
}

std::pair<double, double> free_spot(double rx, double ry, double ryaw, double cx, double cy, double lim, double want_x,
                                    const std::vector<std::pair<double, double>> & others, double ring, double clear) {
  const double fx = std::cos(ryaw), fy = std::sin(ryaw);  // 로봇 앞
  const double lx = -fy, ly = fx;                         // 로봇 왼쪽
  // 바라는 점: 로봇 앞 want_x · 옆은 상자 가운데 줄 — 상자 윗면 안(가운데에서 lim)으로 당김
  const double lat0 = (cx - rx) * lx + (cy - ry) * ly;
  double px = rx + fx * want_x + lx * lat0, py = ry + fy * want_x + ly * lat0;
  if (const double d = std::hypot(px - cx, py - cy); d > lim) { px = cx + (px - cx) * lim / d; py = cy + (py - cy) * lim / d; }
  // 후보 (앞, 옆) 순서: 그 점 → 옆으로 → 옆·조금 앞 → 조금 앞 — 로봇 쪽(팔이 접히는 쪽)으로는 안 감
  static const double off[][2] = {{0, 0}, {0, 1}, {0, -1}, {0.5, 1}, {0.5, -1}, {0.5, 0}, {1, 0}, {1, 1}, {1, -1}};
  std::pair<double, double> best = {px, py};
  double best_d = -1.0;
  for (const auto & o : off) {
    const double x = px + ring * (o[0] * fx + o[1] * lx), y = py + ring * (o[0] * fy + o[1] * ly);
    if (std::hypot(x - cx, y - cy) > lim + 1e-9) continue;  // 상자 윗면 밖
    double d = std::numeric_limits<double>::infinity();
    for (const auto & q : others) d = std::min(d, std::hypot(q.first - x, q.second - y));
    if (d >= clear) return {x, y};
    if (d > best_d) { best_d = d; best = {x, y}; }
  }
  return best;
}

// 놓을 자리: 로봇 앞 put_back_x_m · 상자 윗면 안 · 다른 퍽이 가까우면 옆에서 비어 있는 곳 (free_spot) — 다시 집을 때 옆 퍽에 손가락이 얹히지 않게.
//   다른 퍽 = 지금 상자 윗면에 보이는 퍽 + 상자 위에 있다고 적힌 퍽의 본 자리. 놓은 뒤 inspect로 다시 봐서 다음 pick이 새 자리를 쓰게 함
void MissionManager::put_back(size_t b, int tag, const std::string & why, std::function<void(bool)> next) {
  const auto [bx, by] = here(box_seen_[b]);
  std::vector<std::pair<double, double>> others;
  for (const int t : on_box_) if (t != tag && puck_at_[t].first == b && puck_seen_.count(t)) others.push_back(here(puck_seen_[t]));
  if (objects_) for (const auto & o : objects_->objects) {
    const auto & p = o.pose.pose.position;
    if (!is_puck(o) || detail(o, "tag") == str(tag) || p.z > box_top_z_ + puck_h_ + 0.04) continue;  // 손에 든 퍽은 윗면보다 위
    if (std::hypot(p.x - bx, p.y - by) <= start_boxes_[b].half * M_SQRT2) others.emplace_back(p.x, p.y);
  }
  // 로봇 앞 put_back_x_m 근처 — 상자 가운데는 팔 끝(pick 한계 0.25 m)에 걸리고 tag도 덜 읽힘 (11-21 실기: tag를 잘 읽은 퍽 x 0.20~0.25 m)
  Pose2 r;
  if (!robot_map(r)) r = {bx - put_back_x_m_, by, 0.0};  // TF 없음 — 상자 가운데 쪽으로
  const auto [px, py] = free_spot(r.x, r.y, r.yaw, bx, by, start_boxes_[b].half - 0.04, put_back_x_m_, others, put_back_ring_m_,
                                  put_back_clear_m_);
  const std::string head = "puck " + str(tag) + ": load 실패 " + why;
  note("퍽 " + str(tag) + " 상자 " + str(static_cast<int>(b) + 1) + "에 되돌림" + fmt(" (가운데에서 %.0f mm)", 1000 * std::hypot(px - bx, py - by)));  // fmt는 double만
  run({SubGoal().set__skill("place").set__target(fmt("%.4f,%.4f,%.4f", px, py, box_top_z_))},
    [this, b, tag, bx, by, px, py, head, next](bool ok, const std::string &) {
      if (!ok) { RCLCPP_WARN(get_logger(), "퍽 %d 상자 %zu에 되돌리기 실패", tag, b + 1); next(false); return; }
      puck_seen_[tag] = make_spot(*tf_buffer_, px, py);
      on_box_.insert(tag);
      const rclcpp::Time t0 = now();
      run({box_inspect(b, px, py)},
        [this, b, tag, bx, by, head, next, t0](bool, const std::string &) {
          // 이 inspect 동안 그 tag를 상자 윗면 안(가운데에서 half)에서 봤나 — 끝난 순간의 목록이 아니라 inspect 중 마지막 모습 (last_seen_).
          //   실기 170024 S5: 놓은 퍽 6을 look에서 봤는데 '다시 담기'로 카메라가 돌아가 2 s 뒤 트랙이 지워져 끝에서는 못 찾음
          const auto it = last_seen_.find(tag);
          const TrackedObject * hit = it != last_seen_.end() && rclcpp::Time(it->second.last_seen, RCL_ROS_TIME) >= t0 &&
              std::hypot(it->second.pose.pose.position.x - bx, it->second.pose.pose.position.y - by) <= start_boxes_[b].half ? &it->second : nullptr;
          if (hit) puck_seen_[tag] = make_spot(*tf_buffer_, hit->pose.pose.position.x, hit->pose.pose.position.y);
          else RCLCPP_WARN(get_logger(), "퍽 %d 되돌림 확인 실패 — 상자 %zu 윗면에서 안 보임 (놓은 자리로 다시 찾음)", tag, b + 1);
          result(head + (hit ? " (되돌림 확인 · 다시 집음)" : " (되돌림 확인 실패 · 다시 찾음)"));
          next(true);
        });
    });
}

void MissionManager::solve_allocation() {
  std::string why = order_.size() == 3 ? "" : "no_order";  // 바코드·신호 없음 (diag에서 관측 실패) — 조사만 하고 BUILD 건너뜀
  std::string list;
  for (const auto & p : pucks_) {
    const int slot = basket_slot_of(p.tag);
    list += std::string(list.empty() ? "" : ", ") + "{\"tag\": " + str(p.tag) + ", \"color\": " + q(p.color) +
            ", \"shape\": " + q(p.shape) + ", \"box\": " + str(static_cast<int>(puck_at_[p.tag].first) + 1) +
            ", \"tray_slot\": " + (slot < 0 ? "null" : str(slot)) + "}";
  }
  bool ok = why.empty() && allocate(pucks_, counts_, alloc_, why);
  // 현장 진단만: 퍽이 12개가 아니라 못 나누면 가진 퍽으로 최대한 (다 채운 장소 먼저) — 빈 눈은 기록하고 BUILD는 있는 퍽만
  //   (대회 규칙대로면 할당 실패 = BUILD 없이 RETURN · 평소 실행은 그대로)
  std::map<char, int> filled;
  const bool partial = !ok && diag_run_ && (why == "puck_count" || why == "no_allocation") &&
                       allocate_partial(pucks_, counts_, alloc_, filled);
  if (partial) ok = true;
  std::vector<std::string> empty;
  std::string groups;
  // 적재가 다 끝난 지금 자리에서 가까운 장소부터 (11-22 사용자: 고정 순서 L3 → L2 → L1이면 S1 근처에서 끝나도 L3까지 갔다 옴)
  //   — 그다음도 직전 장소에서 가까운 곳 (욕심 순서). 놓기 시험(start:place)은 정한 장소부터라 그대로
  // ponytail: 직선 거리 — 벽 너머가 가까워 보일 수 있음. 동선이 꼬이면 Nav2 경로 길이(ComputePathToPose)로
  Pose2 rb;
  if (ok && build_nearest_ && !use_given_order_ && robot_arena(rb)) {
    std::vector<size_t> left = build_loc_order_, by;
    double x = rb.x, y = rb.y;
    while (!left.empty()) {
      const auto it = std::min_element(left.begin(), left.end(), [&](size_t a, size_t b) {
        return std::hypot(locations_[a].first - x, locations_[a].second - y) < std::hypot(locations_[b].first - x, locations_[b].second - y);
      });
      by.push_back(*it);
      x = locations_[*it].first; y = locations_[*it].second;
      left.erase(it);
    }
    build_loc_order_ = by;
    std::vector<std::string> names;
    for (const size_t li : by) names.push_back("L" + str(static_cast<int>(li) + 1));
    note("놓는 순서 (지금 자리에서 가까운 장소부터): " + join(names));
  }
  if (ok) {
    for (const size_t li : build_loc_order_) {  // 장소 방문 순서 (build_location_order) — 장소 번호 li는 그대로
      const char g = order_[li];
      const auto cells = dice_cells(group_count(counts_, g));
      std::vector<int> tags;
      for (const auto & [tag, grp] : alloc_) if (grp == g) tags.push_back(tag);
      std::string t;
      for (size_t k = 0; k < cells.size() && k < tags.size(); ++k) {
        jobs_.push_back({tags[k], li, cells[k]});
        t += std::string(t.empty() ? "" : ", ") + "{\"tag\": " + str(tags[k]) + ", \"cell\": " + str(cells[k]) + "}";
      }
      std::vector<std::string> gap;
      for (size_t k = tags.size(); k < cells.size(); ++k) gap.push_back(str(cells[k]));
      if (!gap.empty())
        empty.push_back("L" + str(static_cast<int>(li) + 1) + " " + g + " " + str(static_cast<int>(std::min(tags.size(), cells.size()))) +
                        "/" + str(static_cast<int>(cells.size())) + " (빈 칸 " + join(gap) + ")");
      groups += std::string(groups.empty() ? "" : ", ") + "\"L" + str(static_cast<int>(li) + 1) + "\": {\"group\": " +
                q(std::string(1, g)) + ", \"pucks\": [" + t + "], \"empty_cells\": [" + join(gap) + "]}";
    }
  }
  std::vector<std::string> eq;
  for (const auto & e : empty) eq.push_back(q(e));
  write_log("allocation.json", "{\"pucks\": [" + list + "], \"ok\": " + (ok ? "true" : "false") + ", \"why\": " + q(why) +
            ", \"partial\": " + (partial ? "true" : "false") + ", \"empty\": [" + join(eq) + "], \"locations\": {" + groups + "}}\n");
  if (partial) {
    result("diag: 부분 할당 (" + why + ":" + str(static_cast<int>(pucks_.size())) + ") — 퍽 " + str(static_cast<int>(jobs_.size())) +
           "개로 BUILD" + (empty.empty() ? "" : " · 빈 눈 " + join(empty)));
    RCLCPP_WARN(get_logger(), "부분 할당 — 빈 눈: %s", empty.empty() ? "없음" : join(empty).c_str());
  }
  if (!ok) { finish(false, why + (pucks_.size() != 12 ? ":" + std::to_string(pucks_.size()) : "")); return; }
  finish(true);
}

// ---------- BUILD — 퍽마다 [손에 쥐기 → 놓기 → 확인]. 실패하면 기록하고 다음 퍽
void MissionManager::build(size_t i) {
  if (i == 0 && planner_ == "plan_next" && plan_next_ok_ && build_pn_start()) { build_pn(); return; }
  // 그 자리 재검증 (11-15): 앞 퍽과 장소가 다르거나 다 놓았으면 — 떠나기 전에 지금 선 장소를 봄 (다시 돌아오지 않음)
  if (verify_in_place_ && i > 0 && i <= jobs_.size()) {
    const size_t prev = jobs_[i - 1].loc;
    if ((i == jobs_.size() || jobs_[i].loc != prev) && !verified_.count(prev) &&
        (last_place_loc_ == static_cast<int>(prev) || !back_.empty())) {  // 마지막 퍽을 못 놓았어도 그 장소에 붙어 있으면 그 자리에서 봄
      verify_at(prev, [this, i]() { build(i); });
      return;
    }
  }
  if (i >= jobs_.size()) {
    write_log("build.json", "{\"placed\": " + str(placed_) + ", \"total\": " + str(static_cast<int>(jobs_.size())) +
              ", \"pucks\": [" + build_json_ + "]}\n");
    finish(true);
    return;
  }
  // 장소 상자에 붙어 있으면 그 장소 남은 칸 중 지금 자리에서 팔이 닿는 칸 먼저 (11-22 사용자: 자꾸 exit) — 닿는 칸 중엔 먼 칸부터
  //   (놓인 퍽 위로 손이 안 지나게). 한 변에선 가까운 두 줄만 닿아 R4 · G6은 변이 둘 필요 — 변 옮김을 장소마다 한 번으로
  if (!back_.empty() && !stop_.empty() && stop_[0] == 'L') {
    const size_t li = static_cast<size_t>(std::stoi(stop_.substr(1)) - 1);
    Pose2 r;
    if (jobs_[i].loc == li && robot_arena(r)) {
      size_t best = i;
      double best_d = -1.0;
      for (size_t j = i; j < jobs_.size() && jobs_[j].loc == li; ++j) {
        const auto c = cell_target(li, jobs_[j].cell);
        const double d = std::hypot(c.first - r.x, c.second - r.y);
        if (d <= place_reach_m_ && d > best_d) { best = j; best_d = d; }
      }
      if (best != i) {
        RCLCPP_INFO(get_logger(), "L%zu 칸 %d를 먼저 (지금 자리에서 %.3f m — 칸 %d는 나중에)", li + 1, jobs_[best].cell, best_d, jobs_[i].cell);
        std::swap(jobs_[i], jobs_[best]);
      }
    }
  }
  const Job job = jobs_[i];
  note("퍽 " + str(job.tag) + " → L" + str(static_cast<int>(job.loc) + 1) + " 칸 " + str(job.cell) +
       " (" + str(static_cast<int>(i) + 1) + "/" + str(static_cast<int>(jobs_.size())) + ")");
  bring(i);
}

void MissionManager::bring(size_t i) {
  const Job job = jobs_[i];
  if (place_avoid_ && (place_avoid_->loc != static_cast<size_t>(job.loc) || place_avoid_->cell != job.cell)) place_avoid_.reset();
  last_place_loc_ = -1;  // 장소 앞에서 놓기 시작할 때 다시 정함
  ctx_ = "\"tag\": " + str(job.tag) + ", \"loc\": \"L" + str(static_cast<int>(job.loc) + 1) + "\", \"cell\": " + str(job.cell);
  const auto skip = [this, i, job](const std::string & why) {
    result("puck " + str(job.tag) + ": " + why);
    build_json_ += std::string(build_json_.empty() ? "" : ", ") + "{\"tag\": " + str(job.tag) + ", \"ok\": false, \"why\": " + q(why) + "}";
    job_done(i, false);
  };
  const auto target = cell_target(job.loc, job.cell);
  const double tx = target.first, ty = target.second;
  const int slot = basket_slot_of(job.tag);  // coordinator 장부에서 그 tag가 실린 칸
  // 적재함에 없음 — 시작 박스에서 직접 집어 들고 오지 않음 (11-21 사용자: 들고 다니면 꼬임 · SURVEY에서 다 싣고, 못 실은 퍽은 한 번 더 왔음)
  if (slot < 0) { skip("not_loaded"); return; }
  const auto at_box = [this, i, job, slot, skip](bool ok, const std::string & why) {
    if (!ok) { skip("place_approach_failed " + why); return; }
    correct(job.loc);
    if (!loc_pose_[job.loc]) { skip("place_box_unseen"); return; }  // 상자를 못 봤으면 도면 칸에 놓지 않음 (퍽은 적재함에 그대로)
    // 도킹하자마자(장소 첫 퍽) 남은 칸 중 가장 먼 칸부터 — 놓인 퍽 위로 손이 안 지남 (붙은 뒤 다음 퍽은 build의 같은 정렬)
    {
      Pose2 rr;
      if (robot_arena(rr)) {
        size_t best = i;
        double best_d = -1.0;
        for (size_t j = i; j < jobs_.size() && jobs_[j].loc == job.loc; ++j) {
          const auto cj = cell_target(job.loc, jobs_[j].cell);
          const double d = std::hypot(cj.first - rr.x, cj.second - rr.y);
          if (d <= place_reach_m_ && d > best_d) { best = j; best_d = d; }
        }
        if (best != i) {
          RCLCPP_INFO(get_logger(), "L%d 먼 칸 %d부터 (%.3f m)", static_cast<int>(job.loc) + 1, jobs_[best].cell, best_d);
          std::swap(jobs_[i], jobs_[best]);
          bring(i);  // 붙은 채라 바로 놓기로
          return;
        }
      }
    }
    // 팔이 안 닿는 칸이면 꺼내지 않음 — 꺼낸 뒤 place가 IK로 바로 실패하면 퍽을 쥔 채 다음 칸으로 감 (11-22 실기 nav_1122_025043: L1 칸 2 · 5,
    //   0.282 · 0.263 m — 그 칸 쪽 변은 벽 0.22 m 안이라 빠짐)
    const auto c = cell_target(job.loc, job.cell);
    Pose2 r;
    const bool have_r = robot_arena(r);
    // 이 변에서 안 닿음 → 꺼내지 않고 다른 변으로 한 번 (go_near가 이 변을 맨 뒤로) · 그래도 안 되면 그 칸은 안 놓음 (11-23 사용자)
    const auto other_side = [this, i, job, skip, have_r, r](const std::string & why) {
      if (place_avoid_ || !have_r) { place_avoid_.reset(); skip(why); return; }
      place_avoid_ = PlaceAvoid{static_cast<size_t>(job.loc), job.cell, r.yaw};
      RCLCPP_WARN(get_logger(), "L%d 칸 %d %s — 꺼내지 않고 다른 변으로", static_cast<int>(job.loc) + 1, job.cell, why.c_str());
      bring(i);
    };
    if (have_r && std::hypot(c.first - r.x, c.second - r.y) > place_reach_m_) {
      RCLCPP_WARN(get_logger(), "L%d 칸 %d 팔이 안 닿음 (%.3f m > %.2f)", static_cast<int>(job.loc) + 1, job.cell,
                  std::hypot(c.first - r.x, c.second - r.y), place_reach_m_);
      other_side("place_unreachable");
      return;
    }
    // 거리 안이어도 손 수직 IK는 놓는 높이 0.285 m · 칸 위 2 cm 0.276 m까지 (stand::ik_down) — 꺼내기 전에 skill_place에 물음
    run({SubGoal().set__skill("place").set__target(point(c.first, c.second, box_top_z_)).set__relation("check")},
        [this, i, slot, skip, other_side](bool ok, const std::string & why) {
      if (!ok) {
        if (interrupted(why)) { skip(why); return; }
        other_side("place_unreachable ik");
        return;
      }
      place_avoid_.reset();
      run_retry({SubGoal().set__skill("unload").set__target(str(slot))}, [this, i, skip](bool ok2, const std::string & why2) {
        if (!ok2) { skip("tray_pick_failed " + why2); return; }
        place_check(i, 0);
      });
    });
  };
  // 이미 그 장소 상자에 붙어 있고 다음 칸이 팔 닿는 거리면 물러나지 않고 그 자리에서 (11-22 사용자: 칸마다 exit → 다시 도킹)
  const Box & box = loc_boxes_[job.loc];
  Pose2 r;
  if (!place_avoid_ && !back_.empty() && stop_ == box_label(box) && robot_arena(r) && std::hypot(tx - r.x, ty - r.y) <= place_reach_m_) {
    RCLCPP_INFO(get_logger(), "%s 붙은 채 다음 칸 %d (%.3f m)", stop_.c_str(), job.cell, std::hypot(tx - r.x, ty - r.y));
    at_box(true, "");
    return;
  }
  // 붙은 상자에서 다른 변으로 돌려면 상자 가운데를 알아야 함 — 모르면(도면 기준) Nav2가 실제 상자를 지나는 길로 몰아 박음
  //   (11-22 실기 nav_1122_033953 L3: 도면에서 18.5 cm인 상자를 도면 기준으로 돌다 상자를 밀며 계속 감) → 이 칸은 안 놓음
  if (!back_.empty() && stop_ == box_label(box) && !loc_fix_[job.loc]) {
    RCLCPP_WARN(get_logger(), "%s 칸 %d — 이 변에선 안 닿고 상자 가운데를 몰라 다른 변으로 안 돎", stop_.c_str(), job.cell);
    skip("place_unreachable");
    return;
  }
  // go_near는 점을 도면 기준으로 받아 인지 상자 가운데(near_c_)만큼 옮김 — 인지 상자 위 칸(tx, ty)을 주면 두 번 옮겨짐
  //   무늬를 돌렸으면(#116 loc_rot_) 도면 칸 점도 같이 돌림 — 안 돌리면 닿는 변을 엉뚱하게 고름
  const auto [hx, hy] = box_cell({locations_[job.loc].first, locations_[job.loc].second,
                                  loc_rot_[job.loc] ? *loc_rot_[job.loc] : 0.0}, job.cell, grid_step_);
  go_near(box, hx, hy, at_box);
}

void MissionManager::job_done(size_t i, bool ok) {
  if (pn_action_ < 0) { build(i + 1); return; }  // 규칙 순서
  // plan_next: 이 퍽 결과로 상태를 고치고 다시 고름 — 퍽 하나 처리(bring)에 다시 시도가 들어 있어 실패면 포기(LOST)로 봄
  const int p = PlanNext::puck_of(pn_action_);
  pn_.status[p] = ok ? PlanNext::OK : PlanNext::LOST;
  pn_.tries[p] += 1;
  pn_.at = 6 + static_cast<int>(jobs_[i].loc);
  plan_next_.record(pn_action_, true);
  build_pn();
}

bool MissionManager::build_pn_start() {
  // 퍽 i = 시작 박스 i/2 (가상 미션과 같은 번호) — 박스마다 x 오프셋 순. 12개·박스마다 2개가 아니면 규칙 순서로
  std::vector<size_t> order(jobs_.size());
  for (size_t j = 0; j < order.size(); ++j) order[j] = j;
  std::sort(order.begin(), order.end(), [this](size_t a, size_t b) {
    return puck_at_[jobs_[a].tag] < puck_at_[jobs_[b].tag];
  });
  if (order.size() != static_cast<size_t>(PlanNext::N_PUCK)) return false;
  for (size_t k = 0; k < order.size(); ++k)
    if (puck_at_[jobs_[order[k]].tag].first != k / 2) { RCLCPP_WARN(get_logger(), "plan_next: 박스마다 퍽 2개가 아님 — 규칙 순서로"); return false; }
  pn_ = PlanNext::State{};
  pn_job_ = order;
  for (int p = 0; p < PlanNext::N_PUCK; ++p) {
    const Job & job = jobs_[order[p]];
    pn_.loc[p] = static_cast<int>(job.loc);
    pn_.cell[p] = job.cell;
    pn_.status[p] = basket_slot_of(job.tag) >= 0 ? PlanNext::BASKET : lying_.count(job.tag) ? PlanNext::BOX_LYING : PlanNext::BOX_UP;
  }
  plan_next_.reset();
  build_started_ = now();
  return true;
}

void MissionManager::build_pn() {
  pn_.t = (now() - build_started_).seconds();
  pn_action_ = plan_next_.next(pn_);
  if (pn_action_ == PlanNext::RETURN) {
    write_log("build.json", "{\"placed\": " + str(placed_) + ", \"total\": " + str(static_cast<int>(jobs_.size())) +
              ", \"planner\": \"plan_next\", \"pucks\": [" + build_json_ + "]}\n");
    pn_action_ = -1;
    finish(true);
    return;
  }
  const int p = PlanNext::puck_of(pn_action_);
  const size_t j = pn_job_[p];
  static const char * skills[] = {"pick", "stand", "load", "unload", "place", "nudge", "re-pick", "stand_placed"};
  note("plan_next: " + std::string(skills[PlanNext::skill_of(pn_action_)]) + " 퍽 " + str(jobs_[j].tag) + " → L" +
       str(static_cast<int>(jobs_[j].loc) + 1) + " 칸 " + str(jobs_[j].cell));
  bring(j);  // 손에 쥐기(적재함이면 꺼내기, 누웠으면 세우기) → 놓기 → 확인 — 끝나면 job_done
}

void MissionManager::place_check(size_t i, int tries) {
  const Job job = jobs_[i];
  const auto target = cell_target(job.loc, job.cell);
  const double tx = target.first, ty = target.second;
  const auto next = [this, i, job](bool ok, const std::string & why) {
    if (ok) ++placed_;
    else result("puck " + str(job.tag) + ": " + why);
    build_json_ += std::string(build_json_.empty() ? "" : ", ") + "{\"tag\": " + str(job.tag) + ", \"location\": \"L" +
                   str(static_cast<int>(job.loc) + 1) + "\", \"cell\": " + str(job.cell) + ", \"ok\": " + (ok ? "true" : "false") +
                   ", \"why\": " + q(why) + "}";
    publish_state();
    job_done(i, ok);
  };
  attempt_ = tries;
  last_place_loc_ = static_cast<int>(job.loc);  // 장소 앞에 섬 (go_near 성공 뒤에만 옴)
  (void)tries;
  // 11-22 사용자: 놓은 뒤 확인·다시 놓기 없음 (1 cm 기준으로 다시 집어도 오차가 안 줄고 퍽마다 약 22 s · 장소마다 마지막에 보기만 — verify_at)
  run({SubGoal().set__skill("place").set__target(point(tx, ty, box_top_z_))}, [this, job, next](bool ok, const std::string & why) {
    if (ok) { next(true, "placed"); return; }
    // 손에 쥔 채 실패(IK · 궤적 충돌 검사 — 팔이 움직이기 전)면 적재함 빈 칸에 되돌린 뒤 다음 퍽 (11-22 실기 06:51: 쥔 채로 다음 퍽 unload)
    if (!basket_ || !basket_->held.filled) { next(false, "place_failed " + why); return; }
    RCLCPP_WARN(get_logger(), "퍽 %d 놓기 실패 (%s) — 손에 쥔 퍽을 적재함에 되돌림", job.tag, why.c_str());
    run({SubGoal().set__skill("load").set__target("next")}, [next, why](bool ok2, const std::string & why2) {
      next(false, "place_failed " + why + (ok2 ? " · 적재함에 되돌림" : " · 되돌리기 실패 " + why2));
    });
  });
}

// ---------- VERIFY — 11-22부터 아무것도 안 함 (장소마다 보기는 BUILD에서)
void MissionManager::verify(size_t) {
  // 11-22 사용자: 재검증 단계에서 장소로 다시 가지 않음 — 보는 동작은 BUILD에서 장소마다 다 놓은 뒤 그 자리에서만 (verify_at)
  //   (11-22 실기 05:08: L2 마지막 퍽을 못 놓아 그 자리 보기가 빠지자 VERIFY가 L2에 다시 도킹 · L1로 가다 nav_no_progress)
  finish(true);
}

// 지금 선 자리(장소 li 앞)에서 — 물러나서 한 번 봄
void MissionManager::verify_at(size_t li, std::function<void()> then) {
  // 11-22 사용자: 재검증 판정은 안 함 — 장소에 다 놓으면 똑바로 물러난 뒤 look 자세로 한 번 보고(사진) 접음
  verified_.insert(li);
  const std::string name = "L" + str(static_cast<int>(li) + 1);
  ctx_ = "\"loc\": " + q(name);
  note(name + " 다 놓음 — 물러나서 한 번 봄");
  std::vector<SubGoal> back{stow()};
  if (!back_.empty()) back.push_back(SubGoal().set__skill("approach").set__target(back_).set__relation("exit"));
  back_.clear();
  run(back, [this, name, then](bool ok, const std::string & why) {
    if (!ok) { RCLCPP_WARN(get_logger(), "%s 물러나기 실패 (%s) — 보기 건너뜀", name.c_str(), why.c_str()); then(); return; }
    run({SubGoal().set__skill("pose").set__target("look")}, [this, name, then](bool, const std::string &) {
      snap("look_" + name);
      beep(3000, 70);  // 봤다는 표시 — 짧게 두 번
      beep(3000, 70);
      run({stow()}, [then](bool, const std::string &) { then(); });
    });
  });
}

// ---------- RETURN — 출발 구역 가운데 → 외곽 직사각형이 구역 안인지 확인, 밖이면 한 번 보정 → 정지
void MissionManager::go_home(bool corrected) {
  ctx_.clear();
  if (!back_.empty()) {  // 박스에 붙어 있음 — 먼저 똑바로 물러남
    const std::string back = back_;
    back_.clear();
    run({stow(), SubGoal().set__skill("approach").set__target(back).set__relation("exit")}, [this, corrected](bool ok, const std::string & why) {
      ok ? go_home(corrected) : finish(false, "retreat:" + why);
    });
    return;
  }
  note(corrected ? "출발 구역 가운데로 · 방향 맞춤" : "출발 구역으로");
  const double zx = (start_zone_[0] + start_zone_[2]) / 2, zy = (start_zone_[1] + start_zone_[3]) / 2;
  const auto p = to_map(zx, zy, M_PI);
  const auto arrive = [this, corrected](bool ok, const std::string & why) {
    // 11-18: 실패 한 번으로 미션을 끝내지 않음 — navigate의 끼임 풀기(후진·회전·전진) 뒤에도 안 되면 처음부터 다시 (사용자: 바로 죽으면 안 됨)
    if (!ok && !corrected && ++home_tries_ < 3) {
      RCLCPP_WARN(get_logger(), "출발 구역 복귀 실패 (%s) — 다시 %d/3", why.c_str(), home_tries_ + 1);
      go_home(false);
      return;
    }
    if (!ok && !corrected) { finish(false, "return_nav:" + why); return; }
    if (!corrected) { go_home(true); return; }  // Nav2 도착 → 가운데로 곧게 · 제자리에서 방향 맞춤
    if (!ok) RCLCPP_WARN(get_logger(), "출발 구역 가운데·방향 맞춤 실패 (%s) — 그 자리에서 판정", why.c_str());
    Pose2 r;
    const bool known = robot_arena(r);
    const bool in = known && footprint_in_zone(r, footprint_[0], footprint_[1], footprint_[2], start_zone_);
    note(in ? "출발 구역에서 정지" : "출발 구역 밖 — 그대로 정지");
    write_log("return.json", "{\"in_zone\": " + std::string(in ? "true" : "false") + ", \"corrected\": true" +
              fmt(", \"x_mm\": %.0f, \"y_mm\": %.0f", r.x * 1000, r.y * 1000) + fmt(", \"yaw\": %.3f}\n", r.yaw));
    if (!in) result("return: footprint_out_of_zone");
    snap("return");
    timer_ = create_wall_timer(std::chrono::duration<double>(home_hold_s_), [this]() {
      timer_->cancel();
      finish(true);
    });
  };
  if (!corrected) {  // Nav2로 출발 구역 가운데 근처까지
    run({stow(), SubGoal().set__skill("navigate").set__target(fmt("%.4f,%.4f,%.4f", p.x, p.y, p.yaw)).set__relation(leg_to("start"))}, arrive);
    return;
  }
  // 11-21 실기: park_home(가운데 + 방향 0°를 한 번에 5 cm·10°)은 옆으로 몇 cm 어긋나면 좁은 구역에서 좌우로 비틀기만 하다
  //   approach_blocked (18:58 옆 7.8 cm · 16:16 · 13:45) → ① 가운데를 향해 곧게(뒤면 후진) ② 그 자리에서 제자리 회전으로 방향만
  Pose2 r;
  if (!robot_map(r)) { arrive(false, "no_tf"); return; }
  const double dx = p.x - r.x, dy = p.y - r.y;
  const auto turn = [this, p, arrive](bool, const std::string &) {
    Pose2 q;
    if (!robot_map(q)) { arrive(false, "no_tf"); return; }
    run({SubGoal().set__skill("approach").set__target(fmt("%.4f,%.4f,%.4f", q.x, q.y, p.yaw)).set__relation("park_home")}, arrive);
  };
  if (std::hypot(dx, dy) < 0.02) { turn(true, ""); return; }
  double line = std::atan2(dy, dx);
  if (std::abs(std::remainder(line - r.yaw, 2 * M_PI)) > M_PI / 2) line += M_PI;  // 가운데가 뒤쪽 — 그 방향을 등지고 후진
  run({stow(), SubGoal().set__skill("approach").set__target(fmt("%.4f,%.4f,%.4f", p.x, p.y, std::remainder(line, 2 * M_PI)))
                 .set__relation("park_home")}, turn);
}

void MissionManager::stop(const std::string & why) {
  const bool already_error = phase_ == Phase::ERROR;
  if (active_) {  // 시간 기록 — 진행 중 단계·단계·미션을 실패로 닫음 (finish → ERROR로 왔으면 단계 줄은 이미 씀)
    close_steps(false, why, step_log_.cur);
    if (phase_ != Phase::ERROR) phase_line(false, why, Phase::ERROR);
    mission_line(false, why);
  }
  active_ = false;
  const auto goal = waiting_goal_;
  waiting_goal_.reset();
  waiting_plan_ = 0;
  waiting_done_ = nullptr;
  phase_ = Phase::ERROR;
  if (!already_error) beep_phase(phase_);
  error_ = why;
  if (timer_) timer_->cancel();
  // 관측 중 팔을 뻗은 채 멈추면 다음 시작·주행 때 부딪힘 → 접기(stow)만 새 goal로 보냄 (결과는 안 기다림)
  hires(false);
  const bool fold = (gaze_sent_ || peek_sent_) && !interrupted(why.rfind("interrupted: ", 0) == 0 ? why.substr(13) : why);
  if (goal) execute_client_->async_cancel_goal(goal);
  else if (execute_client_ && !fold) execute_client_->async_cancel_all_goals();  // fold면 새 goal이 진행 중 계획을 preempt
  if (fold) {
    ExecutePlan::Goal stow_goal;
    stow_goal.plan_id = PLAN_ID_BASE + ++plan_seq_;
    stow_goal.goals.push_back(stow());
    stow_goal.goals.back().step = 1;
    execute_client_->async_send_goal(stow_goal);
    gaze_sent_ = peek_sent_ = false;
    RCLCPP_WARN(get_logger(), "정지 — 뻗은 팔 접기(stow)만 보냄");
  }
  write_log("error.json", "{\"why\": " + q(why) + ", \"results\": [" + [&] {
    std::vector<std::string> v;
    for (const auto & r : results_) v.push_back(q(r));
    return join(v);
  }() + "]}\n");
  RCLCPP_ERROR(get_logger(), "미션 오류: %s — 안전 정지", why.c_str());
  note(why);
}

void MissionManager::note(const std::string & text) {
  note_ = text;
  RCLCPP_INFO(get_logger(), "[%s] %s", phase_name(phase_), text.c_str());
  publish_state();
}

void MissionManager::beep_phase(Phase phase) {
  if (!phase_beeps_ || !beep_pub_) return;
  int hz = 0, ms = 120;
  switch (phase) {
    case Phase::GO_OBSERVE: hz = 1100; break;
    case Phase::BARCODE: hz = 1400; break;
    case Phase::READ: hz = 1700; break;
    case Phase::SURVEY: hz = 2000; break;
    case Phase::BUILD: hz = 2300; break;
    case Phase::VERIFY: hz = 2600; break;
    case Phase::RETURN: hz = 2900; break;
    case Phase::DONE: hz = 4200; ms = 320; break;
    case Phase::ERROR: hz = 650; ms = 600; break;
    default: return;  // IDLE에는 소리 내지 않음
  }
  if (phase == Phase::ERROR) {
    beep_queue_.clear();  // 이전 단계 큐보다 오류음을 먼저 알림
    if (beep_timer_active_ && beep_timer_) beep_timer_->cancel();
    beep_timer_active_ = false;
  }
  beep(hz, ms);
}

void MissionManager::beep(int hz, int ms) {
  if (!phase_beeps_ || !beep_pub_) return;
  beep_queue_.emplace_back(hz, ms);
  if (beep_timer_active_) return;
  beep_timer_active_ = true;
  beep_timer_ = create_wall_timer(std::chrono::milliseconds(180), [this]() {
    if (beep_queue_.empty()) {
      beep_timer_->cancel();
      beep_timer_active_ = false;
      return;
    }
    const auto [next_hz, next_ms] = beep_queue_.front();
    beep_queue_.pop_front();
    std_msgs::msg::Int32MultiArray tone;
    tone.data = {next_hz, next_ms};
    beep_pub_->publish(tone);
    if (beep_queue_.empty()) {
      beep_timer_->cancel();
      beep_timer_active_ = false;
    }
  });
}

void MissionManager::result(const std::string & text) {
  results_.push_back(text);
  RCLCPP_WARN(get_logger(), "결과: %s", text.c_str());
  publish_state();
}

void MissionManager::publish_state() {
  MissionState m;
  m.header.stamp = now();
  m.phase = phase_name(phase_);
  m.note = note_;
  m.barcode = barcode_;
  m.count_r = static_cast<uint8_t>(counts_.r);
  m.count_y = static_cast<uint8_t>(counts_.y);
  m.count_g = static_cast<uint8_t>(counts_.g);
  m.order = order_;
  if (basket_) for (const auto & s : basket_->slots)  // coordinator 장부 그대로 — "칸:tag" (tag 모름 ?, 확인 필요 !)
    if (s.filled) m.tray.push_back(str(s.slot) + ":" + (s.tag < 0 ? std::string("?") : str(s.tag)) + (s.check ? "!" : ""));
  m.placed = static_cast<uint16_t>(placed_);
  m.total = static_cast<uint16_t>(jobs_.size());
  m.results = results_;
  m.error = error_;
  state_pub_->publish(m);
}

// ---------- skill 요청
void MissionManager::run(std::vector<SubGoal> goals, Done done) {
  if (!active_) return;  // abort·오류 뒤에 남은 콜백이 새 계획을 보내지 않게
  ExecutePlan::Goal goal;
  goal.plan_id = PLAN_ID_BASE + ++plan_seq_;
  for (size_t i = 0; i < goals.size(); ++i) goals[i].step = static_cast<uint32_t>(i + 1);
  const uint32_t plan_id = goal.plan_id;
  close_steps(false, "superseded", step_log_.cur);  // 앞 계획이 아직 안 끝났는데 새 계획 (coordinator가 앞 것을 취소함)
  step_log_ = {plan_id, goals, 0, wall_s(), attempt_, phase_name(phase_), ctx_};
  attempt_ = 0;
  goal.goals = std::move(goals);
  waiting_plan_ = plan_id;
  waiting_done_ = std::move(done);
  rclcpp_action::Client<ExecutePlan>::SendGoalOptions options;
  options.goal_response_callback = [this, plan_id](const ExecuteGoalHandle::SharedPtr & handle) {
    if (waiting_plan_ != plan_id) return;
    if (!handle) {
      close_steps(false, "coordinator_unavailable", step_log_.cur);
      waiting_plan_ = 0;
      auto done = std::move(waiting_done_);
      done(false, "coordinator_unavailable");
      return;
    }
    waiting_goal_ = handle;
  };
  // 단계 경계 — coordinator가 단계를 시작할 때마다 feedback(current_step)을 냄 → 앞 단계들은 성공으로 닫음
  options.feedback_callback = [this, plan_id](ExecuteGoalHandle::SharedPtr, const std::shared_ptr<const ExecutePlan::Feedback> f) {
    auto & s = step_log_;
    if (s.plan != plan_id || f->current_step == 0 || f->current_step - 1 <= s.cur || f->current_step > s.goals.size()) return;
    const size_t idx = f->current_step - 1;
    const double t = wall_s();
    for (size_t k = s.cur; k < idx; ++k) step_line(k, k == s.cur ? s.t0 : t, t, true, "", k == s.cur && idx == s.cur + 1);
    s.cur = idx;
    s.t0 = t;
  };
  options.result_callback = [this, plan_id](const ExecuteGoalHandle::WrappedResult & result) {
    if (waiting_plan_ != plan_id || !active_) return;
    waiting_plan_ = 0;
    waiting_goal_.reset();
    auto done = std::move(waiting_done_);
    const bool ok = result.code == rclcpp_action::ResultCode::SUCCEEDED && result.result && result.result->success;
    const std::string why = result.result && !result.result->reason.empty() ? result.result->reason :
      (result.code == rclcpp_action::ResultCode::CANCELED ? "cancelled" : "coordinator_failed");
    const uint32_t fs = result.result ? result.result->final_step : 0;
    const std::string detail = result.result ? result.result->detail : "";
    if (!ok) RCLCPP_WARN(get_logger(), "계획 %u 실패 %s%s", plan_id, why.c_str(), detail.empty() ? "" : (" — " + detail).c_str());
    close_steps(ok, ok ? "" : why, ok ? step_log_.goals.size() - 1 : fs ? fs - 1 : step_log_.cur, detail);
    if (!ok && interrupted(why)) { stop("interrupted: " + why); return; }
    if (diag_run_ && diag_fatal(ok, why, stuck_, stuck_limit_)) { stop("fatal: " + why); return; }
    done(ok, why);
  };
  execute_client_->async_send_goal(goal, options);
}

void MissionManager::run_retry(std::vector<SubGoal> goals, Done done, int tries) {
  attempt_ = tries;
  run(goals, [this, goals, done, tries](bool ok, const std::string & why) {
    // load_released: 이미 칸에 놓고 손이 빔 — 다시 하면 빈손으로 다음 칸에 넣음
    // arm_jam (load): 칸 위에서 바로 다시 하면 carry로 되돌아가며 쥔 퍽이 옆 칸 퍽에 부딪혀 돌아감 (11-15) — 다시 하지 않고
    //   위로 올려 보내 상자 제자리에 되돌려 놓게 함
    if (ok || tries >= retries_ || why.rfind("load_released", 0) == 0 || (why == "arm_jam" && goals.front().skill == "load")) {
      done(ok, why);
      return;
    }
    RCLCPP_WARN(get_logger(), "%s %s 실패 (%s) — 다시 (%d/%d)", goals.front().skill.c_str(), goals.front().target.c_str(),
                why.c_str(), tries + 1, retries_);
    run_retry(goals, done, tries + 1);
  });
}

std::string MissionManager::box_label(const Box & box) const {
  for (size_t i = 0; i < start_boxes_.size(); ++i)
    if (std::abs(start_boxes_[i].x - box.x) < 1e-6 && std::abs(start_boxes_[i].y - box.y) < 1e-6) return "S" + str(static_cast<int>(i) + 1);
  for (size_t i = 0; i < loc_boxes_.size(); ++i)
    if (std::abs(loc_boxes_[i].x - box.x) < 1e-6 && std::abs(loc_boxes_[i].y - box.y) < 1e-6) return "L" + str(static_cast<int>(i) + 1);
  return "";
}

// 다가설 자리를 잡을 박스 가운데 (arena) — 이번 회차에 min_hits번 넘게 본 인지 상자(marks_, map 지수 평균)가 도면 가운데에서
//   arena_corr_gate 안이면 그 자리 (to_arena로 보정을 빼서 arena로 — to_map이 다시 얹어 Nav2 목표는 인지 자리 그대로).
//   못 봤거나 gate 밖 · approach_perceived false · 목록에 없는 박스면 도면
std::pair<double, double> MissionManager::near_center(const Box & box) {
  const size_t ns = start_boxes_.size();
  const auto same = [&box](const Box & b) { return std::abs(b.x - box.x) < 1e-6 && std::abs(b.y - box.y) < 1e-6; };
  int i = -1;  // marks_ 번호 (시작 박스 0..ns-1 · 장소 ns..) — box_label과 같은 찾기
  for (size_t j = 0; j < start_boxes_.size() && i < 0; ++j) if (same(start_boxes_[j])) i = static_cast<int>(j);
  for (size_t j = 0; j < loc_boxes_.size() && i < 0; ++j) if (same(loc_boxes_[j])) i = static_cast<int>(ns + j);
  if (!near_perceived_ || i < 0 || static_cast<size_t>(i) >= marks_.size()) return {box.x, box.y};
  const auto & mk = marks_[static_cast<size_t>(i)];
  if (!mk.hits) return {box.x, box.y};
  const auto q = to_arena(mk.qx, mk.qy, 0.0);
  bool used = false;
  const auto c = perceived_center(box.x, box.y, q.x, q.y, mk.hits, acorr_min_hits_, acorr_gate_, used);
  if (used)
    RCLCPP_INFO(get_logger(), "%s 다가설 자리 — 인지 상자 가운데 (도면에서 %+.1f, %+.1f cm · %.1f cm · 본 횟수 %d)", box_label(box).c_str(),
                (c.first - box.x) * 100, (c.second - box.y) * 100, std::hypot(c.first - box.x, c.second - box.y) * 100, mk.hits);
  else
    RCLCPP_INFO(get_logger(), "%s 다가설 자리 — 도면 가운데 (인지 %d번 · 도면에서 %.1f cm — min_hits %d · gate %.0f cm)", box_label(box).c_str(),
                mk.hits, std::hypot(q.x - box.x, q.y - box.y) * 100, acorr_min_hits_, acorr_gate_ * 100);
  return c;
}

void MissionManager::go_near(const Box & box, double px, double py, Done done, size_t k, std::vector<Pose2> poses, bool retried) {
  if (k == 0 && poses.empty() && !back_.empty()) {  // 지금 박스에 붙어 있으면 먼저 standoff까지 똑바로 물러남 (Nav2는 원 0.18이라 붙은 자리에선 출발 못 함)
    const std::string back = back_;  // 후진 거리 [m] — approach exit (odom 기준 똑바로, 11-13: 예전엔 map 자리로 가서 map이 밀리면 비틀림)
    back_.clear();
    run({stow(), SubGoal().set__skill("approach").set__target(back).set__relation("exit")}, [this, box, px, py, done](bool ok, const std::string & why) {
      if (!ok) { done(false, "retreat:" + why); return; }
      go_near(box, px, py, done);
    });
    return;
  }
  if (poses.empty()) {
    near_c_ = near_center(box);  // 인지 평균 또는 도면 — 점(px, py)도 같이 옮김 (장소 칸 등 박스 기준 점)
    {  // 장소 상자는 고정한 가운데(loc_fix_)가 있으면 그것 — 다가설 자리 · Nav2 목표가 실제 상자 기준
      const std::string lb = box_label(box);
      const size_t li = lb[0] == 'L' ? static_cast<size_t>(std::stoi(lb.substr(1)) - 1) : loc_fix_.size();
      if (li < loc_fix_.size() && loc_fix_[li]) {
        double mx, my;
        spot_map(*tf_buffer_, *loc_fix_[li], mx, my, 1e9);
        const auto a = to_arena(mx, my, 0.0);
        near_c_ = {a.x, a.y};
        RCLCPP_INFO(get_logger(), "%s 다가설 자리 — 고정한 상자 가운데 (도면에서 %+.1f, %+.1f cm)", lb.c_str(), (a.x - box.x) * 100, (a.y - box.y) * 100);
      }
    }
    px += near_c_.first - box.x;
    py += near_c_.second - box.y;
    poses = approach_poses(near_c_.first, near_c_.second, box.half, px, py, standoff_);
    Pose2 r;
    if (robot_arena(r)) {  // 점에서 가까운 변 먼저, 같으면(박스 가운데 등) 로봇에서 가까운 변
      std::stable_sort(poses.begin(), poses.end(), [&](const Pose2 & a, const Pose2 & b) {
        const long da = std::lround(std::hypot(a.x - px, a.y - py) * 100), db = std::lround(std::hypot(b.x - px, b.y - py) * 100);
        return da != db ? da < db : std::hypot(a.x - r.x, a.y - r.y) < std::hypot(b.x - r.x, b.y - r.y);
      });
    }
    if (!box.sides.empty()) {  // 지정한 변만 — 다가설 자리 yaw로 변을 앎 (서쪽 변에 서면 +x를 봄 → "-x")
      //   "+x@0.41": 그 변은 박스 가운데에서 0.41 m (변 법선 방향)에 섬 — approach를 시작할 자리 (11-13 사용자 지정, 지도 그림의 빨간 점)
      //   시도 순서 = 적은 순서 (11-14: 앞에 적은 면이 진입 면, 뒤는 실패할 때 대신 — tools/plan 경로 계산과 맞춤)
      std::vector<Pose2> kept;
      for (const auto & t : box.sides) {
        for (auto p : poses) {
          const double c = std::cos(p.yaw), s = std::sin(p.yaw);
          const std::string side = std::abs(c) > std::abs(s) ? (c > 0 ? "-x" : "+x") : (s > 0 ? "-y" : "+y");
          if (t.compare(0, 2, side) != 0) continue;
          if (t.size() > 3 && t[2] == '@') {  // 법선 거리만 바꿈 (옆 위치는 approach_poses 그대로)
            const double d = std::stod(t.substr(3));
            if (std::abs(c) > std::abs(s)) p.x = near_c_.first - d * (c > 0 ? 1 : -1);
            else p.y = near_c_.second - d * (s > 0 ? 1 : -1);
          }
          kept.push_back(p);
          break;
        }
      }
      poses = kept;
      Pose2 rb;
      // ponytail: 직선 거리 — 벽 너머 면이 더 가까워 보일 수 있음. 그런 일이 보이면 Nav2 경로 길이(ComputePathToPose)로
      if (box.nearest && robot_arena(rb))
        std::stable_sort(poses.begin(), poses.end(), [&](const Pose2 & a, const Pose2 & b) {
          return std::hypot(a.x - rb.x, a.y - rb.y) < std::hypot(b.x - rb.x, b.y - rb.y);
        });
    }
    // 벽에 막힌 변은 처음부터 뺌 (예: 시작 박스 1 서쪽 — x 1.101 벽과 박스 사이 0.27 m. 다가설 자리가 벽 위라 Nav2가 벽으로 몰고 감)
    poses.erase(std::remove_if(poses.begin(), poses.end(), [this](const Pose2 & p) {
      const bool blocked = wall_distance(p.x, p.y, walls_) < wall_clearance_;
      if (blocked) RCLCPP_INFO(get_logger(), "다가설 자리 (%.2f, %.2f) — 벽에서 %.2f m 안이라 뺌", p.x, p.y, wall_clearance_);
      else if (in_keepout(p.x, p.y)) {
        RCLCPP_INFO(get_logger(), "다가설 자리 (%.2f, %.2f) — 금지 구역(keepout_zones) 안이라 뺌", p.x, p.y);
        return true;
      }
      return blocked;
    }), poses.end());
    // 장소 상자: 칸(px, py)에 팔이 닿는 변 먼저 — 붙은 자리(가운데에서 반 변 + gap)에서 칸이 place_reach_m 안인지.
    //   한 변에선 먼 줄(0.28 m)이 안 닿음 → 반대 변에서 (11-22 사용자). 안 닿는 변은 뒤로 (대신 시도 · 붙은 뒤 place_unreachable)
    if (box_label(box)[0] == 'L') {
      const double reach_d = box.half + (box.gap >= 0.0 ? box.gap : approach_gap_);
      const auto dock_d = [&](const Pose2 & p) {
        return std::hypot(near_c_.first - reach_d * std::cos(p.yaw) - px, near_c_.second - reach_d * std::sin(p.yaw) - py);
      };
      std::stable_partition(poses.begin(), poses.end(), [&](const Pose2 & p) { return dock_d(p) <= place_reach_m_; });
      if (place_avoid_ && box_label(box) == "L" + str(static_cast<int>(place_avoid_->loc) + 1))  // 칸 IK가 안 풀린 변은 맨 뒤 (11-23)
        std::stable_partition(poses.begin(), poses.end(), [&](const Pose2 & p) {
          return std::abs(std::remainder(p.yaw - place_avoid_->yaw, 2 * M_PI)) > 0.5;
        });
      if (!poses.empty())
        RCLCPP_INFO(get_logger(), "%s 칸 점 (%.3f, %.3f) — 첫 변에서 %.3f m%s", box_label(box).c_str(), px, py, dock_d(poses[0]),
                    dock_d(poses[0]) <= place_reach_m_ ? "" : " (닿는 변 없음)");
    }
  }
  if (k >= poses.size() || static_cast<int>(k) >= approach_tries_) { done(false, "approach_failed"); return; }
  // Nav2로 변 앞 standoff까지 → approach로 approach_gap까지 직선으로 붙음 (Nav2 원형 footprint로는 못 붙는 거리)
  // Nav2 목표는 박스 가운데에서 nav_turn_m까지 법선으로 물림 — 붙은 자리(0.39 m)에선 몸(외접 0.30 m)이 박스에 걸려 못 돌고
  //   등진 채 멈춤 (11-14). 벽 쪽으로는 min(0.32, 원래 자리 벽 거리)보다 안 다가섬. 나머지는 approach가 직선으로 메움
  const double cx = near_c_.first, cy = near_c_.second;  // 박스 가운데 (다가설 자리를 정할 때 정한 것)
  Pose2 n = poses[k];
  {
    const double c = std::cos(n.yaw), s = std::sin(n.yaw);
    const double clear = std::min(0.32, wall_distance(n.x, n.y, walls_));
    for (double d = (cx - n.x) * c + (cy - n.y) * s + 0.02; d <= nav_turn_m_ + 1e-6; d += 0.02) {
      const double x = cx - d * c, y = cy - d * s;
      if (wall_distance(x, y, walls_) < clear || in_keepout(x, y)) break;
      n.x = x; n.y = y;
    }
  }
  const auto m = to_map(n.x, n.y, n.yaw);
  // 다가설 자리 → 도킹 자리 거리 (변마다 다름 — "@d") · 붙은 뒤 exit도 이만큼 물러나 이 자리로
  const double gap = box.gap >= 0.0 ? box.gap : approach_gap_;
  const double push = (cx - poses[k].x) * std::cos(poses[k].yaw) + (cy - poses[k].y) * std::sin(poses[k].yaw)
                      - box.half - gap;
  const auto f = to_map(poses[k].x + push * std::cos(poses[k].yaw), poses[k].y + push * std::sin(poses[k].yaw), poses[k].yaw);
  const double t_start = wall_s();
  // navigate "look"(도착 전 팔 올리기)은 안 씀 (11-13 headless: 팔이 pole_front_cam을 가려 보던 상자를 잃고, 0.25 m에선 gripper_cam에 잘려 보임)
  const std::string to = box_label(box);
  const auto nav = [this, to](const Pose2 & mm) {
    return SubGoal().set__skill("navigate").set__target(fmt("%.3f,%.3f,%.3f", mm.x, mm.y, mm.yaw)).set__relation("near" + leg_to(to));
  };
  // 상자 도킹: 자리(f)는 힌트 — skill_approach가 인지 상자 + gap으로 목표를 정함 (도면은 안 믿음, 11-12).
  //   11-16: f도 같은 가운데(cx, cy — 인지 평균 또는 도면) 기준이라 Nav2 목표 · 힌트 · 인지가 한 상자를 가리킴
  const auto dock = [gap, half = box.half, stop = box.face_stop](const Pose2 & ff) {
    return SubGoal().set__skill("approach").set__target(fmt("%.4f,%.4f,%.4f", ff.x, ff.y, ff.yaw) + fmt(",%.4f,%.4f", gap, half) +
                                                        (stop > 0.0 ? fmt(",%.4f", stop) : std::string())).set__relation("dock_box");
  };
  const std::function<void(bool, const std::string &)> after =
    [this, box, px, py, done, k, poses, push, n, t_start, retried, cx, cy, to, gap](bool ok, const std::string & why) {
      stop_ = ok ? to : "";  // 붙었으면 그 상자 앞 (exit으로 같은 자리로 물러난 뒤 다음 구간 표가 시작) · 실패면 모름
      // 출발하자마자 navigate 실패(5 s 안)는 경로·자리 탓이 아니라 일시 오류 — 같은 자리로 한 번 더 (11-15 실기 S3: 출발 0.4 s 만에
      //   error 102 'Unable to transform goal pose' → 바로 반대 변(+y)으로 넘어가 상자를 빙 돌아 좁은 틈으로 감)
      if (!ok && why == "nav_failed" && !retried && wall_s() - t_start < 5.0) {
        RCLCPP_WARN(get_logger(), "접근 자리 %zu 바로 실패 (%s, %.1f s) — 같은 자리 다시", k + 1, why.c_str(), wall_s() - t_start);
        go_near(box, px, py, done, k, poses, true);
        return;
      }
      // 붙은 뒤 exit은 Nav2 목표 자리(n)까지 물러남 — approach 시작 자리(@d)는 돌 공간이 없을 수 있음
      //   retreat(더 물러남)은 첫 진입 면 뒤를 재서 정한 값 → 대체 면에선 0 (11-18 sim)
      //   11-18: 0.20 m만 물러나게 해봤으나 SmacPlanner2D(방향 없는 경로)에선 상자를 바라본 채 시작 → 회전 기준(60°) 아래 경로면
      //   RPP가 상자 쪽으로 호를 그리다 막히고 BackUp도 뒤 상자에 막혀 갇힘 (sim M5 S4) → 다가설 자리까지 물러남 유지
      // 물러난 자리에서 제자리 회전할 수 있게 — 로봇 중심 ~ 상자 면이 exit_turn_clear_m 이상 (11-22 실기 L3: −y@0.33 · gap 0.119면
      //   0.109 m만 물러나 면까지 0.228 m < 뒤 모서리 회전 반경 0.288 → 복귀 회전에서 상자에 걸림)
      if (ok) {
        creep_used_ = 0.0; backed_ = 0.0;
        //   start_box_retreat · location_box_retreat는 첫 진입 면에서 그 위에 더 (회전 여유를 채운 뒤에도 더 물러날 상자)
        double b = std::max(std::hypot(cx - n.x, cy - n.y) - box.half - gap, exit_turn_clear_ - gap);
        // 더 물러날 거리는 뒤 벽이 회전 여유(exit_turn_clear_m) 밖인 만큼만 — 면마다 뒤 벽 거리가 달라 (L3 +x면 0.42 m 물러나 동쪽 벽 0.19 m)
        if (k == 0 && box.retreat > 0.0) {
          const double c = std::cos(n.yaw), s = std::sin(n.yaw);
          const double dx = cx - (box.half + gap) * c, dy = cy - (box.half + gap) * s;  // 붙은 자리
          double e = 0.0;
          while (e + 0.02 <= box.retreat + 1e-6 && wall_distance(dx - (b + e + 0.02) * c, dy - (b + e + 0.02) * s, walls_) >= exit_turn_clear_) e += 0.02;
          b += e;
        }
        back_ = fmt("%.3f", b);
        done(true, "");
        return;
      }
      RCLCPP_WARN(get_logger(), "접근 자리 %zu 실패 (%s) — 다음 변", k + 1, why.c_str());
      // 상자 앞에서 끝남(붙다 멈춤 · 상자를 못 봐 다가설 자리에 섬) — 그대로 Nav2를 보내면 제자리에서 돌다 상자를 밀거나(11-13),
      //   후진 금지라 몸 뒤가 벽에 걸려 못 빠져나옴 (11-13 headless) → 먼저 똑바로 물러남
      if (why == "approach_blocked" || why == "approach_box_unseen") {
        run({stow(), SubGoal().set__skill("approach").set__target(fmt("%.3f", push)).set__relation("exit")},
            [this, box, px, py, done, k, poses](bool, const std::string &) { go_near(box, px, py, done, k + 1, poses); });
        return;
      }
      go_near(box, px, py, done, k + 1, poses);
    };
  if (to[0] != 'L' || realign_m_ <= 0.0) { run({stow(), nav(m), dock(f)}, after); return; }
  // 장소 상자 (11-22 사용자): Nav2 → 상자 쪽으로 돌아 봄 → 인지 상자 정면에서 옆으로 realign_m 넘게 벗어났으면 인지 가운데 기준
  //   진입점으로 Nav2 다시 (realign_tries번까지) → 직선 도킹. 인지 = 장소 태그(윗면 정중앙)를 읽은 관측, 없으면 잘리지 않은 location_box
  //   (실기 06:51: L3를 도면 진입점으로 가 상자가 12~20 cm 옆 → 대각선으로 꺾다 막힘 · L2 Nav2 도착 옆 11 cm → 51° 틀어져 막힘)
  const size_t li = static_cast<size_t>(std::stoi(to.substr(1)) - 1);
  const Pose2 f_a = to_arena(f.x, f.y, f.yaw);
  // tries −1 = 관측 자리 (place_view_m): 상자를 먼저 보고 그 가운데 기준으로 진입점을 정함 — 상자가 옮겨져 도면 진입점이 상자와 겹쳐도
  //   (시뮬 11-22: L2를 20 cm 옮긴 뒤 도면 진입점 세 변 모두 Nav2 nav_no_progress). 못 보면 도면 진입점으로 (지금과 같음)
  auto step = std::make_shared<std::function<void(const Pose2 &, const Pose2 &, double, double, int, bool)>>();
  *step = [this, step, to, li, n, m, f, cx, cy, f_a, after, nav, dock](const Pose2 & mm, const Pose2 & ff, double bx, double by, int tries, bool precise) {
    // 다시 잡을 때 진입점이 가까우면(0.4 m 안) Nav2 대신 approach 정밀 이동(drive_straight, 1 cm) — Nav2는 목표 허용 오차(약 15 cm) 안이면
    //   '이미 도착'으로 안 움직임 (시뮬 11-22: L1 옆 5.9 cm를 Nav2로 두 번 다시 보내도 그대로)
    Pose2 r0;
    const bool near_entry = precise && robot_map(r0) && std::hypot(mm.x - r0.x, mm.y - r0.y) < 0.4;
    const SubGoal go = near_entry ? SubGoal().set__skill("approach").set__target(fmt("%.4f,%.4f,%.4f", mm.x, mm.y, mm.yaw)).set__relation("drive_straight")
                                  : nav(mm);
    run({stow(), go}, [this, step, to, li, n, m, f, cx, cy, f_a, after, dock, ff, bx, by, tries](bool ok, const std::string & why) {
      if (!ok && tries < 0) {  // 관측 자리에 못 감 — 도면 진입점으로
        RCLCPP_WARN(get_logger(), "%s 관측 자리 못 감 (%s) — 도면 진입점으로", to.c_str(), why.c_str());
        (*step)(m, f, cx, cy, 0, false);
        return;
      }
      if (!ok && tries == 0) { after(false, why); return; }
      if (!ok) RCLCPP_WARN(get_logger(), "%s 진입점 다시 서기 실패 (%s) — 그 자리에서 봄", to.c_str(), why.c_str());
      Pose2 r;
      if (!robot_map(r)) { run({dock(ff)}, after); return; }
      const auto bm = to_map(bx, by);
      const double face = std::atan2(bm.y - r.y, bm.x - r.x);
      const rclcpp::Time t_turn = now();
      run({SubGoal().set__skill("approach").set__target(fmt("%.4f,%.4f,%.4f", r.x, r.y, face)).set__relation("turn")},
          [this, step, to, li, n, m, f, cx, cy, f_a, after, dock, ff, tries, t_turn](bool, const std::string &) {
        realign_timer_ = create_wall_timer(std::chrono::duration<double>(realign_wait_s_),
                                           [this, step, to, li, n, m, f, cx, cy, f_a, after, dock, ff, tries, t_turn]() {
          realign_timer_->cancel();
          const LocSeen * o = nullptr;
          const char * src = "태그";
          if (li < loc_tags_.size()) {
            const auto it = loc_whole_.find(str(loc_tags_[li]));
            if (it != loc_whole_.end() && it->second.t >= t_turn) o = &it->second;
          }
          if (!o && li < loc_any_.size() && loc_any_[li] && loc_any_[li]->t >= t_turn) { o = &*loc_any_[li]; src = "상자"; }
          if (tries < 0) {  // 관측 자리에서 본 상자 기준으로 진입점 (못 봤으면 도면 진입점)
            if (!o) {
              RCLCPP_INFO(get_logger(), "%s 관측 자리 — 상자를 못 봄, 도면 진입점으로", to.c_str());
              (*step)(m, f, cx, cy, 0, false);
              return;
            }
            double vx, vy;
            spot_map(*tf_buffer_, o->s, vx, vy, 1e9);
            const Pose2 av = to_arena(vx, vy, 0.0);
            const double dx = av.x - cx, dy = av.y - cy;
            RCLCPP_INFO(get_logger(), "%s 관측 자리 — 인지 %s가 기준에서 (%+.1f, %+.1f) cm → 그 정면 진입점으로", to.c_str(), src, dx * 100, dy * 100);
            (*step)(to_map(n.x + dx, n.y + dy, n.yaw), to_map(f_a.x + dx, f_a.y + dy, f_a.yaw), av.x, av.y, 0, true);
            return;
          }
          Pose2 r;
          if (!o || !robot_arena(r)) {
            RCLCPP_INFO(get_logger(), "%s 진입점 — 돌아서도 상자를 못 봄, 그대로 approach", to.c_str());
            run({dock(ff)}, after);
            return;
          }
          double mx, my;
          spot_map(*tf_buffer_, o->s, mx, my, 1e9);
          const Pose2 a = to_arena(mx, my, 0.0);
          const double lat = -(r.x - a.x) * std::sin(n.yaw) + (r.y - a.y) * std::cos(n.yaw);  // 변 법선에서 옆으로
          if (std::abs(lat) <= realign_m_ || tries >= realign_tries_) {
            RCLCPP_INFO(get_logger(), "%s 진입점 — 인지 %s 정면에서 옆 %.3f m%s, approach", to.c_str(), src, lat,
                        std::abs(lat) <= realign_m_ ? "" : " (다시 잡기 다 씀)");
            run({dock(ff)}, after);
            return;
          }
          const double dx = a.x - cx, dy = a.y - cy;  // 처음 진입 기준 가운데에서 인지 가운데까지
          // 옆으로 비켜서지 않음 — 차동 구동이라 옆 몇 cm에도 제자리 80° 넘게 돌다 상자 앞에서 막혔음 (실기 L3 3번 · 시뮬 4번).
          //   관측 자리 거리까지 똑바로 물러났다가 보정한 진입점으로 비스듬히 다시 들어감 (0.4 m 안 · 옆 10 cm 안이면 정밀 이동)
          const double back = std::max(0.0, view_m_ - std::hypot(r.x - a.x, r.y - a.y));
          RCLCPP_WARN(get_logger(), "%s 진입점 — 인지 %s 정면에서 옆 %.3f m (> %.2f) · 상자가 기준에서 (%+.1f, %+.1f) cm → %.2f m 물러났다 다시 (%d/%d)",
                      to.c_str(), src, lat, realign_m_, dx * 100, dy * 100, back, tries + 1, realign_tries_);
          const auto again = [this, step, n, f_a, dx, dy, a, tries]() {
            (*step)(to_map(n.x + dx, n.y + dy, n.yaw), to_map(f_a.x + dx, f_a.y + dy, f_a.yaw), a.x, a.y, tries + 1, true);
          };
          if (back < 0.02) { again(); return; }
          run({stow(), SubGoal().set__skill("approach").set__target(fmt("%.3f", back)).set__relation("exit")},
              [again](bool, const std::string &) { again(); });
        });
      });
    });
  };
  // 관측 자리 = 상자 가운데에서 면 법선 뒤로 place_view_m (벽에 가까우면 진입점 쪽으로 줄임) — 0이면 바로 진입점
  double vd = view_m_;
  const double nd = std::hypot(n.x - cx, n.y - cy);
  //   관측 자리에서 상자 쪽으로 크게 돎 → 벽까지 회전 여유(exit_turn_clear_m) 이상인 데까지 줄임 (wall_clearance 0.22면 회전 반경 0.288보다 가까움 ·
  //   11-22 실기 L3 −y: 0.75 m 자리가 벽에서 0.27 m라 90° 돌며 비틀댐)
  while (vd > nd + 0.05 && wall_distance(cx - vd * std::cos(n.yaw), cy - vd * std::sin(n.yaw), walls_) < std::max(wall_clearance_, exit_turn_clear_)) vd -= 0.05;
  if (view_m_ > 0.0 && vd > nd + 0.05)
    (*step)(to_map(cx - vd * std::cos(n.yaw), cy - vd * std::sin(n.yaw), n.yaw), f, cx, cy, -1, false);
  else
    (*step)(m, f, cx, cy, 0, false);
}

// ---------- 장소 상자 — 칸은 인지한 상자 가운데(윗면 정중앙 태그) 위에 (11-22 사용자: 도면 좌표 + 보정은 상자가 도면과 다르면 상자 밖에 놓음)
//   관측은 odom 점으로 기억 → 쓸 때 지금 map으로 (붙어 있는 동안 벽 정합이 map→odom을 3~5 cm씩 옮김)
void MissionManager::correct(size_t li) {
  if (li >= loc_tags_.size() || li >= loc_boxes_.size() || li >= loc_pose_.size()) return;
  const Box & b = loc_boxes_[li];
  const std::string tag = str(loc_tags_[li]);
  const auto fresh = [this](const std::map<std::string, LocSeen> & m, const std::string & k) -> const LocSeen * {
    const auto it = m.find(k);
    return it != m.end() && (now() - it->second.t).seconds() <= loc_keep_s_ ? &it->second : nullptr;
  };
  loc_pose_[li].reset();
  // 장소마다 상자 가운데는 한 번만 정함 (처음 태그를 읽은 관측) — 가운데 칸 퍽이 태그를 가리고 나서도 같은 가운데
  if (!loc_fix_[li]) {
    const LocSeen * o = fresh(loc_whole_, tag);
    if (!o) { RCLCPP_WARN(get_logger(), "L%zu 상자 못 봄 — %.0f s 안에 태그를 읽은 장소 상자 없음", li + 1, loc_keep_s_); return; }
    loc_fix_[li] = o->s;
  }
  double mx, my;
  spot_map(*tf_buffer_, *loc_fix_[li], mx, my, 1e9);  // odom 점을 지금 map으로 (odom은 짧은 동안 안 튐)
  const Pose2 a = to_arena(mx, my, 0.0);
  const double off = std::hypot(a.x - b.x, a.y - b.y);
  if (off > corr_max_) {
    RCLCPP_WARN(get_logger(), "L%zu 상자 도면에서 %.0f cm (> %.0f) — 다른 것을 본 것으로 보고 안 씀", li + 1, off * 100, corr_max_ * 100);
    loc_fix_[li].reset();
    return;
  }
  // 무늬 방향: 처음 도킹한 자리에서 로봇이 보는 쪽이 무늬 위(+Y) — 90° 단위로 맞춰 장소가 끝날 때까지 고정 (11-22 발제 +Y 고정 삭제)
  if (!loc_rot_[li]) {
    Pose2 r;
    loc_rot_[li] = robot_arena(r) ? std::round((r.yaw - M_PI / 2) / (M_PI / 2)) * (M_PI / 2) : 0.0;
  }
  loc_pose_[li] = Pose2{a.x, a.y, *loc_rot_[li]};
  RCLCPP_INFO(get_logger(), "L%zu 상자 가운데 (도면에서 %+.1f, %+.1f mm) · 무늬 %.0f° — 고정", li + 1, (a.x - b.x) * 1000, (a.y - b.y) * 1000,
              *loc_rot_[li] * 180.0 / M_PI);
}

std::pair<double, double> MissionManager::cell_target(size_t li, int cell) const {
  if (li < loc_pose_.size() && loc_pose_[li]) return box_cell(*loc_pose_[li], cell, grid_step_);
  const auto [dx, dy] = cell_offset(cell, grid_step_);
  return {locations_[li].first + dx, locations_[li].second + dy};
}

int MissionManager::basket_slot_of(int tag) const {
  if (basket_ && tag >= 0) for (const auto & s : basket_->slots) if (s.filled && s.tag == tag) return s.slot;
  return -1;
}

bool MissionManager::basket_has_room() const {
  if (!basket_) return true;  // 아직 장부를 못 받음 — coordinator가 load next에서 slot_full로 알려 줌
  for (const auto & s : basket_->slots) if (!s.filled) return true;
  return false;
}

const perception::msg::TrackedObject * MissionManager::puck_near(double mx, double my, double r) const {
  const TrackedObject * best = nullptr;
  double best_d = r > 0 ? r : match_radius_;
  if (objects_) for (const auto & o : objects_->objects) {
    if (!is_puck(o)) continue;
    const double d = std::hypot(o.pose.pose.position.x - mx, o.pose.pose.position.y - my);
    if (d < best_d) { best_d = d; best = &o; }
  }
  return best;
}

// ---------- 집기 · 세우기
void MissionManager::grab(int tag, Done done, int tries) {
  // 다시 시도는 그 자리를 inspect로 먼저 봄 (fresh) — 실패한 pick·세우기가 퍽을 밀거나 넘어뜨렸을 수 있음
  const auto again = [this, tag, done, tries](bool ok, const std::string & why) {
    if (ok || tries >= retries_ || interrupted(why)) { done(ok, why); return; }
    RCLCPP_WARN(get_logger(), "퍽 %d 집기 실패 (%s) — 다시 보고 (%d/%d)", tag, why.c_str(), tries + 1, retries_);
    grab(tag, done, tries + 1);  // 트랙 id가 바뀌었을 수 있음 — 다시 찾음
  };
  locate(tag, tries == 0, [this, tag, tries, again](const TrackedObject * o) {
    if (!o) { again(false, "puck_lost"); return; }
    if (is_lying(*o)) {  // 누움 (조사 뒤에 넘어진 퍽도) — 세워서 쥠 (stand_up 안에 아직 누움이면 다시 세우기가 있음)
      lying_.insert(tag);
      stand_up(o->id, o->pose.pose.position.x, o->pose.pose.position.y, 0, [this, tag, again](bool ok, const std::string & why) {
        if (ok) lying_.erase(tag);
        again(ok, why);  // 세운 뒤 pick 실패 · stand_lost도 다시 봄 (11-21 실기 퍽 5·2: 한 번에 끝)
      });
      return;
    }
    lying_.erase(tag);
    attempt_ = tries;
    pick_near(*o, again);
  }, tries > 0);
}

std::pair<double, double> MissionManager::here(const Spot & s) const {
  double x, y;
  spot_map(*tf_buffer_, s, x, y);
  return {x, y};
}

void MissionManager::locate(int tag, bool memory, std::function<void(const TrackedObject *)> cb, bool fresh) {
  const auto at = puck_at_.find(tag);
  if (at == puck_at_.end()) { cb(puck_for(objects_.get(), tag, nullptr, match_radius_)); return; }
  // 조사 때 본 자리 (map) — 도면 박스 + x 오프셋은 도면과 실제 상자가 어긋난 만큼 비켜 match_radius(4 cm)로 못 짝지음
  const auto seen = puck_seen_.find(tag);
  const Box & box = start_boxes_[at->second.first];
  const auto m = seen != puck_seen_.end() ? Pose2{here(seen->second).first, here(seen->second).second, 0.0}
                                          : to_map(box.x + at->second.second, box.y);
  const double spot[2] = {m.x, m.y};
  if (const auto * o = fresh ? nullptr : puck_for(objects_.get(), tag, spot, match_radius_)) { cb(o); return; }
  // 안 보인 지 2초 넘어 인지 목록에서 빠짐 — 처음엔 마지막으로 본 트랙 그대로 (coordinator가 기억한 자리로 pick, 적재 뒤 다시 보지 않음)
  if (const auto it = last_seen_.find(tag); memory && !fresh && it != last_seen_.end()) {
    note("퍽 " + str(tag) + " 기억한 자리로 (인지 목록에 없음)");
    auto & p = it->second.pose.pose.position;
    std::tie(p.x, p.y) = here(last_spot_[tag]);
    cb(&it->second);
    return;
  }
  // 그래도 못 집었으면 그 자리를 다시 봄
  note("퍽 " + str(tag) + (fresh ? " 다시 보기" : " 다시 보기 (인지 목록에 없음)"));
  const size_t b = at->second.first;
  run({SubGoal().set__skill("inspect").set__target(fmt("%.4f,%.4f,%.4f", m.x, m.y, box_top_z_ + puck_h_))},
      [this, tag, b, mx = m.x, my = m.y, cb](bool, const std::string &) {
        // 실패한 pick·세우기가 퍽을 차체 쪽으로 밀었으면 물러나 다시 봄 (11-21 실기 S5 퍽 6: 0.195 → 0.136 m · tag 못 읽어 puck_lost)
        const auto [bx, by] = box_seen_.count(b) ? here(box_seen_[b]) : std::make_pair(mx, my);
        back_off(b, bx, by, 0, [this, tag, mx, my, cb]() {
          const double spot2[2] = {mx, my};
          cb(puck_for(objects_.get(), tag, spot2, match_radius_));
        });
      });
}

void MissionManager::stand_up(uint32_t id, double mx, double my, int tries, Done done) {
  note("누운 퍽 세우기 (" + str(tries + 1) + "/" + str(retries_ + 1) + ")");
  attempt_ = tries;
  // stand 뒤 look 자세에서 섰는지 봄 — 손을 뺀 자리(퍽 위)에서는 인지가 선 퍽을 '누움'으로 봄 (11-15 실기 bag 3번:
  //   그 자리에서 lying → 선 퍽에 stand를 또 보냄 · stow/look에서는 같은 트랙이 standing)
  run({SubGoal().set__skill("stand").set__target(std::to_string(id)), SubGoal().set__skill("pose").set__target("look")},
      [this, mx, my, tries, done](bool ok, const std::string & why) {
    if (!ok) { done(false, "stand_failed " + why); return; }
    // look 자세 — 인지가 선 퍽을 다시 볼 시간
    timer_ = create_wall_timer(std::chrono::duration<double>(stand_check_s_), [this, mx, my, tries, done]() {
      timer_->cancel();
      if (!active_) return;
      // stand_find_m(10 cm) 안: 실패한 stand는 퍽을 6~7 cm 던짐 (11-15 실기 6.6 cm — 4 cm로 찾다 '섰음'으로 보고 빈손 stand_pick)
      const TrackedObject * o = puck_near(mx, my, stand_find_m_);
      if (!o) { done(false, "stand_lost"); return; }  // look에서도 안 보임 — 빈손으로 집지 않음 (잡기 판정이 빈손을 못 가림: 0.0048 m · 0.09 A)
      if (is_lying(*o)) {  // 아직 누움 — 같은 퍽을 다시 세움
        if (tries >= retries_) { done(false, "stand_still_lying"); return; }
        stand_up(o->id, o->pose.pose.position.x, o->pose.pose.position.y, tries + 1, done);
        return;
      }
      // 섰음: 선 퍽을 일반 pick (인지 위치 · 손 카메라 · pick 정책). 선 퍽은 누운 가운데에서 약 45 mm 옆에 섬
      //   (11-15 실기: 세운 그 자리에서 같은 손목으로 바로 내려가는 stand_pick은 세우며 밀린 만큼 빗나감)
      pick_near(*o, done);
    });
  });
}

// 팔이 닿게 조금 전진 (11-20 사용자: 도킹 뒤 퍽이 멀면 계산해서 조금씩 다가간 뒤 멈추고 pick)
//   박스에 붙은 뒤(back_ 있음)에만 · 퍽(인지, map)이 로봇 중심에서 pick_reach_m 밖이면 그 안에 들 만큼만 곧게 전진 —
//   상자 앞 gap이 creep_min_gap 아래로는 안 감 (approach_gap − creep_min_gap가 이 도킹의 전진 한도) · 움직이며 계산하지 않고
//   한 번 가고 멈춘 뒤 pick (pick은 손 카메라로 다시 잼 · 그래도 못 닿으면 coordinator가 pick_far). 전진한 만큼 물러나는 거리에 더함
//   퍽 자리는 locate가 준 것 — 인지 목록에서 빠진 퍽도 기억한 자리로 (11-22 실기 nav_1122_095737: 두 번째 퍽은 첫 pick 동안 목록에서
//   빠져 id로 못 찾아 전진을 건너뜀 → 0.27~0.28 m에서 grasp_failed_all → pick_far 2번)
void MissionManager::pick_near(const TrackedObject & o, Done done) {
  const auto pick = [this, id = o.id, done]() { run({SubGoal().set__skill("pick").set__target(std::to_string(id))}, done); };
  const double room = approach_gap_ - creep_min_gap_ - creep_used_;
  Pose2 r;
  if (back_.empty() || room < 0.01 || !robot_map(r)) { pick(); return; }
  const double c = std::cos(r.yaw), s = std::sin(r.yaw);
  const double dx = o.pose.pose.position.x - r.x, dy = o.pose.pose.position.y - r.y;
  const double fwd = dx * c + dy * s, lat = -dx * s + dy * c;
  if (std::hypot(fwd, lat) <= pick_reach_m_ || std::abs(lat) >= pick_reach_m_) { pick(); return; }  // 이미 닿음 · 옆이라 앞으로 가도 안 닿음
  const double step = std::min(fwd - std::sqrt(pick_reach_m_ * pick_reach_m_ - lat * lat), room);
  if (step < 0.01) { pick(); return; }
  note(fmt("퍽이 %.3f m — 팔이 닿게 %.0f mm 전진", std::hypot(fwd, lat), step * 1000));
  run({SubGoal().set__skill("approach").set__target(fmt("%.4f,%.4f,%.4f", r.x + step * c, r.y + step * s, r.yaw)).set__relation("drive_straight")},
      [this, step, pick, done](bool ok, const std::string & why) {
        if (!ok && interrupted(why)) { done(false, why); return; }
        // 실패해도 일부 움직였을 수 있음 — 물러날 때 모자라지 않게 다 더함 (더 물러나는 쪽이 안전)
        creep_used_ += step;
        if (!back_.empty()) back_ = fmt("%.3f", std::stod(back_) + step);
        if (!ok) RCLCPP_WARN(get_logger(), "전진 실패 (%s) — 그 자리에서 pick", why.c_str());
        pick();
      });
}

// ---------- 좌표
Pose2 MissionManager::to_map(double x, double y, double yaw) const { return apply_corr(acorr_, arena_to_map({x, y, yaw}, start_)); }

Pose2 MissionManager::to_arena(double mx0, double my0, double myaw0) const {
  return map_to_arena(unapply_corr(acorr_, {mx0, my0, myaw0}), start_);
}

std::string MissionManager::point(double ax, double ay, double z) const {
  const auto m = to_map(ax, ay);
  return fmt("%.4f,%.4f,%.4f", m.x, m.y, z);
}

bool MissionManager::robot_map(Pose2 & p) const {
  try {
    const auto t = tf_buffer_->lookupTransform("map", "base_footprint", tf2::TimePointZero);
    const auto & r = t.transform.rotation;
    p = {t.transform.translation.x, t.transform.translation.y,
         std::atan2(2 * (r.w * r.z + r.x * r.y), 1 - 2 * (r.y * r.y + r.z * r.z))};
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool MissionManager::robot_arena(Pose2 & p) const {
  Pose2 m;
  if (!robot_map(m)) return false;
  p = to_arena(m.x, m.y, m.yaw);
  return true;
}

// ---------- 도면 보정 — 인지 상자(윗면 가운데, map) ↔ 도면 상자 짝으로 arena → map 변환을 고침
//   start_box는 9개 상자 전부와 (태그 못 본 장소 상자도 start_box로 나옴), location_box는 태그의 장소 상자와만 짝지음.
//   gate 안에 하나뿐일 때만 · 로봇에서 arena_corr_range 안 · arena_corr_min_hits번 본 짝만 맞춤에 씀
//   영상 끝에 잘린 시작 박스(cut — 중심을 못 믿음)와 같은 관측의 재발행(last_seen 그대로)은 안 셈 — 트랙은 안 보여도 계속 나와서
//   한 번 본 것이 min_hits를 바로 채웠음 (11-15 실기 approach_loop snap에서 확인한 거름을 여기로)
void MissionManager::update_corr(const perception::msg::TrackedObjectArray & m) {
  if (loc_on_ && wall_s() - loc_last_ < loc_settle_s_) return;  // 로봇 위치를 옮기는 중 — 이 관측은 옛 좌표와 섞임
  const size_t ns = start_boxes_.size();
  if (marks_.size() != ns + loc_boxes_.size()) return;
  std::vector<std::pair<double, double>> raw, cands;  // 도면 상자 map 점 (보정 전 · 지금 보정)
  for (size_t i = 0; i < marks_.size(); ++i) {
    const Box & b = i < ns ? start_boxes_[i] : loc_boxes_[i - ns];
    const auto r = arena_to_map({b.x, b.y, 0}, start_);
    const auto c = apply_corr(acorr_, r);
    raw.push_back({r.x, r.y});
    cands.push_back({c.x, c.y});
  }
  bool seen = false;
  for (const auto & o : m.objects) {
    const bool start_box = o.class_name == "start_box";
    if (!start_box && o.class_name != "location_box") continue;
    const auto dist = detail(o, "distance");
    if (!dist.empty() && std::stod(dist) > acorr_range_) continue;
    if (start_box && detail(o, "cut") == "true") continue;  // 장소 상자는 태그 가운데라 잘려도 씀
    const std::pair<int32_t, uint32_t> stamp{o.last_seen.sec, o.last_seen.nanosec};
    const auto it = acorr_seen_.find(o.id);
    if (it != acorr_seen_.end() && it->second == stamp) continue;
    acorr_seen_[o.id] = stamp;
    const double qx = o.pose.pose.position.x, qy = o.pose.pose.position.y;
    int k = -1;
    // 지금 보정한 자리에서 못 찾으면 도면 그대로의 자리에서도 — 상자 둘로 맞춘 회전이 먼 상자를 0.47 m 옮겨 그 상자가
    //   gate 밖이 되면 다시는 근거가 못 됐음 (11-15 실기 bag approach_1115_094224 재생: S6·L2 → 10° · S3 영영 빠짐)
    if (start_box) {
      k = match_landmark(qx, qy, cands, acorr_gate_);
      if (k < 0) k = match_landmark(qx, qy, raw, acorr_gate_);
    } else {
      const auto t = detail(o, "tag");
      for (size_t li = 0; li < loc_tags_.size() && ns + li < cands.size(); ++li)
        if (t == str(loc_tags_[li]) && (std::hypot(cands[ns + li].first - qx, cands[ns + li].second - qy) < acorr_gate_ ||
                                        std::hypot(raw[ns + li].first - qx, raw[ns + li].second - qy) < acorr_gate_))
          k = static_cast<int>(ns + li);
    }
    if (k < 0) continue;
    auto & mk = marks_[static_cast<size_t>(k)];
    const double a = mk.hits ? acorr_alpha_ : 1.0;
    mk.qx += a * (qx - mk.qx);
    mk.qy += a * (qy - mk.qy);
    ++mk.hits;
    seen = true;
  }
  if (!seen) return;
  std::vector<CorrPair> pairs;
  for (size_t i = 0; i < marks_.size(); ++i)
    if (marks_[i].hits >= acorr_min_hits_) pairs.push_back({raw[i].first, raw[i].second, marks_[i].qx, marks_[i].qy});
  if (pairs.empty()) return;
  std::string used;
  for (size_t i = 0; i < marks_.size(); ++i)
    if (marks_[i].hits >= acorr_min_hits_) used += (used.empty() ? "\"" : ", \"") + (i < ns ? "S" + str(static_cast<int>(i + 1)) : "L" + str(static_cast<int>(i - ns + 1))) + "\"";
  // 벽 정합이 맞으면 보정 자체는 벽으로 (상자 짝은 상자마다 인지 위치 marks만) — 상자 둘로 맞춘 회전은 10°씩 틀림 (bag 094224)
  //   arena_corr_boxes false: 상자로는 보정 안 함 (벽 정합만) — 상자 인지 위치(marks)는 계속 냄
  //   arena_correction false (11-16): marks만 셈 (approach_perceived) — 상자 보정은 안 함
  const bool box_fit = acorr_on_ && acorr_boxes_ && !wall_ok_;
  const auto c = !box_fit ? acorr_ : fit_arena_corr(pairs, acorr_min_sep_, acorr_max_shift_, acorr_max_rot_, acorr_outlier_);
  if (!box_fit || (c.n == acorr_.n && !corr_moved(c, acorr_, acorr_step_m_, acorr_step_rad_))) {  // 작은 흔들림은 버림 (목표·기록이 흔들리지 않게)
    // 근거 상자가 늘거나 상자 하나의 인지 위치가 step만큼 움직이면 알림 (approach_loop는 그 상자 자리를 직접 씀)
    bool mark_moved = used != acorr_used_;
    for (size_t i = 0; i < marks_.size() && i < acorr_sent_.size(); ++i)
      mark_moved |= marks_[i].hits >= acorr_min_hits_ &&
                    std::hypot(marks_[i].qx - acorr_sent_[i].first, marks_[i].qy - acorr_sent_[i].second) > acorr_step_m_;
    if (mark_moved) { acorr_used_ = used; publish_corr(used); }
    return;
  }
  acorr_used_ = used;
  commit_corr(c, "boxes", fmt("근거 %.0f개 · 본 상자 %.0f개", c.n, static_cast<double>(pairs.size())));
}

bool MissionManager::corr_moved(const ArenaCorr & a, const ArenaCorr & b, double m, double r) {
  return std::hypot(a.dx - b.dx, a.dy - b.dy) > m || std::abs(a.dth - b.dth) > r;
}

void MissionManager::commit_corr(const ArenaCorr & c, const char * src, const std::string & why) {
  acorr_ = c;
  acorr_src_ = src;
  RCLCPP_INFO(get_logger(), "도면 보정 dx %+.3f dy %+.3f dθ %+.1f° (%s: %s)%s", c.dx, c.dy, c.dth * 180 / M_PI, src, why.c_str(),
              active_ ? "" : " — 미션 전");
  publish_corr(acorr_used_);
  char line[256];
  std::snprintf(line, sizeof(line), "{\"t\": %.3f, \"phase\": \"%s\", \"src\": \"%s\", \"dx\": %.4f, \"dy\": %.4f, \"dth_deg\": %.2f, \"n\": %d, "
                "\"wall_rms\": %.4f, \"boxes\": [", wall_s() - mission_t0_, phase_name(phase_), src, c.dx, c.dy, c.dth * 180 / M_PI, c.n,
                wall_rms_);
  append_log("arena_correction.jsonl", line + acorr_used_ + "]}");
  // /arena/walls 다시 보내기는 크게 바뀔 때만 — 보낼 때마다 Nav2 costmap 정적 층이 다시 잡힘 (11-14: 주행 중 10번 → 벽에 부딪힘)
  if (acorr_walls_m_ > 0 && corr_moved(acorr_, acorr_walls_, acorr_walls_m_, acorr_walls_rad_)) { acorr_walls_ = acorr_; publish_walls(); }
}

// 보정 → 로봇 위치 — 근거가 충분하면 보정 조각을 cuvslam_slam에 넘기고 (map' = 조각⁻¹ · map) 그만큼 이쪽 보정·관측에서 뺌.
//   조각 = loc_step(5 cm · 1°)까지 · loc_period(2 s)마다 하나 — cuvslam MapGate(초당 5 cm · 5°)가 부드럽게 옮김, 잠금(0.30 m)에 안 걸림
void MissionManager::loc_tick() {
  if (!acorr_on_ || !(active_ || acorr_idle_) || acorr_.n == 0) return;
  if (acorr_.n < loc_min_n_) return;   // 상자 하나 · 둘로는 위치를 안 옮김 (11-15: 상자 1개로 dy −0.2 m)
  if (std::hypot(acorr_.dx, acorr_.dy) < loc_min_m_ && std::abs(acorr_.dth) < loc_min_rad_) return;
  const ArenaCorr st = corr_step(acorr_, loc_step_m_, loc_step_rad_);
  std_msgs::msg::Float64MultiArray req;
  req.data = {st.dx, st.dy, st.dth};
  loc_pub_->publish(req);
  acorr_ = corr_after_step(acorr_, st);
  for (auto & mk : marks_) {
    if (!mk.hits) continue;
    const auto q = unapply_corr(st, {mk.qx, mk.qy, 0.0});   // 같은 상자가 새 좌표에서 보일 자리
    mk.qx = q.x;
    mk.qy = q.y;
  }
  loc_last_ = wall_s();
  RCLCPP_INFO(get_logger(), "도면 보정 → 로봇 위치 %+.3f %+.3f %+.1f° (%s) · 남은 보정 %+.3f %+.3f %+.1f°", st.dx, st.dy, st.dth * 180 / M_PI,
              acorr_src_.c_str(), acorr_.dx, acorr_.dy, acorr_.dth * 180 / M_PI);
  char line[200];
  std::snprintf(line, sizeof(line), "{\"t\": %.3f, \"phase\": \"%s\", \"to_localization\": [%.4f, %.4f, %.3f]}", wall_s() - mission_t0_,
                phase_name(phase_), st.dx, st.dy, st.dth * 180 / M_PI);
  append_log("arena_correction.jsonl", line);
  publish_corr(acorr_used_);
}

void MissionManager::publish_corr(const std::string & used) {
  // used = 근거 상자 이름 '"S3", "L1"' (min_hits 채운 것)
  //   marks = 상자마다 인지 위치 지수 평균 (map) · 본 횟수 — min_hits 넘은 것만
  char head[160];
  std::snprintf(head, sizeof(head), "{\"dx\": %.4f, \"dy\": %.4f, \"dth\": %.5f, \"n\": %d, \"src\": \"%s\", \"active\": %s, \"boxes\": [",
                acorr_.dx, acorr_.dy, acorr_.dth, acorr_.n, acorr_src_.c_str(), active_ ? "true" : "false");
  std::string marks;
  const size_t ns = start_boxes_.size();
  acorr_sent_.assign(marks_.size(), {1e9, 1e9});
  for (size_t i = 0; i < marks_.size(); ++i) {
    if (marks_[i].hits < acorr_min_hits_) continue;
    char e[96];
    std::snprintf(e, sizeof(e), "%s\"%s%zu\": [%.4f, %.4f, %d]", marks.empty() ? "" : ", ", i < ns ? "S" : "L",
                  i < ns ? i + 1 : i - ns + 1, marks_[i].qx, marks_[i].qy, marks_[i].hits);
    marks += e;
    acorr_sent_[i] = {marks_[i].qx, marks_[i].qy};
  }
  std_msgs::msg::String m;
  m.data = head + used + "], \"marks\": {" + marks + "}}";
  corr_pub_->publish(m);
}

// 경기장 벽을 map 격자로 (/arena/walls, latched) — nvblox_map이 /map에 겹쳐 Nav2가 매핑 처음부터 벽을 앎.
//   격자 = 경기장 네 모서리(+여유)를 map으로 옮긴 범위 · 벽 칸 100 · 나머지 −1(모름 — 겹칠 때 nvblox 값을 안 덮음)
void MissionManager::publish_walls() {
  if (walls_.size() < 4) return;
  const double res = 0.05, pad = 0.3;
  double x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (size_t i = 0; i + 1 < walls_.size(); i += 2) {
    const auto m = to_map(walls_[i], walls_[i + 1], 0.0);
    x0 = std::min(x0, m.x); y0 = std::min(y0, m.y); x1 = std::max(x1, m.x); y1 = std::max(y1, m.y);
  }
  x0 -= pad; y0 -= pad;
  nav_msgs::msg::OccupancyGrid g;
  g.header.frame_id = "map";
  g.header.stamp = now();
  g.info.resolution = res;
  g.info.width = static_cast<uint32_t>(std::ceil((x1 + pad - x0) / res));
  g.info.height = static_cast<uint32_t>(std::ceil((y1 + pad - y0) / res));
  g.info.origin.position.x = x0;
  g.info.origin.position.y = y0;
  g.info.origin.orientation.w = 1.0;
  g.data.assign(static_cast<size_t>(g.info.width) * g.info.height, -1);
  for (size_t i = 0; i + 3 < walls_.size(); i += 4) {
    const double len = std::hypot(walls_[i + 2] - walls_[i], walls_[i + 3] - walls_[i + 1]);
    const int n = std::max(1, static_cast<int>(std::ceil(len / (res / 2))));
    for (int k = 0; k <= n; ++k) {
      const double t = static_cast<double>(k) / n;
      const auto m = to_map(walls_[i] + t * (walls_[i + 2] - walls_[i]), walls_[i + 1] + t * (walls_[i + 3] - walls_[i + 1]), 0.0);
      const int cx = static_cast<int>(std::floor((m.x - x0) / res)), cy = static_cast<int>(std::floor((m.y - y0) / res));
      if (cx >= 0 && cy >= 0 && cx < static_cast<int>(g.info.width) && cy < static_cast<int>(g.info.height))
        g.data[static_cast<size_t>(cy) * g.info.width + cx] = 100;
    }
  }
  // 벽만 따로 (11-16) — collision_monitor 옆·뒤 입력(ros_system grid_obstacle_cloud). 상자를 넣으면 상자에 붙는 approach(1.6 cm)를 막음
  walls_only_pub_->publish(g);
  // 시작 박스 · 장소 상자도 칸 100 (11-14) — 7 cm 낮은 상자는 다가가면 카메라 광선이 로컬 코스트맵에서 가까운 면을 지워
  //   DWB가 몸으로 긁고 돌다 밀었다 (시뮬 상자 3을 0.5 m). 붙는 건 approach(코스트맵 안 씀)라 지도에 있어도 붙음
  size_t nb = 0;
  for (const auto * set : {&start_boxes_, &loc_boxes_}) {
    for (const auto & b : *set) {
      ++nb;
      for (double ax = b.x - b.half; ax <= b.x + b.half + 1e-9; ax += res / 2) {
        for (double ay = b.y - b.half; ay <= b.y + b.half + 1e-9; ay += res / 2) {
          const auto m = to_map(ax, ay, 0.0);
          const int cx = static_cast<int>(std::floor((m.x - x0) / res)), cy = static_cast<int>(std::floor((m.y - y0) / res));
          if (cx >= 0 && cy >= 0 && cx < static_cast<int>(g.info.width) && cy < static_cast<int>(g.info.height))
            g.data[static_cast<size_t>(cy) * g.info.width + cx] = 100;
        }
      }
    }
  }
  // 금지 구역 (keepout_zones, 11-15) — 칸 100으로 채움. /map(nvblox)에 그리면 카메라가 빈 바닥을 보며 지우지만 이 격자는 그대로
  for (size_t i = 0; i + 3 < keepout_.size(); i += 4) {
    const double ax0 = std::min(keepout_[i], keepout_[i + 2]), ax1 = std::max(keepout_[i], keepout_[i + 2]);
    const double ay0 = std::min(keepout_[i + 1], keepout_[i + 3]), ay1 = std::max(keepout_[i + 1], keepout_[i + 3]);
    for (double ax = ax0; ax <= ax1 + 1e-9; ax += res / 2) {
      for (double ay = ay0; ay <= ay1 + 1e-9; ay += res / 2) {
        const auto m = to_map(ax, ay, 0.0);
        const int cx = static_cast<int>(std::floor((m.x - x0) / res)), cy = static_cast<int>(std::floor((m.y - y0) / res));
        if (cx >= 0 && cy >= 0 && cx < static_cast<int>(g.info.width) && cy < static_cast<int>(g.info.height))
          g.data[static_cast<size_t>(cy) * g.info.width + cx] = 100;
      }
    }
  }
  walls_pub_->publish(g);
  RCLCPP_INFO(get_logger(), "경기장 벽 %zu개 · 상자 %zu개 · 금지 구역 %zu개 → /arena/walls (%ux%u)", walls_.size() / 4, nb,
              keepout_.size() / 4, g.info.width, g.info.height);
}

bool MissionManager::in_keepout(double x, double y) const {
  for (size_t i = 0; i + 3 < keepout_.size(); i += 4)
    if (x >= std::min(keepout_[i], keepout_[i + 2]) && x <= std::max(keepout_[i], keepout_[i + 2]) &&
        y >= std::min(keepout_[i + 1], keepout_[i + 3]) && y <= std::max(keepout_[i + 1], keepout_[i + 3]))
      return true;
  return false;
}

void MissionManager::write_log(const std::string & name, const std::string & json) const {
  if (run_dir_.empty()) return;
  std::ofstream(run_dir_ + "/" + name) << json;
}

void MissionManager::append_log(const std::string & name, const std::string & line) const {
  if (run_dir_.empty()) return;
  std::ofstream(run_dir_ + "/" + name, std::ios::app) << line << '\n';  // 닫히면서 바로 씀 — 전원이 꺼져도 앞 줄은 남음
}

// ---------- 시간 기록 (steps.jsonl · phases.jsonl) — tools/plan/step_stats.py가 skill별 시도·성공률·시간으로 모음
//   dur_s null = feedback을 놓쳐 단계 경계를 모름 (그 단계는 시간 통계에서 뺌)
//   실패 줄엔 detail(skill 마지막 단계들) · robot(그때 map 자세)도 (11-21 — 원인을 launch 로그에서 찾지 않게)
void MissionManager::step_line(size_t k, double t0, double t1, bool ok, const std::string & why, bool timed, const std::string & detail) {
  const auto & s = step_log_;
  if (k >= s.goals.size()) return;
  const auto & g = s.goals[k];
  append_log("steps.jsonl", fmt("{\"t0\": %.3f, \"t1\": %.3f, \"dur_s\": ", t0, t1) +
    (timed ? fmt("%.3f", t1 - t0) : std::string("null")) + ", \"phase\": " + q(s.phase) +
    ", \"plan\": " + std::to_string(s.plan) + ", \"step\": " + str(static_cast<int>(k) + 1) +
    ", \"steps\": " + str(static_cast<int>(s.goals.size())) + ", \"skill\": " + q(g.skill) + ", \"target\": " + q(g.target) +
    ", \"relation\": " + q(g.relation) + ", \"ok\": " + (ok ? "true" : "false") + ", \"reason\": " + q(why) +
    ", \"retry\": " + str(s.retry) + (s.ctx.empty() ? "" : ", " + s.ctx) + fail_fields(ok, detail) + "}");
}

std::string MissionManager::fail_fields(bool ok, const std::string & detail) const {
  if (ok) return "";
  std::string out = detail.empty() ? "" : ", \"detail\": " + q(detail);
  Pose2 r;
  if (robot_map(r)) out += fmt(", \"robot\": [%.3f, %.3f, %.1f]", r.x, r.y, r.yaw * 180.0 / M_PI);
  return out;
}

void MissionManager::close_steps(bool ok, const std::string & why, size_t last, const std::string & detail) {
  auto & s = step_log_;
  if (!s.plan || s.goals.empty()) return;
  last = std::min(std::max(last, s.cur), s.goals.size() - 1);
  const double t = wall_s();
  for (size_t k = s.cur; k <= last; ++k)  // 경계를 놓친 단계(cur < last)는 시간 모름
    step_line(k, k == s.cur ? s.t0 : t, t, k == last ? ok : true, k == last ? why : "", k == s.cur && last == s.cur, k == last ? detail : "");
  s.plan = 0;
}

void MissionManager::phase_line(bool ok, const std::string & why, Phase next) {
  const double t = wall_s();
  append_log("phases.jsonl", fmt("{\"t0\": %.3f, \"t1\": %.3f, \"dur_s\": %.3f", phase_t0_, t, t - phase_t0_) +
    ", \"phase\": " + q(phase_name(phase_)) + ", \"ok\": " + (ok ? "true" : "false") + ", \"reason\": " + q(why) +
    ", \"next\": " + q(phase_name(next)) + "}");
}

void MissionManager::mission_line(bool ok, const std::string & why) {
  const double t = wall_s();
  std::vector<std::string> r;
  for (const auto & x : results_) r.push_back(q(x));
  append_log("phases.jsonl", fmt("{\"t0\": %.3f, \"t1\": %.3f, \"dur_s\": %.3f", mission_t0_, t, t - mission_t0_) +
    ", \"phase\": \"MISSION\", \"ok\": " + (ok ? "true" : "false") + ", \"reason\": " + q(why) +
    ", \"diag\": " + (diag_run_ ? "true" : "false") +
    ", \"only\": " + q(only_ == Phase::IDLE ? "" : phase_name(only_)) + ", \"placed\": " + str(placed_) +
    ", \"total\": " + str(static_cast<int>(jobs_.size())) + ", \"results\": [" + join(r) + "]}");
}

// ---------- 사진 — 찍을 때만 구독해서 한 장 받고 끊음 (영상을 계속 받지 않게)
void MissionManager::snap(const std::string & name) {
  if (run_dir_.empty() || photo_topic_.empty()) return;
  char prefix[8];
  std::snprintf(prefix, sizeof(prefix), "%03d_", ++photo_n_);
  const std::string path = run_dir_ + "/photos/" + prefix + name + ".jpg";
  photo_sub_ = create_subscription<sensor_msgs::msg::Image>(photo_topic_, rclcpp::SensorDataQoS(),
    [this, path](sensor_msgs::msg::Image::ConstSharedPtr m) {
      if (!photo_sub_) return;
      const bool rgb = m->encoding == "rgb8", bgr = m->encoding == "bgr8";
      if (rgb || bgr) {
        cv::Mat img(static_cast<int>(m->height), static_cast<int>(m->width), CV_8UC3, const_cast<uint8_t *>(m->data.data()), m->step);
        cv::Mat out;
        if (rgb) cv::cvtColor(img, out, cv::COLOR_RGB2BGR); else out = img;
        cv::imwrite(path, out);
      }
      photo_sub_.reset();
    });
}

// ---------- RViz — 출발 구역 · 박스 · 장소 칸 (배정 뒤엔 그룹 색)
void MissionManager::publish_markers() {
  using visualization_msgs::msg::Marker;
  visualization_msgs::msg::MarkerArray arr;
  int id = 0;
  const auto add = [&](int type, double ax, double ay, double z, double sx, double sy, double sz,
                       float r, float g, float b, float a, const std::string & text = "") {
    Marker m;
    m.header.frame_id = "map";
    m.header.stamp = now();
    m.ns = "mission";
    m.id = id++;
    m.type = type;
    const auto p = to_map(ax, ay);
    m.pose.position.x = p.x;
    m.pose.position.y = p.y;
    m.pose.position.z = z;
    m.pose.orientation.z = std::sin(-start_.yaw / 2);  // 박스는 arena 축 정렬
    m.pose.orientation.w = std::cos(-start_.yaw / 2);
    m.scale.x = sx;
    m.scale.y = sy;
    m.scale.z = sz;
    m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = a;
    m.text = text;
    arr.markers.push_back(m);
  };
  if (start_zone_.size() == 4)
    add(Marker::CUBE, (start_zone_[0] + start_zone_[2]) / 2, (start_zone_[1] + start_zone_[3]) / 2, 0.005,
        start_zone_[2] - start_zone_[0], start_zone_[3] - start_zone_[1], 0.01, 0.3f, 0.55f, 0.9f, 0.35f);
  for (const auto & b : start_boxes_)
    add(Marker::CUBE, b.x, b.y, box_top_z_ / 2, 2 * b.half, 2 * b.half, box_top_z_, 0.8f, 0.7f, 0.5f, 0.8f);
  for (size_t li = 0; li < loc_boxes_.size(); ++li) {
    const auto & b = loc_boxes_[li];
    add(Marker::CUBE, b.x, b.y, box_top_z_ / 2, 2 * b.half, 2 * b.half, box_top_z_, 0.55f, 0.8f, 0.65f, 0.6f);
    const char g = order_.size() == 3 ? order_[li] : '?';
    add(Marker::TEXT_VIEW_FACING, locations_[li].first, locations_[li].second, 0.35, 0, 0, 0.12, 1, 1, 1, 1,
        "L" + str(static_cast<int>(li) + 1) + (g == '?' ? "" : " · " + std::string(1, g) + str(group_count(counts_, g))));
    const auto cells = g == '?' ? std::vector<int>{} : dice_cells(group_count(counts_, g));
    for (int c = 0; c < 9; ++c) {  // 3×3 칸 (태그 보정 포함) — 쓰는 칸은 그룹 색, 아니면 흐린 회색
      const auto [x, y] = cell_target(li, c);
      const bool on = std::find(cells.begin(), cells.end(), c) != cells.end();
      const float r = g == 'R' ? 0.9f : g == 'Y' ? 0.95f : 0.2f, gg = g == 'G' ? 0.75f : g == 'Y' ? 0.85f : 0.2f;
      add(Marker::CYLINDER, x, y, box_top_z_ + 0.03, 0.03, 0.03, 0.06,
          on ? r : 0.6f, on ? gg : 0.6f, on ? 0.2f : 0.6f, on ? 0.9f : 0.25f);
    }
  }
  markers_pub_->publish(arr);
}

}  // namespace plan
