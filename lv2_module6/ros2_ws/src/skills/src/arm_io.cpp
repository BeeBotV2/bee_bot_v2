// 팔 skill 공통 — 설명은 include/skills/arm_io.hpp (예전 coordinator.cpp move_arm · move_joints · set_gripper · contact_step …)
#include "skills/arm_io.hpp"
#include "skills/arm_state_cache.hpp"
#include "skills/skill_node.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

#include "shape_msgs/msg/solid_primitive.hpp"
#include "tf2/LinearMath/Transform.hpp"

namespace skills {

bool grasp_held(double pos, double effort, double min_pos, double min_effort) {
  if (pos < min_pos) return false;  // 끝까지 닫힘 — 쥔 것 없음
  return std::isnan(effort) || std::abs(effort) >= min_effort;
}

bool hand_yaw(const geometry_msgs::msg::Quaternion & q, double & psi) {
  if (std::abs(q.z) > 1e-6 || std::abs(q.w) > 1e-6 || std::hypot(q.x, q.y) < 0.5) return false;
  psi = 2.0 * std::atan2(q.y, q.x);
  return true;
}

int grip_state(double pos, double current, double still_s, double t, double min_current, double empty_m, double timeout_s) {
  // 빈손: 목표(stand_pinch_m 0.0050) 근처까지 닫힘 — 0.5 mm 여유. 그 사이(empty_m − 0.5 mm ~ empty_m)는 전류로 가림:
  //   11-15 실기 끌어오기: 퍽을 쥐었는데 손가락 패드가 눌려 0.0054~0.0055 · 10.8~21.5 mA → empty_m 0.0056보다 닫혀 빈손으로 놓침
  if (still_s > 0.2 && pos < empty_m - 0.0005) return 0;
  if (still_s > 0.2 && current >= min_current) return 1;
  if (still_s > 0.2 && pos < empty_m) return 0;
  // 목표(stand_pinch_m 0.0062 = 간격 30 mm)가 퍽 폭과 같아 쥐어도 전류가 0 근처 — 빈손으로 닫히지 않고 0.5 s 멈춰 있으면 쥠
  //   (11-15 실기: 잘 쥐었는데 위치 0.0062 · 0 mA로 4 s 기다려 stand_grip_timeout → 벌리고 실패)
  if (still_s > 0.5 && pos >= empty_m) return 1;
  return t > timeout_s ? -1 : 2;
}

namespace {
constexpr double CONTACT_VS = 0.3;  // 팀 관절 한계 × (시뮬 vel_scale)
constexpr double CONTACT_STOP_S = 3.0;  // 닫기 시작 뒤 손가락이 멈출 때까지 더 기다리는 상한 (실행 속도·그리퍼 속도가 시뮬과 달라도)
constexpr double CONTACT_HOLD_S = 0.3, CONTACT_SETTLE_S = 0.3, CONTACT_CLOSE_S = 0.5, CONTACT_RELEASE_S = 0.4;
constexpr double CONTACT_SETTLE_MAX_S = 0.8;  // 안정 구간 상한 — Δ가 계속 움직여도 이만큼 뒤엔 쥠 (기본 0.3 s + 최대 0.5 s)
constexpr double DELTA_WINDOW_S = 0.15, DELTA_SPREAD = 0.002;  // 최근 0.15 s 동안 관절마다 Δ 최대 − 최소 ≤ 2 mrad = 멈춤
const char * ARM[] = {"joint1", "joint2", "joint3", "joint4", "joint5"};
}  // namespace

ArmIO::ArmIO(rclcpp::Node & node)
: node_(node) {
  arm_plan_tries_ = node.declare_parameter<int>("arm_plan_tries", 3);
  // MoveIt 계획 속도·가속도 비율 (0~1] — 계획할 때마다 읽음 → `ros2 param set /skill_<이름> arm_velocity_scale 1.0`으로 바로 바뀜
  node.declare_parameter<double>("arm_velocity_scale", 0.3);
  // 닿는 구간 2(바닥 안정)에서도 보정 Δ를 계속 바꿀지 — false면 바닥 도착 때 Δ 고정 (3 쥐기와 같음) · 매번 읽음.
  //   11-14 실기 pick: 손 카메라가 ~1 s 못 보다 바닥에서 퍽을 다시 잡으면 v03이 쥐기 0.1~0.4 s 전에 Δ를 17~31 mrad 급히 바꿈 (툭 · 6/12회)
  node.declare_parameter<bool>("contact_bottom_live", true);
  // 닿는 구간 막힘: 쥐기·놓기 직전 joint2+joint3 전류가 위에서 멈췄을 때보다 이만큼 [A] 넘게 오르면 막힘 → 쥐지·놓지 않고 위로
  //   빼고 실패 arm_jam (pick은 다음 후보, coordinator는 단계를 한 번 더). 0 = 끔. 11-08 실기 pick 62판: 막힘 최솟값 +0.64 A ·
  //   정상 최댓값 +0.40 A (arm_safety_monitor 3.0 A는 막혀도 안 닿음 — 실측 최대 0.9~1.1 A)
  node.declare_parameter<double>("jam_current_a", 0.5);
  // 막힘은 전류에 더해 joint2·3이 목표를 이만큼 [rad] 넘게 못 따라갈 때만 — 전류만 오르고 위치는 따라오면 바닥·칸에 닿은 것 (막힘 아님).
  //   0 = 전류만 (예전). 11-21 실기 load: 막힘 4건 joint3 18~44 mrad · 성공 20건 ≤ 11 mrad · 바닥 막힘 판정 1건 0.2 mrad (오판)
  //   추종 값이 없으면(/rl_arm_correction/debug 0.5 s 넘게 끊김) 전류만으로 판정
  node.declare_parameter<double>("jam_lag_rad", 0.0);
  // traj_step(MoveIt 계획을 거치지 않는 궤적) 보내기 전 자기 충돌 검사 — false면 끔 (예전처럼) · 매번 읽음
  node.declare_parameter<bool>("traj_collision_check", true);
  node.declare_parameter<double>("traj_check_step_rad", 0.05);  // [rad] 관절이 이만큼 넘게 움직인 샘플마다 검사 (끝 자세는 늘)
  node.declare_parameter<double>("traj_check_timeout_s", 2.0);  // [s] 검사 전체 상한 — 넘으면 실패
  // 시작 자세가 이미 자기 충돌이라 MoveIt 계획이 거부되면 손끝을 이만큼 곧게 올려 빠져나온 뒤 다시 계획 [m] · 0 = 안 함
  //   11-21 실기 bag: load 막힘 자세(손가락이 적재함 칸 링 = base_link에 걸림)에서 inspect look 계획이 99999로 거부 → 퍽을 쥔 채 다음 상자로
  node.declare_parameter<double>("start_collision_lift_m", 0.03);
  node.declare_parameter<bool>("sag_fade", true);  // pick·load 시작 fade_sag (false = 예전처럼 닿는 구간 시작에서 한 번에 0) · 매번 읽음
  grip_current_a_ = node.declare_parameter<double>("grip_current_a", 0.1);  // 닫을 때 쥐는 힘 상한 [A] — 실기 11/08: gap 26 mm와 같이 0.1 A (약 7 N)
  grasp_check_ = node.declare_parameter<bool>("grasp_check", true);
  grasp_min_pos_m_ = node.declare_parameter<double>("grasp_min_pos_m", 0.003);
  grasp_min_effort_ = node.declare_parameter<double>("grasp_min_effort", 0.25);
  // 닫는 목표 손가락 사이 [m] — 퍽 30 mm보다 조금 좁게 (실기 11/08: 손가락이 퍽에 닿는 곳 ≈ gap 29 mm). 쥐는 힘 = 그 깊이만큼의 위치 오차
  node.declare_parameter<double>("grip_close_gap_m", 0.026);
  // 잡기 판정: 손가락이 닫는 목표보다 이만큼 넘게 못 들어가고 멈췄으면 쥠 (빈손이면 목표까지 감) [m]
  node.declare_parameter<double>("grasp_stop_margin_m", 0.0005);
  // sim: 닿는 구간은 시뮬과 같은 기준 동작 + 학습 보정 (2026-11-03 배포 결정) · moveit: 예전처럼 MoveIt만
  contact_mode_ = node.declare_parameter<std::string>("contact_mode", "sim");
  plan_ = node.create_client<GetMotionPlan>("/plan_kinematic_path");
  validity_ = node.create_client<GetStateValidity>("/check_state_validity");
  exec_ = rclcpp_action::create_client<ExecuteTrajectory>(&node, "/execute_trajectory");
  gripper_ = rclcpp_action::create_client<Gripper>(&node, "/gripper_controller/gripper_cmd");
  reset_sag_ = node.create_client<std_srvs::srv::Trigger>("/rl_arm_correction/reset_sag");
  fade_sag_ = node.create_client<std_srvs::srv::Trigger>("/rl_arm_correction/fade_sag");
  clear_octomap_ = node.create_client<std_srvs::srv::Empty>("/clear_octomap");
  ctx_pub_ = node.create_publisher<ml::msg::SkillContext>("/skill_context", rclcpp::QoS(10).reliable());
  hand_pub_ = node.create_publisher<perception::msg::HandTarget>("/hand_relative/target", rclcpp::QoS(1).reliable().transient_local());
  hr_sub_ = node.create_subscription<perception::msg::HandRelative>("/hand_relative", rclcpp::QoS(10),
    [this](perception::msg::HandRelative::ConstSharedPtr m) { hr_ = m; });
  state_cache_ = ArmStateCache::for_node(node);
  safety_sub_ = node.create_subscription<control::msg::ArmSafetyEvent>("/arm_safety_event", rclcpp::QoS(10).reliable(),
    [this](control::msg::ArmSafetyEvent::ConstSharedPtr m) {
      RCLCPP_WARN(node_.get_logger(), "팔 접촉 %s %.2fA", m->joint_name.c_str(), m->current_a);
      contact_ = true;  // 진행 중 결과가 실패로 오면 reason을 arm_contact로
    });
}

bool ArmIO::moveit_ready() const {
  return plan_->service_is_ready() && exec_->action_server_is_ready() && validity_->service_is_ready();
}
bool ArmIO::gripper_ready() const { return gripper_->action_server_is_ready(); }

void ArmIO::stop() {
  ++gen_;
  if (timer_) timer_->cancel();
  exec_->async_cancel_all_goals();
  gripper_->async_cancel_all_goals();
  ctx_off();
}

bool ArmIO::joint(const std::string & name, double & pos) const {
  return state_cache_->position(name, pos);
}

bool ArmIO::arm_now(Q5 & q) const {
  for (int i = 0; i < 5; ++i)
    if (!joint(ARM[i], q[i])) return false;
  return true;
}

void ArmIO::execute(const moveit_msgs::msg::RobotTrajectory & traj, Cb done) {
  ExecuteTrajectory::Goal goal;
  goal.trajectory = traj;
  const uint64_t gen = gen_;
  rclcpp_action::Client<ExecuteTrajectory>::SendGoalOptions opt;
  opt.goal_response_callback = [this, done, gen](const rclcpp_action::ClientGoalHandle<ExecuteTrajectory>::SharedPtr & h) {
    if (!h && gen == gen_) done(false);
  };
  opt.result_callback = [this, done, gen](const rclcpp_action::ClientGoalHandle<ExecuteTrajectory>::WrappedResult & r) {
    if (gen != gen_) return;  // 취소된 옛 goal의 늦은 결과
    done(r.code == rclcpp_action::ResultCode::SUCCEEDED && r.result->error_code.val == moveit_msgs::msg::MoveItErrorCodes::SUCCESS);
  };
  exec_->async_send_goal(goal, opt);
}

// reset_sag 직후: 처짐 보정이 0이 돼 팔이 처졌다가(퍽을 들면 ~0.04 rad) 다시 올라옴 (~0.4 s). 그 사이 MoveIt이 계획하면
// 실행 시작 검사(allowed_start_tolerance 0.01 rad)에서 거부 — 11/07 Isaac unload → place 5번 중 3번.
// 최소 SETTLE_MIN_S 기다리고(보정 0이 실제로 들어갈 때까지), 최근 SETTLE_N틱(0.2 s) 동안 관절마다 최대 − 최소가 SETTLE_SPREAD 안이면 go
// (상한 SETTLE_MAX_S). 예전 기준(기준 자세에서 4 mrad 안 3번 연속)은 실기 joint1이 멈춰서도 ±4.5 mrad로 떨어(헌팅) 3 s 상한까지
// 기다림 (11/08 pick_loop bag: 퍽 위 3.0 s · pick 뒤 1.9 s). 12 mrad / 0.2 s = 0.06 rad/s보다 빠르게 움직이는 팔은 여전히 기다림
// — 실행 시작 검사(allowed_start_tolerance 0.01 rad)까지 ~0.1 s면 6 mrad (11/08 실기 첫 unload: 천천히 움직이는 팔에 joint3 start point deviates)
void ArmIO::settle(std::function<void()> go) {
  if (!settle_) { go(); return; }
  settle_ = false;
  constexpr double SETTLE_MIN_S = 0.25, SETTLE_MAX_S = 3.0, SETTLE_SPREAD = 0.012, DT = 0.05;
  constexpr size_t SETTLE_N = 4;
  const uint64_t gen = gen_;
  auto hist = std::make_shared<std::vector<Q5>>();
  auto ticks = std::make_shared<int>(0);
  settle_timer_ = node_.create_wall_timer(std::chrono::duration<double>(DT), [this, go, gen, hist, ticks]() {
    if (gen != gen_) { settle_timer_->cancel(); return; }
    Q5 q;
    if (arm_now(q)) {
      hist->push_back(q);
      if (hist->size() > SETTLE_N) hist->erase(hist->begin());
    }
    bool still = hist->size() == SETTLE_N;
    for (int i = 0; i < 5 && still; ++i) {
      double lo = (*hist)[0][i], hi = lo;
      for (const auto & h : *hist) { lo = std::min(lo, h[i]); hi = std::max(hi, h[i]); }
      still = hi - lo <= SETTLE_SPREAD;
    }
    const double t = ++*ticks * DT;
    if ((t >= SETTLE_MIN_S && still) || t >= SETTLE_MAX_S) {
      settle_timer_->cancel();
      go();
    }
  });
}

double ArmIO::velocity_scale() const {
  return std::clamp(node_.get_parameter("arm_velocity_scale").as_double(), 0.01, 1.0);
}

void ArmIO::move_arm(const geometry_msgs::msg::PoseStamped & target, Cb done, double yaw, int tries_left) {
  if (settle_) { settle([this, target, done, yaw, tries_left]() { move_arm(target, done, yaw, tries_left); }); return; }
  if (tries_left < 0) tries_left = arm_plan_tries_ - 1;
  auto req = std::make_shared<GetMotionPlan::Request>();
  auto & r = req->motion_plan_request;
  r.group_name = "arm";
  r.start_state.is_diff = true;
  r.num_planning_attempts = 3;
  r.allowed_planning_time = 5.0;
  r.max_velocity_scaling_factor = r.max_acceleration_scaling_factor = velocity_scale();
  moveit_msgs::msg::Constraints goal;
  moveit_msgs::msg::PositionConstraint pc;
  pc.header = target.header;
  // 목표 = 손가락 끝(end_effector_link, gripper_link에서 0.109m) — 퍽 위치에 손목을 대면 손가락이 바닥을 뚫는다
  pc.link_name = "end_effector_link";
  shape_msgs::msg::SolidPrimitive box;
  box.type = box.BOX;
  box.dimensions = {0.02, 0.02, 0.02};
  pc.constraint_region.primitives.push_back(box);
  pc.constraint_region.primitive_poses.push_back(target.pose);
  pc.weight = 1.0;
  moveit_msgs::msg::OrientationConstraint oc;
  oc.header = target.header;
  oc.link_name = "end_effector_link";
  // 방향 = 늘 위에서 아래로 (손 z축 → 지면 아래, x축 180° 회전). 수직축 둘레는 joint1·joint5 조합이라 풀어 둠 (z = π)
  oc.orientation.x = 1.0;
  oc.orientation.w = 0.0;
  oc.absolute_x_axis_tolerance = 0.15;
  oc.absolute_y_axis_tolerance = 0.15;
  oc.absolute_z_axis_tolerance = M_PI;
  if (!std::isnan(yaw)) {  // 손 방향 고정: Rz(yaw) · Rx(π) = (cos(yaw/2), sin(yaw/2), 0, 0)
    oc.orientation.x = std::cos(yaw / 2);
    oc.orientation.y = std::sin(yaw / 2);
    oc.absolute_z_axis_tolerance = 0.15;
  }
  oc.weight = 1.0;
  goal.position_constraints.push_back(pc);
  goal.orientation_constraints.push_back(oc);
  r.goal_constraints.push_back(goal);
  const uint64_t gen = gen_;
  plan_->async_send_request(req, [this, done, target, tries_left, gen, yaw](rclcpp::Client<GetMotionPlan>::SharedFuture f) {
    if (gen != gen_) return;  // 그새 취소됨
    const auto res = f.get();
    if (res->motion_plan_response.error_code.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
      RCLCPP_WARN(node_.get_logger(), "플랜 실패 (%d) — 남은 재시도 %d", res->motion_plan_response.error_code.val, tries_left);
      if (tries_left > 0 && clear_if_start_in_octomap(res->motion_plan_response.error_code.val,
                                                      [this, target, done, yaw, tries_left]() { move_arm(target, done, yaw, tries_left - 1); }))
        return;
      if (tries_left > 0) move_arm(target, done, yaw, tries_left - 1);
      else done(false);
      return;
    }
    execute(res->motion_plan_response.trajectory, done);
  });
}

void ArmIO::move_joints(const std::vector<double> & q5, Cb done, bool retry, bool escape) {
  if (q5.size() != 5) { done(false); return; }
  if (settle_) { settle([this, q5, done, escape]() { move_joints(q5, done, true, escape); }); return; }
  auto req = std::make_shared<GetMotionPlan::Request>();
  auto & r = req->motion_plan_request;
  r.group_name = "arm";
  r.start_state.is_diff = true;
  r.num_planning_attempts = 3;
  r.allowed_planning_time = 5.0;
  r.max_velocity_scaling_factor = r.max_acceleration_scaling_factor = velocity_scale();
  moveit_msgs::msg::Constraints c;
  for (int i = 0; i < 5; ++i) {
    moveit_msgs::msg::JointConstraint jc;
    jc.joint_name = ARM[i];
    jc.position = q5[i];
    jc.tolerance_above = jc.tolerance_below = 0.01;
    jc.weight = 1.0;
    c.joint_constraints.push_back(jc);
  }
  r.goal_constraints.push_back(c);
  const uint64_t gen = gen_;
  plan_->async_send_request(req, [this, q5, done, gen, retry, escape](rclcpp::Client<GetMotionPlan>::SharedFuture f) {
    if (gen != gen_) return;
    const auto res = f.get();
    const int code = res->motion_plan_response.error_code.val;
    if (code != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
      RCLCPP_WARN(node_.get_logger(), "관절 이동 플랜 실패 (%d)", code);
      if (retry && clear_if_start_in_octomap(code, [this, q5, done, escape]() { move_joints(q5, done, false, escape); }))
        return;
      // octomap을 비워도 거부 — 시작 자세가 로봇 자기 충돌(예: 손가락이 적재함 칸 링에 걸림)일 수 있음 → 곧게 올려 빠져나온 뒤 한 번 다시
      if (escape && start_in_collision(code)) {
        lift_out([this, q5, done](bool ok) {
          if (!ok) { done(false); return; }
          move_joints(q5, done, false, false);
        });
        return;
      }
      done(false);
      return;
    }
    // 실행 거부: 직전 동작·처짐 보정(stow 쉬는 자세에서 풂)으로 팔이 아직 움직이면 'start point deviates' (11/09 Isaac: 첫 라운드
    //   stow → inspect look 자세가 joint4로 거부 → inspect_failed). 닿아서 실패(contact_)면 다시 안 함
    execute(res->motion_plan_response.trajectory, [this, q5, done, retry, escape](bool ok) {
      if (ok || !retry || contact_) { done(ok); return; }
      RCLCPP_WARN(node_.get_logger(), "관절 이동 실행 실패 — 팔이 멈춘 뒤 한 번 다시");
      settle_ = true;
      move_joints(q5, done, false, escape);
    });
  });
}

// 계획이 시작 자세 충돌로 거부되면 octomap을 비우고 바로 한 번 다시 계획 (11-21 Isaac 전체 미션)
//   pick 실패(grasp_empty) 뒤나 손에 퍽을 든 채(퍽이 MoveIt에 붙인 물체가 아니라 octomap 칸으로 찍힘) octomap 칸이
//   gripper_cam_link · finger_left_link와 닿은 채 남아 inspect · place · stow가 모두 CheckStartStateCollision으로 거부 → 연쇄 실패.
//   거부 코드는 START_STATE_IN_COLLISION이 아니라 FAILURE(99999)로 옴 → 둘 다. 비운 바로 뒤(다음 점구름 전)에 계획 — pick 시작과 같은 방식
bool ArmIO::start_in_collision(int code) {
  using E = moveit_msgs::msg::MoveItErrorCodes;
  return code == E::START_STATE_IN_COLLISION || code == E::FAILURE;  // CheckStartStateCollision 어댑터 거부는 99999(FAILURE)로 옴
}

void ArmIO::lift_out(Cb done) {
  const double h = node_.get_parameter("start_collision_lift_m").as_double();
  Q5 now, up;
  if (h <= 0.0 || !arm_now(now) || !stand::raise_hand(now, h, up)) {
    RCLCPP_WARN(node_.get_logger(), "시작 자세 충돌 — 곧게 올리기 못 함 (h %.3f m)", h);
    done(false);
    return;
  }
  if (!validity_->service_is_ready()) { done(false); return; }
  // FAILURE(99999)는 다른 이유일 수도 있음 — 지금 자세가 정말 로봇 자기 충돌일 때만 올림
  auto req = std::make_shared<GetStateValidity::Request>();
  req->group_name = "arm";
  req->robot_state.is_diff = true;
  req->robot_state.joint_state.name.assign(std::begin(ARM), std::end(ARM));
  req->robot_state.joint_state.position.assign(now.begin(), now.end());
  const uint64_t gen = gen_;
  validity_->async_send_request(req, [this, gen, h, up, done](rclcpp::Client<GetStateValidity>::SharedFuture f) {
    if (gen != gen_) return;
    using C = moveit_msgs::msg::ContactInformation;
    const auto self = [](uint32_t t) { return t == C::ROBOT_LINK || t == C::ROBOT_ATTACHED; };
    const auto & cs = f.get()->contacts;
    if (std::none_of(cs.begin(), cs.end(), [&](const auto & c) { return self(c.body_type_1) && self(c.body_type_2); })) {
      done(false);
      return;
    }
    RCLCPP_WARN(node_.get_logger(), "시작 자세 자기 충돌 — 손끝 %.0f cm 곧게 올린 뒤 다시 계획", 100 * h);
    traj_step([this, up](const Q5 & from) { return stand::sample_move(from, up, velocity_scale(), TRAJ_DT); })(done);
  });
}

bool ArmIO::clear_if_start_in_octomap(int code, std::function<void()> retry) {
  if (!start_in_collision(code) || !clear_octomap_->service_is_ready()) return false;
  RCLCPP_WARN(node_.get_logger(), "계획 실패 (%d) — octomap 비우고 바로 다시 계획 (시작 자세가 octomap과 닿았을 수 있음)", code);
  const uint64_t gen = gen_;
  clear_octomap_->async_send_request(std::make_shared<std_srvs::srv::Empty::Request>(),
    [this, gen, retry](rclcpp::Client<std_srvs::srv::Empty>::SharedFuture) {
      if (gen == gen_) retry();
    });
  return true;
}

double ArmIO::grip_current_a() const { return node_.get_parameter("grip_current_a").as_double(); }

double ArmIO::grip_close_m() const {
  return std::max(stand::gap_joint(node_.get_parameter("grip_close_gap_m").as_double()), GRIPPER_CLOSED_M);
}

void ArmIO::set_gripper(double width_m, Cb done, double effort_a) {
  Gripper::Goal goal;
  goal.command.position = {width_m};
  // 쥐는 힘 [A]: goal effort → gripper_controller(max_effort_interface) → 펌웨어 Goal Current (0.00269~1.2 A로 자름)
  if (effort_a > 0.0) goal.command.effort = {effort_a};
  const uint64_t gen = gen_;
  rclcpp_action::Client<Gripper>::SendGoalOptions opt;
  opt.result_callback = [this, done, width_m, gen](const rclcpp_action::ClientGoalHandle<Gripper>::WrappedResult & r) {
    if (gen != gen_) return;
    const bool ok = r.code == rclcpp_action::ResultCode::SUCCEEDED;
    if (!ok) RCLCPP_WARN(node_.get_logger(), "그리퍼 %.3fm 실패", width_m);
    done(ok);
  };
  gripper_->async_send_goal(goal, opt);
}

void ArmIO::run_steps(std::shared_ptr<std::vector<Step>> steps, size_t i, Cb done) {
  if (i >= steps->size()) { done(true); return; }
  (*steps)[i]([this, steps, i, done](bool ok) {
    if (!ok) { done(false); return; }
    run_steps(steps, i + 1, done);
  });
}

ArmIO::Step ArmIO::joints_step(const std::vector<double> & q5) {
  return [this, q5](Cb cb) { move_joints(q5, cb); };
}

ArmIO::Step ArmIO::gripper_step(double width_m, double effort_a) {
  return [this, width_m, effort_a](Cb cb) { set_gripper(width_m, cb, effort_a); };
}

ArmIO::Step ArmIO::held_step() {
  return [this](Cb cb) {
    if (!grasp_check_) { cb(true); return; }
    double pos = 0.0, eff = std::nan("");
    state_cache_->joint("gripper_left_joint", pos, eff);
    // 쥠 = 손가락이 닫는 목표 앞에서 멈춤 (약한 전류) 또는 전류가 하한 이상 (센 전류면 퍽이 눌려 목표 근처까지 들어감 —
    //   11/08 실기 0.35 A: 30 mm 퍽에서 4.7 mm까지 · 0.34 A, 목표 4.2 mm). 빈손은 목표까지 가고 전류 ~0
    const bool stopped = grasp_held(pos, eff, std::max(grasp_min_pos_m_, grip_close_m() + node_.get_parameter("grasp_stop_margin_m").as_double()), 0.0);
    // 전류 기준은 쥐는 전류 상한의 절반까지 낮춤 — grasp_min_effort 0.25 A는 0.35 A로 쥘 때 값. 0.1 A로 쥐면 쥔 퍽도 0.09 ~ 0.10 A라
    // 위치 기준(목표 + 0.5 mm, 0.1 mm 차)만 남아 쥔 퍽을 빈손으로 판정 (11/08 실기 unload 칸 2: 0.0047 m · 0.09 A). 빈손은 목표까지 가고 전류 ~0
    const bool ok = stopped || grasp_held(pos, eff, grasp_min_pos_m_, std::min(grasp_min_effort_, 0.5 * grip_current_a()));
    RCLCPP_INFO(node_.get_logger(), "잡기 판정 %s — 그리퍼 %.4fm · %.2fA", ok ? "쥠" : "빈손", pos, eff);
    grasp_empty_ = !ok;
    cb(ok);
  };
}

ArmIO::Step ArmIO::traj_step(std::function<std::vector<Q5>(const Q5 &)> make) {
  return [this, make](Cb cb) {
    if (settle_) { settle([this, make, cb]() { traj_step(make)(cb); }); return; }
    Q5 now;
    if (!arm_now(now)) { cb(false); return; }
    const auto samples = make(now);
    moveit_msgs::msg::RobotTrajectory traj;
    auto & jt = traj.joint_trajectory;
    jt.joint_names.assign(std::begin(ARM), std::end(ARM));
    const auto point = [](const Q5 & q, double t) {
      trajectory_msgs::msg::JointTrajectoryPoint p;
      p.positions.assign(q.begin(), q.end());
      p.time_from_start = rclcpp::Duration::from_seconds(t);
      return p;
    };
    jt.points.push_back(point(now, 0.0));
    for (size_t k = 0; k < samples.size(); ++k) jt.points.push_back(point(samples[k], (k + 1) * TRAJ_DT));
    // traj_step은 MoveIt 계획을 거치지 않음 — 보내기 전에 샘플마다 자기 충돌만 검사 (11-20 실기: load 실패 뒤 place가 적재함 쪽에서
    //   상자로 되돌리며 joint5 −3.10 해로 가 gripper_cam_link가 shoulder_link를 갈며 감 · MoveIt은 다음 계획에서야 알아챔)
    check_self_collision(samples, [this, traj, cb](bool ok) {
      if (!ok) { cb(false); return; }
      execute(traj, cb);
    });
  };
}

// 궤적 자기 충돌 검사 한 샘플 — 설명은 arm_io.hpp
bool contacts_ok(std::set<ContactPair> & start, const std::set<ContactPair> & now, bool first, bool last, ContactPair & bad) {
  if (first && !last) { start = now; return true; }
  for (const auto & p : now)
    if (last || !start.count(p)) { bad = p; return false; }  // 새 쌍 · 다시 닿음 · 끝 자세에서도 닿음
  for (auto it = start.begin(); it != start.end();)  // 떨어진 쌍은 더는 허용 안 함
    it = now.count(*it) ? std::next(it) : start.erase(it);
  return true;
}

// /check_state_validity로 자세를 하나씩 차례로 검사 — 로봇 링크끼리(붙인 물체 포함) 닿는 자세가 하나라도 있으면 실패. world(octomap·퍽·
//   적재함)와 닿는 건 무시 — 닿는 구간은 일부러 퍽·칸에 닿음. 서비스가 없거나 시간 안에 답이 없으면 실패 (안전 쪽).
//   샘플(0.02 s)을 한꺼번에 보내면 서비스 큐(깊이 10)를 넘은 요청이 버려져 답이 끝내 안 옴 → pick 닿는 구간에서 멈춤 (11-20 실기 2판째).
//   그래서 관절이 traj_check_step_rad 넘게 움직인 샘플만 골라(끝 자세는 늘) 하나씩 보냄
void ArmIO::check_self_collision(const std::vector<Q5> & samples, Cb done) {
  if (samples.empty() || !node_.get_parameter("traj_collision_check").as_bool()) { done(true); return; }
  if (!validity_->service_is_ready()) {
    RCLCPP_ERROR(node_.get_logger(), "궤적 충돌 검사: /check_state_validity 없음 — 궤적 안 보냄");
    done(false);
    return;
  }
  const double step = node_.get_parameter("traj_check_step_rad").as_double();
  auto picks = std::make_shared<std::vector<size_t>>();
  for (size_t k = 0; k < samples.size(); ++k) {
    bool far = picks->empty() || k + 1 == samples.size();
    for (int i = 0; i < 5 && !far; ++i) far = std::abs(samples[k][i] - samples[picks->back()][i]) > step;
    if (far) picks->push_back(k);
  }
  // 첫 샘플(≈ 지금 자세)에 이미 닿은 쌍은 빠져나가는 동안 허용 (11-21 실기: load 막힘 자세에서 '위로'·place 궤적이 샘플 1에서 거부돼
  //   퍽을 쥔 채 갇힘) — 한 번 떨어진 뒤 다시 닿거나 · 새 쌍이 닿거나 · 끝 자세에서도 닿으면 실패
  using Pair = ContactPair;
  struct State { size_t i = 0; bool over = false; rclcpp::TimerBase::SharedPtr timer; std::set<Pair> start; };
  auto st = std::make_shared<State>();
  const uint64_t gen = gen_;
  const auto finish = [this, st, gen, done](bool ok) {
    if (st->over) return;
    st->over = true;
    // 타이머를 st에서 빼서 놓음 — 타이머 콜백이 st를 잡고 있어 그대로 두면 순환 참조 (실행 중인 콜백은 executor가 잡고 있어 안전)
    if (auto t = std::move(st->timer)) t->cancel();
    if (gen == gen_) done(ok);
  };
  const double limit = node_.get_parameter("traj_check_timeout_s").as_double();
  st->timer = node_.create_wall_timer(std::chrono::duration<double>(limit), [this, st, picks, limit, finish]() {
    if (st->over) return;
    RCLCPP_ERROR(node_.get_logger(), "궤적 충돌 검사: %.1f s 안에 답 없음 (%zu/%zu) — 궤적 안 보냄", limit, st->i, picks->size());
    validity_->prune_pending_requests();
    finish(false);
  });
  const auto self = [](uint32_t t) {
    return t == moveit_msgs::msg::ContactInformation::ROBOT_LINK || t == moveit_msgs::msg::ContactInformation::ROBOT_ATTACHED;
  };
  auto next = std::make_shared<std::function<void()>>();
  auto qs = std::make_shared<std::vector<Q5>>(samples);
  // 자기 자신은 weak로 — 진행 중인 요청의 답 콜백만 강하게 잡음 (순환 참조로 남지 않게)
  *next = [this, st, picks, qs, wnext = std::weak_ptr<std::function<void()>>(next), finish, self]() {
    if (st->over) return;
    auto next = wnext.lock();
    if (!next) return;
    if (st->i >= picks->size()) { finish(true); return; }
    const size_t k = (*picks)[st->i];
    auto req = std::make_shared<GetStateValidity::Request>();
    req->group_name = "arm";
    req->robot_state.is_diff = true;
    req->robot_state.joint_state.name.assign(std::begin(ARM), std::end(ARM));
    req->robot_state.joint_state.position.assign((*qs)[k].begin(), (*qs)[k].end());
    validity_->async_send_request(req, [this, st, k, n = qs->size(), next, finish, self](rclcpp::Client<GetStateValidity>::SharedFuture f) {
      if (st->over) return;
      std::set<Pair> now;
      for (const auto & c : f.get()->contacts)
        if (self(c.body_type_1) && self(c.body_type_2))
          now.insert(std::minmax(c.contact_body_1, c.contact_body_2));
      if (st->i == 0)
        for (const auto & p : now)
          if (k + 1 < n)
            RCLCPP_WARN(node_.get_logger(), "궤적 충돌 검사: 시작 자세가 이미 %s - %s 닿음 — 빠져나가는 동안 허용", p.first.c_str(), p.second.c_str());
      Pair bad;
      if (!contacts_ok(st->start, now, st->i == 0, k + 1 == n, bad)) {
        RCLCPP_ERROR(node_.get_logger(), "궤적 충돌 검사: 샘플 %zu/%zu에서 %s - %s 닿음 — 궤적 안 보냄", k + 1, n,
                     bad.first.c_str(), bad.second.c_str());
        finish(false);
        return;
      }
      ++st->i;
      (*next)();
    });
  };
  (*next)();
}

ArmIO::Step ArmIO::delta_settle_step(double min_s, double max_s) {
  return [this, min_s, max_s](Cb cb) {
    const uint64_t gen = gen_;
    const double t0 = node_.now().seconds();
    timer_ = node_.create_timer(std::chrono::duration<double>(0.05), [this, cb, gen, t0, min_s, max_s]() {
      if (gen != gen_) { timer_->cancel(); return; }
      const double now = node_.now().seconds(), t = now - t0;
      if (t < min_s) return;
      // 최근 DELTA_WINDOW_S 안의 Δ가 2개 넘게 있고 관절마다 흩어짐이 DELTA_SPREAD 안이면 멈춤 · debug가 안 오면(정책 없음) 기다릴 것 없음
      const bool still = state_cache_->delta_stable(now, 2 * DELTA_WINDOW_S, DELTA_WINDOW_S, DELTA_SPREAD, 2);
      if (!still && t < max_s) return;
      timer_->cancel();
      if (t > min_s + 0.06)
        RCLCPP_INFO(node_.get_logger(), "닿는 구간: 보정 Δ가 멈출 때까지 %.2f s 더 기다림%s", t - min_s, still ? "" : " (상한 — 아직 움직이는 채로 쥠)");
      cb(true);
    });
  };
}

ArmIO::Step ArmIO::grip_timed_step(double joint_m, double effort_a, double secs) {
  return [this, joint_m, effort_a, secs](Cb cb) {
    Gripper::Goal goal;
    goal.command.position = {joint_m};
    if (effort_a > 0.0) goal.command.effort = {effort_a};
    gripper_->async_send_goal(goal);
    wait_step(secs)(cb);
  };
}

ArmIO::Step ArmIO::wait_step(double secs) {
  return [this, secs](Cb cb) {
    const uint64_t gen = gen_;
    // 노드 시계(Isaac에선 시뮬 시간) — wall이면 시뮬이 느릴 때(0.7배) 멈춤·안정 시간이 짧아져 팔이 흔들리는 채로 쥠
    //   (11/09 Isaac: 닫기 직전 0.5 s 동안 j2·j3 중앙값 37·58 mrad 움직임 · 실기 7·3 mrad)
    timer_ = node_.create_timer(std::chrono::duration<double>(secs), [this, cb, gen]() {
      timer_->cancel();
      if (gen == gen_) cb(true);
    });
  };
}

ArmIO::Step ArmIO::grip_stop_step(double timeout_s) {
  return [this, timeout_s](Cb cb) {
    const uint64_t gen = gen_;
    const auto t0 = std::chrono::steady_clock::now();
    auto still_from = std::make_shared<std::chrono::steady_clock::time_point>(t0);
    auto last = std::make_shared<double>(-1.0);
    timer_ = node_.create_wall_timer(std::chrono::milliseconds(50), [=, this]() {
      if (gen != gen_) { timer_->cancel(); return; }
      const auto now = std::chrono::steady_clock::now();
      double position = 0.0;
      if (state_cache_->position("gripper_left_joint", position) &&
          (*last < 0 || std::abs(position - *last) > 0.0002)) { *still_from = now; *last = position; }
      if (std::chrono::duration<double>(now - *still_from).count() < 0.2 && std::chrono::duration<double>(now - t0).count() < timeout_s) return;
      timer_->cancel();
      cb(true);
    });
  };
}

ArmIO::Step ArmIO::grip_settle_step(double min_current, double empty_m, double timeout_s) {
  return [this, min_current, empty_m, timeout_s](Cb cb) {
    if (!grasp_check_) { cb(true); return; }
    const uint64_t gen = gen_;
    const auto t0 = node_.now();
    auto still_from = std::make_shared<rclcpp::Time>(t0);
    auto last = std::make_shared<double>(-1.0);
    timer_ = node_.create_wall_timer(std::chrono::milliseconds(50), [=, this]() {
      if (gen != gen_) { timer_->cancel(); return; }
      double pos = 0.0, effort = 0.0;
      if (!state_cache_->joint("gripper_left_joint", pos, effort)) return;
      const double cur = std::abs(effort);
      const auto now = node_.now();
      if (*last < 0 || std::abs(pos - *last) > 0.0002) *still_from = now;  // 움직이는 중
      *last = pos;
      const int r = grip_state(pos, cur, (now - *still_from).seconds(), (now - t0).seconds(), min_current, empty_m, timeout_s);
      if (r == 2) return;
      timer_->cancel();
      RCLCPP_INFO(node_.get_logger(), "그리퍼 %s — 위치 %.4f m · 전류 %.1f mA", r == 1 ? "쥠" : r == 0 ? "빈손" : "판단 못 함", pos, 1000 * cur);
      if (r != 1) grip_fail_ = r == 0 ? "stand_no_grip" : "stand_grip_timeout";
      cb(r == 1);
    });
  };
}

void ArmIO::skill_ctx(const SkillModel * policy, uint8_t phase, bool holding, bool live, const Q5 & above, const Q5 & down) {
  ml::msg::SkillContext m;
  m.header.stamp = node_.now();  // 이 단계 시작 시각 (정책 입력 '진행'의 기준)
  m.run_id = run_;
  if (policy) {
    m.skill = policy->name;
    if (policy->use) { m.policy = policy->policy; m.layout = policy->layout; }  // 안 쓰면 "" → rl_arm_correction Δ = 0
    m.phase = phase;
    m.holding = holding;
    m.residual_live = live;
    m.q_above.assign(above.begin(), above.end());
    m.q_down.assign(down.begin(), down.end());
  }
  ctx_pub_->publish(m);
}

void ArmIO::ctx_off() { skill_ctx(nullptr, 0, false, false, {}, {}); }

void ArmIO::hand_target(const std::string & kind, int puck_tag, int slot, double x, double y, double z) {
  perception::msg::HandTarget t;
  t.kind = kind;
  t.puck_tag = puck_tag;
  t.box_tag = -1;  // ponytail: 상자 밀림 보정은 상자 태그·계획 상자 위치를 넘길 때 (mission_manager가 아는 값)
  t.point.x = x;
  t.point.y = y;
  t.point.z = z;
  t.slot = slot;
  hand_pub_->publish(t);
}

void ArmIO::hand_target_end(int puck_tag, double px, double py, double pz, double cx, double cy, double cz) {
  perception::msg::HandTarget t;
  t.kind = "puck";
  t.puck_tag = puck_tag;
  t.box_tag = -2;
  t.point.x = px;
  t.point.y = py;
  t.point.z = pz;
  t.box_point.x = cx;
  t.box_point.y = cy;
  t.box_point.z = cz;
  t.slot = -1;
  hand_pub_->publish(t);
}

void ArmIO::reset_sag() {
  if (reset_sag_->service_is_ready()) {
    reset_sag_->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
    settle_ = true;
  } else if (!sag_warned_) {
    sag_warned_ = true;
    RCLCPP_WARN(node_.get_logger(), "/rl_arm_correction/reset_sag 없음 — 처짐 보정 초기화 없이 계속 (rl_arm_correction 확인)");
  }
}

void ArmIO::fade_sag() {
  // 닿는 구간이 있을 때만 (contact_mode moveit이면 fade가 끝나지 않아 처짐 보정이 꺼진 채로 남음)
  if (sim_contact() && node_.get_parameter("sag_fade").as_bool() && fade_sag_->service_is_ready()) fade_sag_->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
}

bool ArmIO::ik_ee(const geometry_msgs::msg::PoseStamped & ee, const Q5 & near, Q5 & out) {
  geometry_msgs::msg::TransformStamped tf;
  try {
    tf = state_cache_->tf().lookupTransform("base_footprint", ee.header.frame_id, tf2::TimePointZero);
  } catch (const std::exception &) {
    return false;
  }
  const auto & r = tf.transform.rotation;
  const auto & tr = tf.transform.translation;
  const tf2::Transform t(tf2::Quaternion(r.x, r.y, r.z, r.w), tf2::Vector3(tr.x, tr.y, tr.z));
  const tf2::Vector3 p = t * tf2::Vector3(ee.pose.position.x, ee.pose.position.y, ee.pose.position.z);
  stand::V3 axis{};
  stand::fk(near, &axis);  // 손가락이 닫히는 방향 — 지금 손목 그대로
  return stand::ik_down({p.x(), p.y(), p.z()}, std::atan2(axis[1], axis[0]), near, out);
}

bool ArmIO::ik_ee_yaw(const geometry_msgs::msg::PoseStamped & ee, double psi, const Q5 & near, Q5 & out) {
  geometry_msgs::msg::TransformStamped tf;
  try {
    tf = state_cache_->tf().lookupTransform("base_footprint", ee.header.frame_id, tf2::TimePointZero);
  } catch (const std::exception &) {
    return false;
  }
  const auto & r = tf.transform.rotation;
  const auto & tr = tf.transform.translation;
  const tf2::Transform t(tf2::Quaternion(r.x, r.y, r.z, r.w), tf2::Vector3(tr.x, tr.y, tr.z));
  const tf2::Vector3 p = t * tf2::Vector3(ee.pose.position.x, ee.pose.position.y, ee.pose.position.z);
  const double base_yaw = std::atan2(2 * (r.w * r.z + r.x * r.y), 1 - 2 * (r.y * r.y + r.z * r.z));
  // 손가락은 좌우 대칭이라 닫는 축 ±π 두 해 — near(지금 자세)에 가까운 해 대신 |joint5|가 작은 해 (손 카메라가 몸 반대쪽).
  //   적재함 쪽(joint5 ≈ ±π)에서 near로 고르면 앞쪽 자세에서도 손목이 반 바퀴 뒤집혀 카메라가 shoulder_link에 닿음 (11-20 실기 place)
  const stand::V3 at{p.x(), p.y(), p.z()};
  const double close_yaw = psi + base_yaw - M_PI / 2;
  Q5 a{}, b{};
  const bool ok_a = stand::ik_down(at, close_yaw, near, a), ok_b = stand::ik_down(at, close_yaw + M_PI, near, b);
  if (!ok_a && !ok_b) return false;
  out = !ok_b || (ok_a && std::abs(a[4]) <= std::abs(b[4])) ? a : b;
  return true;
}

std::array<double, 5> ArmIO::efforts() {
  std::array<double, 5> a{};
  double q;
  for (int i = 0; i < 5; ++i) state_cache_->joint(ARM[i], q, a[i]);
  return a;
}

// "전류 A j1 0.01→0.02 j2 … · 각 rad [..] · hand puck rel (+12,-3,+40) mm 0.2 s (옆 퍽 (+60,+5) mm)"
std::string ArmIO::jam_note(const std::array<double, 5> & base_a) {
  char b[512];
  int n = std::snprintf(b, sizeof b, "전류 A");
  for (int i = 0; i < 5; ++i) {
    double q = 0, e = 0;
    state_cache_->joint(ARM[i], q, e);
    n += std::snprintf(b + n, sizeof b - n, " j%d %+.2f→%+.2f", i + 1, base_a[i], e);
  }
  n += std::snprintf(b + n, sizeof b - n, " · 각 rad [");
  for (int i = 0; i < 5; ++i) {
    double q = 0;
    state_cache_->position(ARM[i], q);
    n += std::snprintf(b + n, sizeof b - n, "%s%.3f", i ? " " : "", q);
  }
  n += std::snprintf(b + n, sizeof b - n, "]");
  if (hr_ && hr_->valid) {
    n += std::snprintf(b + n, sizeof b - n, " · hand %s rel (%+.0f,%+.0f,%+.0f) mm %.1f s", hr_->kind.c_str(),
                       1000 * hr_->rel.x, 1000 * hr_->rel.y, 1000 * hr_->rel.z, hr_->age);
    if (hr_->neighbour)
      std::snprintf(b + n, sizeof b - n, " (옆 퍽 %+.0f,%+.0f mm)", 1000 * hr_->neighbour_rel.x, 1000 * hr_->neighbour_rel.y);
  } else {
    std::snprintf(b + n, sizeof b - n, " · hand %s 못 잼", hr_ ? hr_->kind.c_str() : "-");
  }
  return b;
}

double ArmIO::jam_mean(double window_s) const {
  return state_cache_->jam_mean(window_s);
}

bool ArmIO::jam_lagging(double & lag) const {
  const double lim = node_.get_parameter("jam_lag_rad").as_double();
  std::array<double, 5> l{};
  lag = std::numeric_limits<double>::quiet_NaN();
  if (lim <= 0.0 || !state_cache_->lag(node_.now().seconds(), 0.5, l)) return true;  // 꺼짐 · 값 없음 → 전류만
  lag = std::max(std::abs(l[1]), std::abs(l[2]));
  return lag > lim;
}

ArmIO::Step ArmIO::contact_step(const SkillModel & policy, std::function<bool(Q5 &, Q5 &)> plan, bool close,
                                double open_m, bool lift_on_empty, bool bottom_ok) {
  return [this, policy, plan, close, open_m, lift_on_empty, bottom_ok](Cb cb) {
    Q5 above{}, down{};
    if (!plan(above, down)) {
      RCLCPP_WARN(node_.get_logger(), "%s: 닿는 구간 자세를 못 구함 (IK·TF)", policy.name.c_str());
      cb(false);
      return;
    }
    // 빠른 이동(load carry → 칸 위, vel_scale 1.0) 끝에 팔이 아직 따라오는 중(11/08 실기: 관절 오차 최대 5° · 떨림)이면
    // 위에서 0.3 s만 멈추고 내려가 칸을 빗나감 → 팔이 멈출 때까지 기다린 뒤 닿는 구간 시작
    settle_ = true;
    settle([this, policy, plan, close, open_m, lift_on_empty, bottom_ok, above, down, cb]() {
    ++run_;
    const auto p = std::make_shared<SkillModel>(policy);
    const auto ctx = [this, p, above, down](uint8_t phase, bool holding, bool live) -> Step {
      return [this, p, phase, holding, live, above, down](Cb c) { skill_ctx(p.get(), phase, holding, live, above, down); c(true); };
    };
    const bool bottom_live = node_.get_parameter("contact_bottom_live").as_bool();
    const auto move = [this](Q5 from, Q5 to) {  // 시뮬처럼 from→to 부드러운 보간 (팀 한계 × 0.3)
      return traj_step([from, to](const Q5 &) { return stand::sample_move(from, to, CONTACT_VS, TRAJ_DT); });
    };
    // 막힘 판정 (jam_current_a): 위에서 멈춘 마지막 0.3 s의 joint2+joint3 전류 → 쥐기·놓기 직전 0.3 s와 비교
    const auto base = std::make_shared<double>(0.0);
    const auto base_a = std::make_shared<std::array<double, 5>>();  // 관절별 전류 (기록용)
    const Step jam_base = [this, base, base_a](Cb c) { *base = jam_mean(0.3); *base_a = efforts(); c(true); };
    const Step jam_check = [this, base, base_a, bottom_ok, name = policy.name](Cb c) {
      const double lim = node_.get_parameter("jam_current_a").as_double(), d = jam_mean(0.3) - *base;
      if (lim <= 0.0 || d <= lim) { c(true); return; }
      if (bottom_ok) {  // 11-22 실기 place: 면에 닿은 전류(+0.57 · +0.64 A)를 막힘으로 보고 쥔 채 올라감
        RCLCPP_WARN(node_.get_logger(), "%s: 바닥 전류 %+.2f A (> %.2f) — 면에 닿음, 그대로 놓음", name.c_str(), d, lim);
        c(true);
        return;
      }
      double lag;
      if (!jam_lagging(lag)) {  // 전류만 오름 — 바닥에 닿은 것으로 보고 그대로 놓기·쥐기
        RCLCPP_WARN(node_.get_logger(), "%s: 바닥 전류 %+.2f A (> %.2f)지만 joint2·3 추종 %.1f mrad — 막힘 아님, 계속", name.c_str(), d, lim, 1000 * lag);
        c(true);
        return;
      }
      jam_ = true;
      const std::string note = jam_note(*base_a);
      RCLCPP_WARN(node_.get_logger(), "%s: 막힘 (바닥) — joint2+3 전류 %+.2f A (> %.2f) · 추종 %.1f mrad · 쥐지·놓지 않고 위로 · %s", name.c_str(), d, lim, 1000 * lag, note.c_str());
      if (auto * s = dynamic_cast<SkillNode *>(&node_)) s->detail("막힘 바닥 " + name + " Δ" + std::to_string(d).substr(0, 5) + " A · " + note);
      c(false);  // 위로 빼기는 끝 callback에서
    };
    // 내려가는 도중 막힘: 같은 기준을 50 ms마다 보다 넘으면 궤적을 취소 (막힌 채 바닥까지 밀지 않게)
    const Step descend = [this, base, base_a, mv = move(above, down), name = policy.name](Cb c) {
      const uint64_t gen = gen_;
      jam_timer_ = node_.create_timer(std::chrono::milliseconds(50), [this, base, base_a, gen, name]() {
        if (gen != gen_) { jam_timer_->cancel(); return; }
        const double lim = node_.get_parameter("jam_current_a").as_double(), d = jam_mean(0.3) - *base;
        if (lim <= 0.0 || d <= lim) return;
        double lag;
        if (!jam_lagging(lag)) return;  // 전류만 오르고 위치는 따라옴 — 계속 내려감 (끝에서 바닥 판정)
        jam_timer_->cancel();
        jam_ = true;
        const std::string note = jam_note(*base_a);
        RCLCPP_WARN(node_.get_logger(), "%s: 막힘 (내려가는 중) — joint2+3 전류 %+.2f A (> %.2f) · 추종 %.1f mrad · 멈추고 위로 · %s", name.c_str(), d, lim, 1000 * lag, note.c_str());
        if (auto * s = dynamic_cast<SkillNode *>(&node_)) s->detail("막힘 내려가는 중 " + name + " Δ" + std::to_string(d).substr(0, 5) + " A · " + note);
        exec_->async_cancel_all_goals();
      });
      mv([this, c](bool ok) {
        if (jam_timer_) jam_timer_->cancel();
        c(ok && !jam_);
      });
    };
    auto steps = std::make_shared<std::vector<Step>>(std::vector<Step>{
      ctx(0, true, true), wait_step(CONTACT_HOLD_S), jam_base,  // 0 위에서 멈춤 (처짐 적분) · 막힘 기준 전류
      ctx(1, false, true), descend,                       // 1 내려가기 (막히면 멈춤)
      ctx(2, true, bottom_live), delta_settle_step(CONTACT_SETTLE_S, CONTACT_SETTLE_MAX_S),  // 2 안정 (+ Δ가 멈출 때까지 · contact_bottom_live false면 고정)
      jam_check,                                          // 막혔으면 위로 빼고 실패 (arm_jam)
      ctx(3, true, false),                                // 3 쥐기 · 놓기 (보정 고정)
      close ? grip_timed_step(grip_close_m(), grip_current_a(), CONTACT_CLOSE_S)
            : grip_timed_step(open_m, -1.0, CONTACT_RELEASE_S)});
    if (close) {
      steps->push_back(grip_stop_step(CONTACT_STOP_S));
      if (lift_on_empty) {  // 빈손이어도 칸 위로 올린 뒤 실패를 알림 (내려간 채 끝나면 다음 이동이 칸을 긁으며 감 — 11/08 실기 unload)
        auto judged = held_step();
        steps->push_back([judged](Cb c) { judged([c](bool) { c(true); }); });
      } else {
        steps->push_back(held_step());
      }
    }
    steps->push_back(ctx(4, false, false));               // 4 올리기
    steps->push_back(move(down, above));
    if (close && !lift_on_empty) steps->push_back(held_step());  // 들어 올린 뒤 한 번 더 — 올리다 놓치면 빈손 (11-15 stand_pick 빈손)
    // 5 대기 (시뮬 판정 구간) — 실기에선 안 기다림: 보정이 고정돼(live false) 하는 일이 없고 다음 이동만 늦춤 (11/08 사용자 — pick 0.3 · load 0.5 s)
    steps->push_back(ctx(5, !close, false));
    RCLCPP_INFO(node_.get_logger(), "%s: 닿는 구간 (시뮬 기준 동작 + %s) run %u", policy.name.c_str(),
                policy.use ? "학습 보정" : "보정 없음", run_);
    run_steps(steps, 0, [this, cb, close, use = policy.use, above, ctx](bool ok) {
      if (jam_) {  // 막힘 (내려가는 중 · 바닥): 지금 자리에서 위로 뺀 뒤 실패 (arm_jam)
        ctx(4, false, false)([](bool) {});
        traj_step([above](const Q5 & now) { return stand::sample_move(now, above, CONTACT_VS, TRAJ_DT); })(
          [this, cb](bool) {
            ctx_off();
            hand_target("none", -1, -1);
            settle_ = true;
            cb(false);
          });
        return;
      }
      ok = ok && !(close && grasp_empty_);
      ctx_off();  // 보정 끔
      hand_target("none", -1, -1);
      // 보정 Δ가 0으로 빠지면 팔이 0.1 s쯤 움직임 (11/08 실기 load: joint1 −0.035 · joint5 +0.06 rad) — 바로 MoveIt을 부르면
      // 출발 자세가 0.01 rad 넘게 어긋나 실행 거부 (Invalid Trajectory: start point deviates) → 다음 동작은 멈춘 뒤 (settle).
      // 정책이 꺼져 있으면 Δ가 늘 0이라 빠질 게 없음 — 바로 다음 동작
      if (use) settle_ = true;
      cb(ok);
    });
    });
  };
}

}  // namespace skills
