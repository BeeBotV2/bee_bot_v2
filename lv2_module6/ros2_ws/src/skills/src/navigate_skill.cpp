// skill_navigate — /skills/navigate: goal.pose(map)로 Nav2 NavigateToPose 한 번 (지도 위 먼 이동, 장애물 피함)
//   목표 자세 계산(물체 앞 · 좌표)은 coordinator가 함. 경로는 Nav2(Lattice, 몸 모양 그대로)가 정하고 RPP가 따라감 — 여기서는 경유점을
//   끼우지 않는다 (11-18 재설계: 문 경유점 · 길목 · 경유점 지도 · 가르친 경로 · 막은 구역을 걷어냄. 몸을 원으로 보던 예전 플래너가
//   문 모서리를 질러서 필요했던 것 — isaac/mission/nav_bench.py 시뮬에서 경유점 없이 8구간 · 동적 장애물 4번 모두 성공, 접촉 0)
//   진척 감시: 남은 경로 거리가 nav_stall_s 동안 nav_stall_gain_m만큼도 안 줄면 취소하고 nav_no_progress (ProgressWatch)
//   goal.target "look" (상자 도킹 앞 주행): 남은 거리가 nav_look_m 안이면 팔을 pose_look으로 (approach가 바로 상자를 봄)
//   goal.target "near" (상자 앞 다가설 자리): 멈추거나 Nav2가 실패해도 목표에서 nav_near_m · nav_near_yaw 안이면 성공 —
//     마지막 몇 cm·방향은 approach가 상자를 보며 맞춤
//   끼임 풀기: 진척 없음 · 출발 칸 막힘(205) · Nav2 실패(앞 충돌로 patience 초과 등)면 nav_unstuck_tries번까지 매번 다른 동작
//     (1: 후진 → 막히면 전진 · 2: 제자리 +40° → 막히면 −40° · 3: 전진 → 막히면 후진, 모두 behavior_server가 몸 모양 충돌 검사)
//     → 같은 목표로 다시. 그래도 안 되면 실패 → mission_manager가 다른 면(plan.yaml start/location_box_sides)으로
//     11-18 sim: 벽 끝·상자 옆에서 꼬리가 걸리면 앞 충돌(nav_failed)이라 예전엔 풀기 대상이 아니었고, 뒤가 막혀 후진 한 번도 실패
//   target에 "leg:<구간>"이 있어도 무시 (예전 가르친 경로 재생 키)
#include <sstream>
#include "skills/arm_io.hpp"
#include "nav2_msgs/action/back_up.hpp"
#include "nav2_msgs/msg/collision_monitor_state.hpp"
#include "nav2_msgs/action/drive_on_heading.hpp"
#include "nav2_msgs/action/spin.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "skills/progress_watch.hpp"
#include "skills/skill_node.hpp"

#include <cmath>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

namespace skills {

class NavigateSkill : public SkillNode {
public:
  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using BackUp = nav2_msgs::action::BackUp;
  using DriveOnHeading = nav2_msgs::action::DriveOnHeading;
  using Spin = nav2_msgs::action::Spin;
  using Pose = geometry_msgs::msg::PoseStamped;
  explicit NavigateSkill(const rclcpp::NodeOptions & options)
  : SkillNode("skill_navigate", options), arm_(*this) {
    nav_ = rclcpp_action::create_client<NavigateToPose>(this, "/navigate_to_pose");
    backup_ = rclcpp_action::create_client<BackUp>(this, "/backup");
    drive_ = rclcpp_action::create_client<DriveOnHeading>(this, "/drive_on_heading");
    spin_ = rclcpp_action::create_client<Spin>(this, "/spin");
    collision_state_sub_ = create_subscription<nav2_msgs::msg::CollisionMonitorState>(
      "/collision_monitor_state", 10, [this](nav2_msgs::msg::CollisionMonitorState::ConstSharedPtr msg) {
        // SLOWDOWN/APPROACH/LIMIT reduce or cap speed; only STOP says the command is blocked.
        const bool stopped = msg->action_type == nav2_msgs::msg::CollisionMonitorState::STOP;
        std::lock_guard<std::mutex> lock(collision_state_mutex_);
        collision_stopped_ = stopped;
        collision_state_received_ = std::chrono::steady_clock::now();
      });
    unstuck_tries_ = declare_parameter<int>("nav_unstuck_tries", 3);   // 0이면 끼임 풀기 안 함
    unstuck_back_m_ = declare_parameter<double>("nav_unstuck_back_m", 0.15);
    unstuck_fwd_m_ = declare_parameter<double>("nav_unstuck_fwd_m", 0.10);
    unstuck_turn_ = declare_parameter<double>("nav_unstuck_turn", 0.7);  // [rad] 약 40°
    stall_s_ = declare_parameter<double>("nav_stall_s", 15.0);        // 0이면 감시 안 함
    stall_gain_m_ = declare_parameter<double>("nav_stall_gain_m", 0.10);
    look_m_ = declare_parameter<double>("nav_look_m", 0.6);
    near_m_ = declare_parameter<double>("nav_near_m", 0.15);
    near_yaw_ = declare_parameter<double>("nav_near_yaw", 0.8);  // [rad]
    look_q_ = declare_parameter<std::vector<double>>("pose_look", {0.0, -1.682497, 0.324631, 1.926843, 0.0});  // arm.yaml이 덮음
    serve("navigate", [this](const SkillGoal & g, Done done) {
      when_ready([this]() { return nav_->action_server_is_ready(); }, "nav_unavailable", done, [this, g, done]() {
        Pose goal = g.pose;
        if (goal.header.frame_id.empty()) goal.header.frame_id = "map";
        std::set<std::string> flags;
        std::istringstream in(g.target);
        for (std::string t; in >> t;) flags.insert(t);
        drive(goal, flags, ++nav_seq_, done);
      });
    });
  }

protected:
  void stop() override {
    ++nav_seq_;  // 응답 전인 goal은 goal_response_callback이 취소
    if (nav_goal_) nav_->async_cancel_goal(nav_goal_);
    nav_goal_.reset();
    if (backup_goal_) backup_->async_cancel_goal(backup_goal_);
    backup_goal_.reset();
    if (drive_goal_) drive_->async_cancel_goal(drive_goal_);
    drive_goal_.reset();
    if (spin_goal_) spin_->async_cancel_goal(spin_goal_);
    spin_goal_.reset();
    arm_.stop();
  }

private:
  void drive(const Pose & target, std::set<std::string> flags, uint64_t seq, Done done, int tries = 0) {
    if (seq != nav_seq_) return;
    NavigateToPose::Goal goal;
    goal.pose = target;
    goal.pose.header.stamp = builtin_interfaces::msg::Time();  // 0 = 가장 최근 TF (예전 실측: 보낸 시각 TF가 없어 error 102)
    stage("Nav2 (" + std::to_string(goal.pose.pose.position.x).substr(0, 5) + ", " +
          std::to_string(goal.pose.pose.position.y).substr(0, 5) + ")" + (tries ? " 다시" : ""));
    rclcpp_action::Client<NavigateToPose>::SendGoalOptions opt;
    auto finished = std::make_shared<bool>(false);  // 감시가 먼저 끝내면 취소 뒤 오는 결과는 무시
    auto canceled_for_stall = std::make_shared<bool>(false);
    auto collision_blocked_at_stall = std::make_shared<bool>(false);
    auto watch = std::make_shared<ProgressWatch>(stall_s_, stall_gain_m_);
    auto look = std::make_shared<bool>(flags.count("look") > 0);  // true = 아직 팔을 안 올림
    const bool near = flags.count("near") > 0;
    auto left = std::make_shared<double>(1e9);      // 목표까지 직선 거리 (마지막 피드백)
    auto yaw_err = std::make_shared<double>(M_PI);  // 목표 방향과 차 [rad]
    const auto gp = target.pose.position;
    const double gyaw = yaw_of(target.pose.orientation);
    auto finish = [this, near, left, yaw_err, done, target, flags, seq, tries, collision_blocked_at_stall](
        bool ok, const std::string & why) {
      if (!ok && near && *left <= near_m_ && *yaw_err <= near_yaw_) {
        RCLCPP_WARN(get_logger(), "%s — 목표 %.2f m · %.0f° 앞이라 성공 처리 (near, 나머지는 approach)", why.c_str(), *left,
                    *yaw_err * 180.0 / M_PI);
        ok = true;
      }
      // 목표가 막힘(206)·거절이면 풀어도 소용없음 → 바로 실패 (mission이 다른 면으로). 11-18 sim: 206에 풀기 3번으로 53 s 버림
      if (!ok && why != "nav_rejected" && why != "nav_goal_occupied" && tries < unstuck_tries_ && seq == nav_seq_) {
        const bool collision_limited = why == "nav_collision_blocked" || *collision_blocked_at_stall;
        unstuck(seq, tries, [this, target, flags, seq, done, tries, why, collision_limited](bool recovered) {
          if (!recovered) {
            done(false, collision_limited ? "nav_collision_recovery_failed" : why);
            return;
          }
          drive(target, flags, seq, done, tries + 1);
        }, collision_limited);
        return;
      }
      done(ok, ok ? "" : why);
    };
    opt.feedback_callback = [this, finished, canceled_for_stall, collision_blocked_at_stall,
        watch, look, left, yaw_err, gp, gyaw, finish](
        rclcpp_action::ClientGoalHandle<NavigateToPose>::SharedPtr h, const std::shared_ptr<const NavigateToPose::Feedback> fb) {
      if (*finished) return;
      *left = std::hypot(fb->current_pose.pose.position.x - gp.x, fb->current_pose.pose.position.y - gp.y);
      *yaw_err = std::abs(std::remainder(yaw_of(fb->current_pose.pose.orientation) - gyaw, 2 * M_PI));
      if (*look && fb->distance_remaining > 0.01 && fb->distance_remaining < look_m_) {
        *look = false;
        if (arm_.moveit_ready())
          arm_.move_joints(look_q_, [this](bool ok) { if (!ok) RCLCPP_WARN(get_logger(), "주행 중 look 자세 실패"); });
        else RCLCPP_WARN(get_logger(), "MoveIt 없음 — 주행 중 look 자세 건너뜀");
      }
      if (stall_s_ <= 0.0 || !watch->update(now().seconds(), fb->distance_remaining)) return;
      *finished = true;
      RCLCPP_WARN(get_logger(), "진척 없음 — 남은 %.2f m가 %.0f s 동안 %.2f m도 안 줄음 (회복 %d회) → 취소",
                  fb->distance_remaining, stall_s_, stall_gain_m_, fb->number_of_recoveries);
      *collision_blocked_at_stall = collision_monitor_stopped_recently();
      if (*collision_blocked_at_stall)
        RCLCPP_WARN(get_logger(), "정체 시점 collision_monitor STOP 활성 — 충돌 검사된 후진만 1회 시도하고 실패하면 즉시 상위로 반환");
      *canceled_for_stall = true;
      nav_->async_cancel_goal(h);
    };
    opt.goal_response_callback = [this, seq, finished, finish](const rclcpp_action::ClientGoalHandle<NavigateToPose>::SharedPtr & h) {
      if (seq != nav_seq_) {  // 응답 전에 멈춤 — 늦게 받아진 goal도 취소
        if (h) nav_->async_cancel_goal(h);
        return;
      }
      if (h) { nav_goal_ = h; return; }
      if (*finished) return;  // 거절되면 결과가 오지 않음 — 여기서 끝내야 skill이 안 멈춤
      *finished = true;
      finish(false, "nav_rejected");
    };
    opt.result_callback = [this, seq, finished, canceled_for_stall, finish](
        const rclcpp_action::ClientGoalHandle<NavigateToPose>::WrappedResult & r) {
      if (seq == nav_seq_) nav_goal_.reset();
      if (seq != nav_seq_ || (*finished && !*canceled_for_stall)) return;
      *finished = true;
      if (*canceled_for_stall) {
        *canceled_for_stall = false;
        finish(false, "nav_no_progress");
        return;
      }
      const bool ok = r.code == rclcpp_action::ResultCode::SUCCEEDED;
      // 205 = 출발 칸 막힘 (상자·벽에 붙은 채 끝난 뒤 다음 이동이 전부 205로 끝났음) → 끼임 풀기 대상
      const int code = r.result ? r.result->error_code : 0;
      finish(ok, ok ? "" : code == 205 ? "nav_start_occupied" : code == 206 ? "nav_goal_occupied" : "nav_failed");
    };
    nav_->async_send_goal(goal, opt);
  }

  // 끼임 풀기 한 번 (try번째): 1 후진→전진 · 2 +40°→−40° · 3 전진→후진. 첫 동작이 충돌 검사로 막히면 두 번째로 → then
  //   behavior_server가 local costmap(도면 벽 + 카메라)·몸 모양으로 검사 — 앞 카메라라 뒤·옆은 도면 벽만 앎
  void unstuck(uint64_t seq, int attempt, std::function<void(bool)> done, bool collision_limited) {
    static const char * names[] = {"후진", "전진", "+40° 회전", "−40° 회전"};
    const int k = attempt % 3;
    const int a = collision_limited ? 0 : k == 0 ? 0 : k == 1 ? 2 : 1;
    const int b = k == 0 ? 1 : k == 1 ? 3 : 0;
    RCLCPP_WARN(get_logger(), "끼임 풀기 %d/%d — %s%s", attempt + 1, unstuck_tries_, names[a],
                collision_limited ? " (충돌 감지 상태라 후진만, 전진·회전 재시도 생략)" : "");
    stage(std::string("끼임 풀기: ") + names[a]);
    escape(seq, a, [this, seq, b, done, collision_limited](bool ok) {
      if (seq != nav_seq_) return;
      if (ok) { done(true); return; }
      if (collision_limited) { done(false); return; }
      stage(std::string("끼임 풀기: ") + names[b]);
      escape(seq, b, [this, seq, done](bool second_ok) { if (seq == nav_seq_) done(second_ok); });
    });
  }

  bool collision_monitor_stopped_recently() const {
    std::lock_guard<std::mutex> lock(collision_state_mutex_);
    return collision_stopped_ &&
      std::chrono::steady_clock::now() - collision_state_received_ <= std::chrono::seconds(1);
  }

  // 0 후진 unstuck_back_m · 1 전진 unstuck_fwd_m · 2 +unstuck_turn · 3 −unstuck_turn (충돌 검사 켬) → done(성공 여부)
  void escape(uint64_t seq, int kind, std::function<void(bool)> done) {
    const auto give_up = [done]() { done(false); };
    if (kind <= 1) {
      const double d = kind == 0 ? unstuck_back_m_ : unstuck_fwd_m_;
      const auto allow = static_cast<int32_t>(d / 0.05 + 4);
      if (kind == 0) {
        if (!backup_->action_server_is_ready()) return give_up();
        BackUp::Goal g;
        g.target.x = d; g.speed = 0.05f; g.time_allowance.sec = allow; g.disable_collision_checks = false;
        send_behavior<BackUp>(backup_, backup_goal_, g, seq, done);
      } else {
        if (!drive_->action_server_is_ready()) return give_up();
        DriveOnHeading::Goal g;
        g.target.x = d; g.speed = 0.05f; g.time_allowance.sec = allow; g.disable_collision_checks = false;
        send_behavior<DriveOnHeading>(drive_, drive_goal_, g, seq, done);
      }
    } else {
      if (!spin_->action_server_is_ready()) return give_up();
      Spin::Goal g;
      g.target_yaw = static_cast<float>(kind == 2 ? unstuck_turn_ : -unstuck_turn_);
      g.time_allowance.sec = 6;
      g.disable_collision_checks = false;
      send_behavior<Spin>(spin_, spin_goal_, g, seq, done);
    }
  }

  template <class A>
  void send_behavior(typename rclcpp_action::Client<A>::SharedPtr cli, typename rclcpp_action::ClientGoalHandle<A>::SharedPtr & slot,
            const typename A::Goal & g, uint64_t seq, std::function<void(bool)> done) {
    typename rclcpp_action::Client<A>::SendGoalOptions o;
    o.goal_response_callback = [this, cli, &slot, seq, done](const typename rclcpp_action::ClientGoalHandle<A>::SharedPtr & h) {
      if (seq != nav_seq_) { if (h) cli->async_cancel_goal(h); return; }
      if (h) slot = h; else done(false);
    };
    o.result_callback = [&slot, done](const typename rclcpp_action::ClientGoalHandle<A>::WrappedResult & r) {
      slot.reset();
      done(r.code == rclcpp_action::ResultCode::SUCCEEDED);
    };
    cli->async_send_goal(g, o);
  }

  static double yaw_of(const geometry_msgs::msg::Quaternion & q) {
    return std::atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
  }

  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_;
  rclcpp_action::Client<BackUp>::SharedPtr backup_;
  rclcpp_action::Client<DriveOnHeading>::SharedPtr drive_;
  rclcpp_action::Client<Spin>::SharedPtr spin_;
  rclcpp::Subscription<nav2_msgs::msg::CollisionMonitorState>::SharedPtr collision_state_sub_;
  mutable std::mutex collision_state_mutex_;
  bool collision_stopped_ = false;
  std::chrono::steady_clock::time_point collision_state_received_{};
  rclcpp_action::ClientGoalHandle<NavigateToPose>::SharedPtr nav_goal_;  // 지금 보낸 Nav2 goal (stop이 이것만 취소)
  rclcpp_action::ClientGoalHandle<BackUp>::SharedPtr backup_goal_;
  rclcpp_action::ClientGoalHandle<DriveOnHeading>::SharedPtr drive_goal_;
  rclcpp_action::ClientGoalHandle<Spin>::SharedPtr spin_goal_;
  uint64_t nav_seq_ = 0;  // 보낸 goal 번호 — stop하면 올라가 옛 응답·결과를 가려냄
  int unstuck_tries_ = 3;
  double unstuck_fwd_m_ = 0.10, unstuck_turn_ = 0.7;
  double unstuck_back_m_ = 0.15, stall_s_ = 15.0, stall_gain_m_ = 0.10, look_m_ = 0.6, near_m_ = 0.15, near_yaw_ = 0.8;
  ArmIO arm_;
  std::vector<double> look_q_;
};

}  // namespace skills

RCLCPP_COMPONENTS_REGISTER_NODE(skills::NavigateSkill)
