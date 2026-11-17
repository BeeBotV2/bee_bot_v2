// skill_navigate — /skills/navigate: goal.pose(map)로 Nav2 NavigateToPose 한 번 (지도 위 먼 이동, 장애물 피함)
//   목표 자세 계산(물체 앞 · 좌표)은 coordinator가 함. 경로는 Nav2(Lattice, 몸 모양 그대로)가 정하고 RPP가 따라감 — 여기서는 경유점을
//   끼우지 않는다 (11-18 재설계: 문 경유점 · 길목 · 경유점 지도 · 가르친 경로 · 막은 구역을 걷어냄. 몸을 원으로 보던 예전 플래너가
//   문 모서리를 질러서 필요했던 것 — isaac/mission/nav_bench.py 시뮬에서 경유점 없이 8구간 · 동적 장애물 4번 모두 성공, 접촉 0)
//   진척 감시: 남은 경로 거리가 nav_stall_s 동안 nav_stall_gain_m만큼도 안 줄면 취소하고 nav_no_progress (ProgressWatch)
//   goal.target "look" (상자 도킹 앞 주행): 남은 거리가 nav_look_m 안이면 팔을 pose_look으로 (approach가 바로 상자를 봄)
//   goal.target "near" (상자 앞 다가설 자리): 멈추거나 Nav2가 실패해도 목표에서 nav_near_m · nav_near_yaw 안이면 성공 —
//     마지막 몇 cm·방향은 approach가 상자를 보며 맞춤
//   끼임 풀기: 진척 없음 · 출발 칸 막힘(205)이면 nav_unstuck_tries번까지 nav_unstuck_back_m 곧게 후진(충돌 검사 켬 — 뒤는 도면 벽만 앎)
//     → 같은 목표로 다시. 그래도 안 되면 실패 → mission_manager가 다른 면(plan.yaml start/location_box_sides)으로
//   target에 "leg:<구간>"이 있어도 무시 (예전 가르친 경로 재생 키)
#include <sstream>
#include "skills/arm_io.hpp"
#include "nav2_msgs/action/back_up.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "skills/progress_watch.hpp"
#include "skills/skill_node.hpp"

#include <cmath>
#include <functional>
#include <memory>
#include <set>
#include <vector>

namespace skills {

class NavigateSkill : public SkillNode {
public:
  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using BackUp = nav2_msgs::action::BackUp;
  using Pose = geometry_msgs::msg::PoseStamped;
  explicit NavigateSkill(const rclcpp::NodeOptions & options)
  : SkillNode("skill_navigate", options), arm_(*this) {
    nav_ = rclcpp_action::create_client<NavigateToPose>(this, "/navigate_to_pose");
    backup_ = rclcpp_action::create_client<BackUp>(this, "/backup");
    unstuck_tries_ = declare_parameter<int>("nav_unstuck_tries", 1);   // 0이면 끼임 풀기 안 함
    unstuck_back_m_ = declare_parameter<double>("nav_unstuck_back_m", 0.15);
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
    auto watch = std::make_shared<ProgressWatch>(stall_s_, stall_gain_m_);
    auto look = std::make_shared<bool>(flags.count("look") > 0);  // true = 아직 팔을 안 올림
    const bool near = flags.count("near") > 0;
    auto left = std::make_shared<double>(1e9);      // 목표까지 직선 거리 (마지막 피드백)
    auto yaw_err = std::make_shared<double>(M_PI);  // 목표 방향과 차 [rad]
    const auto gp = target.pose.position;
    const double gyaw = yaw_of(target.pose.orientation);
    auto finish = [this, near, left, yaw_err, done, target, flags, seq, tries](bool ok, const std::string & why) {
      if (!ok && near && *left <= near_m_ && *yaw_err <= near_yaw_) {
        RCLCPP_WARN(get_logger(), "%s — 목표 %.2f m · %.0f° 앞이라 성공 처리 (near, 나머지는 approach)", why.c_str(), *left,
                    *yaw_err * 180.0 / M_PI);
        ok = true;
      }
      if (!ok && (why == "nav_no_progress" || why == "nav_start_occupied") && tries < unstuck_tries_ && seq == nav_seq_) {
        unstuck(seq, [this, target, flags, seq, done, tries]() { drive(target, flags, seq, done, tries + 1); });
        return;
      }
      done(ok, ok ? "" : why);
    };
    opt.feedback_callback = [this, finished, watch, look, left, yaw_err, gp, gyaw, finish](
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
      nav_->async_cancel_goal(h);
      finish(false, "nav_no_progress");
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
    opt.result_callback = [this, seq, finished, finish](const rclcpp_action::ClientGoalHandle<NavigateToPose>::WrappedResult & r) {
      if (seq == nav_seq_) nav_goal_.reset();
      if (*finished || seq != nav_seq_) return;
      *finished = true;
      const bool ok = r.code == rclcpp_action::ResultCode::SUCCEEDED;
      // 205 = 출발 칸 막힘 (상자·벽에 붙은 채 끝난 뒤 다음 이동이 전부 205로 끝났음) → 끼임 풀기 대상
      finish(ok, ok ? "" : r.result && r.result->error_code == 205 ? "nav_start_occupied" : "nav_failed");
    };
    nav_->async_send_goal(goal, opt);
  }

  // 곧게 후진 (충돌 검사 켬 — 앞 카메라라 뒤는 local costmap의 도면 벽만 보고 막음) → then
  void unstuck(uint64_t seq, std::function<void()> then) {
    if (!backup_->action_server_is_ready()) { then(); return; }
    RCLCPP_WARN(get_logger(), "끼임 풀기 — %.2f m 후진 뒤 같은 목표로 다시", unstuck_back_m_);
    stage("끼임 풀기: 후진");
    BackUp::Goal b;
    b.target.x = unstuck_back_m_;
    b.speed = 0.05f;
    b.time_allowance.sec = static_cast<int32_t>(unstuck_back_m_ / 0.05 + 4);
    b.disable_collision_checks = false;
    rclcpp_action::Client<BackUp>::SendGoalOptions bo;
    bo.goal_response_callback = [this, seq, then](const rclcpp_action::ClientGoalHandle<BackUp>::SharedPtr & h) {
      if (seq != nav_seq_) { if (h) backup_->async_cancel_goal(h); return; }
      if (h) backup_goal_ = h; else then();
    };
    bo.result_callback = [this, seq, then](const rclcpp_action::ClientGoalHandle<BackUp>::WrappedResult &) {
      backup_goal_.reset();
      if (seq == nav_seq_) then();
    };
    backup_->async_send_goal(b, bo);
  }

  static double yaw_of(const geometry_msgs::msg::Quaternion & q) {
    return std::atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
  }

  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_;
  rclcpp_action::Client<BackUp>::SharedPtr backup_;
  rclcpp_action::ClientGoalHandle<NavigateToPose>::SharedPtr nav_goal_;  // 지금 보낸 Nav2 goal (stop이 이것만 취소)
  rclcpp_action::ClientGoalHandle<BackUp>::SharedPtr backup_goal_;
  uint64_t nav_seq_ = 0;  // 보낸 goal 번호 — stop하면 올라가 옛 응답·결과를 가려냄
  int unstuck_tries_ = 1;
  double unstuck_back_m_ = 0.15, stall_s_ = 15.0, stall_gain_m_ = 0.10, look_m_ = 0.6, near_m_ = 0.15, near_yaw_ = 0.8;
  ArmIO arm_;
  std::vector<double> look_q_;
};

}  // namespace skills

RCLCPP_COMPONENTS_REGISTER_NODE(skills::NavigateSkill)
