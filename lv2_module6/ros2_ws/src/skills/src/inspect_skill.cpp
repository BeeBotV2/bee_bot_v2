// skill_inspect — /skills/inspect: gripper_cam으로 보기 (학습 정책 없음). 읽었는지(details tag)는 coordinator · mission_manager도 /tracked_objects로 봄
//   보기 자세는 look 하나 (11-21 사용자: pick 테스트에서 검증한 look으로 통일). 예전 점 보기(점 위 15 cm에 손을 수직 아래로)는 뺌 —
//     11-21 Isaac 전체 미션: 선반 앞 도킹에서 칸까지 수평 0.235~0.27 m라 손을 수직으로 못 둠 (8 cm 높이 한계 0.23 m) → 12번 모두 실패
//   goal.pose = 볼 점 (상자 가운데 · 칸 · 퍽, 퍽 윗면 높이, map), values[0] = 기대 퍽 수 (없으면 2) · target은 안 씀
//     ① pose_look 관절 자세(arm.yaml — skill_pose "look"과 같은 값)로 → inspect_first_wait_s 기다림
//        11-21 사용자: 이미 look이어도(approach가 도킹 중 look으로 올림) 명령을 다시 보냄 — 팔 떨림·처짐을 바로잡고, 기다리는 동안이
//        inspect 단계라 coordinator가 보이는 퍽을 다 기억함 (11-21 실기 bag: '이미 보임'으로 0 s에 끝나 기억 못 함 →
//        load 뒤 pick이 'target 67 not tracked' → 다시 보기 9~15 s)
//     ② 상자 가운데에서 inspect_box_radius_m 안 퍽 트랙이 기대 수 이상이고 모두 태그·자세를 알면 끝
//     ③ 아니면 본 퍽(모자라거나 태그·자세 모르는 퍽이 있으면 상자 가운데도)을 한 장에 담는 자세(look_plan)로 → 기다림 → 다시 셈. 모자라면 inspect_partial
//        퍽 수가 모자라면 상자 가운데에서 로봇 쪽으로 inspect_near_m 당긴 점 둘(옆으로 ±inspect_near_side_m)도 담음 —
//        11-21 실기 bag S1: look에서 차체 쪽 퍽이 화면 아래(손가락 쪽)로 빠져 트랙이 끊김 → 2개 중 1개만 봄
//     11/07 Isaac 전체 스택 퍽 40개: 1차 자세만 85 % · 다시 담으면 92 % (isaac/scripts/demo/inspect_test.py center)
#include <cmath>
#include <mutex>

#include "perception/msg/tracked_object_array.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "skills/arm_io.hpp"
#include "skills/look_plan.hpp"
#include "skills/skill_node.hpp"
#include "tf2/LinearMath/Transform.hpp"

namespace skills {

class InspectSkill : public SkillNode {
public:
  explicit InspectSkill(const rclcpp::NodeOptions & options)
  : SkillNode("skill_inspect", options), arm_(*this) {
    look_q_ = declare_parameter<std::vector<double>>("pose_look", {0.0, -1.682497, 0.324631, 1.926843, 0.0});  // arm.yaml이 덮음
    look_wait_s_ = declare_parameter<double>("inspect_look_wait_s", 1.5);  // [s] 다시 담기 자세 뒤 (궤적 끝 · 처짐 보정 수렴 + 추론)
    first_wait_s_ = declare_parameter<double>("inspect_first_wait_s", 0.5);  // [s] 첫 look 뒤 — 보통 이미 look이라 거의 안 움직임
    near_m_ = declare_parameter<double>("inspect_near_m", 0.05);  // [m] 다시 담기: 상자 가운데에서 로봇 쪽으로 당긴 점
    near_side_m_ = declare_parameter<double>("inspect_near_side_m", 0.04);  // [m] 그 점을 옆으로 ± (퍽 두 개 자리)
    box_radius_m_ = declare_parameter<double>("inspect_box_radius_m", 0.16);  // [m] 상자 가운데에서 이 안 퍽만 셈 (반 변 0.1025 + 여유)
    objects_sub_ = create_subscription<perception::msg::TrackedObjectArray>("/tracked_objects", rclcpp::QoS(5).reliable(),
      [this](perception::msg::TrackedObjectArray::ConstSharedPtr m) { std::lock_guard<std::mutex> lk(mu_); objects_ = m; });
    serve("inspect", [this](const SkillGoal & g, Done done) {
      when_ready([this]() { return arm_.moveit_ready(); }, "arm_unavailable", done, [this, g, done]() {
        look_box(g, done);
      });
    });
  }

protected:
  void stop() override { arm_.stop(); }

private:
  // c 좌표계(보통 map) → base_footprint
  bool to_base(const geometry_msgs::msg::PoseStamped & c, tf2::Transform & t) {
    geometry_msgs::msg::TransformStamped tf;
    try {
      tf = arm_.tf().lookupTransform("base_footprint", c.header.frame_id.empty() ? "map" : c.header.frame_id, tf2::TimePointZero);
    } catch (const std::exception &) {
      return false;
    }
    const auto & r = tf.transform.rotation;
    const auto & tr = tf.transform.translation;
    t = tf2::Transform(tf2::Quaternion(r.x, r.y, r.z, r.w), tf2::Vector3(tr.x, tr.y, tr.z));
    return true;
  }

  // 상자 가운데 c에서 box_radius_m_ 안 퍽 트랙 → base_footprint 윗면 점들 (z = c.z) · 다 본 수 (태그와 자세를 앎)
  //   트랙 중복은 perception이 합침 (merge_tracks · merge_overlap_dets)
  bool box_pucks(const geometry_msgs::msg::PoseStamped & c, std::vector<look::V3> & tops, int & tagged) {
    tops.clear();
    tagged = 0;
    tf2::Transform t;
    if (!to_base(c, t)) return false;
    std::lock_guard<std::mutex> lk(mu_);
    if (!objects_) return true;
    for (const auto & o : objects_->objects) {
      if (o.class_name.rfind("puck", 0) != 0) continue;
      const auto & p = o.pose.pose.position;
      if (std::hypot(p.x - c.pose.position.x, p.y - c.pose.position.y) > box_radius_m_) continue;
      const tf2::Vector3 b = t * tf2::Vector3(p.x, p.y, c.pose.position.z);
      tops.push_back({b.x(), b.y(), b.z()});
      // 다 봤음 = 태그 + 자세(standing · lying)까지 — 자세를 모르면('?') coordinator가 세울지 집을지 못 정함 (11/09 Isaac)
      bool tag = false, posture = false;
      for (const auto & kv : o.details) {
        if (kv.key == "tag" && !kv.value.empty()) tag = true;
        if (kv.key == "posture" && !kv.value.empty()) posture = true;
      }
      if (tag && posture) ++tagged;
    }
    return true;
  }

  void look_box(const SkillGoal & g, Done done) {
    const auto c = g.pose;
    const size_t want = g.values.empty() ? 2 : static_cast<size_t>(std::max(1.0, g.values[0]));
    if (look_q_.size() != 5) { done(false, "inspect_no_look_pose"); return; }
    stage("look 자세");
    arm_.move_joints(look_q_, [this, c, want, done](bool ok) {
      if (!ok) { done(false, "inspect_failed"); return; }
      arm_.wait_step(first_wait_s_)([this, c, want, done](bool) {
        std::vector<look::V3> tops;
        int tagged = 0;
        if (!box_pucks(c, tops, tagged)) { done(false, "inspect_no_tf"); return; }
        if (tops.size() >= want && tagged >= static_cast<int>(tops.size())) { done(true, ""); return; }
        // 다시 담기: 본 퍽 + (모자라거나 태그 못 읽은 퍽이 있으면) 상자 가운데 — 못 본 퍽이 상자 위 어딘가에 있음
        //   태그 못 읽은 퍽 위치는 옆면 depth라 19~39 mm 틀림 — 그 점만 담으면 퍽을 놓친 자세가 됨 (11/09 Isaac: 살짝 보인 퍽)
        tf2::Transform t;
        if ((tops.size() < want || tagged < static_cast<int>(tops.size())) && to_base(c, t)) {
          const tf2::Vector3 b = t * tf2::Vector3(c.pose.position.x, c.pose.position.y, c.pose.position.z);
          tops.push_back({b.x(), b.y(), b.z()});
          if (tops.size() <= want) add_near(b, tops);  // 퍽 수가 모자람 (상자 가운데를 넣기 전 < want)
        }
        look::Q5 q;
        look::Score s;
        if (!look::search(tops, look::Params{}, q, s)) { done(false, "inspect_partial"); return; }
        stage("다시 담기 (점 " + std::to_string(tops.size()) + "개 · 덮음 " + std::to_string(static_cast<int>(100 * s.cover)) + "%)");
        arm_.move_joints(std::vector<double>(q.begin(), q.end()), [this, c, want, done](bool ok2) {
          if (!ok2) { done(false, "inspect_failed"); return; }
          arm_.wait_step(look_wait_s_)([this, c, want, done](bool) {
            std::vector<look::V3> tops2;
            int tagged2 = 0;
            box_pucks(c, tops2, tagged2);
            const bool all = tops2.size() >= want && tagged2 >= static_cast<int>(tops2.size());
            done(all, all ? "" : "inspect_partial");
          });
        });
      });
    });
  }

  // 상자 가운데 b(base_footprint)에서 로봇 쪽으로 near_m_ 당기고 옆으로 ±near_side_m_ 비킨 두 점
  void add_near(const tf2::Vector3 & b, std::vector<look::V3> & tops) const {
    const double r = std::hypot(b.x(), b.y());
    if (r < 1e-6) return;
    const double ux = b.x() / r, uy = b.y() / r;  // 로봇 → 상자
    for (const double s : {-near_side_m_, near_side_m_})
      tops.push_back({b.x() - near_m_ * ux - s * uy, b.y() - near_m_ * uy + s * ux, b.z()});
  }

  ArmIO arm_;
  double look_wait_s_, first_wait_s_, near_m_, near_side_m_, box_radius_m_;
  std::vector<double> look_q_;
  std::mutex mu_;
  perception::msg::TrackedObjectArray::ConstSharedPtr objects_;
  rclcpp::Subscription<perception::msg::TrackedObjectArray>::SharedPtr objects_sub_;
};

}  // namespace skills

RCLCPP_COMPONENTS_REGISTER_NODE(skills::InspectSkill)
