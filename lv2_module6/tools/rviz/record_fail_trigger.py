"""record_fail.sh 짝 — 팔 · 도킹 step이 실패하면 rosbag2 snapshot을 부름 (최근 영상만 디스크로)."""
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from plan.msg import SubGoalArray, TaskStatus
from rosbag2_interfaces.srv import Snapshot

SKILLS = {'pick', 'pick_far', 'load', 'stand', 'stand_pick', 'place', 'approach'}


def main():
    rclpy.init()
    n = rclpy.create_node('record_fail_trigger')
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    plan = {}  # plan_id → {step: skill}
    snap = n.create_client(Snapshot, '/rosbag2_recorder/snapshot')

    def on_plan(m):
        plan.clear()
        plan[m.plan_id] = {g.step: g.skill for g in m.goals}

    def on_status(s):
        if s.state != TaskStatus.FAILED:
            return
        skill = plan.get(s.plan_id, {}).get(s.step, '')
        if skill not in SKILLS:
            return
        if not snap.service_is_ready():
            n.get_logger().warn(f'{skill} 실패 ({s.reason}) — snapshot 서비스 없음')
            return
        snap.call_async(Snapshot.Request())
        n.get_logger().info(f'{skill} 실패 ({s.reason}) — snapshot 씀 (plan {s.plan_id} step {s.step})')

    n.create_subscription(SubGoalArray, '/task/current_plan', on_plan, latched)
    n.create_subscription(TaskStatus, '/task/status', on_status, latched)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
