"""bag 재생용 SLAM만 — 실기에서 녹화한 bag(또는 mapping.sh record)을 PC에서 다시 돌려 매핑 · 위치추정을 시험한다.

  ros2 launch mapping/replay.launch.py map_dir:=<지도들> map_name:=<이름> localization:=false   # 매핑 (끌 때 저장)
  ros2 launch mapping/replay.launch.py map_dir:=<지도들> map_name:=<이름> localization:=true    # 저장 지도로 위치추정
  (보통은 mapping.sh replay가 bag play --clock과 같이 띄운다)

띄우는 것: robot_state_publisher(URDF → 카메라 TF · /joint_states는 bag) + ros_system의 EKF(/odom · /imu는 bag) · cuVSLAM · nvblox.
bag 입력: pole_front_cam 컬러 · 정렬 depth · camera_info 둘, /odom, /imu, /joint_states. /tf는 bag에 넣지 않는다 —
  녹화 때의 odom→base(EKF)가 여기 EKF와 부딪히고, SLAM을 켜고 녹화했으면 map→odom까지 섞인다.
안 띄우는 것: 카메라 드라이버 · ros2_control · Nav2 · twist_mux · 바퀴 명령 체인 (bag이 센서를 대신하고 로봇을 움직이지 않음).
use_sim_time: isaac_system.launch.py처럼 SetParameter로 모든 노드에 — 시계는 bag play --clock
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue


def slam(context):
    d = os.path.expanduser(LaunchConfiguration('map_dir').perform(context))
    name = LaunchConfiguration('map_name').perform(context)
    loc = LaunchConfiguration('localization').perform(context).lower() == 'true'
    folder = os.path.join(d, name)
    nvblox_map = os.path.join(folder, 'nvblox.nvblx')
    # 지도가 있는지는 map.yaml로 — nvblox 5.0(apt)은 끌 때 nvblox.nvblx가 아니라 nvblox.nvblx.png · .yaml(2D 격자)을 씀 (11-19 확인)
    map_yaml = os.path.join(folder, 'map.yaml')
    if loc and not os.path.exists(map_yaml):
        raise RuntimeError(f'저장 지도 없음: {map_yaml} — 먼저 localization:=false로 매핑')
    if not loc:
        if os.path.exists(map_yaml) or os.path.exists(os.path.join(folder, 'cuvslam')):
            raise RuntimeError(f'이미 있는 지도: {folder} — 덮어쓰지 않음, 다른 map_name')
        os.makedirs(folder, exist_ok=True)  # nvblox는 저장할 폴더를 안 만듦 (bringup setup_map과 같음)
    share = get_package_share_directory('ros_system')
    urdf = os.path.join(share, 'urdf', 'default', 'urdf.xacro')
    return [
        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             parameters=[{'robot_description': ParameterValue(Command(
                 ['xacro ', urdf, ' mesh_uri:=package://ros_system/urdf/default/meshes mock_hardware:=true']), value_type=str)}]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'ros_system.launch.py')),
            launch_arguments={
                'use_cameras': 'false', 'use_ros2_control': 'false', 'use_nav2': 'false', 'use_twist_mux': 'false',
                'use_cmd_chain': 'false', 'use_moveit': 'false', 'use_ekf': 'true', 'use_slam': 'true',
                'map_backend': 'nvblox', 'localization': 'true' if loc else 'false',
                'nvblox_map': nvblox_map, 'cuvslam_map': os.path.join(folder, 'cuvslam'),
                'use_pole_back_cam': LaunchConfiguration('use_pole_back_cam').perform(context),
                'pole_back_cam_external': LaunchConfiguration('use_pole_back_cam').perform(context),
            }.items()),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map_dir', default_value='~/.ros/beebot_maps', description='지도 폴더들이 있는 곳'),
        DeclareLaunchArgument('map_name', description='지도 폴더 이름 — <map_dir>/<map_name>/ map.yaml · cuvslam/ · nvblox'),
        DeclareLaunchArgument('localization', default_value='false', description='false = 매핑(끌 때 저장) · true = 저장 지도로 위치추정'),
        DeclareLaunchArgument('use_pole_back_cam', default_value='false', description='bag에 pole_back_cam 토픽이 있으면 true (nvblox camera_1)'),
        SetParameter(name='use_sim_time', value=True),
        OpaqueFunction(function=slam),
    ])
