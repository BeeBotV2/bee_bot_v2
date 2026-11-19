"""Run arena wall localization independently of the mission."""

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def create_node(context):
    config = Path(get_package_share_directory('plan')) / 'config' / 'plan.yaml'
    mission = yaml.safe_load(config.read_text())['mission_manager']['ros__parameters']
    params = {
        key: value for key, value in mission.items()
        if key in ('arena_walls', 'start_pose_arena', 'arena_corr_to_localization',
                   'arena_corr_max_shift', 'arena_corr_max_rot')
        or key.startswith(('arena_corr_wall_', 'arena_corr_loc_'))
    }
    if LaunchConfiguration('apply_to_map').perform(context).lower() == 'false':
        params['arena_corr_to_localization'] = False
    start_pose = LaunchConfiguration('start_pose').perform(context)
    if start_pose:
        params['start_pose_arena'] = [float(value) for value in yaml.safe_load(start_pose)]
    return [Node(package='plan', executable='wall_localization',
                 output='screen', parameters=[params])]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('start_pose', default_value=''),
        DeclareLaunchArgument('apply_to_map', default_value='true'),
        OpaqueFunction(function=create_node),
    ])
