from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            name='image_visualizer_node',
            package='yolopyinference_ros',
            executable='image_visualizer',
            output='screen',
        ),
        Node(
            name='image_view',
            package='rqt_image_view',
            executable='rqt_image_view',
            arguments=['/processed_image']
        )
    ])