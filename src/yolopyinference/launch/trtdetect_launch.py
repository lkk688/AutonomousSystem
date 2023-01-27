import os
import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('yolopyinference_ros'),
        'config',
        'trt_params.yaml'
    )

    trtdetect_node = Node(
        name='trtdetect_node',
        package='yolopyinference_ros',
        executable='TrtDetector',
        parameters=[config],
        output='screen',
        emulate_tty=True,
    )
    final_launch_description = [trtdetect_node] #+ [image_visualizer_node] + [rqt_node]
    return launch.LaunchDescription(final_launch_description)

    # return LaunchDescription([
    #     Node(
    #         name='image_visualizer_node',
    #         package='yolopyinference_ros',
    #         executable='image_visualizer',
    #         output='screen',
    #     ),
    #     Node(
    #         name='image_view',
    #         package='rqt_image_view',
    #         executable='rqt_image_view',
    #         arguments=['/processed_image']
    #     )
    # ])