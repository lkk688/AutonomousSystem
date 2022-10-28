from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="mypypackage",
            executable="talker",
            name="talker",
            output="screen",
            emulate_tty=True
        ),
        Node(
            package="mypypackage",
            executable="listener",
            name="listener",
            output="screen",
            emulate_tty=True
        )
    ])