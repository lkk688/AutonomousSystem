source /opt/ros/${ROS_DISTRO}/setup.bash
rosdep update
rosdep install -i --from-path src --rosdistro humble -y
colcon build --symlink-install
colcon build --packages-select mycpackage

. install/local_setup.bash

ros2 run mycpackage mycnode

ros2 pkg create --build-type ament_cmake --node-name mycnode mycpackage
