#!/bin/bash
set -e

sudo apt-get update

# setup ros environment
source "/opt/ros/$ROS_DISTRO/setup.bash" 
rosdep update
#source "$ROS_WS/devel/setup.bash"
source ~/ws_ros/install/local_setup.bash
rosdep install -i --from-path src --rosdistro humble -y
source install/setup.bash
# if [ ${ROS_DISTRO} == "foxy" ] ; then
# 	source "$ROS2_WS/install/local_setup.bash"
# else
# 	source "/opt/ros/$ROS_DISTRO/setup.bash" 
# 	source "$ROS_WS/devel/setup.bash"
# fi
exec "$@"