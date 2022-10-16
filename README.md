# myROS2

## ROS2 Tutorial
add submodule
```bash
git submodule add -b humble https://github.com/ros2/examples src/examples
```

```bash
git submodule add -b humble-devel https://github.com/ros/ros_tutorials.git src/ros_tutorials
```

Resolve dependencies (under workspace folder):
```bash
rosdep install -i --from-path src --rosdistro humble -y
```

From the root of your workspace, you can now build your packages using the command:
```bash
colcon build --symlink-install
```
Once the build is finished, you will see that colcon has created new directories: build, install, log, src. You can delete this after use: "rm -r build install log"

The install directory is where your workspace’s setup files are, which you can use to source your overlay.

In the new terminal, source your main ROS 2 environment as the “underlay”, so you can build the overlay “on top of” it:
```bash
source /opt/ros/humble/setup.bash
```
Go into the root of your workspace, source your overlay:
```bash
. install/local_setup.bash
```

Now you can run the turtlesim package from the overlay:
```bash
ros2 run turtlesim turtlesim_node
```

## Create own ROS2 package
create ROS2 package (under src folder)
```bash
/myROS2/src$ ros2 pkg create --build-type ament_cmake --node-name mycnode mycpackage
/myROS2/src$ ls ./mycpackage/
CMakeLists.txt	include  package.xml  src
```
package.xml file containing meta information about the package; CMakeLists.txt file that describes how to build the code within the package

To build only the my_package package next time, you can run: 
```bash
colcon build --packages-select mycpackage
```

To use your new package and executable, first open a new terminal and source your main ROS 2 installation: source /opt/ros/humble/setup.bash. Then, from inside the ros2_ws directory, run the following command to source your workspace: 
```bash
. install/local_setup.bash
```

To run the executable you created using the --node-name argument during package creation, enter the command:
```bash
/myROS2$ ros2 run mycpackage mycnode
hello world mycpackage package
```


## Docker
Build the container via [mybuildros2.sh](\scripts\mybuildros2.sh)

Start the container with ROS2 via runcontainer.sh, change the script file if you want to change the path of mounted folders. 
```bash
sudo xhost +si:localuser:root
./scripts/runcontainer.sh
```
after you did changes inside the container, click "control+P+Q" to exit the container without terminate the container. Use "docker ps" to check the container id, then use "docker commit" to commit changes:
```bash
docker commit -a "Kaikai Liu" -m "First ROS2-x86 container" 196073a381b4 myros2:v1
```
Re-enter a container: use the command "docker exec -it container_id /bin/bash" to get a bash shell in the container.

Stop a running container: docker stop container_id