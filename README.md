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

Add publish source code into [mycnode.cpp](src/mycnode.cpp); subscribe code into [mysubscribenode.cpp](src/mysubscribenode.cpp). Add the required packages into package.xml
```bash
  <depend>rclcpp</depend>
  <depend>std_msgs</depend>
```
Add the following things into CMakeLists.txt, one for publisher Node "mycnode", another for subscriber Node "listener":
```bash
find_package(rclcpp REQUIRED)#new add
find_package(std_msgs REQUIRED)#new add
add_executable(mycnode src/mycnode.cpp)#map node name to source code
ament_target_dependencies(mycnode rclcpp std_msgs)#new add

#add for subscriber node
add_executable(listener src/mysubscribenode.cpp)
ament_target_dependencies(listener rclcpp std_msgs)
```
In the "install" section of the CMakeLists.txt, add node names:
```bash
install(TARGETS mycnode
  listener
  DESTINATION lib/${PROJECT_NAME})
```

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
[INFO] [1665902187.007343170] [minimal_publisher]: Publishing: 'Hello, world! 0'
[INFO] [1665902187.507230210] [minimal_publisher]: Publishing: 'Hello, world! 1'
[INFO] [1665902188.007271423] [minimal_publisher]: Publishing: 'Hello, world! 2'
......
```
Open another terminal, open the subscriber
```bash
/myROS2$ ros2 run mycpackage listener
[INFO] [1665902999.743708062] [minimal_subscriber]: I heard: 'Hello, world! 0'
[INFO] [1665903000.243223400] [minimal_subscriber]: I heard: 'Hello, world! 1'
[INFO] [1665903000.743567144] [minimal_subscriber]: I heard: 'Hello, world! 2'
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

### VS Code remote debug:
Install Visual Studio Code Remote-SSH and Dev Containers extension.

Connect to remote ssh targets, "attach" VS Code to an already running Docker container. Once attached, you can install extensions, edit, and debug like you can when you open a folder in a container
<img width="608" alt="image" src="https://user-images.githubusercontent.com/6676586/196021984-800da6ce-ed84-44e6-a68c-2e5d221f97f0.png">
