# myROS2

## ROS2 Installation
Follow [ROS2 Humble](https://docs.ros.org/en/humble/) instruction to install ROS2 humble to Ubuntu22.04 (not Ubuntu20.04 or other versions).

### Local Installation (tested on Ubuntu22.04 and Windows WSL2 Ubuntu22.04)
Run "locale" in a terminal window to view your currently installed locale – if UTF-8 appears in the listed output, you’re all set!

For the purposes of installing ROS, we need to enable the Ubuntu Universe repository (community-maintained open source software) in addition to the Main (Canonical-supported open-source software) repository:
```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
```
Now you can add the ROS 2 repository to your system. Authorize the public GPG (GNU Privacy Guard) key provided by ROS, then add the ROS 2 repository to your sources list:
```bash
$ sudo apt update && sudo apt install curl gnupg lsb-release
$ sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
$ echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

Install your ROS 2 Humble desktop setup
```bash
sudo apt update
sudo apt install ros-humble-desktop
sudo apt install python3-rosdep2 #add rosdep
sudo apt install python3-colcon-common-extensions #add colcon
```
If your face "EasyInstallDeprecationWarning: easy_install" warning after colcon build, you can downgrade the setuptools
```bash
pip3 install setuptools==58.2.0
```


### Container Installation
Check the Docker section for detailed information.

Use the Dockerfile under scripts folder to build the container image:
```bash
myROS2/docker$ docker build .
```

Enter ROS2 container (make sure the current directory is myROS, it will be mounted to the container)
```bash
MyRepo/myROS2$ ./scripts/runcontainer.sh
```
### Test Installation
Check ROS2 packages and source your setup file:
```bash
printenv | grep -i ROS
  ROS_PYTHON_VERSION=3
  PWD=/myROS2
  ROS_ROOT=/opt/ros/humble
  ROS_DISTRO=humble
  
/myROS2$ source /opt/ros/${ROS_DISTRO}/setup.bash
/myROS2$ rosdep update
```
You can automatically trigger this step every time you launch a new shell:
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

You can run some buildin samples to check the ROS installation
```bash
terminal1$ ros2 run demo_nodes_cpp talker
terminal2$ ros2 run demo_nodes_py listener

terminal1$ ros2 multicast receive
terminal2$ ros2 multicast send
```

Clone this repo:
```bash
git clone --recurse-submodules https://github.com/lkk688/myROS2.git
cd myROS2
rosdep install -i --from-path src --rosdistro humble -y
colcon build --symlink-install
```

## Repo Setup
add submodules of ROS2 examples and tutorials:
```bash
git submodule add -b humble https://github.com/ros2/examples src/examples
git submodule add -b humble-devel https://github.com/ros/ros_tutorials.git src/ros_tutorials
```

```bash
/myROS2$ git submodule add -b ros2 https://github.com/ros-drivers/velodyne.git src/velodyne
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
rosdep update
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

Add msp folder
```bash
/myROS2/src/mycpackage$ mkdir msg
```

## Create a new Python Package
Navigate into src folder, and run the package creation command:
```bash
/myROS2/src$ ros2 pkg create --build-type ament_python mypypackage
```
Write the source code [publisher_function.py](./src/mypypackage/mypypackage/publisher_function.py), add dependency packages into "package.xml"
```bash
<exec_depend>rclpy</exec_depend>
<exec_depend>std_msgs</exec_depend>
```
Write the source code [subscriber_function.py](./src/mypypackage/mypypackage/subscriber_function.py)

Add an entry point: open the setup.py file, add the following line within the console_scripts brackets
```bash
entry_points={
        'console_scripts': [
            'talker = mypypackage.publisher_function:main',
            'listener = mypypackage.subscriber_function:main',
        ],
    },
```
Build and run:
```bash
/myROS2$ rosdep install -i --from-path src --rosdistro humble -y
/myROS2$ colcon build --packages-select mypypackage
/myROS2$. install/local_setup.bash
```
Run the talker and listener nodes:
```bash
/myROS2$ ros2 run mypypackage talker

/myROS2$ ros2 run mypypackage listener
```

Create a launch file under launch folder and start two nodes:
```bash
myROS2$ ros2 launch mypypackage mypypackage_launch.py
[INFO] [launch]: All log files can be found below /home/lkk68/.ros/log/2022-10-28-00-12-03-078158-newalienware-15144
[INFO] [launch]: Default logging verbosity is set to INFO
[INFO] [talker-1]: process started with pid [15175]
[INFO] [listener-2]: process started with pid [15177]
[talker-1] [INFO] [1666941127.499735742] [talker]: Publishing: "Hello World: 0"
[listener-2] [INFO] [1666941127.500181698] [listener]: I heard: "Hello World: 0"
[listener-2] [INFO] [1666941127.965901235] [listener]: I heard: "Hello World: 1"
[talker-1] [INFO] [1666941127.966705914] [talker]: Publishing: "Hello World: 1"
```

## Creating custom msg and srv files
Create a new package for the new custom msg, this package is seperate from other packages
```bash
/myROS2/src$ ros2 pkg create --build-type ament_cmake my_interfaces
```
Create folder of msg and srv, add msg files and srv files. Add "rosidl_generate_interfaces" into CmakeLists.txt
```bash
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Num.msg"
  "msg/Sphere.msg"
  "srv/AddThreeInts.srv"
  DEPENDENCIES geometry_msgs # Add packages that above messages depend on, in this case geometry_msgs for Sphere.msg
)
```
Add the following lines to package.xml
```bash
  <depend>geometry_msgs</depend>
  <build_depend>rosidl_default_generators</build_depend>
  <exec_depend>rosidl_default_runtime</exec_depend>
  <member_of_group>rosidl_interface_packages</member_of_group>
```

Build the package, Now the interfaces will be discoverable by other ROS 2 packages.
```bash
colcon build --packages-select my_interfaces
. install/setup.bash
ros2 interface show my_interfaces/msg/Num
ros2 interface show my_interfaces/msg/Sphere
```

To access the custom msg, ref the newly created my_interfaces package in mycpackage. 

## Add Parameters
Create a new package
```bash
/myROS2/src$ ros2 pkg create --build-type ament_cmake cpp_parameters --dependencies rclcpp
```
Your terminal will return a message verifying the creation of your package cpp_parameters and all its necessary files and folders. The --dependencies argument will automatically add the necessary dependency lines to package.xml and CMakeLists.txt.

Create the cpp source file: cpp_parameters_node.cpp

Add following to the CMakeLists.txt
```bash
add_executable(minimal_param_node src/cpp_parameters_node.cpp)
ament_target_dependencies(minimal_param_node rclcpp)

install(TARGETS
  minimal_param_node
  DESTINATION lib/${PROJECT_NAME}
)
```

Build and run
```bash
rosdep install -i --from-path src --rosdistro humble -y
colcon build --packages-select cpp_parameters
. install/setup.bash
ros2 run cpp_parameters minimal_param_node
```

Once you run the node, you can then see the type and description in another terminal
```bash
/myROS2$ ros2 param describe /minimal_param_node my_parameter
Parameter name: my_parameter
  Type: string
  Description: This parameter is mine!
  Constraints:

admin@kaikai-i9new:/myROS2$ ros2 param list
/minimal_param_node:
  my_parameter
  qos_overrides./parameter_events.publisher.depth
  qos_overrides./parameter_events.publisher.durability
  qos_overrides./parameter_events.publisher.history
  qos_overrides./parameter_events.publisher.reliability
  use_sim_time
```
You can also change the custom parameter:
```bash
myROS2$ ros2 param set /minimal_param_node my_parameter earth
Set parameter successful
```

Create a "launch" folder and create a new file called cpp_parameters_launch.py. Add the following into CMakeLists.txt
```bash
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)
```

Run the node using the launch file:
```bash
ros2 launch cpp_parameters cpp_parameters_launch.py
```

## Docker
Install [Docker](https://docs.docker.com/engine/install/ubuntu/) and follow [Post-installation steps for Linux](https://docs.docker.com/engine/install/linux-postinstall/)

Setup Docker and nvidia container runtime via [nvidiacontainer1](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) [nvidiacontainer2](https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/index.html
)

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

## VS Code Remote
Install Visual Studio Code Remote-SSH and Dev Containers extension.

### Option1: Connect to remote ssh targets, "attach" VS Code to an already running Docker container. 
Once attached, you can install extensions, edit, and debug like you can when you open a folder in a container
<img width="608" alt="image" src="https://user-images.githubusercontent.com/6676586/196021984-800da6ce-ed84-44e6-a68c-2e5d221f97f0.png">

You can also see the remote running container in the Remote Explorer (Containers) part:
<img width="393" alt="image" src="https://user-images.githubusercontent.com/6676586/197059167-50a791b1-f9d0-4590-b437-7591fce4ca34.png">

### Option2: Set up the development container in VSCode.
Create a .devcontainer folder under workspace for VSCode to know how to mount your docker container as a workspace. You can click "Dev Containers: Add Dev Container Configuration Files" in command to create such folder. The "devcontainer.json" contains the container options including the Dockerfile and runArgs. With this .devcontainer folder, you can open a new container for development by selecting "Dev Containers: Reopen in Container" -- tested working. Check [VScode Container](https://code.visualstudio.com/docs/devcontainers/containers) for detailed information.

### Windows WSL2
You can use the VS Code Remote extension open the workspace in windows WSL2 (install the ROS2 on Ubuntu22.04), you can see the status bar in the bottom-left corner of VS Code:
<img width="201" alt="image" src="https://user-images.githubusercontent.com/6676586/198738636-2725a72c-214a-451b-bc4a-f8d1884db977.png">


## VS Code ROS Extension
Install extension of Robot Operating System (ROS) with Visual Studio Code, [github](https://github.com/ms-iot/vscode-ros):

<img width="350" alt="image" src="https://user-images.githubusercontent.com/6676586/197295344-ddf86ebb-362b-413c-aaec-74bb6d84fc83.png">

Use VSCode command to run a ros node or launch file:

<img width="429" alt="image" src="https://user-images.githubusercontent.com/6676586/197295957-28a05c92-7f37-45b8-856d-67451a3ee9bf.png">

In the VSCode terminal, open ROS2 daemon:
```bash
ros2 daemon start
ros2 daemon status
```
After ROS2 daemon is started, you can see the status information in the bottom-left corner of VSCode changed from "x" to
<img width="125" alt="image" src="https://user-images.githubusercontent.com/6676586/198738681-b922bb56-5419-45f3-b990-c7b287a607b0.png">

### Python ROS debug
In settings.json under .vscode folder, add the python path of ROS:
```bash
"/opt/ros/humble/lib/python3.10/site-packages",
"/opt/ros/humble/include/**",
```

To enable Python debug, click the Debug icon of VScode, select to create a new "launch.json" (make sure no other python windows is open) based on ROS->launch. After you created the launch.json, you can change the python launch file path in the "target" field in order to do the step by step debug --tested working.

### C++ ROS debug
To enable C++ ROS debug, we need to setup c_cpp_properties.json file (can be created by VSCode ROS extension) first. Add the include path of ROS2 humble.

Make sure the gdb is installed in the container/WSL2 or local system:
```bash
sudo apt-get install libc6-dbg gdb valgrind
```

Compile the ROS2 package with debug options
```bash
colcon build --packages-select cpp_parameters --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
```
Add the C++ ROS2 launch file to the "target" field of "launch.json" (or create a new launch.json, select C++ launch file), then you can click the debug button to start step-by-step debug:
<img width="1718" alt="image" src="https://user-images.githubusercontent.com/6676586/198739861-0f02b139-ff89-4dc6-a5a1-697e9f287005.png">

