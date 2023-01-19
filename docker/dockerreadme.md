## Current Setup
Option1: Ubuntu20.04 based on nvcr.io/nvidia/tritonserver:22.03-py3, you can build the container based on the docker/Dockerfile.triton 

Option2: Ubuntu22.04 based on nvidia/cuda:11.7.1-devel-ubuntu22.04, you can build the container in the following two steps:
 * Build one base container based on Dockerfile: docker/Dockerfile.ubuntu22cu117, the created image named "myros2ubuntu22cuda117:latest"
 * Run build-image.sh (call docker/Dockerfile.ros2humble) to create a ROS2 container (tagged as "myros2humble:latest") based on the "myros2ubuntu22cuda117:latest"

 Option2 version needs additional steps to prevent build error in isaac_ros_nitros (error: ‘unique_lock’ is not a member of ‘std’). Follow the changes [here](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros/pull/8/commits/0e243982f6a6c69ef896b4c621f422d170760825) 

## Docker
Install [Docker](https://docs.docker.com/engine/install/ubuntu/) and follow [Post-installation steps for Linux](https://docs.docker.com/engine/install/linux-postinstall/)

Setup Docker and nvidia container runtime via [nvidiacontainer1](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) [nvidiacontainer2](https://docs.nvidia.com/dgx/nvidia-container-runtime-upgrade/index.html
)


After you build the container, you can check the new container via "docker images", note down the image id, and run this image:
```bash
sudo docker run -it --rm 486a56765aad
```
After you entered the container and did changes inside the container, click "control+P+Q" to exit the container without terminate the container. Use "docker ps" to check the container id, then use "docker commit" to commit changes:
```bash
docker commit -a "Kaikai Liu" -m "First ROS2-x86 container" 196073a381b4 myros2:v1
```
Now, you can see your newly created container image named "myros2:v1" in "docker images".

You can now start your ROS2 container (i.e., myros2:v1) via runcontainer.sh, change the script file if you want to change the path of mounted folders. 
```bash
sudo xhost +si:localuser:root
./scripts/runcontainer.sh [containername]
```
after you 
Re-enter a container: use the command "docker exec -it container_id /bin/bash" to get a bash shell in the container.

Stop a running container: docker stop container_id
Stop all containers not running: docker container prune
Delete docker images: docker image rm dockerimageid

## Container Installation
Check the Docker section for detailed information.

Use the Dockerfile under scripts folder to build the container image:
```bash
myROS2/docker$ docker build -t myros2ubuntu22cuda117 .
```
You can also build the docker image via docker vscode extension. After the extension is installed, simply right click the Dockerfile and select "build image"

Enter ROS2 container (make sure the current directory is myROS, it will be mounted to the container)
```bash
MyRepo/myROS2$ ./scripts/runcontainer.sh myros2ubuntu22cuda117
```
