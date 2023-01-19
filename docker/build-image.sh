#!/bin/bash
set -e
UBUNTU_RELEASE_YEAR=20 	#Specify the Ubunutu release year
CUDA_MAJOR=11 			# CUDA major version
CUDA_MINOR=7 			# CUDA minor version 
ROS_DISTRO_ARG="humble"
ZED_SDK_MAJOR=3 		# ZED SDK major version 
ZED_SDK_MINOR=8 		# ZED SDK minor version

#TAG="ros2 ${ROS_DISTRO_ARG} in cuda${CUDA_MAJOR}.${CUDA_MINOR}-ubuntu${UBUNTU_RELEASE_YEAR}.04"
TAG="myros2${ROS_DISTRO_ARG}"
DOCKERFILE="Dockerfile.ros2humble"

echo "Building '${TAG}'" 

docker build --build-arg UBUNTU_RELEASE_YEAR=${UBUNTU_RELEASE_YEAR} \
--build-arg ROS_DISTRO_ARG=${ROS_DISTRO_ARG} \
--build-arg CUDA_MAJOR=${CUDA_MAJOR} \
--build-arg CUDA_MINOR=${CUDA_MINOR} \
--build-arg ZED_SDK_MAJOR=${ZED_SDK_MAJOR} \
--build-arg ZED_SDK_MINOR=${ZED_SDK_MINOR} \
-t "${TAG}" -f "${DOCKERFILE}" .

#Successfully tagged myros2humble:latest
#xhost +si:localuser:root  # allows container to communicate with X server
#docker run  --gpus all --runtime nvidia --privileged -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix <container_tag> # run the docker container
#docker run  --gpus all --runtime nvidia --privileged -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix myros2humble:latest /bin/bash