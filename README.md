> [!IMPORTANT]
> The base code for docker has been _taken_ from the repository: [avular-robotics:user-container:origin](https://github.com/avular-robotics/user-container/tree/origin)

# Improvement of vSLAM method using instance segmentation and opticalflow

## Installation

Firstly, it is essential to ensure that the GPU requirements are covered by the PC we want to installed the code to:

On a window terminal:
```bash
nvidia-smi
```
The top CUDA version should be greater than or equal to 12.2 for this specific Dockerfile:
![nvidia-smi command output](https://github.com/zuleikarg/user-container/blob/main/imgs/nvidia-smi.png?raw=true)

In the desired directory:
```bash
git clone https://github.com/zuleikarg/user-container.git & cd user-container
```

Build the image with the dockerfile and docker-compose.yml:
```bash
docker compose build
```

Allow GUI access if needed:
```bash
xhost +local:docker
```
Create and run the container:
```bash
sudo docker run -it --rm \
  --gpus all \
  --privileged \
  --device=/dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --name <container_name> robot_flow
```
Once inside:
```bash
source/install/setup.bash
```

Before running the nodes, an external resource for yolact has to be set up:
```bash
cd src/yolact/external/DCNv2
```
```bash
sudo ./make.sh
```

After this step, the nodes are ready to be used. The data from the topics can be published using, for example, rosbag on another container created from the same image.

## Execution

The code included inside _flow_ws_ are related to the combination of OpticalFlow and Instance Segmentation in order to generate and discard the essential ORBs to proceed with the SLAM procedure.

In different terminal windows with containers from the same image, the execution should be as follows:

__For all terminals__

__For different terminals__

Instance segmentation:
```bash
ros2 run yolact seg_node
```
OpticalFlow:
```bash
ros2 run neuflow_v2 infer_hf
```
Integration:
```bash
ros2 run dio motion_removal
```

If only yolact and a camera Intel RealSense are used, another node will be necessary to publish the frames through a topic:
```bash
ros2 run yolact seg_node
```

## How does it work?

For this improvement, [DIO-SLAM](https://pmc.ncbi.nlm.nih.gov/articles/PMC11435655/) was used as a reference. This way, there are two main parts implemented: instance segmentation and opticalflow. Additionally, a visual slip detector was created from the floor segmentation done to improve opticalflow's result.

### Instance Segmentation – [YOLACT++](https://github.com/dbolya/yolact)

| __Subscribers__  | __Publishers__ |
| ------------- | ------------- |
| RGB camera frame  | Nonrigid elements’ mask  |
|   | Rigid elements’ mask  |
|   | Corresponding RGB image  |

### OpticalFlow – [NeuFlow_v2](https://github.com/neufieldrobotics/NeuFlow_v2)

| __Subscribers__  | __Publishers__ |
| ------------- | ------------- |
| Corresponding RGB image  | OpticalFlow final mask  |
| Aligned depth camera frame  |   |
| Odometry  |   |
| Transformations  |   |

### Slip detector

| __Subscribers__  | __Publishers__ |
| ------------- | ------------- |
| Corresponding RGB image  |   |
| Aligned depth camera frame  |   |
| Odometry  |   |
| Transformations  |   |

### Dio node

| __Subscribers__  | __Publishers__ |
| ------------- | ------------- |
| Corresponding RGB image  |   |
| Rigid elements’ mask  |   |
| Corresponding RGB image  |   |
| OPticalFlow final mask  |   |
