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
After the last step, in the yolact/weights folder it is needed to add the desired yolact model. In the GitHub website it is possible to find various of them. But the model yolact_plus_resnet50_54_800000.pth is the one set by default. If desired to change, it is necessary to specify it on the segmentation.py code in yolact/yolact directory.

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
source install/setup.bash
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

[DIO-SLAM](https://pmc.ncbi.nlm.nih.gov/articles/PMC11435655/) presents an improvement from the original pipeline of ORB3-SLAM, adding to the pipeline both instance information from the frames and OpticalFlow to ensure that dynamic objects are not accounted for the localization of the robot on the map and its creation.

The assumption that both rigid and non-rigid objects coexist on the surroundings of the robot is stated and with it, it is claimed that non-rigid objects, such as humans and animals are not reliable for a SLAM task due to its moving nature.

Therefore, the use of an instance segmentation can assist on this labour, by segmenting both people and different types of animals and removing them from the final solution.

On the other hand, not only non-rigid objects can move around the robot, but also rigid objects which movement has been, at least, started by non-rigid entities. That’s where the role of OpticalFlow becomes essential. In this case, dense OpticalFlow will determine the movement of each one of the pixels of the image, allowing us to compare the segments obtained to the rigid object’s segmentations from the previous method.

### Instance Segmentation – [YOLACT++](https://github.com/dbolya/yolact)

| __Subscribers__  | __Publishers__ |
| ------------- | ------------- |
| RGB camera frame  | Nonrigid elements’ mask  |
|   | Rigid elements’ mask  |
|   | Corresponding RGB image  |

As for instance segmentation, YOLACT++ was used. Unlike its predecessor, YOLACT, which was used in DIO-SLAM, this version has a better result/efficiency ratio since trades off slightly more computational cost with best performance.
This method is capable of generating the bbox, segmentation, score and label of each of the objects it has been trained for, in this case, all the objects in the COCO dataset.

With this information in mind, rigid and non-rigid objects were separated for the two masks needed in DIO-SLAM. This way, when the label corresponds to animals or people it would be added as a white segment with black background for the non-rigid object masks. For the rigid object mask, the process would be the same but with all objects detected but people and animals.
The last mask would be compared to the OpticalFlow mask later.


### OpticalFlow – [NeuFlow_v2](https://github.com/neufieldrobotics/NeuFlow_v2)

| __Subscribers__  | __Publishers__ |
| ------------- | ------------- |
| Corresponding RGB image  | OpticalFlow final mask  |
| Aligned depth camera frame  |   |
| Odometry  |   |
| Transformations  |   |

NeoFlow_v2 [3] was selected to take over the task since it demonstrated to provide faster and cleaner results than the method used in the original DIO-SLAM.

Nevertheless, there was still a necessary issue to address. As it was stated on DIO-SLAM paper, the dense OpticalFlow does not account for self-motion flow, which it is required to compensate if it is aimed to work with moving cameras, and therefore with moving robots like the Origin One.

Once estimated, from the total opticalflow, the ego-motion flow is subtracted to get the __residual opticalflow__, which is the result we want to keep.

In addition, a floor segmentation was created and added to the resulting mask at the end, aiming to remove any noise that could appear on the ground surface.

### Slip detector

| __Subscribers__  | __Publishers__ |
| ------------- | ------------- |
| Corresponding RGB image  |   |
| Aligned depth camera frame  |   |
| Odometry  |   |
| Transformations  |   |

The texture of the floor is increased with _cv2.equalizeHist_, on both previous and current images, that only show the floor (the rest in black); especially interesting step for smooth surfaces such as the testing area’s floor. Then, SHIFT points are extracted and the matches of points between both images are found.

Open3d package made possible to directly extract an estimated transformation between the images with the matches and those points once transformed into 3d pointclouds, thanks to their depth information.
This transformation is compared to the one obtained with odometry. Separately, rotation and translation error are calculated, and a specific algorithm is applied to raise a flag when slip is detected.

This code will detect both cases of slip:
 - When camera detects movement, but the robot is not moving, which means that the robot is statically slipping.
 - When the robot is moving but the camera is not detecting movement, which means that the robot is moving on the same spot.

However, for the last case, it was noticed that the odometry was already compensated for this lack of movement of the robot, most likely by using the IMU information.

### Dio node

| __Subscribers__  | __Publishers__ |
| ------------- | ------------- |
| Corresponding RGB image  |   |
| Rigid elements’ mask  |   |
| Corresponding RGB image  |   |
| OPticalFlow final mask  |   |

Dio is a package created as the joint of both YOLACT++ and NeuFlow_v2. Inside the node, the final mask was created by adding the nonrigid mask together with the rigid component’s masks of those with a bigger or equal intersection of 70% with OpticalFlow mask.

