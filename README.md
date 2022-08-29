Dual-Arm Robotic Manipulation of an Elastic Wire
---
Itamar Mishani
---
Partial code implementations of my thesis about dual arm robotic manipulation of elastic wires. 

**paper**: https://ieeexplore.ieee.org/abstract/document/9618864

![](/pic/robot_wire.jpeg)
![](/pic/est.png)
![](/pic/state2_n.png)


### Abstract
Manipulation of wires has been a challenging task and main interest for many decades. There have been many attempts to use visual perception and image segmentation to perform wire manipulation. However, wire manipulation in a cluttered environment where there are many visual occlusions and uncertainties (due to poor lighting or shadows) is hard. Giving a robot the ability to manipulate wires with high certainty is necessary and requires rapid reasoning of its shape in real-time. Furthermore, after having this ability, planning and control of wire manipulations is required. However, there is no efficient ability to do so without visual perception. 

Recent work has shown that the shape of an elastic wire can be defined by a very simple representation. This representation can be interpreted as forces and torques at one end of the wire. To begin with, we experimentally analyzed the theoretical foundation. We deployed a dual-arm robotic system able to accurately manipulate an elastic wire. The system does not require complex visual perception and is able to reason about the shape of the wire by solely sensing forces and torques on one arm. Furthermore, we proposed a full framework in which the mechanical properties of the wire are rapidly approximated in real-time. Then, a simple control rule based on Force/Torque (F/T) feedback is used to manipulate the wire to some goal or track a planned path. 

However, the model used to develop the system relies on assumptions that may not be met in real-world wires and does not take gravity into account. Therefore, the model cannot be applied to any wire with accurate shape estimation. Additionally, as the model does not consider the non-linearity of the F/T sensor, it is necessary to solve the non-linear non convex inverse problem, i.e. from wire shape to the F/T state, which is computationally expensive. As a consequence, we investigated the learning of a model to estimate the shape of a wire solely from measurements of the F/T state. We propose to train a novel learning model which can both act as a descriptor of the wire where F/T states can be mapped to its shape and as a solver of the inverse problem where a desired goal shape can be mapped to an F/T state. Additionally, we trained a different model, with the same dataset, which gives the robot the ability to execute a planned path.

### Requirements
* Ubuntu 18.04
* ROS (Robotic Operating System)
* python2.7 and python3

**Note:** packages requirements are inside each ROS package.

## ROS Packages
* [moveit_config_cable](moveit_config_cable): Launching robot to Rviz and Moveit pipline to control motoman SDA10F dual-arm robot. **YASKAWA ROS package (motoman) must also be installed in your workspace**
```sh
roslaunch moveit_config_cable launch_robot_moveit_config.launch
```
This launch file loads the entire system:
1. SDA10F motoman
2. Rviz
3. Moveit
4. Motion capture system [natent ros package](natnet_ros)
After launching the robot and the motion capture system, run the following in order to start streaming 
calibrated F/T Sensor data ([bota driver](https://gitlab.com/imishani/bota_driver/-/tree/main/bota_demo) package)
```sh
roslaunch bota_demo bota_demo.launch
```

* [natnet_ros](natnet_ros): Motion capture system package.
* [motoman_description](motoman_description): Contains all description files of the system (URDF files).
* [motoman_simulation](motoman_simulation): Rviz cable visualization scripts and robot movements and control scripts. See [motoman_simulation readme](motoman_simulation/README.md)
* [motoman_cable](motoman_cable): Cable properties approximator using PSO algorithm (Particle Swarm Optimization), recorder and [planner](motoman_cable/scripts/path_plan) (RRT and RRT*) scripts.

Exapmle: 
cable properties approximator
```sh
rosrun motoman_cable approx_c_live.py
```
* [learning_wire](learning_wire): Deep learning techniques to predict the shape of the wire based on Force Torque feedback.
