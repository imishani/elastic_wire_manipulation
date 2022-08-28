Dual-Arm Robotic Manipulation of an Elastic Wire
---
Itamar Mishani
---

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
5. F/T Sensor [bota driver](https://gitlab.com/imishani/bota_driver)

* [natnet_ros](natnet_ros): Motion capture system package.
* [motoman_description](motoman_description): Contains all description files of the system (URDF files).
* [motoman_simulation](motoman_simulation): Rviz cable visualization scripts and robot movements and control scripts. See [motoman_simulation readme](motoman_simulation/README.md)
* [motoman_cable](motoman_cable): Cable properties approximator, recorder and planner scripts.
Exapmle: 
cable properties approximator
```sh
rosrun motoman_cable approx_c_live.py
```
* [learning_wire](learning_wire): Learning techniques to predict the shape of the wire based on Force Torque feedback.
