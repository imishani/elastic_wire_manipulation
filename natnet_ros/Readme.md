# NatNet ROS driver
### Communicate with Motive app

This package is an update of [NatNet 3 ROS driver](https://github.com/mje-nz/natnet_ros)

The NatNet protocol is used for streaming live motion capture data (rigid bodies, skeletons etc) across a network.
If you're just looking to record some motion capture data and you don't need to synchronise it with other sensors, the easiest method is to record it in Motive directly.

Only tested on Motive 2.2
Skeletons, force plates, and other peripherals probably mostly work in the underlying library but are not tested and are not published as ROS topics.

Supported on ROS Melodic and Kinetic, but it works on Indigo and probably on newer distributions too.
The underlying library supports Python 2.7 and 3.4+ on Linux, Windows and macOS.
Both this package and the library have CI for all supported platforms.


## TODO

* Add duplicate marker ID check and report bug
* Document "solver replaces occlusion" behaviour and implement workaround
* Rename MocapFrame to FrameOfData
* Make clock sync optional
* Release python_natnet on PyPI
* Add CONTRIBUTING etc
* Make sure unknown message IDs, failure to decode messages etc doesn't result in a crash
* Make sure to use the same variable signedness as the SDK
* Use rospy.spin
* Add warning when unnecessary data is being streamed
* Add guidance for appropriate Motive settings


## Installation

```
mkdir -p catkin_ws/src/
cd catkin_ws/src
git clone --recursive https://gitlab.com/imishani/natnet_ros.git
cd ..
rosdep install -y --from-paths src --ignore-src
catkin_make
source devel/setup.bash
cd catkin_ws/src/natnet_ros/python_natnet
sudo python setup.py install
```

Run with _Motive instance_ connected to the same network as your computer:
_Automatic search for IP:_
```
rosrun natnet_ros client 
```
_With specific IP:_
```
rosrun natnet_ros client _server:=(Motive IP)
```

Else, if you have a ROS Master running, you can test it with a fake server:

```
rosrun natnet_ros client _fake:=true
```


## ROS API

The `client` node connects to a NatNet server and publishes the data as ROS topics.
The data and descriptions are used directly, so make to give your rigid bodies unique streaming IDs and sensible names, and set the axis convention to z-up.
If you need Motive to use y-up for some reason, you can fix it with:

```
rosrun tf static_transform_publisher 0 0 0 0 0 1.57079632679 mocap_z_up mocap 100
```


### Published topics

* `~rigid_bodies/<name>/pose` (geometry_msgs/PoseStamped)

  Rigid body pose.
  The name is taken from the NatNet stream (i.e., from Motive), with a bit of an attempt to make sure it's a valid ROS name.

* `~rigid_bodies/<name>/marker<id>` (geometry_msgs/PointStamped)

  Position of each marker of each rigid body.
  The ID is taken from the NatNet stream (i.e., from Motive).

* `~rigid_bodies/<name>/markers` (natnet_msgs/MarkerList)

  Position and ID of each marker of each marker of each rigid body as a list.

* `~markers/leftovers` (natnet_msgs/MarkerList)

  Position and ID of any markers that aren't in a rigid body.

* `~markers/vis` (visualization_msgs/Marker)

  Position of all markers, for visualization with Rviz.
  The size is set to the average estimated marker size.


### Parameters

* `~server` (`string`, optional)

  NatNet server to connect to; will autodetect if not provided.

* `~fake` (`bool`, default: false)

  Use fake data instead of connecting to a real server.

* `~debug` (`bool`, default: false)

  Enable debug logging.

* `~rate` (`int`, default: 100)

  If `fake` is true, the rate at which to publish the fake data (in Hz).

* `~mocap_frame` (`string`, default: 'mocap')

  The name of the `tf` frame for the mocap data.

* `~tf_remap` (`dict`, optional)

  A dictionary mapping rigid body names to the name of the `tf` frame they should be published as.
  For example, `{'UAV': 'base_link'}` would cause the pose for the rigid body named `UAV` to be published as a transform from `mocap` to `base_link`.

  Dict params are a bit tricky to use in Kinetic or earlier: a literal works with `rosrun`, but using a dict literal as a parameter value in a launch file doesn't work until Lunar.
  Using a `rosparam` tag works though.

