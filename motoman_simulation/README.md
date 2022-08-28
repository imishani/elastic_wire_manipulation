Visualization and Control Package
---

After loading the environment use the following scripts as follows:

* [cable_markers](scripts/cable_markers.py): For adding marker line which uses the analytical model [rodeval.py](scripts/rodeval.py) to predict the shape of the wire based on the F/T feedback.
```sh
rosrun motoman_simulation cable_makers.py
```
* [cable_markers_nn_direct](scripts/cable_markers_nn_direct.py): For adding marker line which uses the learned model [learning_wire](../learning_wire) to predict the shape of the wire based on the F/T feedback.
```sh
rosrun motoman_simulation cable_makers_nn_direct.py
```
* [move_agro_rectangle](scripts/move_agro_rectangle.py): Moves the arms randomly in the space.
```sh
rosrun motoman_simulation move_agro_rectangle
```
* [Movement_RRTStar](scripts/Movement_RRTStar.py): Plan a path in the F/T space, visualize the path and perform the manipulation in open loop.
```sh
rosrun motoman_simulation Movement_RRTStar.py
```
* [control](scripts/control.py): Plan a path in the F/T space, visualize the path and perform the manipulation in closed loop control.
```sh
rosrun motoman_simulation control.py
```