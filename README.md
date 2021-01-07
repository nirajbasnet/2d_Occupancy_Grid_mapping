# 2d Occupancy Gridmap implementation 
This code consists of implementation of 2D occupancy grid mapping. Turtlebot in a Gazebo environment is used for the demo.

## Usage 
__1. Launch the turtlebot demo world.__
```sh
roslaunch mapping turtlebot_in_stage.launch
```
__2. Run occupancy grid mapping node.__
```sh
roslaunch mapping mapping.launch
```
This uses ground truth to localize the robot and update the map. This is just for understanding the implementation of occupancy grid mapping. To create map while localizing the robot at the same time, 
SLAM algorithms are used. 

