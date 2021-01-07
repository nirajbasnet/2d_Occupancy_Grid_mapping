#!/usr/bin/env python
import rospy
import tf.transformations as tr
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped, Point,PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from math import cos, sin
import numpy as np
import copy
import numpy as np
import matplotlib.pyplot as plt
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Point, Quaternion


class Map():
    def __init__(self, map_x, map_y, grid_size):
        xsize = int(map_x/grid_size); ysize = int(map_y/grid_size)
        self.xsize = xsize # 200
        self.ysize = ysize # 200
        self.grid_size = grid_size # save this off for future use

        self.log_prob_map = np.zeros((self.xsize, self.ysize)) # set all to zero

        self.alpha = 1.0 # The assumed thickness of obstacles
        self.beta = 1.0*np.pi/180.0 # The assumed width of the laser beam
        self.z_max = 30.0 # The max reading from the laser

        # Log-Probabilities to add or remove from the map 
        self.l_occ = np.log(0.9/0.1)
        self.l_free = np.log(0.1/0.9)

        self.robot_x = 0.0
        self.robot_y = 0.0

    def update_map(self, odom_msg, laser_msg):
        robot_x = odom_msg.pose.pose.position.x
        robot_y = odom_msg.pose.pose.position.y

        self.robot_x = robot_x
        self.robot_y = robot_y

        quaternion = np.array([odom_msg.pose.pose.orientation.x,
                              odom_msg.pose.pose.orientation.y,
                              odom_msg.pose.pose.orientation.z,
                              odom_msg.pose.pose.orientation.w])
        _, _, robot_theta = tr.euler_from_quaternion(quaternion)

        robot_x_pixel = int(robot_x/self.grid_size)
        robot_y_pixel = int(robot_y/self.grid_size)

        # Bound it between [200,200], the map size in pixels
        robot_x_pixel = max(0, min(199, robot_x_pixel))
        robot_y_pixel = max(0, min(199, robot_y_pixel))

        unoccupied_pixel_list_xy = []
        occupied_pixel_list_xy = []

        for i in range(len(laser_msg.ranges)):
            scan_range = laser_msg.ranges[i]  # range measured for the particular scan
            scan_angle = laser_msg.angle_min + i*laser_msg.angle_increment  # bearing measured

            # scans at the max range are not obstacles
            if scan_range == laser_msg.range_max:
                continue

            # find position of cells in the global frame
            occupied_x = scan_range * cos(robot_theta + scan_angle) + robot_x # sth funky, should've been cos i thought
            occupied_y = scan_range * sin(robot_theta + scan_angle) + robot_y

            negative = False
            if (occupied_x < 0) or (occupied_y < 0):
                negative = True
            if negative == True:
                continue

            pixel_occupied_x = int(occupied_x/self.grid_size)
            pixel_occupied_y = int(occupied_y/self.grid_size)

            pixel_occupied_x = max(0, min(199, pixel_occupied_x))
            pixel_occupied_y = max(0, min(199, pixel_occupied_y))

            # Now get the points in-between that cover the line segment
            unoccupied_temp = list(self.bresenham(pixel_occupied_x, pixel_occupied_y,
                                                           robot_x_pixel, robot_y_pixel))
            unoccupied_temp.remove((pixel_occupied_x, pixel_occupied_y))  # since occupied pixels need to be removed.
            # Open space is the space "between" the robot and the occupied pixels
            unoccupied_pixel_list_xy += unoccupied_temp # Addition of list
            occupied_pixel_list_xy = occupied_pixel_list_xy + [(pixel_occupied_x, pixel_occupied_y)]

        for i in range(len(unoccupied_pixel_list_xy)-1):  # don't consider end corrdinates returned by bresenham
            index_x = unoccupied_pixel_list_xy[i][0]
            index_y = unoccupied_pixel_list_xy[i][1]
            self.log_prob_map[index_x][index_y] += copy.deepcopy(self.l_free)

        for i in range(len(occupied_pixel_list_xy)):
            index_x = occupied_pixel_list_xy[i][0]
            index_y = occupied_pixel_list_xy[i][1]
            self.log_prob_map[index_x][index_y] += copy.deepcopy(self.l_occ)

        print("x:"+str(self.robot_x)+" y:"+str(self.robot_y))

    def bresenham(self, x0, y0, x1, y1):
        """Yield integer coordinates on the line from (x0, y0) to (x1, y1).
        Input coordinates should be integers.
        The result will contain both the start and the end point.
        https://github.com/encukou/bresenham
        """
        dx = x1 - x0
        dy = y1 - y0

        xsign = 1 if dx > 0 else -1
        ysign = 1 if dy > 0 else -1

        dx = abs(dx)
        dy = abs(dy)

        if dx > dy:
            xx, xy, yx, yy = xsign, 0, 0, ysign
        else:
            dx, dy = dy, dx
            xx, xy, yx, yy = 0, ysign, xsign, 0

        D = 2 * dy - dx
        y = 0

        for x in range(dx + 1):
            yield x0 + x * xx + y * yx, y0 + x * xy + y * yy
            if D >= 0:
                y += 1
                D -= 2 * dx
            D += 2 * dy

    def to_message(self):
        """ Return a nav_msgs/OccupancyGrid representation of this map.
        https://w3.cs.jmu.edu/spragunr/CS354/labs/mapping/mapper.py
        """
        grid_msg = OccupancyGrid()

        # Set up the header.
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = "map"

        # .info is a nav_msgs/MapMetaData message. 
        grid_msg.info.resolution = self.grid_size
        grid_msg.info.width = self.xsize
        grid_msg.info.height = self.ysize
        
        # Rotated maps are not supported... quaternion represents no
        # rotation. 
        grid_msg.info.origin = Pose(Point(0, 0, 0),
                               Quaternion(0,0,0,1 ))

        # Flatten the numpy array into a list of integers from 0-100.
        # This assumes that the grid entries are probalities in the
        # range 0-1. This code will need to be modified if the grid
        # entries are given a different interpretation (like
        # log-odds).

        #flat_grid = np.array(self.log_prob_map.reshape((self.log_prob_map.size,)))
        flat_grid = np.array(self.log_prob_map.flatten(order='F'))

        flat_grid = np.clip(flat_grid, -100, 100) # too big or too small numbers will cause NaNs while calculating probabilitites
        
        probabilities_flat_grid = np.exp(flat_grid)/(1+np.exp(flat_grid))
        probabilities_flat_grid = probabilities_flat_grid*100 # 0 to 100 range
        grid_msg.data = list(np.round(probabilities_flat_grid))
        return grid_msg


class OccupancyGridMapping:
    def __init__(self, x, y, grid_size):
        
        # Initialize node
        rospy.init_node("occupancy_grid_mapping")

        # Get map from map server
        print("Wait for static_map from map server...")
        rospy.wait_for_service('static_map')
        map = rospy.ServiceProxy("static_map", GetMap)
        resp1 = map()
        self.grid_map = resp1.map
        print("Map resolution: " + str(self.grid_map.info.resolution))
        print("Map loaded.")

        # Subscribers
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.odometry_callback, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.laser_callback, queue_size=1)

        # Publisher
        self.publish_new_map = rospy.Publisher('/ogm_map', OccupancyGrid, queue_size=1)

        # Flags
        self.odom_initialized = False
        self.laser_initialized = False

        # Map dimensions and resolution (meters)
        self.map_x = x
        self.map_y = y
        self.grid_size = grid_size

        # Create empty map
        self.map = Map(self.map_x, self.map_y, self.grid_size)

        # Callback placeholders
        self.robot_odom = None
        self.laser_data = None

        self.laser_min_angle = None
        self.laser_max_angle = None
        self.laser_min_range = None
        self.laser_max_range = None

    def odometry_callback(self, msg):
        self.robot_odom = msg
        if not self.odom_initialized:
            self.odom_initialized = True

    def laser_callback(self, msg):
        if not self.laser_initialized:
            print("Got first laser callback.")
            self.laser_min_angle = msg.angle_min
            self.laser_max_angle = msg.angle_max
            self.laser_min_range = msg.range_min
            self.laser_max_range = msg.range_max
            self.laser_initialized = True
        self.laser_data = msg

    def update(self):
        if self.odom_initialized and self.laser_initialized:
            odom = copy.deepcopy(self.robot_odom)
            laser = copy.deepcopy(self.laser_data)

            self.map.update_map(odom, laser)

    def publish_map(self):
        self.publish_new_map.publish(self.map.to_message())

    def run(self):
        rate = rospy.Rate(0.5)
        while not rospy.is_shutdown():
            self.update()
            self.publish_map()
            rate.sleep()


if __name__ == '__main__':
    print("Start mapping")
    x_size = 10.0
    y_size = 10.0
    grid_size = 0.05
    og_mapping = OccupancyGridMapping(x_size, y_size, grid_size)
    og_mapping.run()
