#!/usr/bin/env python3

import logging
import threading
from typing import Tuple

import matplotlib.cm as cm
import numpy as np
import rospy
import tf
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
from nav_msgs.msg import Odometry
from rtree import index
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker, MarkerArray

from gmms_exploration.msg import GaussianComponent, GaussianMixtureModel, Grid
from gmms_exploration.srv import FlyTrajectory, GetViewpoint
from scripts.topological_graph import TopoTree

class SOGMMExplorationNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("sogmm_exploration_node", anonymous=True)

        # Set logging level based on parameter
        log_level = rospy.get_param("~log_level", "INFO").upper()
        if log_level == "DEBUG":
            rospy.loginfo("Setting log level to DEBUG")

            logging.getLogger("rosout").setLevel(logging.DEBUG)

        self.reached_target = True

        self.topo_tree = TopoTree()

        self.fly_trajectory_client = rospy.ServiceProxy("/starling1/fly_trajectory", FlyTrajectory)
        self.get_viewpoint_client = rospy.ServiceProxy("get_viewpoint", GetViewpoint)

        self.grid_sub = rospy.Subscriber("/starling1/mpa/grid", Grid, self.grid_callback)

        self.viewpoint_marker_pub = rospy.Publisher("viewpoint_marker", Marker, queue_size=1)
        self.path_marker_pub = rospy.Publisher("path_marker", Marker, queue_size=1)

        self.viewpoint_list = [Point(0.0, -1.0, 1.0), Point(3.0, -1.0, 1.0), Point(3.0, 2.0, 1.0), Point(0.0, 2.0, 1.0)]
        self.last_id = 0

        self.ftr_goal_tol_ = rospy.get_param('~ftr_goal_tol', 1.0)
        self.fail_pos_tol_ = rospy.get_param('~fail_pos_tol', 0.1)
        self.fail_yaw_tol_ = rospy.get_param('~fail_yaw_tol', 0.1)

    def execute_exploration(self):

        if self.reached_target:
            if self.path_to_ftr is not None and self.current_waypoint_index < len(self.path_to_ftr):
                # Get the next viewpoint from the path
                next_waypoint = self.path_to_ftr[self.current_waypoint_index]
                self.viewpoint = Point()
                self.viewpoint.x = next_waypoint[0]
                self.viewpoint.y = next_waypoint[1]
                self.viewpoint.z = 1.0  # Assuming a fixed height

                rospy.loginfo(f"Sending waypoint {self.current_waypoint_index + 1}/{len(self.path_to_ftr)}: {self.viewpoint}")

                self.viewpoint_marker_pub.publish(self.create_viewpoint_marker(self.viewpoint))

                # Pass the extracted Point object to the fly service
                response = self.fly_trajectory_client(self.viewpoint, False)
                self.reached_target = response.success

                if self.reached_target:
                    self.current_waypoint_index += 1
                
                if self.current_waypoint_index >= len(self.path_to_ftr):
                    rospy.loginfo("Completed path.")
                    self.path_to_ftr = None # Clear path after completion
            else:
                rospy.loginfo("No active path to follow. Waiting for new path.")
                # Optionally, you can add logic here to request a new plan if idle.
                rospy.sleep(1.0) # Wait before checking again

        else:
            # Retry the previously stored viewpoint
            if hasattr(self, 'viewpoint'):
                rospy.loginfo("Retrying previous viewpoint...")
                self.viewpoint_marker_pub.publish(self.create_viewpoint_marker(self.viewpoint))
                response = self.fly_trajectory_client(self.viewpoint, False)
                self.reached_target = response.success
            else:
                # Fallback if no viewpoint is stored
                self.reached_target = True
    
    def grid_callback(self, msg):
        means = msg.means.data
        uncertainties = msg.uncertainties.data

        means = np.array(means).reshape((-1,3))

        path = self.topo_tree.spin(means, uncertainties, self.ftr_goal_tol_, self.fail_pos_tol_, self.fail_yaw_tol_)

        if path is None:
            rospy.logerr("No path to frontier found")
        else:
            # New path received, reset and store it
            self.path_to_ftr = path
            self.current_waypoint_index = 0
            self.reached_target = True # Trigger execution of the new path
            print("path to ftr: ", self.path_to_ftr)
            path_marker = self.create_path_marker(self.path_to_ftr)
            self.path_marker_pub.publish(path_marker)

    @staticmethod
    def create_viewpoint_marker(viewpoint):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "sogmm_ellipsoids"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(2.0)

        # Set position
        marker.pose.position.x = viewpoint.x
        marker.pose.position.y = viewpoint.y
        marker.pose.position.z = viewpoint.z
        
        # Set Orientation (Identity)
        marker.pose.orientation.w = 1.0

        # Set Scale
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5

        # Set Color
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        return marker

    @staticmethod
    def create_path_marker(path):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "sogmm_path"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(0)

        # Set scale
        marker.scale.x = 0.1

        # Set Color
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.
        marker.color.a = 1.0

        # Set points
        for point in path:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = 1.0  # Assuming a fixed height for visualization
            marker.points.append(p)

        return marker

    def run(self):
        """
        Main execution loop - keeps the node running
        """
        rospy.loginfo("SOGMM Exploration Node is running.")
        
        rate = rospy.Rate(1.0) 

        while not rospy.is_shutdown():
            try:
                self.execute_exploration()
            except rospy.ServiceException as e:
                rospy.logerr(f"Service call failed: {e}")
                rospy.sleep(1.0) # Wait a bit before retrying
            except rospy.ROSInterruptException:
                break
            except Exception as e:
                rospy.logerr(f"Unexpected error in loop: {e}")
                rospy.sleep(1.0)
            
            rate.sleep()


def main():
    """
    Main function
    """
    try:
        node = SOGMMExplorationNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("SOGMM Exploration Node interrupted")
    except Exception as e:
        rospy.logerr(f"Error in SOGMM Exploration Node: {str(e)}")


if __name__ == "__main__":
    main()
