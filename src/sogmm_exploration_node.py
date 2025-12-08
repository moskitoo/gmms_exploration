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

from gmms_exploration.msg import GaussianComponent, GaussianMixtureModel
from gmms_exploration.srv import FlyTrajectory, GetViewpoint



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


        self.fly_trajectory_client = rospy.ServiceProxy("/starling1/fly_trajectory", FlyTrajectory)
        self.get_viewpoint_client = rospy.ServiceProxy("get_viewpoint", GetViewpoint)

        self.viewpoint_marker_pub = rospy.Publisher("viewpoint_marker", Marker, queue_size=1)

        self.viewpoint_list = [Point(0.0, -1.0, 1.0), Point(3.0, -1.0, 1.0), Point(3.0, 2.0, 1.0), Point(0.0, 2.0, 1.0)]
        self.last_id = 0

    def execute_exploration(self):

        if self.reached_target:
            # Call service to get the response object
            viewpoint_resp = self.get_viewpoint_client()
            # Extract the 'goal' (geometry_msgs/Point) from the response
            self.viewpoint = viewpoint_resp.goal

            # if self.last_id == len(self.viewpoint_list):
            #     self.last_id = 0
            # self.viewpoint = self.viewpoint_list[self.last_id]
            # self.last_id += 1
            
            rospy.loginfo(f"New viewpoint received: {self.viewpoint}")

            self.viewpoint_marker_pub.publish(self.create_viewpoint_marker(self.viewpoint))

            # Pass the extracted Point object to the fly service
            response = self.fly_trajectory_client(self.viewpoint, False)
            self.reached_target = response.success
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

        rospy.loginfo(f"Response message: {response.message}")

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
