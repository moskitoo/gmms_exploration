#!/usr/bin/env python3
"""
ROS Node for Real-time Point Cloud SOGMM Processing and Visualization

This node subscribes to PointCloud2 messages, processes them using SOGMM,
and publishes visualization markers for RViz.
"""

import threading

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class SOGMMOdomNode:
    """
    ROS Node for processing point clouds with SOGMM and visualizing results
    """

    def __init__(self):
        # Initialize ROS node
        rospy.init_node("sogmm_odom_node", anonymous=True)

        # Set logging level based on parameter
        log_level = rospy.get_param("~log_level", "INFO").upper()
        if log_level == "DEBUG":
            rospy.loginfo("Setting log level to DEBUG")
            import logging
            logging.getLogger('rosout').setLevel(logging.DEBUG)

        # Parameters
        self.mavros_pose_topic = rospy.get_param(
            "~mavros_pose_topic", "/starling1/mavros/local_position/pose"
        )

        # Publishers
        self.odom_pub = rospy.Publisher("/starling1/odometry", Odometry, queue_size=1)

        # Subscribers
        self.mavros_pose_sub = rospy.Subscriber(
            self.mavros_pose_topic, PoseStamped, self.mavros_pose_callback, queue_size=1
        )

        # Threading for non-blocking processing
        self.processing_lock = threading.Lock()
        self.latest_pose = None
        self.processing_thread = None

        rospy.loginfo("SOGMM Odom Node initialized")

    def mavros_pose_callback(self, msg):
        """
        Callback for PoseStamped messages
        """
        rospy.loginfo("Received pose message from MAVROS")
        with self.processing_lock:
            self.latest_pose = msg

        # Start processing in separate thread if not already running
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self.process_pose)
            self.processing_thread.daemon = True
            self.processing_thread.start()

    def process_pose(self):
        """
        Process the latest pose with SOGMM
        """
        with self.processing_lock:
            if self.latest_pose is None:
                return
            msg = self.latest_pose
            self.latest_pose = None

        try:
            rospy.logdebug(
                f"Received Pose: Position({msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z})"
            )
            rospy.logdebug(
                f"Received Pose: Orientation({msg.pose.orientation.x}, {msg.pose.orientation.y}, {msg.pose.orientation.z}, {msg.pose.orientation.w})"
            )
            # Convert Pose to Odometry message
            odom_msg = Odometry()
            odom_msg.header = msg.header
            odom_msg.child_frame_id = "odometry_1"
            odom_msg.pose.pose = msg.pose

            # Publish Odometry message
            self.odom_pub.publish(odom_msg)

        except Exception as e:
            rospy.logerr(f"Error processing pose: {str(e)}")

    def run(self):
        """
        Main execution loop
        """
        rospy.loginfo("SOGMM Odom Node is running. Waiting for pose data...")
        rospy.spin()


def main():
    """
    Main function
    """
    try:
        node = SOGMMOdomNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("SOGMM Odom Node interrupted")
    except Exception as e:
        rospy.logerr(f"Error in SOGMM Odom Node: {str(e)}")


if __name__ == "__main__":
    main()
