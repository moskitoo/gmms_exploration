#!/usr/bin/env python3
"""
ROS Node for processing MAVROS pose data and publishing odometry

This node subscribes to MAVROS pose messages, publishes odometry topic,
and broadcasts tf transforms for map, odometry_1, and tof_1 frames.
"""

import threading

import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
import tf2_ros


class SOGMMOdomNode:
    """
    ROS Node for processing MAVROS pose data and publishing odometry with tf transforms
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
        self.odom_topic = rospy.get_param("~odom_topic", "/odometry")
        
        # Frame IDs
        self.map_frame = "map"
        self.odom_frame = "odometry_1"
        self.tof_frame = "tof_1"

        # Publishers
        self.odom_pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=1)

        # Subscribers
        self.mavros_pose_sub = rospy.Subscriber(
            self.mavros_pose_topic, PoseStamped, self.mavros_pose_callback, queue_size=1
        )

        # TF2 broadcaster for transforms
        self.br = tf2_ros.TransformBroadcaster()

        # Threading for non-blocking processing
        self.processing_lock = threading.Lock()
        self.latest_pose = None
        self.processing_thread = None

        rospy.loginfo("SOGMM Odom Node initialized with frames: map=%s, odom=%s, tof=%s", 
                     self.map_frame, self.odom_frame, self.tof_frame)

    def mavros_pose_callback(self, msg):
        """
        Callback for PoseStamped messages
        """
        rospy.logdebug("Received pose message from MAVROS")
        with self.processing_lock:
            self.latest_pose = msg

        # Start processing in separate thread if not already running
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self.process_pose)
            self.processing_thread.daemon = True
            self.processing_thread.start()

    def process_pose(self):
        """
        Process the latest pose and publish odometry with tf transforms
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
            
            current_time = rospy.Time.now()
            
            # Convert Pose to Odometry message
            odom_msg = Odometry()
            odom_msg.header.stamp = current_time
            odom_msg.header.frame_id = self.odom_frame  # Parent frame
            odom_msg.child_frame_id = self.tof_frame    # Child frame
            odom_msg.pose.pose = msg.pose
            
            # Since this is ideal odometry, we can set covariance to very small values
            # Position covariance (x, y, z, roll, pitch, yaw) - 6x6 matrix flattened
            odom_msg.pose.covariance = [0.001, 0, 0, 0, 0, 0,  # x
                                       0, 0.001, 0, 0, 0, 0,    # y  
                                       0, 0, 0.001, 0, 0, 0,    # z
                                       0, 0, 0, 0.001, 0, 0,    # roll
                                       0, 0, 0, 0, 0.001, 0,    # pitch
                                       0, 0, 0, 0, 0, 0.001]    # yaw

            # Publish Odometry message
            self.odom_pub.publish(odom_msg)
            
            # Publish TF transforms
            self.publish_transforms(msg.pose, current_time)

        except Exception as e:
            rospy.logerr(f"Error processing pose: {str(e)}")
    
    def publish_transforms(self, pose, timestamp):
        """
        Publish tf2 transforms for the frame hierarchy
        """
        try:
            # 1. Transform from map to odometry_1 (identity since they're in the same place)
            map_to_odom = TransformStamped()
            map_to_odom.header.stamp = timestamp
            map_to_odom.header.frame_id = self.map_frame
            map_to_odom.child_frame_id = self.odom_frame
            map_to_odom.transform.translation.x = 0.0
            map_to_odom.transform.translation.y = 0.0
            map_to_odom.transform.translation.z = 0.0
            map_to_odom.transform.rotation.x = 0.0
            map_to_odom.transform.rotation.y = 0.0
            map_to_odom.transform.rotation.z = 0.0
            map_to_odom.transform.rotation.w = 1.0
            
            # 2. Transform from odometry_1 to tof_1 (the actual pose from MAVROS)
            odom_to_tof = TransformStamped()
            odom_to_tof.header.stamp = timestamp
            odom_to_tof.header.frame_id = self.odom_frame
            odom_to_tof.child_frame_id = self.tof_frame
            odom_to_tof.transform.translation.x = pose.position.x
            odom_to_tof.transform.translation.y = pose.position.y
            odom_to_tof.transform.translation.z = pose.position.z
            odom_to_tof.transform.rotation.x = pose.orientation.x
            odom_to_tof.transform.rotation.y = pose.orientation.y
            odom_to_tof.transform.rotation.z = pose.orientation.z
            odom_to_tof.transform.rotation.w = pose.orientation.w
            
            # Send both transforms
            self.br.sendTransform([map_to_odom, odom_to_tof])
            
        except Exception as e:
            rospy.logerr(f"Error publishing transforms: {str(e)}")

    def run(self):
        """
        Main execution loop - keeps the node running
        """
        rospy.loginfo("SOGMM Odom Node is running. Publishing odometry and tf transforms.")
        rospy.loginfo("Frame hierarchy: map -> odometry_1 -> tof_1")
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
