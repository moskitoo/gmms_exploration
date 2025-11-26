#!/usr/bin/env python3

import threading

import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry

from gmms_exploration.msg import GaussianComponent, GaussianMixtureModel
from std_msgs.msg import Int32


class SOGMMExplorationNode:


    def __init__(self):
        # Initialize ROS node
        rospy.init_node("sogmm_exploration_node", anonymous=True)

        # Set logging level based on parameter
        log_level = rospy.get_param("~log_level", "INFO").upper()
        if log_level == "DEBUG":
            rospy.loginfo("Setting log level to DEBUG")
            import logging
            logging.getLogger('rosout').setLevel(logging.DEBUG)

        # Parameters
        self.gmm_topic = rospy.get_param("~gmm_topic", "/starling1/mpa/gmm")

        # Subscribers
        self.gmm_sub = rospy.Subscriber(
            self.gmm_topic, GaussianMixtureModel, self.gmm_callback, queue_size=1
        )

        #Publishers
        self.uct_id_pub = rospy.Publisher("/starling1/mpa/uct_id", Int32)

    def gmm_callback(self, msg: GaussianMixtureModel):
        
        rospy.logdebug("Received GMM")
        
        # Process pose directly in callback to avoid threading issues
        self.process_gmm(msg)

    def process_gmm(self, msg: GaussianMixtureModel):
        
        gmm = msg.components
        n_components = msg.n_components

        means = np.empty([n_components, 3])
        covs = np.empty([n_components, 9])
        uct = np.empty(n_components)

        for i in range(n_components):
            means[i] = gmm[i].mean
            covs[i] = gmm[i].covariance
            uct[i] = gmm[i].uncertainty

        rospy.logdebug(f"means shape: {means.shape}")
        rospy.logdebug(f"covs shape: {covs.shape}")
        rospy.logdebug(f"unct shape: {uct.shape}")

        max_uct_id = np.argmax(uct)

        rospy.logdebug(f"max_uct_id: {max_uct_id}")
        uct_msg = Int32()
        uct_msg.data = max_uct_id
        self.uct_id_pub.publish(uct_msg)

        rospy.logdebug(f"max_uct_id mean: {gmm[max_uct_id].mean}")
        rospy.logdebug(f"max_uct_id covariance: {gmm[max_uct_id].covariance}")
        rospy.logdebug(f"max_uct_id uncertainty: {gmm[max_uct_id].uncertainty}")

        rospy.logdebug(f"max_uct_id fusion count: {gmm[max_uct_id].fusion_count}")
        rospy.logdebug(f"max_uct_id observation_count: {gmm[max_uct_id].observation_count}")
        rospy.logdebug(f"max_uct_id last_displacement: {gmm[max_uct_id].last_displacement}")

        #here we can also sample viewpoints around that gaussian
        # we need to check if there is no collision with other etc?

        # then we can send a command to the poly traj server 


    def run(self):
        """
        Main execution loop - keeps the node running
        """
        rospy.loginfo("SOGMM Exploration Node is running.")
        rospy.spin()


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