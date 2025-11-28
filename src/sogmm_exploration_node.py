#!/usr/bin/env python3

import threading

import matplotlib.cm as cm
import numpy as np
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker, MarkerArray

from gmms_exploration.msg import GaussianComponent, GaussianMixtureModel


class ExplorationGrid():
    def __init__(self, resolution=2.0):
        self.resolution = resolution
        self.cells = {}  # Dictionary to store sparse grid data: (ix, iy) -> list of uncertainties

    def update_grid(self, means, uncertainties, n_components):
        """
        Updates the sparse grid with uncertainty values from the GMM.
        
        Args:
            gmm: The GMM model object (expected to have means_ and uncertainty_ attributes).
            rtree: The R-tree index (unused in this implementation but kept for interface compatibility).
        """
        self.cells.clear()
        
        if means is None or uncertainties is None or n_components == 0:
            return

        # Extract means and uncertainties
        # means_ is (N, 4) where columns are x, y, z, intensity
        # means = gmm.means_
        # uncertainties = gmm.uncertainty_

        # Vectorized calculation of grid indices
        # We only care about x and y for the 2D grid
        ixs = np.floor(means[:, 0] / self.resolution).astype(int)
        iys = np.floor(means[:, 1] / self.resolution).astype(int)

        # Aggregate uncertainties into cells
        for i in range(n_components):
            key = (ixs[i], iys[i])
            if key not in self.cells:
                self.cells[key] = []
            self.cells[key].append(uncertainties[i])

    def get_grid_vis(self, frame_id="map", timestamp=None):
        """
        Generates a MarkerArray for visualizing the uncertainty grid.
        """
        if timestamp is None:
            timestamp = rospy.Time.now()

        marker_array = MarkerArray()
        
        # Add a marker to delete all previous markers in this namespace
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_marker.header.frame_id = frame_id
        delete_marker.header.stamp = timestamp
        delete_marker.ns = "exploration_grid"
        marker_array.markers.append(delete_marker)

        marker_id = 0
        for (ix, iy), u_values in self.cells.items():
            if not u_values:
                continue

            mean_uncertainty = np.mean(u_values)
            
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = timestamp
            marker.ns = "exploration_grid"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.lifetime = rospy.Duration(0)

            # Position: Center of the grid cell
            marker.pose.position.x = (ix + 0.5) * self.resolution
            marker.pose.position.y = (iy + 0.5) * self.resolution
            # Height (z) is scaled by uncertainty. 
            # We place the base at z=0 (or relative to map). 
            height = max(0.05, mean_uncertainty) # Minimum height for visibility
            marker.pose.position.z = height / 2.0

            marker.pose.orientation.w = 1.0

            # Scale: x and y match the grid resolution, z matches uncertainty
            marker.scale.x = self.resolution
            marker.scale.y = self.resolution
            marker.scale.z = height

            # Color: Map uncertainty to color
            # High uncertainty (1.0) -> Hot/Red, Low (0.0) -> Cool/Blue
            color = cm.plasma(mean_uncertainty)
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        return marker_array


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

        self.exploration_grid = ExplorationGrid()

        # Subscribers
        self.gmm_sub = rospy.Subscriber(
            self.gmm_topic, GaussianMixtureModel, self.gmm_callback, queue_size=1
        )

        #Publishers
        self.uct_id_pub = rospy.Publisher("/starling1/mpa/uct_id", Int32)

        self.grid_marker_pub = rospy.Publisher(
            "/starling1/mpa/grid_markers", MarkerArray, queue_size=1
        )

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

        self.exploration_grid.update_grid(means, uct, n_components)

        self.grid_marker_pub.publish(self.exploration_grid.get_grid_vis("map", msg.header.stamp))

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