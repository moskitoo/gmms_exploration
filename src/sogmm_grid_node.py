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
from gmms_exploration.srv import FlyTrajectory, GetViewpoint, GetViewpointResponse
from rospy.impl.tcpros_service import Service


class ExplorationGrid:
    def __init__(
        self,
        grid_cell_size: float = 0.05,
    ):
        self.grid_cell_size = grid_cell_size

        self.means = None
        self.covs = None
        self.uncertainties = None
        self.weights = None
        
        self.cluster_centroids = np.array([])
        self.average_gradient_magnitudes = np.array([])

    def update_grid(self, means, covs, uncertainties, weights):
        self.means = means
        self.covs = covs
        self.uncertainties = uncertainties
        self.weights = weights

        self.cluster_centroids, self.average_gradient_magnitudes = self.get_gaussian_frontiers(means, uncertainties, cluster_size=self.grid_cell_size, filter_grad_mean=False)


    def get_gaussian_frontiers(self, means, uncertainties, cluster_size=0.1, nms_radius=1.5, filter_grad_mean=False):

        # cluster size
        inverse_cluster_size = 1.0 / cluster_size

        # compute cluster indices
        cluster_indices = np.floor(means * inverse_cluster_size).astype(int)
        unique_cluster_indices, inverse_indices = np.unique(cluster_indices, axis=0, return_inverse=True)

        num_unique = len(unique_cluster_indices)
        sum_means = np.zeros((num_unique, 3))
        sum_grads = np.zeros((num_unique, 1))
        counts = np.zeros(num_unique, dtype=int)

        # compute means for each cluster
        np.add.at(sum_means, inverse_indices, means)
        
        if uncertainties.ndim == 1:
            uncertainties_reshaped = uncertainties[:, np.newaxis]
        else:
            uncertainties_reshaped = uncertainties
            
        np.add.at(sum_grads, inverse_indices, uncertainties_reshaped)
        np.add.at(counts, inverse_indices, 1)
        
        mean_positions = sum_means / counts[:, np.newaxis]
        mean_grads = sum_grads / counts[:, np.newaxis]

        # only keep centroids above grad mean
        if filter_grad_mean:
            avg_grad = np.mean(mean_grads)
            # std_grad = np.std(mean_grads)
            ths_grad = avg_grad# + std_grad
            grads_mask = (mean_grads > ths_grad).flatten()
            mean_positions = mean_positions[grads_mask]
            mean_grads = mean_grads[grads_mask]

        cluster_centroids = mean_positions
        average_gradient_magnitudes = mean_grads.flatten()
        
        sorted_indices = np.argsort(average_gradient_magnitudes)
        cluster_centroids = cluster_centroids[sorted_indices]
        average_gradient_magnitudes = average_gradient_magnitudes[sorted_indices]

        if len(cluster_centroids) > 0:
            distances = np.linalg.norm(cluster_centroids[:, np.newaxis] - cluster_centroids[np.newaxis, :], axis=2)
            keep_indices = np.ones(len(cluster_centroids), dtype=bool)

            for i in range(len(cluster_centroids)):
                if not keep_indices[i]:
                    continue

                # check if within radius
                within_radius = distances[i] < nms_radius
                keep_indices[within_radius] = False
                keep_indices[i] = True

            # filter
            nms_clusters = cluster_centroids[keep_indices]
            nms_gradient_magnitudes = average_gradient_magnitudes[keep_indices]

            # sort by gradient magnitude
            num_clusters = len(nms_clusters)
            top_indices = np.argsort(nms_gradient_magnitudes)[-num_clusters:]
            cluster_centroids = nms_clusters[top_indices]
            average_gradient_magnitudes = nms_gradient_magnitudes[top_indices]
        
        return cluster_centroids, average_gradient_magnitudes
    
    def get_viewpoint(self):
        if len(self.average_gradient_magnitudes) == 0:
            rospy.logwarn("No frontiers found, returning default viewpoint")
            return 0.0, 0.0

        max_idx = np.argmax(self.average_gradient_magnitudes)
        
        world_x, world_y, world_z = self.cluster_centroids[max_idx]
        uncertainty = self.average_gradient_magnitudes[max_idx]

        rospy.logdebug(f"Most uncertain cell index: {max_idx}")
        rospy.logdebug(f"Most uncertain cell uncertainty: {uncertainty}")
        rospy.logdebug(f"Most uncertain cell coordinates: {world_x}, {world_y}")

        return world_x, world_y

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

        for i in range(self.average_gradient_magnitudes.shape[0]):
            world_x, world_y, world_z = self.cluster_centroids[i]
            uncertainty = self.average_gradient_magnitudes[i]

            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = timestamp
            marker.ns = "exploration_grid"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.lifetime = rospy.Duration(0)

            marker.pose.position.x = world_x
            marker.pose.position.y = world_y
            marker.pose.position.z = world_z

            # Set orientation
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            # Scale: x and y match the grid resolution, z matches uncertainty
            marker.scale.x = self.grid_cell_size
            marker.scale.y = self.grid_cell_size
            marker.scale.z = self.grid_cell_size

            # Color: Map uncertainty to color
            color = cm.plasma(1-uncertainty)
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        return marker_array
    
    def get_means_vis(self, means, frame_id="map", timestamp=None):
        """
        Generates a MarkerArray for visualizing the means as spheres.
        """
        if timestamp is None:
            timestamp = rospy.Time.now()

        marker_array = MarkerArray()

        # Add a marker to delete all previous markers in this namespace
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_marker.header.frame_id = frame_id
        delete_marker.header.stamp = timestamp
        delete_marker.ns = "means_vis"
        marker_array.markers.append(delete_marker)

        marker_id = 0

        for i in range(means.shape[0]):
            world_x, world_y, world_z = means[i]

            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = timestamp
            marker.ns = "means_vis"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.lifetime = rospy.Duration(0)

            marker.pose.position.x = world_x
            marker.pose.position.y = world_y
            marker.pose.position.z = world_z

            # Set orientation
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            # Scale
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            # Color
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        return marker_array

    def compose_grid_msg(self):
        msg = Grid()

        msg.means.data = self.cluster_centroids.flatten()
        msg.uncertainties.data = self.average_gradient_magnitudes

        return msg


class SOGMMGridNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("sogmm_grid_node", anonymous=False)

        # Set logging level based on parameter
        log_level = rospy.get_param("~log_level", "INFO").upper()
        if log_level == "DEBUG":
            rospy.loginfo("Setting log level to DEBUG")
            
            logging.getLogger("rosout").setLevel(logging.DEBUG)

        # Parameters
        self.gmm_topic = rospy.get_param("~gmm_topic", "/starling1/mpa/gmm")

        self.exploration_grid = ExplorationGrid(
            grid_cell_size=2.0
        )

        # Subscribers
        self.gmm_sub = rospy.Subscriber(
            self.gmm_topic, GaussianMixtureModel, self.gmm_callback, queue_size=1
        )

        self.viewpoint_service = rospy.Service("get_viewpoint", GetViewpoint, self.get_viewpoint_callback)

        # Publishers
        self.uct_id_pub = rospy.Publisher("/starling1/mpa/uct_id", Int32)

        self.grid_marker_pub = rospy.Publisher(
            "/starling1/mpa/grid_markers", MarkerArray, queue_size=1
        )

        self.means_marker_pub = rospy.Publisher(
            "/starling1/mpa/means_markers", MarkerArray, queue_size=1
        )

        self.grid_pub = rospy.Publisher(
            "/starling1/mpa/grid", Grid, queue_size=1
        )

        self.reached_target = True

    def gmm_callback(self, msg: GaussianMixtureModel):
        rospy.logdebug("Received GMM")

        # Process pose directly in callback to avoid threading issues
        self.process_gmm(msg)

    def process_gmm(self, msg: GaussianMixtureModel):
        gmm = msg.components
        n_components = len(gmm)

        if n_components > 0:
            means = np.empty([n_components, 3])
            covs = np.empty([n_components, 9])
            uct = np.empty(n_components)
            weights = np.empty(n_components)

            for i in range(n_components):
                means[i] = gmm[i].mean
                covs[i] = gmm[i].covariance
                uct[i] = gmm[i].uncertainty
                weights[i] = gmm[i].weight

            # rospy.logdebug(f"means shape: {means.shape}")
            # rospy.logdebug(f"covs shape: {covs.shape}")
            # rospy.logdebug(f"unct shape: {uct.shape}")

            max_uct_id = np.argmax(uct)

            # rospy.logdebug(f"max_uct_id: {max_uct_id}")
            uct_msg = Int32()
            uct_msg.data = max_uct_id
            self.uct_id_pub.publish(uct_msg)

            # rospy.logdebug(f"max_uct_id mean: {gmm[max_uct_id].mean}")
            # rospy.logdebug(f"max_uct_id covariance: {gmm[max_uct_id].covariance}")
            # rospy.logdebug(f"max_uct_id uncertainty: {gmm[max_uct_id].uncertainty}")

            # rospy.logdebug(f"max_uct_id fusion count: {gmm[max_uct_id].fusion_count}")
            # rospy.logdebug(
            #     f"max_uct_id observation_count: {gmm[max_uct_id].observation_count}"
            # )
            # rospy.logdebug(
            #     f"max_uct_id last_displacement: {gmm[max_uct_id].last_displacement}"
            # )

            # rospy.logdebug(f"grid: \n{self.exploration_grid.grid}")

            self.exploration_grid.update_grid(means, covs, uct, weights)

            self.grid_pub.publish(self.exploration_grid.compose_grid_msg())

            self.grid_marker_pub.publish(
                self.exploration_grid.get_grid_vis("map", msg.header.stamp)
            )

            self.means_marker_pub.publish(
                self.exploration_grid.get_means_vis(means, "map", msg.header.stamp)
            )

    def get_viewpoint_callback(self, req):
        x, y = self.exploration_grid.get_viewpoint()
        goal = Point()
        goal.x = x
        goal.y = y
        goal.z = 1.0
        return GetViewpointResponse(goal)

    def run(self):
        """
        Main execution loop - keeps the node running
        """
        rospy.loginfo("SOGMM Grid Node is running.")
        rospy.spin()


def main():
    """
    Main function
    """
    try:
        node = SOGMMGridNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("SOGMM Grid Node interrupted")
    except Exception as e:
        rospy.logerr(f"Error in SOGMM Grid Node: {str(e)}")


if __name__ == "__main__":
    main()
