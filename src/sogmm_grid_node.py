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
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker, MarkerArray

from gmms_exploration.msg import GaussianComponent, GaussianMixtureModel, Grid
from gmms_exploration.srv import FlyTrajectory, GetViewpoint, GetViewpointResponse
from rospy.impl.tcpros_service import Service
from typing import List, Tuple


class ExplorationGrid:
    def __init__(
        self,
        grid_cell_size: float = 0.05,
        static_cluster_center: bool = False,
        map_bounds: List[Tuple] = [(-0.65, 9.0), (-1.0, 4.5), (0.0, 3.0)],
        unexplored_uncertainty: float = 1.0,
        grid_offset = 0.5
    ):
        self.grid_cell_size = grid_cell_size

        self.means = None
        self.covs = None
        self.uncertainties = None
        self.weights = None
        
        self.static_cluster_center = static_cluster_center
        self.unexplored_uncertainty = unexplored_uncertainty
        
        self.map_bounds = map_bounds
        self.map_bounds = [
            [map_bounds[0][0] - grid_offset, map_bounds[0][1] + grid_offset],
            [map_bounds[1][0] - grid_offset, map_bounds[1][1] + grid_offset],
            [map_bounds[2][0] - grid_offset, map_bounds[2][1] + grid_offset]
        ]

        if self.static_cluster_center:
            # Calculate number of cells in each dimension
            grid_length = int(np.round((self.map_bounds[0][1] - self.map_bounds[0][0]) / self.grid_cell_size))
            grid_width = int(np.round((self.map_bounds[1][1] - self.map_bounds[1][0]) / self.grid_cell_size))
            grid_height = int(np.round((self.map_bounds[2][1] - self.map_bounds[2][0]) / self.grid_cell_size))
            
            # FIX: Create coordinates at cell CENTERS using arange for consistency
            x_coords = self.map_bounds[0][0] + (np.arange(grid_length) + 0.5) * self.grid_cell_size
            y_coords = self.map_bounds[1][0] + (np.arange(grid_width) + 0.5) * self.grid_cell_size
            z_coords = self.map_bounds[2][0] + (np.arange(grid_height) + 0.5) * self.grid_cell_size
            
            # Create meshgrid and reshape to (N, 3) array
            xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
            self.cluster_centroids = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=-1)
            
            # Initialize gradient magnitudes to unexplored value
            self.average_gradient_magnitudes = np.full(len(self.cluster_centroids), unexplored_uncertainty)
            
            # Store grid dimensions for indexing (x, y, z order)
            self.grid_dims = (grid_length, grid_width, grid_height)
            
            # FIX: min_bounds should be the START of the first cell (grid edge)
            self.min_bounds = np.array([
                self.map_bounds[0][0],
                self.map_bounds[1][0],
                self.map_bounds[2][0]
            ])
            
            rospy.loginfo(f"Grid dimensions (x,y,z): {self.grid_dims}")
            rospy.loginfo(f"Grid bounds: {self.map_bounds}")
            rospy.loginfo(f"Min bounds: {self.min_bounds}")
            rospy.loginfo(f"Total grid cells: {len(self.cluster_centroids)}")
        else:
            self.cluster_centroids = np.array([])
            self.average_gradient_magnitudes = np.array([])


    def update_grid(self, means, covs, uncertainties, weights):
        self.means = means
        self.covs = covs
        self.uncertainties = uncertainties
        self.weights = weights

        if self.static_cluster_center:
            # Don't reset - grid persists values from previous iterations
            # Only updated cells will change based on new measurements
            self.get_gaussian_frontiers(means, uncertainties, cluster_size=self.grid_cell_size, filter_grad_mean=True, nms_filter=False)
        else:
            self.cluster_centroids, self.average_gradient_magnitudes = self.get_gaussian_frontiers(means, uncertainties, cluster_size=self.grid_cell_size, filter_grad_mean=True, nms_filter=False)
        # self.cluster_centroids_not_filtered, self.average_gradient_magnitudes_not_filtered = self.get_gaussian_frontiers(means, uncertainties, cluster_size=self.grid_cell_size, filter_grad_mean=False)

        # rospy.logdebug(f"centroid number         : {self.cluster_centroids_not_filtered.shape}\n")
        rospy.logdebug(f"centroids shape: {self.cluster_centroids.shape}")


    def get_gaussian_frontiers(self, means, uncertainties, cluster_size=0.1, nms_radius=1.5, filter_grad_mean=False, nms_filter=False):

        # cluster size
        inverse_cluster_size = 1.0 / cluster_size

        # compute cluster indices
        if self.static_cluster_center:
            # For static mode, compute indices relative to min_bounds
            cluster_indices = np.floor((means - self.min_bounds) * inverse_cluster_size).astype(int)
            # Clip to valid range
            cluster_indices = np.clip(cluster_indices, 
                                    [0, 0, 0], 
                                    [self.grid_dims[0]-1, self.grid_dims[1]-1, self.grid_dims[2]-1])
        else:
            cluster_indices = np.floor(means * inverse_cluster_size).astype(int)
        
        unique_cluster_indices, inverse_indices = np.unique(cluster_indices, axis=0, return_inverse=True)

        # rospy.logdebug(f"cluster_indices: {cluster_indices}")
        # rospy.logdebug(f"unique_cluster_indices: {unique_cluster_indices}")

        # rospy.logdebug(f"cluster_centroids: {unique_cluster_indices * cluster_size}")

        # rospy.logdebug(f"means: {means}")

        num_unique = len(unique_cluster_indices)
        sum_means = np.zeros((num_unique, 3))
        sum_grads = np.zeros((num_unique, 1))
        counts = np.zeros(num_unique, dtype=int)

        # rospy.logdebug(f"num_unique: {num_unique}")

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

        # rospy.logdebug(f"mean_positions: {mean_positions}")
        # rospy.logdebug(f"mean_grads: {mean_grads}")

        if self.static_cluster_center:
            # Update the pre-initialized grid with new gradient values
            # Apply gradient filtering if needed
            if filter_grad_mean:
                avg_grad = np.mean(mean_grads)
                ths_grad = avg_grad
                grads_mask = (mean_grads > ths_grad).flatten()
                filtered_cluster_indices = unique_cluster_indices[grads_mask]
                filtered_mean_grads = mean_grads[grads_mask]
            else:
                filtered_cluster_indices = unique_cluster_indices
                filtered_mean_grads = mean_grads
            
            # Update grid cells with gradient values
            for cluster_idx, grad in zip(filtered_cluster_indices, filtered_mean_grads):
                # FIX: Check bounds properly and convert to flat index
                if (cluster_idx >= 0).all() and \
                cluster_idx[0] < self.grid_dims[0] and \
                cluster_idx[1] < self.grid_dims[1] and \
                cluster_idx[2] < self.grid_dims[2]:
                    # Convert 3D grid index to flat index (row-major order)
                    flat_idx = (cluster_idx[0] * self.grid_dims[1] * self.grid_dims[2] + 
                                cluster_idx[1] * self.grid_dims[2] + 
                                cluster_idx[2])
                    
                    rospy.logdebug(f"Updating grid cell [{cluster_idx[0]}, {cluster_idx[1]}, {cluster_idx[2]}] "
                                f"(flat idx {flat_idx}) with uncertainty {grad[0]:.3f}")
                    rospy.logdebug(f"Cell center: {self.cluster_centroids[flat_idx]}")
                    
                    self.average_gradient_magnitudes[flat_idx] = grad[0]
                else:
                    rospy.logwarn(f"Cluster index {cluster_idx} out of bounds! Grid dims: {self.grid_dims}")
            
            # Don't return anything, grid is updated in place
            return None, None
        
        # For dynamic mode, apply filtering after computing cluster centroids
        if filter_grad_mean:
            avg_grad = np.mean(mean_grads)
            ths_grad = avg_grad
            grads_mask = (mean_grads > ths_grad).flatten()
            mean_positions = mean_positions[grads_mask]
            mean_grads = mean_grads[grads_mask]

        cluster_centroids = mean_positions

        average_gradient_magnitudes = mean_grads.flatten()

        # rospy.logdebug(f"cluster_centroids: {cluster_centroids}")
        
        sorted_indices = np.argsort(average_gradient_magnitudes)
        cluster_centroids = cluster_centroids[sorted_indices]
        average_gradient_magnitudes = average_gradient_magnitudes[sorted_indices]

        if len(cluster_centroids) > 0 and nms_filter:
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

    def get_grid_vis(self, frame_id="map", timestamp=None, marker_scale=0.5):
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

            if uncertainty >= 0.0:

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
                marker.scale.x = self.grid_cell_size * marker_scale
                marker.scale.y = self.grid_cell_size * marker_scale
                marker.scale.z = self.grid_cell_size * marker_scale

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

        centroids_for_msg = self.cluster_centroids.copy()
        centroids_for_msg[:, 2] += self.grid_cell_size / 2 

        msg.means.data = centroids_for_msg.flatten()
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
        self.static_cluster_center = rospy.get_param("/simple_exploration", False)
        self.map_bounds = rospy.get_param("map_bounds", [(-0.65, 9.0, 0.0), (-1.0, 4.5, 0.0)])
        self.grid_marker_scale = rospy.get_param("~grid_marker_scale", 1.0)
        self.grid_cell_size = rospy.get_param("~grid_cell_size", 1.0)
        self.unexplored_uncertainty = rospy.get_param("~unexplored_uncertainty", 1.0)
        self.grid_offset = rospy.get_param("~grid_offset", 0.5)

        rospy.logdebug(f"self.grid_marker_scale: {self.grid_marker_scale}")

        self.exploration_grid = ExplorationGrid(
            grid_cell_size=self.grid_cell_size, 
            static_cluster_center=self.static_cluster_center, 
            map_bounds=self.map_bounds, 
            unexplored_uncertainty=self.unexplored_uncertainty,
            grid_offset=self.grid_offset
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
        # rospy.logdebug("Received GMM")

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

            self.exploration_grid.update_grid(means, covs, uct, weights)

            self.grid_pub.publish(self.exploration_grid.compose_grid_msg())

            self.grid_marker_pub.publish(
                self.exploration_grid.get_grid_vis("map", msg.header.stamp, marker_scale=self.grid_marker_scale)
            )

            self.means_marker_pub.publish(
                self.exploration_grid.get_means_vis(means, "map", msg.header.stamp)
            )

        rospy.logdebug(f"First 5 means from measurement: {means[:5]}")
        rospy.logdebug(f"First 5 uncertainties: {uct[:5]}")
        rospy.logdebug(f"Min/Max grid uncertainty after update: "
                    f"{np.min(self.exploration_grid.average_gradient_magnitudes):.3f} / "
                    f"{np.max(self.exploration_grid.average_gradient_magnitudes):.3f}")
        rospy.logdebug(f"Number of cells with uncertainty != {self.unexplored_uncertainty}: "
                    f"{np.sum(self.exploration_grid.average_gradient_magnitudes != self.unexplored_uncertainty)}")

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
