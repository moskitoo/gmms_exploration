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
from gmms_exploration.srv import FlyTrajectory, GetViewpoint, GetViewpointResponse
from rospy.impl.tcpros_service import Service


class ExplorationGrid:
    def __init__(
        self,
        center_coorinates: Tuple[float] = (0.0, 0.0),
        width: float = 3.0,
        height: float = 6.0,
        rotation_angle: float = 0.0,
        grid_cell_size: float = 0.25,
    ):
        self.center_coorinates = center_coorinates
        self.width = width
        self.height = height
        self.rotation_angle = rotation_angle
        self.grid_cell_size = grid_cell_size

        self.n_rows = int(self.height // self.grid_cell_size)
        self.n_cols = int(self.width // self.grid_cell_size)
        self.grid = np.full((self.n_rows, self.n_cols), 0.5)

        self.means = None
        self.covs = None
        self.uncertainties = None
        self.weights = None

    def update_rtree(self, rtree, means):
        for i in range(len(means)):
            mean = means[i]
            rtree.insert(i, (*mean, *mean))

    def init_r_tree(self):
        p = index.Property()
        p.dimension = 3
        return index.Index(properties=p)

    def grid_to_world(self, x_grid, y_grid):
        """
        Converts points from the grid frame to the global frame.
        """
        cos_a = np.cos(self.rotation_angle)
        sin_a = np.sin(self.rotation_angle)
        cx, cy = self.center_coorinates

        local_x = x_grid - self.width / 2.0
        local_y = y_grid - self.height / 2.0

        # Rotate back to world frame
        dx = local_x * cos_a - local_y * sin_a
        dy = local_x * sin_a + local_y * cos_a

        # Translate back to world frame
        world_x = dx + cx
        world_y = dy + cy

        return world_x, world_y

    def update_grid(self, means, covs, uncertainties, weights):
        rtree = self.init_r_tree()
        self.update_rtree(rtree, means)

        self.means = means
        self.covs = covs
        self.uncertainties = uncertainties
        self.weights = weights

        # Reset grid to default value (0.5) as we are recalculating from the whole GMM
        self.grid.fill(0.5)

        # Search radius for R-tree query (half diagonal of the cell)
        search_radius = (self.grid_cell_size * np.sqrt(2)) / 2.0

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                # Calculate local coordinates of the cell center
                x_grid = (j + 0.5) * self.grid_cell_size
                y_grid = (i + 0.5) * self.grid_cell_size

                world_x, world_y = self.grid_to_world(x_grid, y_grid)

                # Define bounding box for R-tree query
                min_bounds = (world_x - search_radius, world_y - search_radius, -100.0)
                max_bounds = (world_x + search_radius, world_y + search_radius, 100.0)

                # Query R-tree for components intersecting the cell's bounding box
                indices = list(rtree.intersection((*min_bounds, *max_bounds)))

                if indices:
                    # Update cell with mean uncertainty of found components
                    self.grid[i, j] = np.mean(uncertainties[indices])
                    # self.grid[i, j] = np.average(a=uncertainties[indices],weights=weights[indices])
    
    def get_viewpoint(self):
        max_uct_cell = np.unravel_index(self.grid.argmax(), self.grid.shape)

        rospy.logdebug(f"Most uncertain cell: {max_uct_cell}")
        rospy.logdebug(f"Most uncertain cell uncertainty: {self.grid[max_uct_cell]}")

        x_grid = (max_uct_cell[1] + 0.5) * self.grid_cell_size
        y_grid = (max_uct_cell[0] + 0.5) * self.grid_cell_size
        world_x, world_y = self.grid_to_world(x_grid, y_grid)     

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

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                uncertainty = self.grid[i, j]

                # Calculate local coordinates of the cell center
                x_grid = (j + 0.5) * self.grid_cell_size
                y_grid = (i + 0.5) * self.grid_cell_size

                world_x, world_y = self.grid_to_world(x_grid, y_grid)

                marker = Marker()
                marker.header.frame_id = frame_id
                marker.header.stamp = timestamp
                marker.ns = "exploration_grid"
                marker.id = marker_id
                marker_id += 1
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                marker.lifetime = rospy.Duration(0)

                height = max(0.05, uncertainty)
                marker.pose.position.x = world_x
                marker.pose.position.y = world_y
                marker.pose.position.z = height / 2.0

                # Set orientation based on grid rotation
                q = tf.transformations.quaternion_from_euler(0, 0, self.rotation_angle)
                marker.pose.orientation.x = q[0]
                marker.pose.orientation.y = q[1]
                marker.pose.orientation.z = q[2]
                marker.pose.orientation.w = q[3]

                # Scale: x and y match the grid resolution, z matches uncertainty
                marker.scale.x = self.grid_cell_size
                marker.scale.y = self.grid_cell_size
                marker.scale.z = height

                # Color: Map uncertainty to color
                # High uncertainty (1.0) -> Hot/Red, Low (0.0) -> Cool/Blue
                color = cm.plasma(uncertainty)
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = 0.8

                marker_array.markers.append(marker)

        return marker_array


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
            height=7.0, width=11, center_coorinates=(4.0, 0.0), grid_cell_size=1.75
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

            rospy.logdebug(f"grid: \n{self.exploration_grid.grid}")

            self.exploration_grid.update_grid(means, covs, uct, weights)

            self.grid_marker_pub.publish(
                self.exploration_grid.get_grid_vis("map", msg.header.stamp)
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
