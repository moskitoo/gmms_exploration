#!/usr/bin/env python3
"""
ROS Node for Real-time Point Cloud SOGMM Processing and Visualization

This node subscribes to PointCloud2 messages, processes them using SOGMM,
and publishes visualization markers for RViz.
"""

import copy
import threading
import time
from typing import List

import numpy as np
import ros_numpy
import rospy
import sogmm_cpu
import sogmm_gpu
import tf2_ros
from rtree import index
from scipy.spatial.transform import Rotation as R
from scripts.sogmm_hash_table import GMMSpatialHash
from sensor_msgs.msg import PointCloud2
from sogmm_cpu import SOGMMf3Host as CPUContainerf3
from sogmm_cpu import SOGMMf4Host as CPUContainerf4
from sogmm_gpu import SOGMMf3Device as GPUContainerf3
from sogmm_gpu import SOGMMf4Device as GPUContainerf4
from sogmm_gpu import SOGMMInference as GPUInference
from sogmm_gpu import SOGMMLearner as GPUFit
from sogmm_py.sogmm import SOGMM
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker, MarkerArray


class SOGMMROSNode:
    """
    ROS Node for processing point clouds with SOGMM and visualizing results
    """

    def __init__(self):
        # Initialize ROS node
        rospy.init_node("sogmm_ros_node", anonymous=True)

        # Parameters
        self.bandwidth = rospy.get_param("~bandwidth", 0.02)
        self.visualization_scale = rospy.get_param("~visualization_scale", 2.0)
        self.processing_decimation = rospy.get_param("~processing_decimation", 1)
        self.enable_visualization = rospy.get_param("~enable_visualization", True)
        self.target_frame = rospy.get_param("~target_frame", "map")
        self.min_novel_points = rospy.get_param("~min_novel_points", 500)

        # Publishers
        self.marker_pub = rospy.Publisher(
            "/starling1/mpa/gmm_markers", MarkerArray, queue_size=1
        )

        self.gmm_size_pub = rospy.Publisher(
            "/starling1/mpa/gmm_size", Int32, queue_size=1
        )

        # Subscribers
        self.pc_sub = rospy.Subscriber(
            "/starling1/mpa/tof_pc", PointCloud2, self.pointcloud_callback, queue_size=1
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # self.sogmm = None
        # self.gsh = GMMSpatialHash(width=50, height=30, depth=15, resolution=0.2)

        # R-tree properties for 3D data
        p = index.Property()
        p.dimension = 3
        # The 'id' is the GMM component index, 'object' can be used if needed

        self.rtree = index.Index(properties=p)
        self.l_thres = 0.05
        self.novel_pts_placeholder = None
        self.global_model_cpu = None
        self.learner = GPUFit(self.bandwidth)
        self.inference = GPUInference()

        # Threading for non-blocking processing
        self.processing_lock = threading.Lock()
        self.latest_pointcloud = None
        self.processing_thread = None

        rospy.loginfo("SOGMM ROS Node initialized with parameters:")
        rospy.loginfo(f"  - Bandwidth: {self.bandwidth}")
        rospy.loginfo(f"  - Visualization scale: {self.visualization_scale}")
        rospy.loginfo(f"  - Processing decimation: {self.processing_decimation}")

    def pointcloud_callback(self, msg):
        """
        Callback for PointCloud2 messages
        """
        with self.processing_lock:
            self.latest_pointcloud = msg

        # Start processing in separate thread if not already running
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self.process_pointcloud)
            self.processing_thread.daemon = True
            self.processing_thread.start()

    def process_pointcloud(self):
        """
        Process the latest point cloud with SOGMM
        """
        with self.processing_lock:
            if self.latest_pointcloud is None:
                return
            msg = self.latest_pointcloud
            self.latest_pointcloud = None

        try:
            start_time = time.time()

            pcld = self.preprocess_point_cloud(msg)

            # Fit SOGMM model
            gmm_start_time = time.time()

            local_model_gpu = None

            if self.global_model_cpu is None:
                local_model_gpu = GPUContainerf4()
                self.learner.fit(self.extract_ms_data(pcld), pcld, local_model_gpu)

                local_model_cpu = CPUContainerf4(local_model_gpu.n_components_)
                local_model_gpu.to_host(local_model_cpu)

                # Add new components to the R-tree
                for i in range(local_model_cpu.n_components_):
                    mean = local_model_cpu.means_[i, :3]
                    # R-tree stores (min_x, min_y, min_z, max_x, max_y, max_z)
                    # For a point, min and max are the same.
                    self.rtree.insert(i, (*mean, *mean))

                self.global_model_cpu = copy.deepcopy(local_model_cpu)

            else:
                # fov_comp_indices = self.gsh.find_points(pcld)
                # Find components in the point cloud's bounding box using the R-tree
                min_bounds = pcld[:, :3].min(axis=0)
                max_bounds = pcld[:, :3].max(axis=0)
                fov_comp_indices = list(self.rtree.intersection((*min_bounds, *max_bounds)))

                submap = self.global_model_cpu.submap_from_indices(list(fov_comp_indices))

                # take it to the GPU
                submap_gpu = GPUContainerf4(submap.n_components_)
                submap_gpu.from_host(submap)

                old_n_components = self.global_model_cpu.n_components_

                local_model_gpu = GPUContainerf4()
                self.learner.fit(self.extract_ms_data(pcld), pcld, local_model_gpu)
                local_model_cpu = CPUContainerf4(local_model_gpu.n_components_)
                local_model_gpu.to_host(local_model_cpu)

                

                self.global_model_cpu.merge(local_model_cpu)

                new_n_components = self.global_model_cpu.n_components_

                # self.gsh.add_points(
                #     local_model_cpu.means_,
                #     np.arange(old_n_components, new_n_components, dtype=int),
                # )
                # Add new components to the R-tree
                for i in range(local_model_cpu.n_components_):
                    idx = old_n_components + i
                    mean = local_model_cpu.means_[i, :3]
                    self.rtree.insert(idx, (*mean, *mean))

            gmm_time = time.time() - gmm_start_time

            # Visualize results - only if we have a model
            viz_start_time = time.time()
            if self.enable_visualization and self.global_model_cpu is not None:
                self.visualize_gmm(self.global_model_cpu, self.target_frame, msg.header.stamp)
            viz_time = time.time() - viz_start_time

            processing_time = time.time() - start_time
            n_components = (
                self.global_model_cpu.n_components_ if self.global_model_cpu is not None else 0
            )
            rospy.loginfo(
                f"Processed point cloud with {n_components} components in {processing_time:.3f}s (GMM: {gmm_time:.3f}s, Viz: {viz_time:.3f}s)"
            )

            gmm_stats_msg = Int32()
            gmm_stats_msg.data = n_components

            self.gmm_size_pub.publish(gmm_stats_msg)

        except Exception as e:
            rospy.logerr(f"Error processing point cloud: {str(e)}")

    def transform_point_cloud(self, msg, points_3d_original, target_frame):
        try:
            # Lookup transform from cloud frame to target frame
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                msg.header.frame_id,
                # rospy.Time(0),  # get the latest transform
                msg.header.stamp,
                rospy.Duration(1.0),
            )

            # Extract transformation parameters
            translation = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ]
            )

            rotation_quat = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ]

            # Convert quaternion to rotation matrix
            rotation_matrix = R.from_quat(rotation_quat).as_matrix()

            # Apply transformation: R * points + t
            points_3d = (rotation_matrix @ points_3d_original.T).T + translation

            rospy.loginfo(
                f"Successfully transformed point cloud from {msg.header.frame_id} to map frame"
            )

        except Exception as e:
            rospy.logwarn("Transform failed: %s, using original points" % str(e))
            points_3d = points_3d_original

        return points_3d

    def preprocess_point_cloud(self, msg):
        # Convert PointCloud2 to numpy array first
        pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(msg)

        # Extract XYZ coordinates before transformation
        points_3d_original = np.column_stack(
            [
                pc_array["x"].flatten(),
                pc_array["y"].flatten(),
                pc_array["z"].flatten(),
            ]
        )

        # Remove NaN and infinite values before transformation
        valid_mask = np.isfinite(points_3d_original).all(axis=1)
        points_3d_original = points_3d_original[valid_mask]

        if len(points_3d_original) == 0:
            rospy.logwarn("No valid points in point cloud")
            return

        points_3d = self.transform_point_cloud(
            msg, points_3d_original, self.target_frame
        )

        # Decimate points for faster processing
        if self.processing_decimation > 1:
            indices = np.arange(0, len(points_3d), self.processing_decimation)
            points_3d = points_3d[indices]

        rospy.loginfo(f"Processing {len(points_3d)} points")

        # SOGMM requires 4D, so add dummy intensity for 3D point clouds
        intensity = np.ones(len(points_3d)) * 0.5
        return np.column_stack([points_3d, intensity])

    def extract_ms_data(self, X):
        d = np.array([np.linalg.norm(x) for x in X[:, 0:3]])[:, np.newaxis]
        g = X[:, 3][:, np.newaxis]
        return np.concatenate((d, g), axis=1)

    def extract_novel(self, model, pcld, comp_indices):
        # create a GMM submap using component indices
        submap = model.submap_from_indices(list(comp_indices))

        # take it to the GPU
        submap_gpu = GPUContainerf4(submap.n_components_)
        submap_gpu.from_host(submap)

        # perform likelihood score computation on the GPU
        scores = self.inference.score_3d(pcld[:, :3], submap_gpu)

        # filter the point cloud and return novel points
        scores = scores.flatten()
        return pcld[scores < self.l_thres, :]

    # def submap_from_indices(self, model, indices: List[int]) -> SOGMM:
    #     weights = model.weights_[indices]
    #     means = model.means_[indices, :]
    #     covariances = model.covariances_[indices, :]
    #     return SOGMM(weights, means, covariances)

    def visualize_gmm(self, model, frame_id, timestamp):
        """
        Publish MarkerArray for RViz visualization
        """
        if model is None or model.n_components_ == 0:
            return

        marker_array = MarkerArray()

        for i in range(len(model.means_)):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = timestamp
            marker.ns = "sogmm_ellipsoids"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Extract mean (first 3 components are XYZ, 4th is intensity)
            mean = model.means_[i, :3]
            intensity = model.means_[i, 3]

            try:
                # Reshape covariance from flat array to 4x4 and extract 3x3 spatial part
                cov_flat = model.covariances_[i]
                cov_4x4 = cov_flat.reshape(4, 4)
                cov = cov_4x4[:3, :3]

            except (np.linalg.LinAlgError, ValueError) as e:
                rospy.logwarn(f"Error with component {i}: {e}")
                continue

            # Set position (mean of 3D coordinates)
            marker.pose.position.x = float(mean[0])
            marker.pose.position.y = float(mean[1])
            marker.pose.position.z = float(mean[2])

            # Extract 3D spatial covariance
            cov_3d = cov[:3, :3] if cov.shape[0] >= 3 else cov

            try:
                # Compute eigenvalues and eigenvectors for scale and orientation
                eigenvals, eigenvecs = np.linalg.eigh(cov_3d)

                # Ensure positive eigenvalues
                eigenvals = np.maximum(eigenvals, 1e-6)

                # Set scale based on eigenvalues
                scale_factor = self.visualization_scale
                marker.scale.x = 2 * scale_factor * np.sqrt(eigenvals[0])
                marker.scale.y = 2 * scale_factor * np.sqrt(eigenvals[1])
                marker.scale.z = 2 * scale_factor * np.sqrt(eigenvals[2])

                # Set orientation based on eigenvectors
                # Convert rotation matrix to quaternion
                rot_matrix = eigenvecs
                r = R.from_matrix(rot_matrix)
                quat = r.as_quat()  # Returns [x, y, z, w]

                marker.pose.orientation.x = float(quat[0])
                marker.pose.orientation.y = float(quat[1])
                marker.pose.orientation.z = float(quat[2])
                marker.pose.orientation.w = float(quat[3])

            except np.linalg.LinAlgError:
                # Fallback to spherical visualization
                avg_scale = self.visualization_scale * 0.1
                marker.scale.x = marker.scale.y = marker.scale.z = avg_scale
                marker.pose.orientation.w = 1.0

            # Set color based on intensity (grayscale)
            gray_val = float(intensity)  # Intensity is already in 0-1 range
            marker.color.r = gray_val
            marker.color.g = gray_val
            marker.color.b = gray_val

            # Set alpha based on intensity with minimum visibility
            marker.color.a = float(min(1.0, max(0.3, intensity * 2.0)))

            marker.lifetime = rospy.Duration(2.0)  # Markers last 2 seconds

            marker_array.markers.append(marker)

        # Clear old markers if we have fewer components now
        if hasattr(self, "last_marker_count"):
            for i in range(len(marker_array.markers), self.last_marker_count):
                clear_marker = Marker()
                clear_marker.header.frame_id = frame_id
                clear_marker.header.stamp = timestamp
                clear_marker.ns = "sogmm_ellipsoids"
                clear_marker.id = i
                clear_marker.action = Marker.DELETE
                marker_array.markers.append(clear_marker)

        self.last_marker_count = len(model.means_)

        # Publish markers
        self.marker_pub.publish(marker_array)
        rospy.logdebug(f"Published {len(model.means_)} ellipsoid markers")

    def run(self):
        """
        Main execution loop
        """
        rospy.loginfo("SOGMM ROS Node is running. Waiting for point cloud data...")
        rospy.spin()


def main():
    """
    Main function
    """
    try:
        node = SOGMMROSNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("SOGMM ROS Node interrupted")
    except Exception as e:
        rospy.logerr(f"Error in SOGMM ROS Node: {str(e)}")


if __name__ == "__main__":
    main()
