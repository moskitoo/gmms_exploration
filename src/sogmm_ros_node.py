#!/usr/bin/env python3
"""
ROS Node for Real-time Point Cloud SOGMM Processing and Visualization

This node subscribes to PointCloud2 messages, processes them using SOGMM,
and publishes visualization markers for RViz.
"""


import threading
import time
import traceback

import matplotlib.cm as cm
import numpy as np
import ros_numpy
import rospy
import tf2_ros
from rtree import index
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2
from sogmm_cpu import SOGMMf4Host as CPUContainerf4
from sogmm_gpu import SOGMMf4Device as GPUContainerf4
from sogmm_gpu import SOGMMInference as GPUInference
from sogmm_gpu import SOGMMLearner as GPUFit
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker, MarkerArray


class MasterGMM:
    def __init__(self):
        # self.model = None
        # Store GMM components as numpy arrays we control
        self.weights = None
        self.means = None
        self.covariances = None
        self.n_components = 0

        self.fusion_counts = []
        self.last_displacements = []
        self.observation_counts = []

        # R-tree for spatial indexing of GMM components
        p = index.Property()
        p.dimension = 3
        self.rtree = index.Index(properties=p)

    def update(self, gmm_measurement, match_threshold=5.0):
        """
        Updates the master GMM with a new measurement GMM.
        
        Args:
            gmm_measurement: Incoming GMM measurement to integrate
            match_threshold: KL divergence threshold for component matching
        """
        n_incoming = len(gmm_measurement.weights_)
        rospy.loginfo(f"Processing {n_incoming} incoming Gaussians")
        
        # Handle empty master GMM - initialize with first measurement
        if self.n_components == 0:
            self._initialize_from_measurement(gmm_measurement)
            return
        
        candidate_indices = self._find_spatial_candidates(gmm_measurement)
        self._update_observation_counts(candidate_indices)
        
        n_fused, new_components = self._match_and_fuse_components(
            gmm_measurement, candidate_indices, match_threshold
        )
        
        if new_components:
            self._add_new_components(new_components)
        
        self._normalize_weights()
        
        # Log results
        n_novel = len(new_components)
        rospy.loginfo(f"Fused: {n_fused}, Novel: {n_novel}")


    def _make_writable(self, gmm):
        # Create deep writable NumPy copies detached from pybind11
        gmm.weights_ = np.array(gmm.weights_, copy=True)
        gmm.means_ = np.array(gmm.means_, copy=True)
        gmm.covariances_ = np.array(gmm.covariances_, copy=True)
        return gmm

    def _initialize_from_measurement(self, gmm_measurement):
        """Initialize master GMM from first measurement."""
        self.weights = np.array(gmm_measurement.weights_, copy=True)
        self.means = np.array(gmm_measurement.means_, copy=True) 
        self.covariances = np.array(gmm_measurement.covariances_, copy=True)
        self.n_components = gmm_measurement.n_components_
        
        # Initialize tracking arrays
        self.fusion_counts = [1] * self.n_components
        self.observation_counts = [1] * self.n_components
        self.last_displacements = [0.0] * self.n_components
        
        # Build spatial index
        for i in range(self.n_components):
            mean = self.means[i, :3]
            self.rtree.insert(i, (*mean, *mean))
        
        self._normalize_weights()

    def _find_spatial_candidates(self, gmm_measurement):
        """Find master components spatially overlapping with measurement."""
        # Get bounding box of incoming measurement
        min_bounds = gmm_measurement.means_[:, :3].min(axis=0)
        max_bounds = gmm_measurement.means_[:, :3].max(axis=0)
        
        # Query R-tree for overlapping components
        return list(self.rtree.intersection((*min_bounds, *max_bounds)))

    def _update_observation_counts(self, candidate_indices):
        """Update observation counts for components in re-observed region."""
        for idx in candidate_indices:
            self.observation_counts[idx] += 1

    def _match_and_fuse_components(self, gmm_measurement, candidate_indices, match_threshold):
        """
        Match measurement components to master components and fuse them.
        
        Returns:
            tuple: (number_fused, list_of_unmatched_components)
        """
        fused_master_indices = set()
        unmatched_components = []
        
        for j in range(gmm_measurement.n_components_):
            best_match_idx = self._find_best_match(
                gmm_measurement, j, candidate_indices, match_threshold
            )
            
            if best_match_idx is not None:
                # Fuse with matched master component
                meas_comp = self._make_writable(gmm_measurement.submap_from_indices([j]))
                self._fuse_components(best_match_idx, meas_comp)
                fused_master_indices.add(best_match_idx)
            else:
                # No match found - add to unmatched list
                unmatched_comp = self._make_writable(gmm_measurement.submap_from_indices([j]))
                unmatched_components.append(unmatched_comp)
        
        return len(fused_master_indices), unmatched_components

    def _find_best_match(self, gmm_measurement, meas_idx, candidate_indices, match_threshold):
        """
        Find the best matching master component for a measurement component.
        
        Returns:
            int or None: Index of best matching master component, or None if no match
        """
        meas_mu = gmm_measurement.means_[meas_idx]
        meas_cov = gmm_measurement.covariances_[meas_idx].reshape(4, 4)
        
        min_distance = float("inf")
        best_master_idx = None
        
        for master_idx in candidate_indices:
            master_mu = self.means[master_idx]
            master_cov = self.covariances[master_idx].reshape(4, 4)
            
            distance = self._calculate_kl(meas_mu, meas_cov, master_mu, master_cov)
            
            if distance < min_distance:
                min_distance = distance
                best_master_idx = master_idx
        
        # Return match only if below threshold
        return best_master_idx if min_distance < match_threshold else None

    def _add_new_components(self, new_components):
        """Add unmatched components as new master components."""
        # Create temporary container with current master state
        temp_model = CPUContainerf4(self.n_components)
        temp_model.weights_ = self.weights
        temp_model.means_ = self.means
        temp_model.covariances_ = self.covariances
        
        old_n_components = self.n_components
        
        # Merge each new component
        for new_comp in new_components:
            temp_model.merge(new_comp)
            self.fusion_counts.append(1)
            self.last_displacements.append(0.0)
            self.observation_counts.append(1)
        
        # Extract updated data back to numpy arrays
        self.n_components = temp_model.n_components_
        self.weights = np.array(temp_model.weights_, copy=True)
        self.means = np.array(temp_model.means_, copy=True)
        self.covariances = np.array(temp_model.covariances_, copy=True)
        
        # Update spatial index with new components
        for i in range(old_n_components, self.n_components):
            mean = self.means[i, :3]
            self.rtree.insert(i, (*mean, *mean))

    def _normalize_weights(self):
        """Normalize component weights to sum to 1."""
        total_weight = np.sum(self.weights)
        if total_weight > 0:
            self.weights /= total_weight

    def prune_stale_components(self, min_observations=5, max_fusion_ratio=0.4):
        """
        Remove Gaussians that are stale outliers.
        
        A Gaussian is considered stale if:
        1. It has been observed multiple times (in observation space of measurements)
        2. But hasn't been fused recently (hasn't matched any incoming Gaussians)
        
        This means:
        - Single-visit regions are NEVER pruned (they're just under-explored)
        - Only Gaussians in re-observed regions that don't match are pruned (likely outliers)
        """
        
        if self.n_components == 0:
            return 0
        
        # Determine which components to keep
        keep_mask = []
        for i in range(self.n_components):
            obs_count = self.observation_counts[i]
            fus_count = self.fusion_counts[i]
            
            # Keep if under-observed OR has good fusion ratio
            fusion_ratio = fus_count / obs_count if obs_count > 0 else 1.0
            
            keep = obs_count < min_observations or fusion_ratio >= max_fusion_ratio
            keep_mask.append(keep)
        
        keep_mask = np.array(keep_mask)
        n_pruned = np.sum(~keep_mask)
        
        if n_pruned == 0:
            return 0
        
        # Get indices to keep
        keep_indices = np.where(keep_mask)[0]
        
        # Update all arrays
        self.weights = self.weights[keep_indices]
        self.means = self.means[keep_indices]
        self.covariances = self.covariances[keep_indices]
        self.fusion_counts = [self.fusion_counts[i] for i in keep_indices]
        self.last_displacements = [self.last_displacements[i] for i in keep_indices]
        self.n_components = len(keep_indices)
        
        # Rebuild R-tree with remaining components
        p = index.Property()
        p.dimension = 3
        self.rtree = index.Index(properties=p)
        for i in range(self.n_components):
            mean = self.means[i, :3]
            self.rtree.insert(i, (*mean, *mean))
        
        # Renormalize weights
        if self.n_components > 0:
            self.weights /= np.sum(self.weights)
        
        rospy.loginfo("=========================================================================")
        rospy.loginfo(f"Pruned {n_pruned} stale outlier Gaussians, {self.n_components} remaining")
        rospy.loginfo("=========================================================================")
        return n_pruned

    def _calculate_kl(self, mu1, cov1, mu2, cov2, symmetric=True):
        p = mu1.shape[0]

        def kl_divergence(m1, c1, m2, c2):
            c2_inv = np.linalg.inv(c2)
            trace_term = np.trace(c2_inv @ c1)
            mean_diff = m2 - m1
            mahalanobis_term = mean_diff.T @ c2_inv @ mean_diff
            log_det_term = np.log(np.linalg.det(c2) / np.linalg.det(c1))
            return 0.5 * (trace_term + mahalanobis_term - p + log_det_term)

        try:
            d1 = kl_divergence(mu1, cov1, mu2, cov2)

            if symmetric:
                d2 = kl_divergence(mu2, cov2, mu1, cov1)
                return d1 + d2
            else:
                return d1
        except np.linalg.LinAlgError:
            return float("inf")

    def _fuse_components(self, master_idx, incoming_measurement):
        """Merges a measurement component into a master component."""
        w_i = self.weights[master_idx]
        mu_i = self.means[master_idx]
        cov_i = self.covariances[master_idx].reshape(4, 4)

        w_j = incoming_measurement.weights_[0]
        mu_j = incoming_measurement.means_[0]
        cov_j = incoming_measurement.covariances_[0].reshape(4, 4)

        w_new = w_i + w_j
        mu_new = (w_i * mu_i + w_j * mu_j) / w_new

        displacement = np.linalg.norm(mu_new[:3] - mu_i[:3])

        diff_i = (mu_i - mu_new).reshape(4, 1)
        diff_j = (mu_j - mu_new).reshape(4, 1)

        cov_new = (
            w_i * (cov_i + diff_i @ diff_i.T) + w_j * (cov_j + diff_j @ diff_j.T)
        ) / w_new

        # Update master model in place
        self.weights[master_idx] = w_new
        self.means[master_idx] = mu_new
        self.covariances[master_idx] = cov_new.flatten()
        self.fusion_counts[master_idx] += 1
        self.last_displacements[master_idx] = displacement

class SOGMMROSNode:
    """
    ROS Node for processing point clouds with SOGMM and visualizing results
    """
    def __init__(self):
        """Initialize the SOGMM ROS Node with parameters, publishers, subscribers, and core components."""
        # Initialize ROS node
        rospy.init_node("sogmm_ros_node", anonymous=True)
        
        # Load ROS parameters
        self._load_parameters()
        
        # Initialize publishers and subscribers
        self._setup_communication()
        
        # Initialize core components
        self._setup_core_components()
        
        # Log initialization summary
        self._log_initialization()

    def _load_parameters(self):
        """Load all ROS parameters with default values."""
        # SOGMM algorithm parameters
        self.bandwidth = rospy.get_param("~bandwidth", 0.02)
        self.kl_div_match_thresh = rospy.get_param("~kl_div_match_thresh", 5.0)
        self.l_thres = 0.05
        
        # Visualization parameters
        self.enable_visualization = rospy.get_param("~enable_visualization", True)
        self.visualization_scale = rospy.get_param("~visualization_scale", 2.0)
        self.color_by = rospy.get_param("~color_by", "intensity")  # Options: intensity, confidence, stability, combined
        self.add_metric_text = rospy.get_param("~add_metric_text", True)
        
        # Processing parameters
        self.processing_decimation = rospy.get_param("~processing_decimation", 1)
        self.min_novel_points = rospy.get_param("~min_novel_points", 500)
        self.target_frame = rospy.get_param("~target_frame", "map")
        
        # Pruning parameters
        self.max_fusion_ratio = rospy.get_param("~max_fusion_ratio", 0.4)
        self.prune_min_observations = rospy.get_param("~prune_min_observations", 5)
        self.prune_interval_frames = rospy.get_param("~prune_interval_frames", 10)

    def _setup_communication(self):
        """Initialize ROS publishers and subscribers."""
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
        
        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def _setup_core_components(self):
        """Initialize core SOGMM processing components and threading."""
        # Core SOGMM components
        self.master_gmm = MasterGMM()
        self.learner = GPUFit(self.bandwidth)
        self.inference = GPUInference()
        
        # Processing state
        self.frame_count = 0
        self.novel_pts_placeholder = None
        
        # Threading for non-blocking processing
        self.processing_lock = threading.Lock()
        self.latest_pointcloud = None
        self.processing_thread = None
        
        # Timing
        self.last_prune_time = rospy.Time.now().to_sec()

    def _log_initialization(self):
        """Log initialization parameters for debugging."""
        rospy.loginfo("SOGMM ROS Node initialized with parameters:")
        rospy.loginfo(f"  - Bandwidth: {self.bandwidth}")
        rospy.loginfo(f"  - KL divergence match threshold: {self.kl_div_match_thresh}")
        rospy.loginfo(f"  - Target frame: {self.target_frame}")
        rospy.loginfo(f"  - Processing decimation: {self.processing_decimation}")
        rospy.loginfo(f"  - Visualization enabled: {self.enable_visualization}")
        rospy.loginfo(f"  - Visualization scale: {self.visualization_scale}")
        rospy.loginfo(f"  - Color by: {self.color_by}")
        rospy.loginfo(f"  - Add metric text: {self.add_metric_text}")
        rospy.loginfo("  - Pruning parameters:")
        rospy.loginfo(f"    * Min observations: {self.prune_min_observations}")
        rospy.loginfo(f"    * Max fusion ratio: {self.max_fusion_ratio}")
        rospy.loginfo(f"    * Prune interval (frames): {self.prune_interval_frames}")

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

            # 1. Generate a local GMM from the current point cloud
            local_model_gpu = GPUContainerf4()
            self.learner.fit(self.extract_ms_data(pcld), pcld, local_model_gpu)
            local_model_cpu = CPUContainerf4(local_model_gpu.n_components_)
            local_model_gpu.to_host(local_model_cpu)

            # 2. Update the master GMM with the new local measurement
            self.master_gmm.update(
                gmm_measurement=local_model_cpu, 
                match_threshold=self.kl_div_match_thresh,
            )

            # 3. Periodically prune outliers (every N frames)
            self.frame_count += 1
            if self.frame_count % self.prune_interval_frames == 0:
                self.master_gmm.prune_stale_components(
                    min_observations=self.prune_min_observations,
                    max_fusion_ratio=self.max_fusion_ratio
                )

            gmm_time = time.time() - gmm_start_time

            # Visualize results - only if we have a model
            viz_start_time = time.time()
            if self.enable_visualization and self.master_gmm.n_components > 0:
                self.visualize_gmm(self.master_gmm, self.target_frame, msg.header.stamp)
            viz_time = time.time() - viz_start_time

            processing_time = time.time() - start_time
            n_components = self.master_gmm.n_components
            rospy.loginfo(
                f"Processed point cloud with {n_components} components in {processing_time:.3f}s (GMM: {gmm_time:.3f}s, Viz: {viz_time:.3f}s)"
            )

            gmm_stats_msg = Int32()
            gmm_stats_msg.data = n_components

            self.gmm_size_pub.publish(gmm_stats_msg)

        except Exception as e:
            rospy.logerr(f"Error processing point cloud: {e}\n{traceback.format_exc()}")

    def transform_point_cloud(self, msg, points_3d_original, target_frame):
        try:
            # Lookup transform from cloud frame to target frame
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                msg.header.frame_id,
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

            # rospy.loginfo(f"Successfully transformed point cloud from {msg.header.frame_id} to map frame")

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

    def visualize_gmm(self, master_gmm, frame_id, timestamp):
        """
        Publish MarkerArray for RViz visualization
        """
        if master_gmm.n_components == 0:
            return

        marker_array = MarkerArray()

        # Normalize values for coloring based on selected mode
        norm_values = []
        
        if self.color_by == "confidence":
            # Apply logarithmic scaling to fusion counts
            log_counts = [np.log1p(c) for c in master_gmm.fusion_counts]
            max_log_count = max(log_counts) if log_counts else 1.0
            norm_values = [c / max(1.0, max_log_count) for c in log_counts]
            
        elif self.color_by == "stability":
            # Stability based on displacement only
            for idx in range(len(master_gmm.means)):
                displacement = master_gmm.last_displacements[idx]
                
                # Score inversely proportional to displacement
                # Lower displacement = higher stability score
                score = np.exp(-displacement / 0.05)  # 0.05 = threshold for "small"
                norm_values.append(score)
                
            # Normalize to 0-1 range
            if norm_values:
                max_score = max(norm_values) if max(norm_values) > 0 else 1.0
                norm_values = [s / max_score for s in norm_values]
                
        elif self.color_by == "combined":
            # Combined stability metric: fusion_count AND displacement
            for idx in range(len(master_gmm.means)):
                fusion_count = master_gmm.fusion_counts[idx]
                displacement = master_gmm.last_displacements[idx]
                
                if fusion_count <= 1:
                    # Unverified (only seen once) - score = 0
                    score = 0.0
                elif displacement < 0.001:
                    # Suspicious (multiple fusions but nearly zero movement) - score = 0.3
                    score = 0.3
                elif displacement > 0.1:
                    # Large displacement (unstable) - score = 0.5
                    score = 0.5
                else:
                    # Good stability: small non-zero displacement
                    # Higher fusion count = higher score
                    displacement_score = np.exp(-displacement / 0.05)  # 0.05 = threshold for "small"
                    fusion_score = np.log1p(fusion_count) / np.log1p(10)  # Normalized to ~0-1
                    score = fusion_score * displacement_score
                    
                norm_values.append(score)
                
            # Normalize to 0-1 range
            if norm_values:
                max_score = max(norm_values) if max(norm_values) > 0 else 1.0
                norm_values = [s / max_score for s in norm_values]

        for i in range(len(master_gmm.means)):
            # Create sphere marker for Gaussian
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = timestamp
            marker.ns = "sogmm_ellipsoids"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Extract mean (first 3 components are XYZ, 4th is intensity)
            mean = master_gmm.means[i, :3]

            try:
                # Reshape covariance from flat array to 4x4 and extract 3x3 spatial part
                cov_flat = master_gmm.covariances[i]
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

            # Set color based on the selected mode
            if self.color_by == "confidence":
                # Colormap: Purple (low confidence) -> Yellow (high confidence)
                color = cm.plasma(norm_values[i])
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = 0.8
                
            elif self.color_by == "stability":
                # Colormap: Red (unstable) -> Green (stable)
                color = cm.RdYlGn(norm_values[i])
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = 0.8
                
            elif self.color_by == "combined":
                # Custom color scheme for combined metric
                fusion_count = master_gmm.fusion_counts[i]
                displacement = master_gmm.last_displacements[i]
                
                if fusion_count <= 1:
                    # Red = unverified (only seen once)
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 0.6
                elif displacement < 0.001:
                    # Yellow = suspicious (multiple fusions but no movement)
                    marker.color.r = 1.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 0.7
                elif displacement > 0.2:
                    # Orange = unstable (large displacement)
                    marker.color.r = 1.0
                    marker.color.g = 0.5
                    marker.color.b = 0.0
                    marker.color.a = 0.7
                else:
                    # Green = stable (good fusion count + small non-zero displacement)
                    # Intensity based on normalized score
                    green_intensity = norm_values[i]
                    marker.color.r = 0.0
                    marker.color.g = float(green_intensity)
                    marker.color.b = 0.0
                    marker.color.a = 0.8
                    
            else:  # Default to intensity
                intensity = master_gmm.means[i, 3]
                gray_val = float(intensity)
                marker.color.r = gray_val
                marker.color.g = gray_val
                marker.color.b = gray_val
                marker.color.a = float(min(1.0, max(0.3, intensity * 2.0)))

            marker.lifetime = rospy.Duration(2.0)  # Markers last 2 seconds

            marker_array.markers.append(marker)

            # Add text marker for metric value
            if self.add_metric_text:
                text_marker = Marker()
                text_marker.header.frame_id = frame_id
                text_marker.header.stamp = timestamp
                text_marker.ns = "sogmm_text"
                text_marker.id = i
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD

                # Position text slightly above the ellipsoid
                text_marker.pose.position.x = float(mean[0])
                text_marker.pose.position.y = float(mean[1])
                text_marker.pose.position.z = float(mean[2])

                text_marker.pose.orientation.w = 1.0

                # Set text content based on color mode
                if self.color_by == "confidence":
                    count = master_gmm.fusion_counts[i]
                    text_marker.text = f"C:{count}"
                    # text_marker.text = f"{norm_values[i]}"
                elif self.color_by == "stability":
                    disp = master_gmm.last_displacements[i]
                    text_marker.text = f"D:{disp:.3f}"
                elif self.color_by == "combined":
                    count = master_gmm.fusion_counts[i]
                    disp = master_gmm.last_displacements[i]
                    text_marker.text = f"C:{count} D:{disp:.3f}"
                else:
                    text_marker.text = f"{master_gmm.means[i, 3]:.2f}"

                # Set text size
                text_marker.scale.z = 0.05  # Text height

                # Set text color (white with good visibility)
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                text_marker.color.a = 1.0

                text_marker.lifetime = rospy.Duration(2.0)

                marker_array.markers.append(text_marker)

        # Clear old markers if we have fewer components now
        if hasattr(self, "last_marker_count"):
            for i in range(len(master_gmm.means), self.last_marker_count):
                # Clear sphere marker
                clear_marker = Marker()
                clear_marker.header.frame_id = frame_id
                clear_marker.header.stamp = timestamp
                clear_marker.ns = "sogmm_ellipsoids"
                clear_marker.id = i
                clear_marker.action = Marker.DELETE
                marker_array.markers.append(clear_marker)

                # Clear text marker
                clear_text = Marker()
                clear_text.header.frame_id = frame_id
                clear_text.header.stamp = timestamp
                clear_text.ns = "sogmm_text"
                clear_text.id = i
                clear_text.action = Marker.DELETE
                marker_array.markers.append(clear_text)

        self.last_marker_count = len(master_gmm.means)

        # Publish markers
        self.marker_pub.publish(marker_array)
        rospy.logdebug(f"Published {len(master_gmm.means)} ellipsoid markers with text")

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