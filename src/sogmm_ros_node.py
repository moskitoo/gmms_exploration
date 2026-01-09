#!/usr/bin/env python3
"""
ROS Node for Real-time Point Cloud SOGMM Processing and Visualization

This node subscribes to PointCloud2 messages, processes them using SOGMM,
and publishes visualization markers for RViz.
"""

import logging
import os
import threading
import time
import traceback
from typing import List, Optional

import matplotlib.cm as cm
import numpy as np
import ros_numpy
import rospy
import tf2_ros
from gmms_exploration.msg import GaussianComponent, GaussianMixtureModel
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2
from sogmm_cpu import SOGMMf4Host as CPUContainerf4
from sogmm_cpu import SOGMMInference as CPUInference
from sogmm_cpu import SOGMMLearner as CPUFit
from sogmm_gpu import SOGMMf4Device as GPUContainerf4
from sogmm_gpu import SOGMMInference as GPUInference
from sogmm_gpu import SOGMMLearner as GPUFit
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker, MarkerArray

from scripts.master_gmm import MasterGMM


class SOGMMROSNode:
    """ROS Node for processing point clouds with SOGMM and visualizing results."""
    
    # Visualization constants
    MAX_UCT_SCALE_FACTOR = 3.0  # Scale multiplier for max uncertainty component
    MARKER_LIFETIME_SECS = 2.0  # Marker lifetime in seconds

    def __init__(self):
        """Initialize the SOGMM ROS Node with parameters, publishers, subscribers, and core components."""
        # Initialize ROS node
        rospy.init_node("sogmm_ros_node", anonymous=True)

        # Set log level from parameter
        self._set_log_level()

        # Load ROS parameters
        self._load_parameters()

        # Initialize publishers and subscribers
        self._setup_communication()

        # Initialize core components
        self._setup_core_components()

        # Log initialization summary
        self._log_initialization()
        
        # Register shutdown hook for timing statistics
        rospy.on_shutdown(self.shutdown_hook)

    def _set_log_level(self):
        """Set ROS logger level from parameter."""
        import logging

        log_level_str = rospy.get_param("~log_level", "INFO").upper()

        # Map string to Python logging level
        log_level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO, 
            "WARN": logging.WARN,
            "ERROR": logging.ERROR,
            "FATAL": logging.FATAL
        }

        if log_level_str not in log_level_map:
            rospy.logwarn(f"Invalid log level '{log_level_str}', defaulting to INFO")
            log_level_str = "INFO"

        # Set the logger level using Python's logging module
        logging.getLogger("rosout").setLevel(log_level_map[log_level_str])

        rospy.loginfo(f"Log level set to: {log_level_str}")
        rospy.logdebug("This is a debug message - if you see this, DEBUG logging is working!")

    def _load_parameters(self):
        """Load all ROS parameters with default values."""
        # Hardware selection
        self.use_gpu = rospy.get_param("~use_gpu", True)
        
        # SOGMM algorithm parameters
        self.bandwidth = rospy.get_param("~bandwidth", 0.02)
        self.tolerance = rospy.get_param("~tolerance", 1e-3)
        self.reg_covar = rospy.get_param("~reg_covar", 1e-6)
        self.max_iter = rospy.get_param("~max_iter", 100)
        self.kl_div_match_thresh = rospy.get_param("~kl_div_match_thresh", 5.0)
        self.l_thres = 0.05

        # Visualization parameters
        self.enable_visualization = rospy.get_param("~enable_visualization", True)
        self.visualization_scale = rospy.get_param("~visualization_scale", 2.0)
        self.color_by = rospy.get_param(
            "~color_by", "intensity"
        )  # Options: intensity, confidence, stability, combined
        self.add_metric_text = rospy.get_param("~add_metric_text", True)
        self.suspicious_displacement = rospy.get_param("~suspicious_displacement", 0.01)
        self.unstable_displacement = rospy.get_param("~unstable_displacement", 0.2)
        self.displacement_score_factor = rospy.get_param(
            "~displacement_score_factor", 0.05
        )

        self.uncertainty_heuristic = rospy.get_param(
            "~uncertainty_heuristic", "confidence"
        )  # Options: intensity, confidence, stability, combined
        self.uncertainty_scaler = rospy.get_param("~uncertainty_scaler", 2.0)

        # Processing parameters
        self.processing_decimation = rospy.get_param("~processing_decimation", 1)
        self.min_novel_points = rospy.get_param("~min_novel_points", 500)
        self.target_frame = rospy.get_param("~target_frame", "map")

        # Pruning parameters
        self.max_fusion_ratio = rospy.get_param("~max_fusion_ratio", 0.4)
        self.prune_min_observations = rospy.get_param("~prune_min_observations", 5)
        self.prune_interval_frames = rospy.get_param("~prune_interval_frames", 10)

        # Freezing parameters
        self.enable_freeze = rospy.get_param("~enable_freeze", True)
        self.freeze_interval_frames = rospy.get_param("~freeze_interval_frames", 5)
        self.freeze_fusion_threshold = rospy.get_param("~freeze_fusion_threshold", 20)

        self.publish_whole_gmm = rospy.get_param("~publish_whole_gmm", False)
        self.gmm_publish_rate = rospy.get_param("~gmm_publish_rate", 10.0)  # Hz

        self.fusion_weight_update = rospy.get_param("~fusion_weight_update", False)

        # Output file for saving GMM on shutdown
        self.output_file = rospy.get_param("~output_file", "gmm_output.txt")

        rospy.logdebug(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")

    def _setup_communication(self):
        """Initialize ROS publishers and subscribers."""
        # Publishers
        self.marker_pub = rospy.Publisher(
            "/starling1/mpa/gmm_markers", MarkerArray, queue_size=1
        )
        self.grid_marker_pub = rospy.Publisher(
            "/starling1/mpa/grid_markers", MarkerArray, queue_size=1
        )
        self.gmm_size_pub = rospy.Publisher(
            "/starling1/mpa/gmm_size", Int32, queue_size=1
        )

        self.gmm_pub = rospy.Publisher(
            "/starling1/mpa/gmm", GaussianMixtureModel, queue_size=1
        )

        self.updated_gmm_pub = rospy.Publisher(
            "/starling1/mpa/updated_gmm", GaussianMixtureModel, queue_size=1
        )

        # Subscribers
        self.pc_sub = rospy.Subscriber(
            "/starling1/mpa/tof_pc", PointCloud2, self.pointcloud_callback, queue_size=1
        )

        self.uct_id_sub = rospy.Subscriber(
            "/starling1/mpa/uct_id", Int32, self.uct_id_callback, queue_size=1
        )

        # TF setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def _setup_core_components(self):
        """Initialize core SOGMM processing components and threading."""
        # Core SOGMM components
        self.master_gmm = MasterGMM(
            self.uncertainty_heuristic,
            self.uncertainty_scaler,
            self.enable_freeze,
            self.fusion_weight_update,
        )
        
        # Select CPU or GPU implementation based on parameter
        if self.use_gpu:
            self.learner = GPUFit(
                bandwidth=self.bandwidth,
                tolerance=self.tolerance,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
            )
            self.fallback_learner = GPUFit(
                bandwidth=self.bandwidth,
                tolerance=self.tolerance,
                reg_covar=max(self.reg_covar * 100.0, 1e-3),
                max_iter=self.max_iter,
            )
            self.inference = GPUInference()
            self.ContainerClass = GPUContainerf4
            rospy.loginfo("Using GPU implementation for SOGMM")
        else:
            self.learner = CPUFit(
                bandwidth=self.bandwidth,
                tolerance=self.tolerance,
                reg_covar=self.reg_covar,
                max_iter=self.max_iter,
            )
            self.fallback_learner = CPUFit(
                bandwidth=self.bandwidth,
                tolerance=self.tolerance,
                reg_covar=max(self.reg_covar * 100.0, 1e-3),
                max_iter=self.max_iter,
            )
            self.inference = CPUInference()
            self.ContainerClass = CPUContainerf4
            rospy.loginfo("Using CPU implementation for SOGMM")

        # Processing state
        self.frame_count = 0
        self.novel_pts_placeholder = None

        # Threading for non-blocking processing
        self.processing_lock = threading.Lock()
        self.latest_pointcloud = None
        self.processing_thread = None

        # Timing
        self.last_prune_time = rospy.Time.now().to_sec()

        # Timing statistics
        self.gira_times = []
        self.processing_times = []
        self.viz_times = []
        self.full_processing_times = []

        self.max_uct_id = 0
        
        # Track updated indices for publishing
        self.updated_indices_lock = threading.Lock()
        self.latest_updated_indices = []
        
        # Setup GMM publishing timer
        if self.gmm_publish_rate > 0:
            self.gmm_publish_timer = rospy.Timer(
                rospy.Duration(1.0 / self.gmm_publish_rate),
                self.gmm_publish_timer_callback
            )

    def _log_initialization(self):
        """Log initialization parameters for debugging."""
        rospy.loginfo("SOGMM ROS Node initialized with parameters:")
        rospy.loginfo(f"  - Bandwidth: {self.bandwidth}")
        rospy.loginfo(f"  - KL divergence match threshold: {self.kl_div_match_thresh}")
        rospy.loginfo(f"  - Target frame: {self.target_frame}")
        rospy.loginfo(f"  - Processing decimation: {self.processing_decimation}")
        rospy.loginfo(f"  - Use GPU: {self.use_gpu}")
        rospy.loginfo(f"  - Visualization enabled: {self.enable_visualization}")
        rospy.loginfo(f"  - Visualization scale: {self.visualization_scale}")
        rospy.loginfo(f"  - Color by: {self.color_by}")
        rospy.loginfo(f"  - Add metric text: {self.add_metric_text}")
        rospy.loginfo("  - Pruning parameters:")
        rospy.loginfo(f"    * Min observations: {self.prune_min_observations}")
        rospy.loginfo(f"    * Max fusion ratio: {self.max_fusion_ratio}")
        rospy.loginfo(f"    * Prune interval (frames): {self.prune_interval_frames}")

    def uct_id_callback(self, msg: Int32):
        self.max_uct_id = msg.data

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
            rospy.logdebug("Entered process_pointcloud")

            pcld = self.preprocess_point_cloud(msg)
            if pcld is None:
                return

            # Fit SOGMM model
            gmm_start_time = time.time()

            # 1. Generate a local GMM from the current point cloud
            local_model = self.ContainerClass()
            ms_data = self.extract_ms_data(pcld)

            try:
                self.learner.fit(ms_data, pcld, local_model)
            except RuntimeError as e:
                # Catch Open3D/CUDA Cholesky decomposition failure
                if "potrfBatched failed" in str(e):
                    rospy.logwarn("GMM fitting failed (singular covariance). Retrying with higher regularization...")
                    # Create a temporary learner with higher regularization (e.g. 1e-3) to ensure stability
                    self.fallback_learner.fit(ms_data, pcld, local_model)
                else:
                    raise e
            
            # Convert to CPU if using GPU (MasterGMM expects CPU containers)
            if self.use_gpu:
                local_model_cpu = CPUContainerf4(local_model.n_components_)
                local_model.to_host(local_model_cpu)
            else:
                local_model_cpu = local_model

            gira_timestamp = time.time()
            gira_time = gira_timestamp - gmm_start_time

            # 2. Update the master GMM with the new local measurement
            updated_indices = self.master_gmm.update(
                gmm_measurement=local_model_cpu,
                match_threshold=self.kl_div_match_thresh,
            )

            # 3. Periodically prune outliers (every N frames)
            self.frame_count += 1
            if self.frame_count % self.prune_interval_frames == 0:
                self.master_gmm.prune_stale_components(
                    min_observations=self.prune_min_observations,
                    max_fusion_ratio=self.max_fusion_ratio,
                )
                # Filter out indices that were pruned
                current_n_components = self.master_gmm.model.n_components_
                updated_indices = [idx for idx in updated_indices if idx < current_n_components]

            if (
                self.frame_count % self.freeze_interval_frames == 0
                and self.enable_freeze
            ):
                self.master_gmm.freeze_components(
                    freeze_fusion_threshold=self.freeze_fusion_threshold
                )

            proc_timestamp = time.time()
            processing_time = proc_timestamp - gira_timestamp

            # Store updated indices for periodic publishing (AFTER pruning)
            with self.updated_indices_lock:
                self.latest_updated_indices = updated_indices

            # Visualize results - only if we have a model

            viz_start_time = time.time()
            if self.enable_visualization and self.master_gmm.model.n_components_ > 0:
                self.visualize_gmm(self.master_gmm, self.target_frame, msg.header.stamp, updated_indices)
            viz_time = time.time() - viz_start_time

            full_processing_time = time.time() - start_time
            
            # Store timing statistics
            self.gira_times.append(gira_time)
            self.processing_times.append(processing_time)
            self.viz_times.append(viz_time)
            self.full_processing_times.append(full_processing_time)
            
            n_components = self.master_gmm.model.n_components_
            rospy.loginfo(
                f"Processed point cloud with {n_components} components in {full_processing_time:.3f}s (GIRA: {gira_time:.3f}s, Processing: {processing_time:.3f}s, Viz: {viz_time:.3f}s)"
            )

            gmm_stats_msg = Int32()
            gmm_stats_msg.data = n_components

            self.gmm_size_pub.publish(gmm_stats_msg)

        except Exception as e:
            rospy.logerr(f"Error processing point cloud: {e}\n{traceback.format_exc()}")

    def gmm_publish_timer_callback(self, event):
        """Timer callback to periodically publish the latest GMM."""

        rospy.logdebug("Entered gmm_publish_timer_callback")

        if self.master_gmm.model is None or self.master_gmm.model.n_components_ == 0:
            return
        
        # Get the latest updated indices
        with self.updated_indices_lock:
            updated_indices = self.latest_updated_indices.copy()
        
        self.publish_gmm(updated_indices)
    
    def publish_gmm(self, updated_indices):
        """Publish the current state of the master GMM."""
        if self.master_gmm.model is None or self.master_gmm.model.n_components_ == 0:
            return

        rospy.logdebug("Entered publish_gmm")
        
        # Acquire lock to prevent model modification during publishing
        with self.processing_lock:
            # Check again after acquiring lock
            if self.master_gmm.model is None or self.master_gmm.model.n_components_ == 0:
                return
                
            # Filter out invalid indices that may have been pruned
            current_n_components = self.master_gmm.model.n_components_
            valid_updated_indices = [idx for idx in updated_indices if idx < current_n_components]
            
            if len(valid_updated_indices) < len(updated_indices):
                rospy.logdebug(f"Filtered out {len(updated_indices) - len(valid_updated_indices)} invalid indices due to pruning")
            
            # Create message for complete GMM
            complete_gmm_msg = GaussianMixtureModel()
            complete_gmm_msg.n_components = current_n_components
            
            # Create message for updated Gaussians only
            updated_gmm_msg = GaussianMixtureModel()
            updated_gmm_msg.n_components = len(valid_updated_indices)

            try:
                for idx in range(current_n_components):
                    gaussian_component = GaussianComponent()
                    # use the first 3 dimensions of the mean (x,y,z)
                    gaussian_component.mean = self.master_gmm.model.means_[idx, :3].tolist()
                    # extract the 4x4 covariance for this component and take the 3x3 spatial part
                    cov3 = self.master_gmm.model.covariances_.reshape(-1, 4, 4)[idx, :3, :3]
                    gaussian_component.covariance = cov3.flatten().tolist()
                    gaussian_component.weight = float(self.master_gmm.model.weights_[idx])
                    gaussian_component.fusion_count = int(self.master_gmm.model.fusion_counts_[idx])
                    gaussian_component.observation_count = int(self.master_gmm.model.observation_counts_[idx])
                    gaussian_component.last_displacement = float(self.master_gmm.model.last_displacements_[idx])
                    gaussian_component.uncertainty = float(self.master_gmm.model.uncertainty_[idx])
                    
                    # Add to complete GMM message
                    complete_gmm_msg.components.append(gaussian_component)
                    
                    # Add to updated GMM message only if it was updated
                    if idx in valid_updated_indices:
                        # Create a separate component for updated message
                        updated_component = GaussianComponent()
                        updated_component.mean = gaussian_component.mean
                        updated_component.covariance = gaussian_component.covariance
                        updated_component.weight = gaussian_component.weight
                        updated_component.fusion_count = gaussian_component.fusion_count
                        updated_component.observation_count = gaussian_component.observation_count
                        updated_component.last_displacement = gaussian_component.last_displacement
                        updated_component.uncertainty = gaussian_component.uncertainty
                        updated_gmm_msg.components.append(updated_component)
            except IndexError as e:
                rospy.logwarn(f"IndexError during GMM publishing (likely due to concurrent pruning): {e}. Skipping this publish cycle.")
                return  # Skip publishing this cycle, next one will be correct

        # Publish outside the lock to avoid blocking processing
        self.gmm_pub.publish(complete_gmm_msg)
        self.updated_gmm_pub.publish(updated_gmm_msg)

    def transform_point_cloud(
        self, msg: PointCloud2, points_3d_original: np.ndarray, target_frame: str
    ) -> np.ndarray:
        """Transform point cloud to target frame using TF.
        
        Args:
            msg: Original PointCloud2 message
            points_3d_original: Points in original frame
            target_frame: Target TF frame
            
        Returns:
            Transformed points
        """
        try:
            # Lookup transform
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                msg.header.frame_id,
                msg.header.stamp,
                rospy.Duration(1.0),
            )

            # Extract transformation
            translation = np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z,
            ])

            rotation_quat = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ]

            # Apply transformation: R * points + t
            rotation_matrix = R.from_quat(rotation_quat).as_matrix()
            return (rotation_matrix @ points_3d_original.T).T + translation

        except Exception as e:
            rospy.logwarn(f"Transform failed: {e}, using original points")
            return points_3d_original

    def preprocess_point_cloud(self, msg: PointCloud2) -> Optional[np.ndarray]:
        """Preprocess point cloud: extract points, remove NaNs, transform, and decimate.
        
        Args:
            msg: Input PointCloud2 message
            
        Returns:
            Preprocessed point cloud array (N, 4) or None if invalid
        """
        # Convert to numpy array
        pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(msg)

        # Extract XYZ coordinates
        points_3d_original = np.column_stack([
            pc_array["x"].flatten(),
            pc_array["y"].flatten(),
            pc_array["z"].flatten(),
        ])

        # Remove invalid points
        valid_mask = np.isfinite(points_3d_original).all(axis=1)
        points_3d_original = points_3d_original[valid_mask]

        if len(points_3d_original) == 0:
            rospy.logwarn("No valid points in point cloud")
            return None

        # Transform to target frame
        points_3d = self.transform_point_cloud(
            msg, points_3d_original, self.target_frame
        )

        # Decimate for faster processing
        if self.processing_decimation > 1:
            indices = np.arange(0, len(points_3d), self.processing_decimation)
            points_3d = points_3d[indices]

        rospy.loginfo(f"Processing {len(points_3d)} points")

        # Add dummy intensity channel (SOGMM requires 4D)
        intensity = np.ones(len(points_3d)) * 0.5
        return np.column_stack([points_3d, intensity])

    def extract_ms_data(self, X: np.ndarray) -> np.ndarray:
        """Extract mean-shift data (distance and intensity) from point cloud.
        
        Args:
            X: Point cloud array with shape (N, 4) where columns are [x, y, z, intensity]
            
        Returns:
            Array with shape (N, 2) containing [distance, intensity]
        """
        d = np.array([np.linalg.norm(x) for x in X[:, 0:3]])[:, np.newaxis]
        g = X[:, 3][:, np.newaxis]
        return np.concatenate((d, g), axis=1)

    def extract_novel(self, model: CPUContainerf4, pcld: np.ndarray, comp_indices: List[int]) -> np.ndarray:
        """Extract novel points not well-explained by specified GMM components.
        
        Args:
            model: GMM model to score against
            pcld: Point cloud array
            comp_indices: Indices of components to use for novelty detection
            
        Returns:
            Filtered point cloud containing only novel points
        """
        # create a GMM submap using component indices
        submap = model.submap_from_indices(list(comp_indices))

        # transfer to GPU/CPU container based on configuration
        submap_device = self.ContainerClass(submap.n_components_)
        submap_device.from_host(submap)

        # perform likelihood score computation
        scores = self.inference.score_3d(pcld[:, :3], submap_device)

        # filter the point cloud and return novel points
        scores = scores.flatten()
        return pcld[scores < self.l_thres, :]

    def visualize_gmm(self, master_gmm, frame_id, timestamp, updated_indices=None):
        """
        Publish MarkerArray for RViz visualization
        """
        if master_gmm.model.n_components_ == 0:
            return

        marker_array = MarkerArray()

        # Create markers for each Gaussian component
        for i in range(len(master_gmm.model.means_)):
            updated_marker = updated_indices is not None and i in updated_indices
            # Create sphere marker
            sphere_marker = self._create_sphere_marker(
                master_gmm, i, frame_id, timestamp, updated_marker
            )
            if sphere_marker:
                marker_array.markers.append(sphere_marker)

            # Create text marker if enabled
            if self.add_metric_text:
                text_marker = self._create_text_marker(
                    master_gmm, i, frame_id, timestamp
                )
                marker_array.markers.append(text_marker)

        # Clear old markers if component count decreased
        self._add_clear_markers(
            marker_array, len(master_gmm.model.means_), frame_id, timestamp
        )

        # Publish and update state
        self.marker_pub.publish(marker_array)
        self.last_marker_count = len(master_gmm.model.means_)
        rospy.logdebug(
            f"Published {len(master_gmm.model.means_)} ellipsoid markers with text"
        )

    def _calculate_normalization_values(self, master_gmm):
        """Calculate normalized values for color mapping based on selected mode."""
        norm_values = []

        if self.color_by == "confidence":
            # log_counts = [np.log1p(c) for c in master_gmm.model.fusion_counts_]
            # max_log_count = max(log_counts) if log_counts else 1.0
            # norm_values = [c / max(1.0, max_log_count) for c in log_counts]

            norm_values = [
                -np.exp(-c / 2.0) + 1 for c in master_gmm.model.fusion_counts_
            ]

        elif self.color_by == "stability":
            for idx in range(len(master_gmm.model.means_)):
                displacement = master_gmm.model.last_displacements_[idx]
                score = np.exp(-displacement / 0.05)
                norm_values.append(score)

            if norm_values:
                max_score = max(norm_values) if max(norm_values) > 0 else 1.0
                norm_values = [s / max_score for s in norm_values]

        elif self.color_by == "combined":
            norm_values = self._calculate_combined_scores(master_gmm)

        return norm_values

    def _calculate_combined_scores(self, master_gmm):
        """Calculate combined stability scores based on fusion count and displacement."""
        norm_values = []

        for idx in range(len(master_gmm.model.means_)):
            fusion_count = master_gmm.model.fusion_counts_[idx]
            displacement = master_gmm.model.last_displacements_[idx]

            if fusion_count <= 1:
                score = 0.0  # Unverified
            elif displacement < self.suspicious_displacement:
                score = 0.3  # Suspicious (no movement)
            elif displacement > self.unstable_displacement:
                score = 0.5  # Unstable (large displacement)
            else:
                # Good stability: combine displacement and fusion scores
                displacement_score = np.exp(
                    -displacement / self.displacement_score_factor
                )
                fusion_score = np.log1p(fusion_count) / np.log1p(10)
                score = fusion_score * displacement_score

            norm_values.append(score)

        # Normalize to 0-1 range
        if norm_values:
            max_score = max(norm_values) if max(norm_values) > 0 else 1.0
            norm_values = [s / max_score for s in norm_values]

        return norm_values

    def _create_sphere_marker(
        self, master_gmm: MasterGMM, i: int, frame_id: str, timestamp, updated_marker: bool
    ) -> Optional[Marker]:
        """Create a sphere marker for a Gaussian component.
        
        Args:
            master_gmm: Master GMM containing the component
            i: Index of component to visualize
            frame_id: TF frame for the marker
            timestamp: Timestamp for the marker
            updated_marker: Whether this component was recently updated
            
        Returns:
            Marker object, or None if creation failed
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = timestamp
        marker.ns = "sogmm_ellipsoids"
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(self.MARKER_LIFETIME_SECS)

        # Set position
        mean = master_gmm.model.means_[i, :3]
        marker.pose.position.x = float(mean[0])
        marker.pose.position.y = float(mean[1])
        marker.pose.position.z = float(mean[2])

        # Set scale and orientation based on covariance
        if not self._set_marker_geometry(marker, master_gmm.model.covariances_[i]):
            return None

        # Set color based on visualization mode
        self._set_marker_color(marker, master_gmm, i, updated_marker)

        return marker

    def _set_marker_geometry(self, marker: Marker, covariance_flat: np.ndarray) -> bool:
        """Set marker scale and orientation based on covariance matrix.
        
        Args:
            marker: Marker to modify
            covariance_flat: Flattened 16-element covariance array
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract 3D spatial covariance
            cov_4x4 = covariance_flat.reshape(4, 4)
            cov_3d = cov_4x4[:3, :3]

            # Compute eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(cov_3d)

            # Ensure a right-handed coordinate system
            eigenvecs[:, 2] = np.cross(eigenvecs[:, 0], eigenvecs[:, 1])

            # Ensure eigenvalues are non-negative
            eigenvals = np.maximum(eigenvals, 1e-9)

            # Set scale based on eigenvalues (2 standard deviations)
            scale_factor = 2.0 * self.visualization_scale
            marker.scale.x = scale_factor * np.sqrt(eigenvals[0])
            marker.scale.y = scale_factor * np.sqrt(eigenvals[1])
            marker.scale.z = scale_factor * np.sqrt(eigenvals[2])

            # Set orientation based on eigenvectors
            r = R.from_matrix(eigenvecs)
            quat = r.as_quat()
            marker.pose.orientation.x = float(quat[0])
            marker.pose.orientation.y = float(quat[1])
            marker.pose.orientation.z = float(quat[2])
            marker.pose.orientation.w = float(quat[3])

            return True

        except (np.linalg.LinAlgError, ValueError) as e:
            rospy.logwarn(f"Error processing covariance: {e}")
            # Fallback to spherical visualization
            avg_scale = self.visualization_scale * 0.1
            marker.scale.x = marker.scale.y = marker.scale.z = avg_scale
            marker.pose.orientation.w = 1.0
            return True

    def _set_marker_color(self, marker, master_gmm, i, updated_marker):
        """Set marker color based on visualization mode."""
        
        # Highlight max uncertainty component
        # if i == self.max_uct_id:
        #     marker.color.r = 0.0
        #     marker.color.g = 1.0
        #     marker.color.b = 0.0
        #     marker.color.a = 1.0
        #     marker.scale.x *= self.MAX_UCT_SCALE_FACTOR
        #     marker.scale.y *= self.MAX_UCT_SCALE_FACTOR
        #     marker.scale.z *= self.MAX_UCT_SCALE_FACTOR
        #     return

        # Highlight recently updated components
        # if updated_marker:
        #     marker.color.r = 0.0
        #     marker.color.g = 1.0
        #     marker.color.b = 0.9
        #     marker.color.a = 1.0
        #     return

        # Color based on selected visualization mode
        if self.color_by == "confidence":
            color = cm.plasma(-master_gmm.model.uncertainty_[i] + 1.0)
            marker.color.r, marker.color.g, marker.color.b = color[0], color[1], color[2]
            marker.color.a = 0.8

        elif self.color_by in ["stability", "combined"]:
            # Use same inversion as confidence: low uncertainty (high stored value) → yellow, high uncertainty (low stored value) → blue
            color = cm.plasma(-master_gmm.model.uncertainty_[i] + 1.0)
            marker.color.r, marker.color.g, marker.color.b = color[0], color[1], color[2]
            marker.color.a = 0.8

        else:  # Default to intensity
            intensity = master_gmm.model.means_[i, 3]
            gray_val = float(intensity)
            marker.color.r = marker.color.g = marker.color.b = gray_val
            marker.color.a = float(min(1.0, max(0.3, intensity * 2.0)))

    def _set_combined_color(self, marker, master_gmm, i):
        """Set color for combined visualization mode."""
        fusion_count = master_gmm.model.fusion_counts_[i]
        displacement = master_gmm.model.last_displacements_[i]

        if fusion_count <= 1:
            # Red = unverified
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
            marker.color.a = 0.6
        elif displacement < self.suspicious_displacement:
            # Yellow = suspicious
            marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
            marker.color.a = 0.7
        elif displacement > self.unstable_displacement:
            # Orange = unstable
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
            marker.color.a = 0.7
        else:
            # Green = stable (intensity based on score)
            green_intensity = master_gmm.model.uncertainty_[i]
            marker.color.r, marker.color.g, marker.color.b = (
                0.0,
                float(green_intensity),
                0.0,
            )
            marker.color.a = 0.8

    def _create_text_marker(self, master_gmm, i, frame_id, timestamp):
        """Create a text marker for a Gaussian component."""
        text_marker = Marker()
        text_marker.header.frame_id = frame_id
        text_marker.header.stamp = timestamp
        text_marker.ns = "sogmm_text"
        text_marker.id = i
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.lifetime = rospy.Duration(self.MARKER_LIFETIME_SECS)

        # Position at component mean
        mean = master_gmm.model.means_[i, :3]
        text_marker.pose.position.x = float(mean[0])
        text_marker.pose.position.y = float(mean[1])
        text_marker.pose.position.z = float(mean[2])
        text_marker.pose.orientation.w = 1.0

        # Set text content based on visualization mode
        text_marker.text = self._get_text_content(master_gmm, i)

        # Set text properties
        text_marker.scale.z = 0.05
        text_marker.color.r = text_marker.color.g = text_marker.color.b = 1.0
        text_marker.color.a = 1.0

        return text_marker

    def _get_text_content(self, master_gmm, i):
        """Get text content for marker based on visualization mode."""
        suffix = " [MAX]" if i == self.max_uct_id else ""

        if self.color_by == "confidence":
            return f"C:{master_gmm.model.uncertainty_[i]:.3f}{suffix}"
        elif self.color_by == "stability":
            return f"D:{master_gmm.model.last_displacements_[i]:.3f}{suffix}\nC:{master_gmm.model.uncertainty_[i]:.3f}{suffix}"
        elif self.color_by == "combined":
            count = master_gmm.model.fusion_counts_[i]
            disp = master_gmm.model.last_displacements_[i]
            return f"C:{count} D:{disp:.3f}{suffix}"
        else:
            return f"{master_gmm.model.means_[i, 3]:.2f}{suffix}"

    def _add_clear_markers(self, marker_array, current_count, frame_id, timestamp):
        """Add markers to clear old components if count decreased."""
        if not hasattr(self, "last_marker_count"):
            return

        for i in range(current_count, self.last_marker_count):
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

    def save_gmm_to_file(self, filepath: str):
        """
        Save the master GMM to a file in the specified format.
        Format:
        mean: x y z
        cov: c11 c12 c13
        cov: c21 c22 c23
        cov: c31 c32 c33
        """
        if self.master_gmm.model is None or self.master_gmm.model.n_components_ == 0:
            rospy.logwarn("No GMM to save (model is empty)")
            return
        
        try:
            with open(filepath, 'w') as f:
                for i in range(self.master_gmm.model.n_components_):
                    # Extract mean (first 3 dimensions: x, y, z)
                    mean = self.master_gmm.model.means_[i, :3]
                    f.write(f"mean: {mean[0]:.6f} {mean[1]:.6f} {mean[2]:.6f}\n")
                    
                    # Extract 3x3 spatial covariance from 4x4 covariance matrix
                    cov_4x4 = self.master_gmm.model.covariances_[i].reshape(4, 4)
                    cov_3x3 = cov_4x4[:3, :3]
                    
                    # Write covariance matrix row by row
                    for row in range(3):
                        f.write(f"cov: {cov_3x3[row, 0]:.6f} {cov_3x3[row, 1]:.6f} {cov_3x3[row, 2]:.6f}\n")
            
            rospy.loginfo(f"GMM saved to {filepath} ({self.master_gmm.model.n_components_} components)")
        except Exception as e:
            rospy.logerr(f"Failed to save GMM to file: {e}")

    def shutdown_hook(self):
        """
        Called when node is shutting down. Saves GMM and logs timing statistics.
        """
        # Save GMM to file
        self.save_gmm_to_file(self.output_file)
        
        rospy.loginfo("="*80)
        rospy.loginfo("SOGMM Node Shutdown - Timing Statistics Summary")
        rospy.loginfo("="*80)
        
        if len(self.gira_times) > 0:
            rospy.loginfo(f"Total frames processed: {len(self.gira_times)}")
            rospy.loginfo("")
            
            gira_mean = np.mean(self.gira_times)
            gira_std = np.std(self.gira_times)
            rospy.loginfo(f"GIRA (GMM Generation) Time:")
            rospy.loginfo(f"  Mean: {gira_mean:.4f}s")
            rospy.loginfo(f"  Std:  {gira_std:.4f}s")
            rospy.loginfo("")
            
            proc_mean = np.mean(self.processing_times)
            proc_std = np.std(self.processing_times)
            rospy.loginfo(f"Processing Time:")
            rospy.loginfo(f"  Mean: {proc_mean:.4f}s")
            rospy.loginfo(f"  Std:  {proc_std:.4f}s")
            rospy.loginfo("")
            
            viz_mean = np.mean(self.viz_times)
            viz_std = np.std(self.viz_times)
            rospy.loginfo(f"Visualization Time:")
            rospy.loginfo(f"  Mean: {viz_mean:.4f}s")
            rospy.loginfo(f"  Std:  {viz_std:.4f}s")
            rospy.loginfo("")
            
            full_mean = np.mean(self.full_processing_times)
            full_std = np.std(self.full_processing_times)
            rospy.loginfo(f"Total Processing Time:")
            rospy.loginfo(f"  Mean: {full_mean:.4f}s")
            rospy.loginfo(f"  Std:  {full_std:.4f}s")
        else:
            rospy.loginfo("No timing data collected (no frames processed)")
        
        rospy.loginfo("="*80)

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
