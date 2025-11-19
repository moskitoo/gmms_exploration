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
    def __init__(self, uncertainty_heuristic, uncertainty_scaler):
        self.model = None

        # R-tree for spatial indexing of GMM components
        self.init_r_tree()

    def init_r_tree(self):
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
        if self.model is None:
            self._initialize_from_measurement(gmm_measurement)
            return
        
        candidate_indices = self._find_spatial_candidates(gmm_measurement)
        self._update_observation_counts(candidate_indices)
        
        n_fused, new_components_ids = self._match_and_fuse_components(
            gmm_measurement, candidate_indices, match_threshold
        )
        
        if len(new_components_ids) > 0:
            self._add_new_components(gmm_measurement, new_components_ids)
        
        # self._normalize_weights()
        self.model.normalize_weights()
        
        # Log results
        n_novel = len(new_components_ids)
        rospy.loginfo(f"Fused: {n_fused}, Novel: {n_novel}")

    def _initialize_from_measurement(self, gmm_measurement):
        """Initialize master GMM from first measurement."""

        self.model = CPUContainerf4(gmm_measurement)
        
        # Build spatial index
        for i in range(self.model.n_components_):
            mean = self.model.means_[i, :3]
            self.rtree.insert(i, (*mean, *mean))
        
        # self._normalize_weights() its probably not needed because newly created model will alreadyhave a normalised weightse

    def _find_spatial_candidates(self, gmm_measurement):
        """Find master components spatially overlapping with measurement."""
        # Get bounding box of incoming measurement
        min_bounds = gmm_measurement.means_[:, :3].min(axis=0)
        max_bounds = gmm_measurement.means_[:, :3].max(axis=0)
        
        # Query R-tree for overlapping components
        return list(self.rtree.intersection((*min_bounds, *max_bounds)))

    def _update_observation_counts(self, candidate_indices):
        """Update observation counts for components in re-observed region."""
        if candidate_indices:
            self.model.observation_counts_[candidate_indices] += 1

    def _match_and_fuse_components(self, gmm_measurement, candidate_indices, match_threshold):
        """
        Match measurement components to master components and fuse them using vectorized operations.
        
        Returns:
            tuple: (number_fused, list_of_unmatched_components)
        """
        if not candidate_indices:
            return 0, list(range(gmm_measurement.n_components_))

        # 1. Create a subset of the master model for matching
        model_subset = self.model.submap_from_indices(candidate_indices)

        # 2. Calculate the pairwise KL divergence matrix
        kl_matrix = self._calculate_kl_matrix(
            gmm_measurement.means_, gmm_measurement.covariances_,
            model_subset.means_, model_subset.covariances_
        )

        # 3. Find the best matches for each measurement component
        best_master_indices_in_subset = np.argmin(kl_matrix, axis=1)
        min_kl_values = np.min(kl_matrix, axis=1)

        # 4. Identify which measurement components have a match below the threshold
        matched_mask = min_kl_values < match_threshold
        meas_indices_to_fuse = np.where(matched_mask)[0]
        
        unmatched_measurement_indices = np.where(~matched_mask)[0].tolist()
        
        if len(meas_indices_to_fuse) == 0:
            return 0, unmatched_measurement_indices

        # 5. Get the corresponding master components to fuse with
        subset_indices_to_fuse_with = best_master_indices_in_subset[matched_mask]
        # Map subset indices back to original master model indices
        master_indices_to_fuse_with = np.array(candidate_indices)[subset_indices_to_fuse_with]

        # 6. Perform vectorized fusion for all matched pairs
        self._fuse_components_vectorized(master_indices_to_fuse_with, gmm_measurement, meas_indices_to_fuse)
        
        # The number of fused components is the number of unique master components that were updated
        n_fused = len(np.unique(master_indices_to_fuse_with))
        
        return n_fused, unmatched_measurement_indices
    
    def _calculate_kl_matrix(self, means1, covs1, means2, covs2, symmetric=True):
        """
        Calculates a matrix of pairwise KL divergences between two sets of Gaussians.
        
        Args:
            means1 (np.ndarray): Means of the first set (N, D).
            covs1 (np.ndarray): Covariances of the first set (N, D*D).
            means2 (np.ndarray): Means of the second set (M, D).
            covs2 (np.ndarray): Covariances of the second set (M, D*D).
        
        Returns:
            np.ndarray: A (N, M) matrix where entry (i, j) is the KL divergence
                        between Gaussian i from set 1 and Gaussian j from set 2.
        """
        n1, d = means1.shape
        n2, _ = means2.shape
        
        # Reshape flattened covariances to (N, D, D) matrices
        covs1_mat = covs1.reshape(n1, d, d)
        covs2_mat = covs2.reshape(n2, d, d)

        try:
            # Pre-compute inverses and log-determinants for all components in set 2
            inv_covs2 = np.linalg.inv(covs2_mat)
            _, log_det_covs2 = np.linalg.slogdet(covs2_mat)
            
            # Pre-compute log-determinants for all components in set 1
            _, log_det_covs1 = np.linalg.slogdet(covs1_mat)

            # Expand dimensions for broadcasting
            m1_exp = means1[:, np.newaxis, :]          # (N, 1, D)
            c1_exp = covs1_mat[:, np.newaxis, :, :]    # (N, 1, D, D)
            m2_exp = means2[np.newaxis, :, :]          # (1, M, D)
            inv_c2_exp = inv_covs2[np.newaxis, :, :, :]# (1, M, D, D)

            # Vectorized computation of KL divergence terms
            # Term 1: Trace term
            trace_term = np.einsum('...ij,...ji->...', inv_c2_exp, c1_exp) # (N, M)
            
            # Term 2: Mahalanobis distance term
            mean_diff = m2_exp - m1_exp # (N, M, D)
            mahalanobis_term = np.einsum('...i,...ij,...j->...', mean_diff, inv_c2_exp, mean_diff) # (N, M)

            # Term 3: Log determinant term
            log_det_term = log_det_covs2[np.newaxis, :] - log_det_covs1[:, np.newaxis] # (N, M)

            # KL divergence D(1 || 2)
            kl_12 = 0.5 * (trace_term + mahalanobis_term - d + log_det_term)

            if not symmetric:
                return kl_12
            
            # For symmetric KL, calculate D(2 || 1)
            inv_covs1 = np.linalg.inv(covs1_mat)
            inv_c1_exp = inv_covs1[:, np.newaxis, :, :] # (N, 1, D, D)
            c2_exp = covs2_mat[np.newaxis, :, :, :]     # (1, M, D, D)
            
            trace_term_21 = np.einsum('...ij,...ji->...', inv_c1_exp, c2_exp)
            mahalanobis_term_21 = np.einsum('...i,...ij,...j->...', -mean_diff, inv_c1_exp, -mean_diff)
            log_det_term_21 = -log_det_term

            kl_21 = 0.5 * (trace_term_21 + mahalanobis_term_21 - d + log_det_term_21)
            
            return kl_12 + kl_21

        except np.linalg.LinAlgError:
            # Fallback to infinity if any matrix is singular
            return np.full((n1, n2), float("inf"))   
    
    def _fuse_components_vectorized(self, master_indices, meas_gmm, meas_indices):
        """
        Vectorized fusion of multiple measurement components into master components.
        """
        # --- Gather data for all components to be fused ---
        # Master components
        w_i = self.model.weights_[master_indices]
        mu_i = self.model.means_[master_indices]
        cov_i = self.model.covariances_[master_indices].reshape(-1, 4, 4)

        # Measurement components
        meas_comps_to_fuse = meas_gmm.submap_from_indices(meas_indices)
        w_j = meas_comps_to_fuse.weights_
        mu_j = meas_comps_to_fuse.means_
        cov_j = meas_comps_to_fuse.covariances_.reshape(-1, 4, 4)

        # --- Perform fusion calculations in a vectorized manner ---
        w_new = w_i + w_j
        mu_new = (w_i[:, np.newaxis] * mu_i + w_j[:, np.newaxis] * mu_j) / w_new[:, np.newaxis]

        diff_i = mu_i - mu_new
        diff_j = mu_j - mu_new

        # Use einsum for batched outer product: diff @ diff.T
        # '...i,...j->...ij' computes the outer product for each vector in the batch
        outer_i = np.einsum('...i,...j->...ij', diff_i, diff_i)
        outer_j = np.einsum('...i,...j->...ij', diff_j, diff_j)

        cov_new = (w_i[:, np.newaxis, np.newaxis] * (cov_i + outer_i) + 
                   w_j[:, np.newaxis, np.newaxis] * (cov_j + outer_j)) / w_new[:, np.newaxis, np.newaxis]

        # --- Update master model state ---
        # This is tricky because multiple `meas_indices` can map to the same `master_index`.
        # We need to sum the contributions for each unique master component.
        unique_master_indices, inverse_indices = np.unique(master_indices, return_inverse=True)

        # Update weights by adding all contributions from new measurements
        np.add.at(self.model.weights_, master_indices, w_j)

        # Update means and covariances using the final fused values
        # For master components matched multiple times, this uses the result from the *last* match.
        # A more complex (but slower) approach would be to iteratively fuse. This is a good approximation.
        self.model.means_[master_indices] = mu_new
        self.model.covariances_[master_indices] = cov_new.reshape(-1, 16)

        # Update fusion counts
        np.add.at(self.model.fusion_counts_, master_indices, 1)

        # Update displacements (store the last displacement for each master component)
        displacements = np.linalg.norm(mu_new[:, :3] - mu_i[:, :3], axis=1)
        self.model.last_displacements_[master_indices] = displacements

        self.model.uncertainty_[master_indices] = self._calculate_uncertainty(master_indices)


    def _add_new_components(self, gmm_measurement, new_components_ids):
        """Add unmatched components as new master components."""

        old_n_components = self.model.n_components_

        new_gmm = gmm_measurement.submap_from_indices(new_components_ids)

        print("==================================")
        print(f"new gaussians uncerainties: {new_gmm.uncertainty_}")
        print("==================================")

        self.model.merge(new_gmm)
        
        # Update spatial index with new components
        for i in range(old_n_components, self.model.n_components_):
            mean = self.model.means_[i, :3]
            self.rtree.insert(i, (*mean, *mean))

    def prune_stale_components(self, min_observations=5, max_fusion_ratio=0.4):
        """
        Remove Gaussians that are stale outliers using vectorized operations.
        
        A Gaussian is considered stale if it has been observed multiple times but has a low
        fusion ratio, indicating it's likely an outlier in a re-observed area.
        """
        if self.model.n_components_ == 0:
            return 0
        
        # --- Vectorized Pruning Logic ---
        # Convert tracking lists to numpy arrays for vectorized operations
        obs_counts = self.model.observation_counts_
        fus_counts = self.model.fusion_counts_
        
        # Calculate fusion ratio for all components, avoiding division by zero
        # Where obs_counts is 0, the ratio is irrelevant as it will be kept anyway.
        # We can safely set the ratio to 1.0 in these cases.
        fusion_ratios = np.divide(fus_counts, obs_counts, 
                                  out=np.ones_like(fus_counts, dtype=float), 
                                  where=obs_counts!=0)
        
        # Determine which components to keep based on the criteria
        # Keep if: under-observed OR has a good fusion ratio
        keep_mask = (obs_counts < min_observations) | (fusion_ratios >= max_fusion_ratio)
        
        n_pruned = np.sum(~keep_mask)
        
        if n_pruned == 0:
            return 0
        
        # --- Apply Pruning ---
        # Get indices of components to keep
        keep_indices = np.where(keep_mask)[0].tolist()
        
        # Prune the model directly using the list of indices
        pruned_model = self.model.submap_from_indices(keep_indices)
        self.model = pruned_model
        
        # Rebuild the R-tree from scratch with the remaining components
        p = index.Property()
        p.dimension = 3
        self.rtree = index.Index(properties=p)
        for i in range(self.model.n_components_):
            mean = self.model.means_[i, :3]
            self.rtree.insert(i, (*mean, *mean))
        
        # Renormalize weights after pruning
        self.model.normalize_weights()
        
        rospy.loginfo("=========================================================================")
        rospy.loginfo(f"Pruned {n_pruned} stale outlier Gaussians, {self.model.n_components_} remaining")
        rospy.loginfo("=========================================================================")
        return n_pruned

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

    def _calculate_uncertainty(self, master_indices):
        """Calculate normalized values for color mapping based on selected mode."""
        norm_values = []
        
        if self.uncertainty_heuristic == "confidence":
            # log_counts = [np.log1p(c) for c in master_gmm.model.fusion_counts_]
            # max_log_count = max(log_counts) if log_counts else 1.0
            # norm_values = [c / max(1.0, max_log_count) for c in log_counts]

            # norm_values = [-np.exp(-c/2.0)+1 for c in self.model.fusion_counts_]
            # return -np.exp(-self.model.fusion_counts_[master_indices]/2.0)+1
            return np.exp(-self.model.fusion_counts_[master_indices]/self.uncertainty_scaler)
            
        elif self.uncertainty_heuristic == "stability":
            for idx in range(len(self.model.means_)):
                displacement = self.model.last_displacements_[idx]
                score = np.exp(-displacement / 0.05)
                norm_values.append(score)
                
            if norm_values:
                max_score = max(norm_values) if max(norm_values) > 0 else 1.0
                norm_values = [s / max_score for s in norm_values]
                
        elif self.uncertainty_heuristic == "combined":
            norm_values = self._calculate_combined_scores(self)
            
        return norm_values

    def _calculate_combined_scores(self):
        """Calculate combined stability scores based on fusion count and displacement."""
        norm_values = []
        
        for idx in range(len(self.model.means_)):
            fusion_count = self.model.fusion_counts_[idx]
            displacement = self.model.last_displacements_[idx]
            
            if fusion_count <= 1:
                score = 0.0  # Unverified
            elif displacement < self.suspicious_displacement:
                score = 0.3  # Suspicious (no movement)
            elif displacement > self.unstable_displacement:
                score = 0.5  # Unstable (large displacement)
            else:
                # Good stability: combine displacement and fusion scores
                displacement_score = np.exp(-displacement / self.displacement_score_factor)
                fusion_score = np.log1p(fusion_count) / np.log1p(10)
                score = fusion_score * displacement_score
                
            norm_values.append(score)
            
        # Normalize to 0-1 range
        if norm_values:
            max_score = max(norm_values) if max(norm_values) > 0 else 1.0
            norm_values = [s / max_score for s in norm_values]
            
        return norm_values

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
        self.tolerance = rospy.get_param("~tolerance", 1e-3)
        self.reg_covar = rospy.get_param("~reg_covar", 1e-6)
        self.max_iter = rospy.get_param("~max_iter", 100)
        self.kl_div_match_thresh = rospy.get_param("~kl_div_match_thresh", 5.0)
        self.l_thres = 0.05
        
        # Visualization parameters
        self.enable_visualization = rospy.get_param("~enable_visualization", True)
        self.visualization_scale = rospy.get_param("~visualization_scale", 2.0)
        self.color_by = rospy.get_param("~color_by", "intensity")  # Options: intensity, confidence, stability, combined
        self.add_metric_text = rospy.get_param("~add_metric_text", True)
        self.suspicious_displacement = rospy.get_param("~suspicious_displacement", 0.01)
        self.unstable_displacement = rospy.get_param("~unstable_displacement", 0.2)
        self.displacement_score_factor = rospy.get_param("~displacement_score_factor", 0.05)

        self.uncertainty_heuristic = rospy.get_param("~uncertainty_heuristic", "confidence")  # Options: intensity, confidence, stability, combined
        self.uncertainty_scaler = rospy.get_param("~uncertainty_scaler", 2.0)
        
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
        self.master_gmm = MasterGMM(self.uncertainty_heuristic, self.uncertainty_scaler)
        # self.learner = GPUFit(bandwidth=self.bandwidth, tolerance=1e-2, reg_covar=1e-6, max_iter=100)
        self.learner = GPUFit(bandwidth=self.bandwidth, tolerance=self.tolerance, reg_covar=self.reg_covar, max_iter=self.max_iter)
        # self.learner = GPUFit(bandwidth=self.bandwidth)
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

            # rospy.loginfo("===============================================")
            # rospy.loginfo(f"CONFIG: {self.learner.tol}, {self.learner.reg_covar}, {self.learner.max_iter}")
            # rospy.loginfo("===============================================")


            pcld = self.preprocess_point_cloud(msg)

            # Fit SOGMM model
            gmm_start_time = time.time()

            # 1. Generate a local GMM from the current point cloud
            local_model_gpu = GPUContainerf4()
            self.learner.fit(self.extract_ms_data(pcld), pcld, local_model_gpu)
            local_model_cpu = CPUContainerf4(local_model_gpu.n_components_)
            local_model_gpu.to_host(local_model_cpu)

            # cov_reshaped = local_model_cpu.covariances_.reshape(local_model_cpu.covariances_.shape[0],4,4)
            # cov_3_3 = cov_reshaped[:,0:3, 0:3]
            # mean_cov = cov_3_3.mean(axis=0)

            # rospy.loginfo("===============================================")
            # # rospy.loginfo(f"COVARIANCE: {mean_cov}")
            # rospy.loginfo(f"COVARIANCE: {local_model_cpu.covariances_}")
            # rospy.loginfo("===============================================")

            gira_timestamp = time.time()
            gira_time = gira_timestamp - gmm_start_time

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

            proc_timestamp = time.time()
            processing_time = proc_timestamp - gira_timestamp

            # Visualize results - only if we have a model
            viz_start_time = time.time()
            if self.enable_visualization and self.master_gmm.model.n_components_ > 0:
                self.visualize_gmm(self.master_gmm, self.target_frame, msg.header.stamp)
            viz_time = time.time() - viz_start_time

            full_processing_time = time.time() - start_time
            n_components = self.master_gmm.model.n_components_
            rospy.loginfo(
                f"Processed point cloud with {n_components} components in {full_processing_time:.3f}s (GIRA: {gira_time:.3f}s, Processing: {processing_time:.3f}s, Viz: {viz_time:.3f}s)"
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
        if master_gmm.model.n_components_ == 0:
            return

        marker_array = MarkerArray()
        # norm_values = self._calculate_normalization_values(master_gmm)

        # Create markers for each Gaussian component
        for i in range(len(master_gmm.model.means_)):
            # Create sphere marker
            sphere_marker = self._create_sphere_marker(master_gmm, i, frame_id, timestamp)
            if sphere_marker:
                marker_array.markers.append(sphere_marker)

            # Create text marker if enabled
            if self.add_metric_text:
                text_marker = self._create_text_marker(master_gmm, i, frame_id, timestamp)
                marker_array.markers.append(text_marker)

        # Clear old markers if component count decreased
        self._add_clear_markers(marker_array, len(master_gmm.model.means_), frame_id, timestamp)

        # Publish and update state
        self.marker_pub.publish(marker_array)
        self.last_marker_count = len(master_gmm.model.means_)
        rospy.logdebug(f"Published {len(master_gmm.model.means_)} ellipsoid markers with text")

    def _calculate_normalization_values(self, master_gmm):
        """Calculate normalized values for color mapping based on selected mode."""
        norm_values = []
        
        if self.color_by == "confidence":
            # log_counts = [np.log1p(c) for c in master_gmm.model.fusion_counts_]
            # max_log_count = max(log_counts) if log_counts else 1.0
            # norm_values = [c / max(1.0, max_log_count) for c in log_counts]

            norm_values = [-np.exp(-c/2.0)+1 for c in master_gmm.model.fusion_counts_]
            
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
                displacement_score = np.exp(-displacement / self.displacement_score_factor)
                fusion_score = np.log1p(fusion_count) / np.log1p(10)
                score = fusion_score * displacement_score
                
            norm_values.append(score)
            
        # Normalize to 0-1 range
        if norm_values:
            max_score = max(norm_values) if max(norm_values) > 0 else 1.0
            norm_values = [s / max_score for s in norm_values]
            
        return norm_values

    def _create_sphere_marker(self, master_gmm, i, frame_id, timestamp):
        """Create a sphere marker for a Gaussian component."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = timestamp
        marker.ns = "sogmm_ellipsoids"
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(2.0)

        # Set position
        mean = master_gmm.model.means_[i, :3]
        marker.pose.position.x = float(mean[0])
        marker.pose.position.y = float(mean[1])
        marker.pose.position.z = float(mean[2])

        # Set scale and orientation based on covariance
        if not self._set_marker_geometry(marker, master_gmm.model.covariances_[i]):
            return None

        # Set color based on visualization mode
        self._set_marker_color(marker, master_gmm, i)

        return marker

    def _set_marker_geometry(self, marker, covariance_flat):
        """Set marker scale and orientation based on covariance matrix."""
        try:

             # Extract 3D spatial covariance
            cov_4x4 = covariance_flat.reshape(4, 4)
            cov_3d = cov_4x4[:3, :3]

            # Compute eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(cov_3d)

            # eigh returns them in ascending order. We want descending for convention.
            # sorted_indices = np.argsort(eigenvals)[::-1]
            # eigenvals = eigenvals[sorted_indices]
            # eigenvecs = eigenvecs[:, sorted_indices]

            # Ensure a right-handed coordinate system (important for rotation matrix)
            # The third eigenvector is the cross product of the first two.
            eigenvecs[:, 2] = np.cross(eigenvecs[:, 0], eigenvecs[:, 1])
            
            # Ensure eigenvalues are non-negative for sqrt
            eigenvals = np.maximum(eigenvals, 1e-9)

            # Set scale based on eigenvalues (now sorted largest to smallest)
            scale_factor = self.visualization_scale
            marker.scale.x = 2 * scale_factor * np.sqrt(eigenvals[0]) # Largest variance
            marker.scale.y = 2 * scale_factor * np.sqrt(eigenvals[1]) # Middle variance
            marker.scale.z = 2 * scale_factor * np.sqrt(eigenvals[2]) # Smallest variance

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

    def _set_marker_color(self, marker, master_gmm, i):
        """Set marker color based on visualization mode."""
        if self.color_by == "confidence":
            color = cm.plasma(-master_gmm.model.uncertainty_[i]+1.)
            marker.color.r, marker.color.g, marker.color.b = color[0], color[1], color[2]
            marker.color.a = 0.8
            
        elif self.color_by == "stability":
            color = cm.RdYlGn(master_gmm.model.last_displacements_[i])
            marker.color.r, marker.color.g, marker.color.b = color[0], color[1], color[2]
            marker.color.a = 0.8
            
        elif self.color_by == "combined":
            self._set_combined_color(marker, master_gmm, i, master_gmm.model.uncertainty_)
            
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
            marker.color.r, marker.color.g, marker.color.b = 0.0, float(green_intensity), 0.0
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
        text_marker.lifetime = rospy.Duration(2.0)

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
        if self.color_by == "confidence":
            # return f"C:{master_gmm.model.fusion_counts_[i]}"
            return f"C:{master_gmm.model.uncertainty_[i]}"
        elif self.color_by == "stability":
            return f"D:{master_gmm.model.last_displacements_[i]:.3f}"
        elif self.color_by == "combined":
            count = master_gmm.model.fusion_counts_[i]
            disp = master_gmm.model.last_displacements_[i]
            return f"C:{count} D:{disp:.3f}"
        else:
            return f"{master_gmm.model.means_[i, 3]:.2f}"

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