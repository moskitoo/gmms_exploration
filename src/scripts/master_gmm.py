#!/usr/bin/env python3

from typing import List, Optional

import numpy as np
import rospy
from rtree import index
from sogmm_cpu import SOGMMf4Host as CPUContainerf4


class MasterGMM:
    """Maintains a global GMM by fusing local measurements with spatial indexing."""
    
    # Constants for outlier detection
    MAX_EIGENVALUE_RATIO = 10.0  # Maximum ratio between largest and middle eigenvalues
    
    def __init__(
        self,
        uncertainty_heuristic: str,
        uncertainty_scaler: float,
        enable_freeze: bool,
        fusion_weight_update: bool,
    ):
        """Initialize the master GMM with configuration parameters.
        
        Args:
            uncertainty_heuristic: Method for uncertainty calculation ('confidence', 'stability', or 'combined')
            uncertainty_scaler: Scaling factor for uncertainty computation
            enable_freeze: Whether to freeze highly-fused components
            fusion_weight_update: Whether to use fusion-count-based weight updates
        """
        self.model: Optional[CPUContainerf4] = None

        self.uncertainty_heuristic = uncertainty_heuristic
        self.uncertainty_scaler = uncertainty_scaler
        self.enable_freeze = enable_freeze

        self.fusion_weight_update = fusion_weight_update

        # R-tree for spatial indexing of GMM components
        self.init_r_tree()

    def init_r_tree(self):
        p = index.Property()
        p.dimension = 3
        self.rtree = index.Index(properties=p)

    def update(self, gmm_measurement: CPUContainerf4, match_threshold: float = 5.0) -> List[int]:
        """
        Update the master GMM with a new measurement GMM using spatial matching and fusion.

        Args:
            gmm_measurement: Incoming GMM measurement to integrate
            match_threshold: KL divergence threshold for component matching (default: 5.0)
            
        Returns:
            List of indices in master model that were updated or added
        """
        n_incoming = len(gmm_measurement.weights_)
        rospy.loginfo(f"Processing {n_incoming} incoming Gaussians")

        # Handle empty master GMM - initialize with first measurement"
        if self.model is None:
            self._initialize_from_measurement(gmm_measurement)
            return list(range(self.model.n_components_))

        candidate_indices = self.find_spatial_candidates(gmm_measurement)
        self._update_observation_counts(candidate_indices)

        n_fused, master_indices_to_fuse_with, new_components_ids = self._match_and_fuse_components(
            gmm_measurement, candidate_indices, match_threshold
        )

        updated_indices = []
        
        # Add indices of components that were fused (updated)
        if len(master_indices_to_fuse_with) > 0:
            updated_indices.extend(np.unique(master_indices_to_fuse_with).tolist())

        # Add indices of components that were added (novel)
        if len(new_components_ids) > 0:
            start_idx = self.model.n_components_
            self._add_new_components(gmm_measurement, new_components_ids)
            end_idx = self.model.n_components_
            updated_indices.extend(list(range(start_idx, end_idx)))

        self.model.normalize_weights()

        # Log results
        n_novel = len(new_components_ids)
        rospy.loginfo(f"Fused: {n_fused}, Novel: {n_novel}")
        
        return updated_indices

    def _initialize_from_measurement(self, gmm_measurement):
        """Initialize master GMM from first measurement."""

        self.model = CPUContainerf4(gmm_measurement)

        # Build spatial index
        for i in range(self.model.n_components_):
            mean = self.model.means_[i, :3]
            self.rtree.insert(i, (*mean, *mean))

    def find_spatial_candidates(self, gmm_measurement: CPUContainerf4) -> List[int]:
        """Find master components spatially overlapping with measurement GMM.
        
        Args:
            gmm_measurement: The incoming GMM to find overlaps for
            
        Returns:
            List of indices of master components that spatially overlap
        """
        # Get bounding box of incoming measurement
        min_bounds = gmm_measurement.means_[:, :3].min(axis=0)
        max_bounds = gmm_measurement.means_[:, :3].max(axis=0)

        # Query R-tree for overlapping components
        return list(self.rtree.intersection((*min_bounds, *max_bounds)))

    def _update_observation_counts(self, candidate_indices):
        """Update observation counts for components in re-observed region."""
        if candidate_indices and self.model is not None:
            self.model.observation_counts_[candidate_indices] += 1

    def _match_and_fuse_components(
        self, gmm_measurement, candidate_indices, match_threshold
    ):
        """
        Match measurement components to master components and fuse them using vectorized operations.

        Returns:
            tuple: (number_fused, list_of_fused_master_indices, list_of_unmatched_measurement_indices)
        """
        if not candidate_indices:
            return 0, [], list(range(gmm_measurement.n_components_))

        # Ensure model is initialized (should always be true when this method is called)
        assert self.model is not None, "Model must be initialized before fusion"

        # 1. Create a subset of the master model for matching (now includes frozen components)
        model_subset = self.model.submap_from_indices(candidate_indices)

        # 2. Calculate the pairwise KL divergence matrix
        kl_matrix = self._calculate_kl_matrix(
            gmm_measurement.means_,
            gmm_measurement.covariances_,
            model_subset.means_,
            model_subset.covariances_,
        )

        # Check if kl_matrix is valid for argmin operation
        if kl_matrix.shape[1] == 0:
            # No master components to match against, all incoming are novel
            return 0, [], list(range(gmm_measurement.n_components_))

        # 3. Find the best matches for each measurement component
        best_master_indices_in_subset = np.argmin(kl_matrix, axis=1)
        min_kl_values = np.min(kl_matrix, axis=1)

        # 4. Identify which measurement components have a match below the threshold
        matched_mask = min_kl_values < match_threshold
        meas_indices_matched = np.where(matched_mask)[0]
        unmatched_measurement_indices = np.where(~matched_mask)[0].tolist()

        if len(meas_indices_matched) == 0:
            return 0, [], unmatched_measurement_indices

        # 5. For the matched components, separate them based on whether the master component is frozen
        subset_indices_matched_with = best_master_indices_in_subset[matched_mask]

        if self.enable_freeze:
            is_frozen_mask = model_subset.freeze_[subset_indices_matched_with] == 1.0

            # Indices for fusion (master component is NOT frozen)
            meas_indices_to_fuse = meas_indices_matched[~is_frozen_mask]
            subset_indices_to_fuse_with = subset_indices_matched_with[~is_frozen_mask]
        else:
            meas_indices_to_fuse = meas_indices_matched
            subset_indices_to_fuse_with = subset_indices_matched_with

        # New Gaussians that matched a frozen component are simply discarded (not fused, not added as novel)

        if len(meas_indices_to_fuse) == 0:
            return 0, [], unmatched_measurement_indices

        # 6. Get the corresponding master components to fuse with
        # Map subset indices back to original master model indices
        master_indices_to_fuse_with = np.array(candidate_indices)[
            subset_indices_to_fuse_with
        ]

        # 7. Perform vectorized fusion for all matched pairs with non-frozen components
        self._fuse_components_vectorized(
            master_indices_to_fuse_with, gmm_measurement, meas_indices_to_fuse
        )

        # The number of fused components is the number of unique master components that were updated
        n_fused = len(np.unique(master_indices_to_fuse_with))

        return n_fused, master_indices_to_fuse_with, unmatched_measurement_indices

    def _calculate_kl_matrix(self, means1, covs1, means2, covs2, symmetric=True):
        """
        Calculate pairwise KL divergences between two sets of Gaussians.

        Args:
            means1: Means of first set (N, D)
            covs1: Covariances of first set (N, D*D) 
            means2: Means of second set (M, D)
            covs2: Covariances of second set (M, D*D)
            symmetric: Whether to compute symmetric KL divergence

        Returns:
            (N, M) matrix of KL divergences
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
            m1_exp = means1[:, np.newaxis, :]  # (N, 1, D)
            c1_exp = covs1_mat[:, np.newaxis, :, :]  # (N, 1, D, D)
            m2_exp = means2[np.newaxis, :, :]  # (1, M, D)
            inv_c2_exp = inv_covs2[np.newaxis, :, :, :]  # (1, M, D, D)

            # Vectorized computation of KL divergence terms
            # Term 1: Trace term
            trace_term = np.einsum("...ij,...ji->...", inv_c2_exp, c1_exp)  # (N, M)

            # Term 2: Mahalanobis distance term
            mean_diff = m2_exp - m1_exp  # (N, M, D)
            mahalanobis_term = np.einsum(
                "...i,...ij,...j->...", mean_diff, inv_c2_exp, mean_diff
            )  # (N, M)

            # Term 3: Log determinant term
            log_det_term = (
                log_det_covs2[np.newaxis, :] - log_det_covs1[:, np.newaxis]
            )  # (N, M)

            # KL divergence D(1 || 2)
            kl_12 = 0.5 * (trace_term + mahalanobis_term - d + log_det_term)

            if not symmetric:
                return kl_12

            # For symmetric KL, calculate D(2 || 1)
            inv_covs1 = np.linalg.inv(covs1_mat)
            inv_c1_exp = inv_covs1[:, np.newaxis, :, :]  # (N, 1, D, D)
            c2_exp = covs2_mat[np.newaxis, :, :, :]  # (1, M, D, D)

            trace_term_21 = np.einsum("...ij,...ji->...", inv_c1_exp, c2_exp)
            mahalanobis_term_21 = np.einsum(
                "...i,...ij,...j->...", -mean_diff, inv_c1_exp, -mean_diff
            )
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
        assert self.model is not None, "Model must be initialized before fusion"
        
        # --- Gather data for all components to be fused ---
        # Master components
        w_i = self.model.weights_[master_indices]
        mu_i = self.model.means_[master_indices]
        cov_i = self.model.covariances_[master_indices].reshape(-1, 4, 4)
        fusion_counts = self.model.fusion_counts_[master_indices]

        # Measurement components
        meas_comps_to_fuse = meas_gmm.submap_from_indices(meas_indices)
        w_j = meas_comps_to_fuse.weights_
        mu_j = meas_comps_to_fuse.means_
        cov_j = meas_comps_to_fuse.covariances_.reshape(-1, 4, 4)

        # --- Perform fusion calculations in a vectorized manner ---
        if self.fusion_weight_update:
            # Use fusion counts as effective weights for mean/covariance calculation
            effective_w_i = fusion_counts.astype(float)
            effective_w_j = np.ones_like(w_j, dtype=float)
            w_sum_for_fusion = effective_w_i + effective_w_j

            mu_new = (
                effective_w_i[:, np.newaxis] * mu_i
                + effective_w_j[:, np.newaxis] * mu_j
            ) / w_sum_for_fusion[:, np.newaxis]

            cov_new = (
                effective_w_i[:, np.newaxis, np.newaxis]
                * (cov_i + np.einsum("...i,...j->...ij", mu_i - mu_new, mu_i - mu_new))
                + effective_w_j[:, np.newaxis, np.newaxis]
                * (cov_j + np.einsum("...i,...j->...ij", mu_j - mu_new, mu_j - mu_new))
            ) / w_sum_for_fusion[:, np.newaxis, np.newaxis]

        else:
            w_new = w_i + w_j
            mu_new = (w_i[:, np.newaxis] * mu_i + w_j[:, np.newaxis] * mu_j) / w_new[
                :, np.newaxis
            ]

            diff_i = mu_i - mu_new
            diff_j = mu_j - mu_new

            # Use einsum for batched outer product: diff @ diff.T
            # '...i,...j->...ij' computes the outer product for each vector in the batch
            outer_i = np.einsum("...i,...j->...ij", diff_i, diff_i)
            outer_j = np.einsum("...i,...j->...ij", diff_j, diff_j)

            cov_new = (
                w_i[:, np.newaxis, np.newaxis] * (cov_i + outer_i)
                + w_j[:, np.newaxis, np.newaxis] * (cov_j + outer_j)
            ) / w_new[:, np.newaxis, np.newaxis]

        # --- Update master model state ---
        # This is tricky because multiple `meas_indices` can map to the same `master_index`.
        # We need to sum the contributions for each unique master component.
        unique_master_indices, inverse_indices = np.unique(
            master_indices, return_inverse=True
        )

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

        self.model.uncertainty_[master_indices] = self._calculate_uncertainty(
            master_indices
        )

    def _add_new_components(self, gmm_measurement, new_components_ids):
        """Add unmatched components as new master components."""

        assert self.model is not None, "Model must be initialized before adding new components"
        old_n_components = self.model.n_components_

        new_gmm = gmm_measurement.submap_from_indices(new_components_ids)

        new_gmm = self.remove_outliers(new_gmm)

        self.model.merge(new_gmm)

        # Update spatial index with new components
        for i in range(old_n_components, self.model.n_components_):
            mean = self.model.means_[i, :3]
            self.rtree.insert(i, (*mean, *mean))

    def remove_outliers(self, gmm: CPUContainerf4) -> CPUContainerf4:
        """Remove Gaussian components with extreme eigenvalue ratios (likely outliers).
        
        Args:
            gmm: Input GMM to filter
            
        Returns:
            Filtered GMM with outliers removed
        """
        valid_indices = []
        
        for idx in range(gmm.n_components_):
            cov_4x4 = gmm.covariances_[idx].reshape(4, 4)
            cov_3d = cov_4x4[:3, :3]

            eigenvals, _ = np.linalg.eigh(cov_3d)
            eigenvals = np.abs(eigenvals)

            # Keep components where largest eigenvalue < threshold * middle eigenvalue
            if eigenvals[2] < self.MAX_EIGENVALUE_RATIO * eigenvals[1]:
                valid_indices.append(idx)

        return gmm.submap_from_indices(valid_indices)

    def freeze_components(self, freeze_fusion_threshold: int) -> None:
        """Freeze components that have been fused many times.
        
        Args:
            freeze_fusion_threshold: Fusion count above which to freeze components
        """
        if self.model is None:
            return
        
        freeze_indices = self.model.fusion_counts_ > freeze_fusion_threshold
        self.model.freeze_[freeze_indices] = 1.0

    def prune_stale_components(
        self, min_observations: int = 5, max_fusion_ratio: float = 0.4
    ) -> int:
        """
        Remove stale outlier Gaussians using vectorized operations.

        A Gaussian is considered stale if it has been observed multiple times but has a low
        fusion ratio, indicating it's likely an outlier in a re-observed area.
        
        Args:
            min_observations: Minimum observations before considering for pruning
            max_fusion_ratio: Maximum allowed fusion/observation ratio before pruning
            
        Returns:
            Number of components pruned
        """
        if self.model is None or self.model.n_components_ == 0:
            return 0

        # --- Vectorized Pruning Logic ---
        # Convert tracking lists to numpy arrays for vectorized operations
        obs_counts = self.model.observation_counts_
        fus_counts = self.model.fusion_counts_

        # Calculate fusion ratio for all components, avoiding division by zero
        # Where obs_counts is 0, the ratio is irrelevant as it will be kept anyway.
        # We can safely set the ratio to 1.0 in these cases.
        fusion_ratios = np.divide(
            fus_counts,
            obs_counts,
            out=np.ones_like(fus_counts, dtype=float),
            where=obs_counts != 0,
        )

        # Determine which components to keep based on the criteria
        # Keep if: under-observed OR has a good fusion ratio
        keep_mask = (
            (obs_counts < min_observations)
            | (fusion_ratios >= max_fusion_ratio)
            | (self.model.freeze_ == 1.0)
        )

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

        rospy.loginfo(
            "========================================================================="
        )
        rospy.loginfo(
            f"Pruned {n_pruned} stale outlier Gaussians, {self.model.n_components_} remaining"
        )
        rospy.loginfo(
            "========================================================================="
        )
        return n_pruned

    def _calculate_uncertainty(self, master_indices):
        """Calculate uncertainty values for specified components based on heuristic mode.
        
        Uncertainty calculation modes:
        - confidence: Based on fusion count (higher fusion = lower uncertainty)
                     Formula: exp(-fusion_count / scaler)
        - stability: Based on displacement (higher displacement = higher uncertainty)
                    Formula: 1 - exp(-displacement / scaler)
        - combined: Combines both fusion count and displacement
        
        Returns values in range [0, 1] where 0 = certain, 1 = uncertain
        """
        assert self.model is not None, "Model must be initialized before calculating uncertainty"
        
        if self.uncertainty_heuristic == "confidence":
            return np.exp(
                -self.model.fusion_counts_[master_indices] / self.uncertainty_scaler
            )
        
        elif self.uncertainty_heuristic == "stability":
            displacements = self.model.last_displacements_[master_indices]
            # Lower displacement = lower uncertainty (more stable = more certain)
            # Small displacement → low uncertainty (≈0), large displacement → high uncertainty (≈1)
            # Formula: 1 - exp(-displacement/scale)
            # This gives: displacement=0 → uncertainty≈0, large displacement → uncertainty→1
            return 1.0 - np.exp(-displacements / self.uncertainty_scaler)
        
        elif self.uncertainty_heuristic == "combined":
            # Combine fusion count and displacement information
            fusion_counts = self.model.fusion_counts_[master_indices]
            displacements = self.model.last_displacements_[master_indices]
            
            # Confidence component (lower is better)
            confidence_uncertainty = np.exp(
                -fusion_counts / self.uncertainty_scaler
            )
            
            # Stability component (lower is better)
            stability_uncertainty = 1.0 - np.exp(-displacements / self.uncertainty_scaler)
            
            # Average the two uncertainty measures
            return (confidence_uncertainty + stability_uncertainty) / 2.0
        
        # Fallback for unknown modes
        rospy.logwarn(f"Unknown uncertainty heuristic: {self.uncertainty_heuristic}, using zeros")
        return np.zeros(len(master_indices))