import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def cluster_signals(voxel_info, voxel_signals, N_clusters=50, rho_s=1.0, rho_p=1.0, max_iter=100, tol=1e-4):
    """
    Perform clustering based on position, reflected energy, and time-domain signal correlation.
    Plot original signal positions (including excluded signals) and clustered positions.
    
    Parameters:
    - voxel_info: List of voxel positions and signal power in the format [(x, y, z, p), ...].
    - voxel_signals: List of corresponding signals in the format (num_voxels, num_time_frames).
    - N_clusters: Number of clusters.
    - rho_s: Scaling factor for signal motion difference.
    - rho_p: Scaling factor for positional difference.
    - max_iter: Maximum number of iterations for the EM algorithm.
    - tol: Tolerance for convergence.

    Returns:
    - merged_signals: Array of merged signals for each cluster.
    - centroids: Final centroids for signal motion.
    - position_centroids: Final centroids for position.
    - clustered_signals: List of signals for each cluster, where each entry contains a dictionary 
                        with 'positions' and 'signals' for that cluster.
    """
    
    # Extract signal positions (x, y, z) and power (p) from voxel_info
    positions = np.array([[x, y, z] for (x, y, z, p) in voxel_info])
    powers = np.array([p for (x, y, z, p) in voxel_info])
    
    num_voxels, num_frames = voxel_signals.shape
    
    # Initialize centroids for both signal and position
    centroids = np.random.randn(N_clusters, num_frames)  # Randomly initialize signal centroids
    position_centroids = np.random.randn(N_clusters, 3)  # Randomly initialize position centroids
    
    # Initialize cluster assignments
    cluster_assignments = -1 * np.ones(num_voxels, dtype=int)  # -1 means not assigned to any cluster
    
    for iteration in range(max_iter):
        # E-Step: Assign voxels to clusters based on the objective function J
        for i in range(num_voxels):
            min_distance = np.inf
            for k in range(N_clusters):
                # Compute signal motion difference
                signal_diff = np.linalg.norm(voxel_signals[i] - centroids[k]) ** 2
                
                # Compute position difference
                position_diff = np.linalg.norm(positions[i] - position_centroids[k]) ** 2
                
                # Objective function J
                J = rho_s * signal_diff + rho_p * position_diff
                
                # Check if this is the closest cluster
                if J < min_distance:
                    min_distance = J
                    cluster_assignments[i] = k  # Assign voxel to this cluster
        
        # M-Step: Update centroids based on the current cluster assignments
        new_centroids = np.zeros_like(centroids)
        new_position_centroids = np.zeros_like(position_centroids)
        for k in range(N_clusters):
            cluster_indices = np.where(cluster_assignments == k)[0]
            if len(cluster_indices) > 0:
                # Update signal centroids based on the weighted sum of signals
                new_centroids[k] = np.sum(voxel_signals[cluster_indices] * powers[cluster_indices, None], axis=0) / np.sum(powers[cluster_indices])
                
                # Update position centroids based on the weighted sum of positions
                new_position_centroids[k] = np.sum(positions[cluster_indices] * powers[cluster_indices, None], axis=0) / np.sum(powers[cluster_indices])
        
        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol and np.linalg.norm(new_position_centroids - position_centroids) < tol:
            break
        
        # Update centroids for next iteration
        centroids = new_centroids
        position_centroids = new_position_centroids

    # Plotting original and clustered voxel positions
    fig = plt.figure(figsize=(14, 7))

    # Subplot 1: Plot all voxel positions (including excluded signals) in 3D
    ax1 = fig.add_subplot(121, projection='3d')
    xs = positions[:, 0]
    ys = positions[:, 1]
    zs = positions[:, 2]
    
    # Plot excluded voxels (not assigned to any cluster)
    excluded_indices = np.where(cluster_assignments == -1)[0]
    ax1.scatter(positions[excluded_indices, 0], positions[excluded_indices, 1], positions[excluded_indices, 2], 
                c='gray', marker='o', alpha=0.3, label='Excluded Voxels')
    
    # Plot original voxel positions that are assigned to clusters
    included_indices = np.where(cluster_assignments != -1)[0]
    ax1.scatter(positions[included_indices, 0], positions[included_indices, 1], positions[included_indices, 2], 
                c='b', marker='o', alpha=0.7, label='Included Voxels')
    
    ax1.set_title('Original Voxel Positions (Included and Excluded)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()

    # Subplot 2: Plot clustered voxel positions with centroids in 3D
    ax2 = fig.add_subplot(122, projection='3d')

    # Plot each cluster with a different color
    colors = plt.cm.get_cmap('tab10', N_clusters)  # Use a colormap to differentiate clusters
    for k in range(N_clusters):
        cluster_indices = np.where(cluster_assignments == k)[0]
        ax2.scatter(positions[cluster_indices, 0], positions[cluster_indices, 1], positions[cluster_indices, 2], 
                    color=colors(k), s=60, alpha=0.7, label=f'Cluster {k}')
        
        # Plot centroids for each cluster
        ax2.scatter(position_centroids[k, 0], position_centroids[k, 1], position_centroids[k, 2], 
                    c='r', marker='x', s=100, label=f'Centroid {k}')

    ax2.set_title('Clustered Voxel Positions and Centroids')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()

    # Prepare to store signals and locations for each cluster
    clustered_signals = []
    
    # Merge signals within each cluster
    merged_signals = np.zeros((N_clusters, num_frames))
    for k in range(N_clusters):
        cluster_indices = np.where(cluster_assignments == k)[0]
        if len(cluster_indices) > 0:
            merged_signals[k] = np.mean(voxel_signals[cluster_indices], axis=0)
            
            # Store the signals and locations of the voxels in the current cluster
            clustered_signals.append({
                'positions': positions[cluster_indices],
                'signals': voxel_signals[cluster_indices]
            })
    
    # Show the plots
    plt.tight_layout()
    plt.show()
    
    return merged_signals, centroids, position_centroids, clustered_signals
