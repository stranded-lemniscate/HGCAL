import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging

# Configure logging for debug output
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def flatten_nested_list(nested_list):
    """Flatten a potentially nested list structure"""
    # Handle scalar values directly
    if not isinstance(nested_list, (list, tuple, np.ndarray)):
        return [nested_list]
    result = []
    for item in nested_list:
        if isinstance(item, (list, tuple, np.ndarray)):
            result.extend(flatten_nested_list(item))
        else:
            result.append(item)
    return result

class LayerCluster:
    """Represents layer clusters with 3D positions and energies (can handle nested arrays for multiple events)"""
    def __init__(self, x: Union[float, List], y: Union[float, List], z: Union[float, List], 
                 energy: Union[float, List], cluster_type: Union[str, List] = "default", 
                 cluster_n_hits: Union[int, List, np.ndarray] = 1):
        # Flatten all inputs to handle nested structures
        self.x_values = flatten_nested_list([x]) 
        self.y_values = flatten_nested_list([y]) 
        self.z_values = flatten_nested_list([z]) 
        self.energy_values = flatten_nested_list([energy]) 
        
        # Handle cluster types
        if isinstance(cluster_type, (str, int, float, np.integer, np.floating)):
            self.cluster_types = [cluster_type] * len(self.x_values)
        else:
            self.cluster_types = flatten_nested_list(cluster_type)
        # Pad with default if not enough types provided
        while len(self.cluster_types) < len(self.x_values):
            self.cluster_types.append("default")
        
        # Handle cluster_n_hits
        if isinstance(cluster_n_hits, (int, float, np.integer, np.floating)):
            self.cluster_n_hits_values = [int(cluster_n_hits)] * len(self.x_values)
        else:
            self.cluster_n_hits_values = [int(x) for x in flatten_nested_list(cluster_n_hits)]
        # Pad with 1 if not enough values provided
        while len(self.cluster_n_hits_values) < len(self.x_values):
            self.cluster_n_hits_values.append(1)
        
        # Validate that all arrays have the same length
        lengths = [len(self.x_values), len(self.y_values), len(self.z_values), 
                   len(self.energy_values), len(self.cluster_n_hits_values)]
        if not all(l == lengths[0] for l in lengths):
            raise ValueError(f"All input arrays must have the same length. Got lengths: {lengths}")
        
        self.n_clusters = len(self.x_values)
        
    def get_cluster(self, index: int):
        """Get individual cluster data by index"""
        if index >= self.n_clusters:
            raise IndexError(f"Index {index} out of range for {self.n_clusters} clusters")
        return {
            'x': self.x_values[index],
            'y': self.y_values[index], 
            'z': self.z_values[index],
            'energy': self.energy_values[index],
            'cluster_type': self.cluster_types[index],
            'cluster_n_hits': self.cluster_n_hits_values[index]
        }

    @property
    def x(self):
        """For backward compatibility - returns first x value"""
        return self.x_values[0] if self.x_values else 0.0

    @property
    def y(self):
        """For backward compatibility - returns first y value"""
        return self.y_values[0] if self.y_values else 0.0

    @property
    def z(self):
        """For backward compatibility - returns first z value"""
        return self.z_values[0] if self.z_values else 0.0

    @property
    def energy(self):
        """For backward compatibility - returns first energy value"""
        return self.energy_values[0] if self.energy_values else 0.0

    @property
    def cluster_type(self):
        """For backward compatibility - returns first cluster type"""
        return self.cluster_types[0] if self.cluster_types else "default"

    @property
    def cluster_n_hits(self):
        """For backward compatibility - returns first cluster_n_hits value"""
        return self.cluster_n_hits_values[0] if self.cluster_n_hits_values else 1

class Vector3D:
    """Simple 3D vector class"""
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

class Trackster:
    """Represents a trackster with vertices, energy, and PCA properties"""
    def __init__(self):
        self._vertices = []
        self._vertex_multiplicities = []
        self._NCLusters_for_PCA = 0
        self._raw_energy = 0.0
        self._raw_em_energy = 0.0
        self._raw_pt = 0.0
        self._raw_em_pt = 0.0
        self._barycenter = Vector3D()
        self._eigenvalues = np.zeros(3)
        self._eigenvectors = np.zeros((3, 3))
        self._sigmas = np.zeros(3)
        self._sigmas_eigen = np.zeros(3)
        
    def vertices(self, i: Optional[int] = None):
        if i is None:
            return self._vertices
        return self._vertices[i]
    
    def vertex_multiplicity(self, i: int) -> float:
        return self._vertex_multiplicities[i] if i < len(self._vertex_multiplicities) else 1.0
    
    def set_vertex_multiplicity(self, multiplicities: List[float]):
        """Set vertex multiplicities"""
        self._vertex_multiplicities = multiplicities
        
    def compute_vertex_multiplicity(self, n_vertices: int, default_value: float = 1.0) -> List[float]:
        """Compute vertex multiplicities - in your case always 1.0"""
        return [default_value] * n_vertices
        
    def set_raw_energy(self, energy: float):
        self._raw_energy = energy
        
    def add_to_raw_energy(self, energy: float):
        self._raw_energy += energy
        
    def set_raw_em_energy(self, energy: float):
        self._raw_em_energy = energy
        
    def add_to_raw_em_energy(self, energy: float):
        self._raw_em_energy += energy
        
    def set_raw_pt(self, pt: float):
        self._raw_pt = pt
        
    def set_raw_em_pt(self, pt: float):
        self._raw_em_pt = pt

    def set_NCLusters_for_PCA(self, NCluster: int):
        self._NCLusters_for_PCA = NCluster
        
    def raw_energy(self) -> float:
        return self._raw_energy
        
    def set_barycenter(self, barycenter: Vector3D):
        self._barycenter = barycenter
        
    def fill_pca_variables(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray, 
                           sigmas: np.ndarray, sigmas_eigen: np.ndarray):
        # Sort eigenvalues and eigenvectors in ascending order
        idx = np.argsort(eigenvalues)
        self._eigenvalues = eigenvalues[idx]
        self._eigenvectors = eigenvectors[:, idx]
        self._sigmas = sigmas
        self._sigmas_eigen = sigmas_eigen

def get_layer_from_lc(cluster_data: dict, rh_tools) -> int:
    """Extract layer information from layer cluster (placeholder implementation)"""
    # This would contain actual layer extraction logic
    return int(abs(cluster_data['z']) / 10)  # Simplified layer calculation

def sort_by_layer(trackster: Trackster, layer_clusters: List[LayerCluster], rh_tools) -> Dict[int, List[int]]:
    """Sort trackster vertices by layer"""
    vertices_by_layer = {}
    for i, vertex_idx in enumerate(trackster.vertices()):
        cluster_data = layer_clusters[vertex_idx].get_cluster(0)
        layer = get_layer_from_lc(cluster_data, rh_tools)
        if layer not in vertices_by_layer:
            vertices_by_layer[layer] = []
        vertices_by_layer[layer].append(i)
    return vertices_by_layer

def should_mask_cluster(cluster_data: dict, masked_cluster_types: List[str] = None) -> bool:
    """
    Determine if a cluster should be masked based on cluster_n_hits and cluster_type.
    
    Masking rule: Mask LayerClusters with cluster_n_hits == 1 for all cluster_types 
    EXCEPT when cluster_type == 6.
    
    Args:
        cluster_data: Dictionary containing cluster information
        masked_cluster_types: Legacy parameter for backward compatibility
    
    Returns:
        bool: True if cluster should be masked, False otherwise
    """
    cluster_type = cluster_data.get('cluster_type', 'default')
    cluster_n_hits = cluster_data.get('cluster_n_hits', 1)

    # return False

    # Mask clusters with n_hits == 1, EXCEPT when cluster_type == 6
    if cluster_n_hits == 1:
        if cluster_type == 6: 
            return False  # Don't mask cluster_type 6 even if n_hits == 1
        else:
            return True   # Mask all other cluster types when n_hits == 1
        
    # Also apply legacy masking based on masked_cluster_types if provided
    if masked_cluster_types and cluster_data.get('cluster_type') in masked_cluster_types:
        return True
        
    return False

def assign_pca_to_tracksters(tracksters: List[Trackster],
                            layer_clusters: List[LayerCluster],
                            z_limit_em: float,
                            rh_tools=None,
                            energy_weight: bool = True,
                            clean: bool = False,
                            min_layer: int = 10,
                            max_layer: int = 10,
                            masked_cluster_types: List[str] = None) -> None:

    if masked_cluster_types is None:
        masked_cluster_types = []
        
    # logger.debug("------- Eigen -------")
    
    for trackster in tracksters:
        # logger.debug(f"start testing trackster with size: {len(trackster.vertices())}")
        
        # Initialize vectors
        barycenter = np.zeros(3)
        filtered_barycenter = np.zeros(3)
        
        # Initialize trackster with default values
        trackster.set_raw_energy(0.0)
        trackster.set_raw_em_energy(0.0)
        trackster.set_raw_pt(0.0)
        trackster.set_raw_em_pt(0.0)
        trackster.set_NCLusters_for_PCA(0)
        
        # Set vertex multiplicities (always 1.0 in your case)
        N = len(trackster.vertices())
        trackster.set_vertex_multiplicity(trackster.compute_vertex_multiplicity(N))
                
        if N == 0:
            continue
            
        weight = 1.0 / N
        weights2_sum = 0.0
        layer_cluster_energies = []

        # Helper function to check if a cluster should be masked
        def is_cluster_masked(cluster_idx: int, sub_cluster_idx: int = 0) -> bool:
            cluster_data = layer_clusters[cluster_idx].get_cluster(sub_cluster_idx)
            return should_mask_cluster(cluster_data, masked_cluster_types)
            
        def process_cluster(i):
            # print('vertex_idx ' , vertex_idx)
            layer_cluster = layer_clusters[i]
            
            if layer_cluster.n_clusters == 1:
                cluster_data = layer_cluster.get_cluster(0)
                energy = cluster_data['energy']
                pos = np.array([cluster_data['x'], cluster_data['y'], cluster_data['z']])
                return energy, pos
            else:
                # Multiple clusters in single LayerCluster
                total_energy = sum(layer_cluster.energy_values)
                if energy_weight:
                    weighted_pos = np.zeros(3)
                    total_weight = 0.0
                    for j in range(layer_cluster.n_clusters):
                        cluster_data = layer_cluster.get_cluster(j)
                        w = cluster_data['energy']
                        weighted_pos += w * np.array([cluster_data['x'], cluster_data['y'], cluster_data['z']])
                        total_weight += w
                    if total_weight > 0:
                        weighted_pos /= total_weight
                    else:
                        # Simple average
                        weighted_pos = np.array([
                            np.mean(layer_cluster.x_values),
                            np.mean(layer_cluster.y_values), 
                            np.mean(layer_cluster.z_values)
                        ])
                    return total_energy, weighted_pos
                else:
                    # Simple average
                    avg_pos = np.array([
                        np.mean(layer_cluster.x_values),
                        np.mean(layer_cluster.y_values), 
                        np.mean(layer_cluster.z_values)
                    ])
                    return total_energy, avg_pos

        # Calculate raw energies and barycenter (INCLUDE all clusters for energy)
        for i in range(N):
            # vertex_idx = trackster.vertices(i)
            fraction = 1.0 / trackster.vertex_multiplicity(i)
            
            cluster_energy, cluster_pos = process_cluster(i)
            
            trackster.add_to_raw_energy(cluster_energy * fraction)
            if abs(cluster_pos[2]) <= z_limit_em:  # z-coordinate check
                trackster.add_to_raw_em_energy(cluster_energy * fraction)
            
            # Compute weighted barycenter (INCLUDE all clusters for barycenter)
            if energy_weight:
                weight = cluster_energy * fraction
                point = weight * cluster_pos
                barycenter += point
            
            layer_cluster_energies.append(cluster_energy)
            
        raw_energy = trackster.raw_energy()
        inv_raw_energy = 1.0 / raw_energy if raw_energy > 0 else 0.0
        
        if energy_weight and raw_energy > 0:
            barycenter *= inv_raw_energy
        
        trackster.set_barycenter(Vector3D(barycenter[0], barycenter[1], barycenter[2]))
        
        # logger.debug(f"cleaning is: {clean}")
        
        # Filtering for cleaned PCA
        filtered_idx = []
        filtered_energy = 0.0
        inv_filtered_energy = 0.0
        
        logger.debug(f"min, max {min_layer}  {max_layer}")
        logger.debug(f"Use energy weighting: {energy_weight}")
        logger.debug(f"Trackster characteristics:")
        logger.debug(f"Size: {N}")
        logger.debug(f"Energy: {trackster.raw_energy()}")
        logger.debug(f"Means: {barycenter[0]}, {barycenter[1]}, {barycenter[2]}")
        
        # PCA computation (only for tracksters with more than 2 clusters)
        if N > 2:
            sigmas = np.zeros(3)
            sigmas_eigen = np.zeros(3)
            cov_matrix = np.zeros((3, 3))
            weights2_sum = 0.0
            
            # Filter indices based on masking criteria (including cluster_n_hits)
            indices_to_use = []
            masked_count = 0

            for i in range(N):
                # vertex_idx = trackster.vertices(i)
                # Single vertex
                layer_cluster = layer_clusters[i]
                should_mask = False
                for j in range(layer_cluster.n_clusters):
                    cluster_data = layer_cluster.get_cluster(j)
                    if should_mask_cluster(cluster_data, masked_cluster_types):
                        should_mask = True
                        break
                if not should_mask:
                    indices_to_use.append(i)
                else:
                    masked_count += 1

            # print('indices_to_use', indices_to_use)
            trackster._NCLusters_for_PCA = len(indices_to_use)
            # logger.debug(f"Masked {masked_count} clusters based on n_hits and cluster_type criteria")
            # logger.debug(f"Using {len(indices_to_use)} clusters for PCA calculation")
            
            reference_barycenter = barycenter
            reference_inv_energy = inv_raw_energy
            
            # Skip PCA if no valid clusters remain after masking
            if len(indices_to_use) <= 2:
                # logger.warning(f"Not enough unmasked clusters for PCA: {len(indices_to_use)}")
                trackster.fill_pca_variables(np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3))
                continue
            
            # Compute covariance matrix
            positions = []
            weights = []
            for i in indices_to_use:
                # vertex_idx = trackster.vertices(i)
                cluster_energy, cluster_pos = process_cluster(i)
                
                if energy_weight and raw_energy > 0:
                    weight = (cluster_energy / trackster.vertex_multiplicity(i)) * reference_inv_energy
                else:
                    weight = 1.0 / len(indices_to_use)
                positions.append(cluster_pos)
                weights.append(weight)
            
            positions = np.array(positions)
            weights = np.array(weights)
            mean = np.average(positions, axis=0, weights=weights)
            centered = positions - mean
            cov_matrix = np.cov(centered.T, aweights=weights, bias=True)
            
            # Eigen decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            # Sort in ascending order
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Fill sigmas
            sigmas = np.sqrt(np.abs(eigenvalues))
            sigmas_eigen = sigmas.copy()
            
            trackster.fill_pca_variables(eigenvalues, eigenvectors, sigmas, sigmas_eigen)
        else:
            # Not enough clusters for PCA
            trackster.fill_pca_variables(np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3))