#!/usr/bin/env python3
"""Heusden snapshot loader for spatial inference.

Input and target come from the same timestep. Input depths are masked outside
the sensor set; targets cover all manholes.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import warnings
from scipy.spatial import distance_matrix
import networkx as nx


class SpatialHeusdenLoader:
    """
    Data loader for spatial inference from sparse sensors.
    
    Creates graphs where:
    - Input: Depths at 2% of manholes (sensor locations)
    - Target: Depths at ALL manholes (including 98% without sensors)
    - Same timestep for input and target (spatial task, not temporal)
    """
    
    def __init__(self, topology_dir: str, data_dir: str, 
                 sensor_strategy: str = 'random',
                 sensor_coverage: float = 0.02,
                 temporal_window: int = 20,
                 use_temporal_aggregation: bool = True,
                 verbose: bool = True,
                 relation_naming: str = "behavior"):
        """
        Args:
            topology_dir: Path to topology files
            data_dir: Path to depnod data files
            sensor_strategy: 'random', 'strategic', or 'graph_based'
            sensor_coverage: Fraction of nodes with sensors (default 0.02 = 2%)
            temporal_window: Number of previous timesteps to use (default 20 - 2/3 of available)
            use_temporal_aggregation: Aggregate temporal features (mean, max, recent)
            verbose: Print detailed information
        """
        self.topology_dir = Path(topology_dir)
        self.data_dir = Path(data_dir)
        self.sensor_strategy = sensor_strategy
        self.sensor_coverage = sensor_coverage
        self.temporal_window = temporal_window
        self.use_temporal_aggregation = use_temporal_aggregation
        self.verbose = verbose
        # Relation naming for heterogeneous edge types.
        #   "behavior"   -> CONVEYANCE / CONTROL_ACTIVE / CONTROL_GRAVITY (14 types)
        #   "asset_type" -> conduit / channel / weir / orifice / flap_valve / pump
        #                   (22 types), matching the v3 schema.
        # hydrognn.data_loader.ASSET_GROUP keys on asset-type names, so
        # lines_as_nodes() hoists no asset nodes at all under "behavior" naming.
        # Anything feeding HydroGNN must use "asset_type".
        if relation_naming not in ("behavior", "asset_type"):
            raise ValueError(f"relation_naming must be 'behavior' or 'asset_type', "
                             f"got {relation_naming!r}")
        self.relation_naming = relation_naming
        self.sensor_distances = None  # Will be computed after sensor placement
        
        # Import from original loader
        from data_loaders.base_loader import CompleteHeusdenLoader
        
        # Use original loader for topology (it works fine)
        # Use preprocessed data for faster loading!
        self.base_loader = CompleteHeusdenLoader(
            topology_dir=topology_dir,
            data_dir=data_dir,
            verbose=verbose,
            use_preprocessed=True
        )
        
        # Get topology info
        self.nodes = self.base_loader.nodes
        self.edges_by_type = self.base_loader.edges_by_type
        self.depnod_files = self.base_loader.depnod_files
        self.asset_to_behavior = self.base_loader.asset_to_behavior
        self.node_type_mapping = self.base_loader.node_type_mapping
        
        # Get manhole information for sensor placement
        self.manholes = self.nodes[
            self.nodes['node_type'] == 'Manhole'
        ].reset_index(drop=True)
        
        self.num_manholes = len(self.manholes)
        self.num_sensors = max(1, int(self.num_manholes * self.sensor_coverage))
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"SpatialHeusdenLoader initialized")
            print(f"{'='*70}")
            print(f"Total manholes: {self.num_manholes}")
            print(f"Sensor coverage: {self.sensor_coverage*100:.1f}% ({self.num_sensors} sensors)")
            print(f"Sensor strategy: {self.sensor_strategy}")
            print(f"{'='*70}\n")
        
        # Generate sensor placement based on strategy
        self.sensor_indices = self._generate_sensor_placement()
        
        # Compute distances from each node to nearest sensor
        self.sensor_distances = self._compute_sensor_distances()
        
    def _generate_sensor_placement(self) -> List[int]:
        """
        Generate sensor placement indices based on selected strategy.
        
        Returns:
            List of manhole indices that have sensors
        """
        if self.sensor_strategy == 'all' or self.sensor_coverage >= 1.0:
            # Use ALL manholes as sensors (100% coverage for training)
            return self._all_sensors_placement()
        elif self.sensor_strategy == 'random':
            return self._random_placement()
        elif self.sensor_strategy == 'strategic':
            return self._strategic_placement()
        elif self.sensor_strategy == 'graph_based':
            return self._graph_based_placement()
        else:
            raise ValueError(f"Unknown sensor strategy: {self.sensor_strategy}")
    
    def _all_sensors_placement(self) -> List[int]:
        """Use ALL manholes as sensors (100% coverage for training)."""
        if self.verbose:
            print(f"✓ All nodes as sensors: {self.num_manholes} sensors (100% coverage)")
            print(f"  Purpose: Training with complete network state")
        
        return list(range(self.num_manholes))
    
    def _random_placement(self) -> List[int]:
        """Random sensor placement (baseline)."""
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(
            self.num_manholes, 
            size=self.num_sensors, 
            replace=False
        )
        
        if self.verbose:
            print(f"✓ Random placement: {self.num_sensors} sensors")
        
        return sorted(indices.tolist())
    
    def _strategic_placement(self) -> List[int]:
        """
        Strategic placement: Maximize spatial coverage.
        
        Algorithm: Greedy farthest-point sampling
        - Start with random point
        - Iteratively add point that is farthest from all selected points
        - Ensures even spatial distribution
        """
        if self.verbose:
            print(f"Computing strategic sensor placement...")
        
        # Get coordinates
        coords = self.manholes[['x', 'y']].values
        
        # Normalize coordinates
        coords = (coords - coords.mean(axis=0)) / coords.std(axis=0)
        
        # Farthest point sampling
        selected = []
        
        # Start with random point
        np.random.seed(42)
        first_idx = np.random.randint(0, len(coords))
        selected.append(first_idx)
        
        # Iteratively add farthest point
        for _ in range(self.num_sensors - 1):
            # Compute distances from each point to nearest selected point
            distances_to_selected = distance_matrix(coords, coords[selected])
            min_distances = distances_to_selected.min(axis=1)
            
            # Add farthest point
            farthest_idx = min_distances.argmax()
            selected.append(farthest_idx)
        
        if self.verbose:
            # Compute average distance between sensors
            sensor_coords = coords[selected]
            sensor_distances = distance_matrix(sensor_coords, sensor_coords)
            avg_distance = sensor_distances[sensor_distances > 0].mean()
            print(f"✓ Strategic placement: {self.num_sensors} sensors")
            print(f"  Average inter-sensor distance: {avg_distance:.2f} (normalized)")
        
        return sorted(selected)
    
    def _graph_based_placement(self) -> List[int]:
        """
        Graph-based placement: Use network topology.
        
        Algorithm: Graph dominating set approximation
        - Construct graph from drainage network
        - Find dominating set (nodes where every node is within k-hops)
        - Select subset with best coverage
        
        This ensures sensors are placed at topologically important locations.
        """
        if self.verbose:
            print(f"Computing graph-based sensor placement...")
        
        # Build NetworkX graph from drainage network
        G = nx.DiGraph()
        
        # Add manhole nodes
        manhole_ids = self.manholes['node_id'].tolist()
        for idx, node_id in enumerate(manhole_ids):
            G.add_node(idx, node_id=node_id)
        
        # Add edges from drainage network
        for asset_type, edges_df in self.edges_by_type.items():
            for _, edge in edges_df.iterrows():
                us_node_id = edge.get('us_node_id') or edge.get('from_node')
                ds_node_id = edge.get('ds_node_id') or edge.get('to_node')
                
                if pd.isna(us_node_id) or pd.isna(ds_node_id):
                    continue
                
                # Check if both are manholes
                if us_node_id in manhole_ids and ds_node_id in manhole_ids:
                    us_idx = manhole_ids.index(us_node_id)
                    ds_idx = manhole_ids.index(ds_node_id)
                    G.add_edge(us_idx, ds_idx)
        
        # Convert to undirected for dominating set
        G_undirected = G.to_undirected()
        
        if self.verbose:
            print(f"  Network graph: {G_undirected.number_of_nodes()} nodes, "
                  f"{G_undirected.number_of_edges()} edges")
        
        # Compute node importance metrics
        # 1. Degree centrality (how connected)
        degree_centrality = nx.degree_centrality(G_undirected)
        
        # 2. Betweenness centrality (how many shortest paths pass through)
        # Note: Expensive for large graphs, use approximation
        if G_undirected.number_of_nodes() > 1000:
            # Sample-based approximation for large graphs (use even smaller sample)
            betweenness = nx.betweenness_centrality(G_undirected, k=50)
        else:
            betweenness = nx.betweenness_centrality(G_undirected)
        
        # 3. Closeness centrality (average distance to all nodes)
        try:
            # For very large graphs, skip closeness or use approximation
            if G_undirected.number_of_nodes() > 5000:
                # Set uniform closeness for very large graphs
                closeness = {node: 1.0 / G_undirected.number_of_nodes() for node in G_undirected.nodes()}
                if self.verbose:
                    print(f"  Warning: Using uniform closeness for large graph")
            else:
                closeness = nx.closeness_centrality(G_undirected)
        except:
            # If graph is disconnected, use per-component closeness
            closeness = {node: 0.0 for node in G_undirected.nodes()}
        
        # Combined importance score
        importance = {}
        for node in G_undirected.nodes():
            importance[node] = (
                0.4 * degree_centrality.get(node, 0) +
                0.4 * betweenness.get(node, 0) +
                0.2 * closeness.get(node, 0)
            )
        
        # Greedy selection with diversity
        selected = []
        remaining = set(G_undirected.nodes())
        
        while len(selected) < self.num_sensors and remaining:
            # For each remaining node, compute score
            scores = {}
            for node in remaining:
                # Base importance
                base_score = importance[node]
                
                # Penalty for being close to already selected sensors
                if selected:
                    min_distance = float('inf')
                    for selected_node in selected:
                        try:
                            dist = nx.shortest_path_length(G_undirected, selected_node, node)
                            min_distance = min(min_distance, dist)
                        except nx.NetworkXNoPath:
                            pass
                    
                    # Prefer nodes farther from existing sensors
                    diversity_bonus = min(min_distance / 5.0, 1.0)  # Cap at 1.0
                else:
                    diversity_bonus = 1.0
                
                scores[node] = base_score * (1 + diversity_bonus)
            
            # Select node with highest score
            best_node = max(scores, key=scores.get)
            selected.append(best_node)
            remaining.remove(best_node)
        
        if self.verbose:
            avg_importance = np.mean([importance[n] for n in selected])
            print(f"✓ Graph-based placement: {len(selected)} sensors")
            print(f"  Average node importance: {avg_importance:.3f}")
        
        return sorted(selected)
    
    def _compute_sensor_distances(self):
        """
        Compute shortest path distance from each manhole to nearest sensor.
        Uses graph topology (conduits/channels only).
        """
        import networkx as nx
        
        if self.verbose:
            print("Computing distances to nearest sensors...")
        
        # Build undirected graph of manhole connections
        G = nx.Graph()
        
        # Add all manholes as nodes
        manholes = self.nodes[self.nodes['node_type'] == 'Manhole']
        manhole_ids = list(manholes['node_id'])
        manhole_id_to_idx = {mid: idx for idx, mid in enumerate(manhole_ids)}
        
        for idx in range(len(manhole_ids)):
            G.add_node(idx)
        
        # Add edges from conduits and channels (main flow paths)
        for asset_type in ['conduit', 'channel']:
            if asset_type in self.edges_by_type:
                edges_df = self.edges_by_type[asset_type]
                for _, edge in edges_df.iterrows():
                    from_node = edge.get('us_node_id') or edge.get('from_node')
                    to_node = edge.get('ds_node_id') or edge.get('to_node')
                    
                    if from_node in manhole_id_to_idx and to_node in manhole_id_to_idx:
                        i = manhole_id_to_idx[from_node]
                        j = manhole_id_to_idx[to_node]
                        G.add_edge(i, j)
        
        # Compute shortest distance from each node to nearest sensor
        distances = np.zeros(self.num_manholes)
        
        for i in range(self.num_manholes):
            if i in self.sensor_indices:
                distances[i] = 0  # This is a sensor node
            else:
                # Find shortest path to any sensor
                min_dist = float('inf')
                for sensor_idx in self.sensor_indices:
                    try:
                        dist = nx.shortest_path_length(G, i, sensor_idx)
                        min_dist = min(min_dist, dist)
                    except nx.NetworkXNoPath:
                        pass  # No path exists
                
                # Cap at max distance 20 for numerical stability
                distances[i] = min(min_dist, 20.0) if min_dist < float('inf') else 20.0
        
        if self.verbose:
            print(f"✓ Sensor distances computed:")
            print(f"  Min: {distances.min():.1f}, Max: {distances.max():.1f}, Mean: {distances.mean():.1f}")
        
        return distances
    
    def create_heterogeneous_graph(self, 
                                   depnod_file: Path, 
                                   timestep: int = 0,
                                   sensor_indices: Optional[List[int]] = None) -> HeteroData:
        """
        Create heterogeneous graph for SPATIAL INFERENCE with temporal context.
        
        Args:
            depnod_file: Path to depnod CSV file
            timestep: Timestep index (SAME for input and target)
            sensor_indices: Indices of manholes with sensors (None = use default)
            
        Returns:
            HeteroData with:
            - Input features: Current depths at sensor locations + temporal context
            - Target values: Depths at ALL locations (100%)
        """
        # Load depnod data
        depnod_df = pd.read_csv(depnod_file, encoding='latin-1', header=0, skiprows=[1])
        
        # Validate timestep
        max_timestep = len(depnod_df) - 1
        if timestep >= max_timestep:
            raise ValueError(f"Timestep {timestep} out of range. Max: {max_timestep}")
        
        # Get depths at current and previous timesteps for temporal context
        current_depnod = depnod_df.iloc[timestep]
        
        # Collect historical depths (always available - no masking on history)
        historical_depnods = []
        for t in range(1, self.temporal_window):
            if timestep >= t:
                historical_depnods.append(depnod_df.iloc[timestep - t])
            else:
                # Pad with current timestep for early timesteps
                historical_depnods.append(current_depnod)
        
        # Use default sensor placement if not provided
        if sensor_indices is None:
            sensor_indices = self.sensor_indices
        
        # Create sensor mask
        sensor_mask = np.zeros(self.num_manholes, dtype=bool)
        sensor_mask[sensor_indices] = True
        
        # Initialize heterogeneous graph
        data = HeteroData()
        
        # Store sensor mask and distances in graph for later use
        data.sensor_mask = torch.tensor(sensor_mask, dtype=torch.bool)
        data.sensor_indices = torch.tensor(sensor_indices, dtype=torch.long)
        data.sensor_distances = torch.tensor(self.sensor_distances, dtype=torch.float32)
        
        # Create node features and targets
        node_id_to_idx = {}
        manhole_idx = 0
        
        for original_type in self.nodes['node_type'].unique():
            if original_type == 'node_type':
                continue
            
            node_type = self.node_type_mapping.get(original_type, original_type.lower())
            nodes_of_type = self.nodes[self.nodes['node_type'] == original_type]
            
            if len(nodes_of_type) == 0:
                continue
            
            node_features = []
            node_targets = []
            
            for idx, (_, node) in enumerate(nodes_of_type.iterrows()):
                node_id = node['node_id']
                node_id_to_idx[f"{node_type}_{node_id}"] = idx
                
                if node_type == 'manhole':
                    # Get true depth at current timestep
                    true_depth = float(current_depnod.get(node_id, 0.0)) if node_id in current_depnod else 0.0
                    
                    # SPATIAL MASKING: Only sensors see current depth
                    if sensor_mask[manhole_idx]:
                        input_depth = true_depth  # Has sensor: Use actual depth
                    else:
                        input_depth = 0.0  # No sensor: Masked to 0
                    
                    # TEMPORAL FEATURES: Historical depths
                    # Note: Masking behavior depends on sensor_mask
                    # During training (100% sensors): All nodes have history
                    # During testing (2% sensors): Only sensors have history
                    historical_depths = []
                    for hist_depnod in historical_depnods:
                        hist_depth = float(hist_depnod.get(node_id, 0.0)) if node_id in hist_depnod else 0.0
                        # Apply masking based on sensor availability
                        if not sensor_mask[manhole_idx]:
                            hist_depth = 0.0  # No sensor = no measurements
                        historical_depths.append(hist_depth)
                    
                    # Temporal aggregation for efficient representation
                    if self.use_temporal_aggregation and len(historical_depths) > 0:
                        # Recent depth (t-1)
                        depth_recent = historical_depths[0] if len(historical_depths) > 0 else 0.0
                        # Mean depth over window
                        depth_mean = np.mean(historical_depths) if len(historical_depths) > 0 else 0.0
                        # Max depth over window (peak flooding)
                        depth_max = np.max(historical_depths) if len(historical_depths) > 0 else 0.0
                        # Depth velocity (recent change)
                        depth_velocity = historical_depths[0] - (historical_depths[1] if len(historical_depths) > 1 else historical_depths[0])
                        # Depth trend (slope over window)
                        if len(historical_depths) >= 3:
                            # Simple linear trend
                            depth_trend = (historical_depths[0] - historical_depths[-1]) / len(historical_depths)
                        else:
                            depth_trend = 0.0
                        
                        temporal_features = [depth_recent, depth_mean, depth_max, depth_velocity, depth_trend]
                    else:
                        # Fall back to raw historical depths
                        temporal_features = historical_depths[:min(5, len(historical_depths))]  # Cap at 5 timesteps
                        # Pad if needed
                        while len(temporal_features) < 5:
                            temporal_features.append(0.0)
                        depth_velocity = temporal_features[0] - temporal_features[1] if len(temporal_features) > 1 else 0.0
                    
                    # Get sensor distance for this manhole
                    sensor_distance = float(self.sensor_distances[manhole_idx]) / 20.0  # Normalize by max distance
                    
                    # Build feature vector with temporal context and sensor distance
                    # CRITICAL: Must match training feature structure exactly!
                    # Training expects: [ground_level, chamber_area, shaft_area, x, y, depth, temp1-5, sensor_flag]
                    features = [
                        float(node.get('ground_level', 0.0)),
                        float(node.get('chamber_area', 1.0)),
                        float(node.get('shaft_area', 1.0)),
                        float(node.get('x', 0.0)) / 1000.0,
                        float(node.get('y', 0.0)) / 1000.0,
                        input_depth,  # Current depth (MASKED - only sensors have real values)
                    ]
                    
                    # Add aggregated temporal features (efficient representation of history)
                    features.extend(temporal_features)  # 5 features: recent, mean, max, velocity, trend
                    
                    # CRITICAL FIX: Use sensor_flag (0.0 or 1.0) instead of sensor_distance
                    is_sensor = 1.0 if sensor_mask[manhole_idx] else 0.0
                    features.append(is_sensor)  # index 11: sensor_flag
                    
                    # Target is SAME timestep, ALL nodes
                    target = true_depth
                    
                    manhole_idx += 1
                    
                elif node_type == 'outfall':
                    features = [
                        float(node.get('ground_level', 0.0)),
                        float(node.get('x', 0.0)) / 1000.0,
                        float(node.get('y', 0.0)) / 1000.0,
                    ]
                    target = 0.0
                    
                else:  # storage
                    features = [
                        float(node.get('ground_level', 0.0)),
                        float(node.get('x', 0.0)) / 1000.0,
                        float(node.get('y', 0.0)) / 1000.0,
                        float(node.get('chamber_area', 1.0))
                    ]
                    target = 0.0
                
                node_features.append(features)
                node_targets.append(target)
            
            data[node_type].x = torch.tensor(node_features, dtype=torch.float32)
            data[node_type].y = torch.tensor(node_targets, dtype=torch.float32).unsqueeze(1)
        
        # Create edges (same as original loader)
        for asset_type, edges_df in self.edges_by_type.items():
            behavior = self.asset_to_behavior.get(asset_type, 'CONVEYANCE')
            edge_groups = {}
            
            for _, edge in edges_df.iterrows():
                us_node_id = edge.get('us_node_id') or edge.get('from_node')
                ds_node_id = edge.get('ds_node_id') or edge.get('to_node')
                
                if pd.isna(us_node_id) or pd.isna(ds_node_id):
                    continue
                
                us_type = self.base_loader._get_node_type(us_node_id)
                ds_type = self.base_loader._get_node_type(ds_node_id)
                
                if not us_type or not ds_type:
                    continue
                
                us_idx = self.base_loader._get_node_index(us_node_id, us_type, node_id_to_idx)
                ds_idx = self.base_loader._get_node_index(ds_node_id, ds_type, node_id_to_idx)
                
                if us_idx is None or ds_idx is None:
                    continue
                
                relation = behavior if self.relation_naming == "behavior" else asset_type
                edge_type = (us_type, relation, ds_type)
                
                if edge_type not in edge_groups:
                    edge_groups[edge_type] = {'indices': [], 'features': []}
                
                edge_groups[edge_type]['indices'].append([us_idx, ds_idx])
                edge_groups[edge_type]['features'].append(
                    self.base_loader._extract_edge_features(edge, asset_type, behavior)
                )
            
            for edge_type, edge_data in edge_groups.items():
                if len(edge_data['indices']) > 0:
                    data[edge_type].edge_index = torch.tensor(
                        edge_data['indices'], dtype=torch.long
                    ).t().contiguous()
                    data[edge_type].edge_attr = torch.tensor(
                        edge_data['features'], dtype=torch.float32
                    )
        
        return data
    
    def create_dataset(self,
                      max_simulations: int = 0,
                      timesteps_per_simulation: int = 30) -> List[HeteroData]:
        """
        Create dataset for spatial inference.
        
        Args:
            max_simulations: Max simulations to use (0 = all)
            timesteps_per_simulation: Number of timesteps per simulation
            
        Returns:
            List of HeteroData graphs
        """
        dataset = []
        
        num_sims = len(self.depnod_files) if max_simulations == 0 else min(max_simulations, len(self.depnod_files))
        files_to_process = self.depnod_files[:num_sims]
        
        if self.verbose:
            print(f"\nCreating spatial inference dataset...")
            print(f"  Processing {num_sims} simulations")
            print(f"  {timesteps_per_simulation} timesteps per simulation")
            print(f"  Sensor strategy: {self.sensor_strategy}")
            print(f"  Sensor coverage: {self.sensor_coverage*100:.1f}%")
        
        for sim_idx, file in enumerate(files_to_process):
            depnod_df = pd.read_csv(file, encoding='latin-1', header=0, skiprows=[1])
            num_timesteps = len(depnod_df)
            
            # Sample timesteps
            timestep_indices = np.linspace(
                0, 
                num_timesteps - 1,
                min(timesteps_per_simulation, num_timesteps),
                dtype=int
            )
            
            for timestep in timestep_indices:
                try:
                    graph = self.create_heterogeneous_graph(
                        depnod_file=file,
                        timestep=timestep,
                        sensor_indices=self.sensor_indices
                    )
                    dataset.append(graph)
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: Skipped sim {sim_idx}, timestep {timestep}: {e}")
                    continue
            
            if self.verbose and (sim_idx + 1) % 50 == 0:
                print(f"  Processed {sim_idx + 1}/{num_sims} simulations...")
        
        if self.verbose:
            print(f"\n✓ Dataset created: {len(dataset)} graphs")
            print(f"  Each graph: {self.num_sensors} sensors, {self.num_manholes} total manholes")
        
        return dataset



