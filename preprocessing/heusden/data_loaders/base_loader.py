#!/usr/bin/env python3
"""Heusden topology and depnod reader.

Loads the node table, the six link tables of the InfoWorks export and the
depnod time series, and assembles HeteroData graphs.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import warnings


class CompleteHeusdenLoader:
    """
    Complete data loader for Heusden urban drainage network.
    
    Features:
    - Loads ALL 7 topology files (nodes, conduits, channels, weirs, pumps, orifices, flap_valves)
    - Processes ALL node types (manholes, outfalls, storage, pumps, etc.)
    - Maps assets to hydraulic behaviors (CONVEYANCE, CONTROL_GRAVITY, CONTROL_ACTIVE, etc.)
    - Preserves original node IDs for depnod data matching
    - Creates proper heterogeneous graphs for HydroGNN
    
    Args:
        topology_dir: Path to directory containing topology CSV files
        data_dir: Path to directory containing depnod time series files
        verbose: Whether to print detailed loading information
    """
    
    def __init__(self, topology_dir: str, data_dir: str, verbose: bool = True, use_preprocessed: bool = True):
        self.topology_dir = Path(topology_dir)
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.use_preprocessed = use_preprocessed

        # Complete hydraulic behavior mapping based on InfoWorks ICM asset types
        self.asset_to_behavior = {
            # CONVEYANCE: Free-flowing conduits and channels (Manning's equation)
            'conduit': 'CONVEYANCE',
            'channel': 'CONVEYANCE',
            'Channl': 'CONVEYANCE',  # Alternative spelling in Heusden data

            # CONTROL_GRAVITY: Threshold-driven overflow structures (Weir/Orifice equations)
            'weir': 'CONTROL_GRAVITY',
            'WEIR': 'CONTROL_GRAVITY',
            'orifice': 'CONTROL_GRAVITY',
            'ORIFIC': 'CONTROL_GRAVITY',
            'flap_valve': 'CONTROL_GRAVITY',

            # CONTROL_ACTIVE: Energy-driven pump systems (Pump curves)
            'pump': 'CONTROL_ACTIVE',
            'SCRPMP': 'CONTROL_ACTIVE',  # Screw pump
            'FIXPMP': 'CONTROL_ACTIVE',  # Fixed pump

            # RUNOFF: Rainfall-runoff connections (Rational method)
            'subcatchment': 'RUNOFF',

            # STORAGE: Volume-depth storage (Storage curves)
            'storage': 'STORAGE',

            # OUTFALL: Boundary conditions (Tailwater effects)
            'outfall': 'OUTFALL',
        }

        # Node type mapping for heterogeneous graph
        self.node_type_mapping = {
            'Manhole': 'manhole',
            'Outfall': 'outfall',
            'Storage': 'storage',
            'Pond': 'storage',
            'Pump': 'pump',
            'Subcatchment': 'subcatchment',
        }

        # Load complete topology
        self._load_complete_topology()

        # Load depnod data information
        self._load_depnod_info(use_preprocessed=self.use_preprocessed)

        # Create node ID to index mapping for graph creation
        self.node_id_to_idx = self._create_node_id_to_idx_mapping()

        # B10 FIX: build O(1) dict for node-type + index lookups.
        # The old _get_node_type() did a pandas boolean-mask scan on every call,
        # producing O(E × N) complexity during graph construction.
        self._node_id_to_type_and_idx: dict = {}
        for original_type in self.nodes["node_type"].unique():
            if original_type == "node_type":
                continue
            mapped = self.node_type_mapping.get(original_type, original_type.lower())
            sub    = self.nodes[self.nodes["node_type"] == original_type]
            for local_idx, (_, row) in enumerate(sub.iterrows()):
                nid = row["node_id"]
                self._node_id_to_type_and_idx[nid] = (mapped, local_idx)

        if self.verbose:
            data_type = "preprocessed" if self.use_preprocessed else "raw"
            print(f"CompleteHeusdenLoader initialized successfully with {data_type} data!")
    
    def _load_complete_topology(self):
        """Load ALL 7 topology files from Heusden dataset."""
        if self.verbose:
            print("=" * 70)
            print("LOADING COMPLETE HEUSDEN TOPOLOGY")
            print("=" * 70)
        
        # 1. Load nodes (ALL types with preserved IDs)
        node_file = self.topology_dir / "1D rioleringsmodel Heusden!_node.csv"
        if not node_file.exists():
            raise FileNotFoundError(f"Node file not found: {node_file}")
        
        self.nodes = pd.read_csv(node_file, encoding='latin-1')
        
        # Remove header row if present
        if 'node_type' in self.nodes['node_type'].values:
            self.nodes = self.nodes[self.nodes['node_type'] != 'node_type']
        
        if self.verbose:
            print(f"\nLoaded {len(self.nodes)} nodes")
            print("\nNode Type Distribution:")
            for node_type, count in self.nodes['node_type'].value_counts().items():
                print(f"   {node_type:15s}: {count:5d} nodes")
        
        # 2. Load ALL edge files (complete network topology)
        edge_files = {
            'conduit': "1D rioleringsmodel Heusden!_conduit.csv",
            'channel': "1D rioleringsmodel Heusden!_channel.csv",
            'weir': "1D rioleringsmodel Heusden!_weir.csv",
            'pump': "1D rioleringsmodel Heusden!_pump.csv",
            'orifice': "1D rioleringsmodel Heusden!_orifice.csv",
            'flap_valve': "1D rioleringsmodel Heusden!_flap_valve.csv",
        }
        
        self.edges_by_type = {}
        total_edges = 0
        
        if self.verbose:
            print("\nLoading Edge Files:")
        
        for edge_type, filename in edge_files.items():
            file_path = self.topology_dir / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, encoding='latin-1')
                    # Remove header row if present
                    if edge_type in df.iloc[0].values:
                        df = df.iloc[1:]
                    self.edges_by_type[edge_type] = df
                    total_edges += len(df)
                    behavior = self.asset_to_behavior.get(edge_type, 'UNKNOWN')
                    if self.verbose:
                        print(f"   {edge_type:12s}: {len(df):5d} edges → {behavior}")
                except Exception as e:
                    if self.verbose:
                        print(f"   ⚠️  {edge_type:12s}: Failed to load ({e})")
            else:
                if self.verbose:
                    print(f"   ⚠️  {edge_type:12s}: File not found")
        
        if self.verbose:
            print(f"\nTotal Edges Loaded: {total_edges}")
            print("=" * 70)
    
    def _load_depnod_info(self, use_preprocessed: bool = True, max_files: int = None):
        """
        Load depnod file information and node ID mapping.

        Args:
            use_preprocessed: Whether to expect preprocessed files or raw files
            max_files: Maximum number of files to load (for quick testing)
        """
        if self.verbose:
            print("\nLOADING DEPNOD DATA INFORMATION")
            print("=" * 70)

        if use_preprocessed:
            # Look for preprocessed files
            self.depnod_files = sorted(self.data_dir.glob("preprocessed_*.csv"))
            if max_files:
                self.depnod_files = self.depnod_files[:max_files]

            if len(self.depnod_files) == 0:
                warnings.warn(f"No preprocessed files found in {self.data_dir}")
                self.depnod_node_ids = []
                self.node_to_depnod_idx = {}
                return

            if self.verbose:
                print(f"Found {len(self.depnod_files)} preprocessed simulation files")

            # Load first preprocessed file to get node IDs
            sample_file = self.depnod_files[0]
            try:
                sample_df = pd.read_csv(sample_file, nrows=1)
                # All columns except 'Time' and 'Seconds' are node IDs
                self.depnod_node_ids = [col for col in sample_df.columns
                                        if col not in ['Time', 'Seconds']]

                # Create mapping from node_id to depnod column index
                self.node_to_depnod_idx = {node_id: idx for idx, node_id in enumerate(self.depnod_node_ids)}

                if self.verbose:
                    print(f"Found depnod data for {len(self.depnod_node_ids)} nodes")
                    print(f"   Sample node IDs: {self.depnod_node_ids[:10]}")

            except Exception as e:
                warnings.warn(f"Error loading preprocessed file headers: {e}")
                self.depnod_node_ids = []
                self.node_to_depnod_idx = {}

        else:
            # Original behavior for raw files
            self.depnod_files = sorted(self.data_dir.glob("*depnod*.csv"))

            if len(self.depnod_files) == 0:
                warnings.warn(f"No depnod files found in {self.data_dir}")
                self.depnod_node_ids = []
                self.node_to_depnod_idx = {}
                return

            if self.verbose:
                print(f"Found {len(self.depnod_files)} depnod simulation files")

            # Load first file to get node IDs from headers
            sample_file = self.depnod_files[0]
            try:
                # Read header from first line (node IDs), then skip units line for data
                sample_df = pd.read_csv(sample_file, nrows=1, encoding='latin-1')
                # All columns except 'Time' and 'Seconds' are node IDs
                self.depnod_node_ids = [col for col in sample_df.columns
                                        if col not in ['Time', 'Seconds']]

                # Create mapping from node_id to depnod column index
                self.node_to_depnod_idx = {node_id: idx for idx, node_id in enumerate(self.depnod_node_ids)}

                if self.verbose:
                    print(f"Found depnod data for {len(self.depnod_node_ids)} nodes")
                    print(f"   Sample node IDs: {self.depnod_node_ids[:10]}")

            except Exception as e:
                warnings.warn(f"Error loading depnod file headers: {e}")
                self.depnod_node_ids = []
                self.node_to_depnod_idx = {}

        if self.verbose:
            print("=" * 70)

    def _create_node_id_to_idx_mapping(self) -> Dict[str, int]:
        """Create mapping from node_type_node_id to sequential index for graph creation."""
        node_id_to_idx = {}
        idx = 0

        for node_type in self.get_node_types():
            node_ids = self.get_node_ids_by_type(node_type)

            for node_id in node_ids:
                key = f"{node_type}_{node_id}"
                node_id_to_idx[key] = idx
                idx += 1

        return node_id_to_idx

    def get_node_types(self) -> List[str]:
        """Get list of unique node types in the network."""
        unique_types = set()
        for node_type in self.nodes['node_type'].unique():
            mapped_type = self.node_type_mapping.get(node_type, node_type.lower())
            unique_types.add(mapped_type)
        return sorted(list(unique_types))
    
    def get_edge_types(self) -> List[Tuple[str, str, str]]:
        """Get list of edge types in format (src_type, relation, dst_type)."""
        edge_types = set()

        for edge_asset_type, edges_df in self.edges_by_type.items():
            behavior = self.asset_to_behavior.get(edge_asset_type, 'CONVEYANCE')

            # Sample a few edges to determine node types
            for _, edge in edges_df.head(10).iterrows():
                us_node_id = edge.get('us_node_id') or edge.get('from_node')
                ds_node_id = edge.get('ds_node_id') or edge.get('to_node')

                if pd.isna(us_node_id) or pd.isna(ds_node_id):
                    continue

                # Get node types
                us_node_type = self._get_node_type(us_node_id)
                ds_node_type = self._get_node_type(ds_node_id)

                if us_node_type and ds_node_type:
                    edge_types.add((us_node_type, behavior, ds_node_type))

        return sorted(list(edge_types))

    def get_node_ids_by_type(self, node_type: str) -> List[str]:
        """Get list of node IDs for a specific node type."""
        if node_type not in self.get_node_types():
            return []

        # Filter nodes by type
        type_nodes = self.nodes[self.nodes['node_type'].str.lower() == node_type.lower()]
        return list(type_nodes.index)

    def get_edges_by_type(self, behavior: str) -> pd.DataFrame:
        """Get edges DataFrame for a specific behavior type."""
        return self.edges_by_type.get(behavior, pd.DataFrame())

    def get_edge_indices_dict(self) -> Dict[Tuple[str, str, str], torch.Tensor]:
        """Get edge indices dictionary for heterogeneous graph creation."""
        edge_indices = {}

        for edge_type in self.get_edge_types():
            source_type, behavior, target_type = edge_type

            # Get edges for this behavior
            edges_df = self.get_edges_by_type(behavior)

            if not edges_df.empty:
                # Create edge index tensor [2, num_edges]
                # Convert node IDs to indices
                source_indices = []
                target_indices = []

                for _, edge in edges_df.iterrows():
                    us_node_id = edge.get('us_node_id') or edge.get('from_node')
                    ds_node_id = edge.get('ds_node_id') or edge.get('to_node')

                    if pd.isna(us_node_id) or pd.isna(ds_node_id):
                        continue

                    # Get node indices
                    us_idx = self._get_node_index(us_node_id, source_type, self.node_id_to_idx)
                    ds_idx = self._get_node_index(ds_node_id, target_type, self.node_id_to_idx)

                    if us_idx is not None and ds_idx is not None:
                        source_indices.append(us_idx)
                        target_indices.append(ds_idx)

                if source_indices:
                    edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)
                    edge_indices[edge_type] = edge_index

        return edge_indices
    
    def _get_node_type(self, node_id: str) -> Optional[str]:
        """Return the mapped node type for node_id in O(1) via pre-built dict."""
        entry = self._node_id_to_type_and_idx.get(node_id)
        return entry[0] if entry is not None else None
    
    def _get_node_index(self, node_id: str, node_type: str, node_id_to_idx: Dict) -> Optional[int]:
        """Get the index of a node within its type group."""
        key = f"{node_type}_{node_id}"
        return node_id_to_idx.get(key)
    
    def _extract_edge_features(self, edge: pd.Series, asset_type: str, behavior: str) -> List[float]:
        """Extract behavior-specific edge features from edge data with proper NaN and type handling."""
        
        def safe_float(value, default):
            """Safely convert to float, handling NaN, None, and non-numeric strings."""
            if value is None or pd.isna(value):
                return default
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
        
        if behavior == 'CONVEYANCE':
            # Pipe/channel features for Manning's equation
            # Handle different asset types with different column names
            
            # Width/Diameter: try multiple column names
            width = edge.get('conduit_width')
            if pd.isna(width):
                width = edge.get('diameter')
            if pd.isna(width):
                width = edge.get('width')  # For channels
            if pd.isna(width):
                width = edge.get('shape')  # Alternative name
            width = safe_float(width, 1.0)
            
            # Length: try multiple column names
            length = edge.get('conduit_length')
            if pd.isna(length):
                length = edge.get('length')
            length = safe_float(length, 100.0)
            length = length / 100.0  # Normalized
            
            # Roughness: try multiple column names
            roughness = edge.get('bottom_roughness_Manning')
            if pd.isna(roughness):
                roughness = edge.get('roughness')
            if pd.isna(roughness):
                roughness = edge.get('mannings_n')
            roughness = safe_float(roughness, 0.015)
            
            # Gradient: try to compute or use default
            gradient = edge.get('gradient')
            if pd.isna(gradient):
                # Try computing from invert elevations
                us_invert = edge.get('us_invert')
                ds_invert = edge.get('ds_invert')
                if not pd.isna(us_invert) and not pd.isna(ds_invert) and length > 0:
                    gradient = (safe_float(us_invert, 0.0) - safe_float(ds_invert, 0.0)) / (length * 100.0)
                else:
                    gradient = 0.001
            gradient = safe_float(gradient, 0.001)
            
            return [width, length, roughness, gradient]
        
        elif behavior == 'CONTROL_GRAVITY':
            # Weir/orifice features for overflow equations
            crest = edge.get('crest')
            crest = safe_float(crest, 0.0)
            
            width = edge.get('width')
            if pd.isna(width):
                width = edge.get('crest_width')
            width = safe_float(width, 1.0)
            
            discharge_coeff = edge.get('discharge_coeff')
            if pd.isna(discharge_coeff):
                discharge_coeff = edge.get('coefficient')
            discharge_coeff = safe_float(discharge_coeff, 0.6)
            
            height = edge.get('height')
            if pd.isna(height):
                height = edge.get('gate_height')
            height = safe_float(height, 1.0)
            
            return [crest, width, discharge_coeff, height]
        
        elif behavior == 'CONTROL_ACTIVE':
            # Pump features for pump curves
            discharge = edge.get('discharge')
            if pd.isna(discharge):
                discharge = edge.get('max_flow')
            discharge = safe_float(discharge, 1.0)
            
            switch_on = edge.get('switch_on_level')
            if pd.isna(switch_on):
                switch_on = edge.get('on_level')
            switch_on = safe_float(switch_on, 0.0)
            
            switch_off = edge.get('switch_off_level')
            if pd.isna(switch_off):
                switch_off = edge.get('off_level')
            switch_off = safe_float(switch_off, 0.0)
            
            efficiency = edge.get('efficiency')
            efficiency = safe_float(efficiency, 0.8)
            
            return [discharge, switch_on, switch_off, efficiency]
        
        else:
            # Default features for unknown types
            return [1.0, 1.0, 1.0, 1.0]
    
    def create_heterogeneous_graph(self, 
                                   depnod_file: Path, 
                                   timestep: int = 0,
                                   target_timestep: int = 1) -> HeteroData:
        """
        Create a heterogeneous graph for a specific simulation and timestep.
        
        Args:
            depnod_file: Path to depnod CSV file for a specific simulation
            timestep: Input timestep index
            target_timestep: Target timestep index for prediction
            
        Returns:
            HeteroData object ready for HydroGNN training
        """
        # Load depnod data - use row 0 as headers (node IDs), skip row 1 (units)
        # This preserves node IDs as column names while skipping the units row
        depnod_df = pd.read_csv(depnod_file, encoding='latin-1', header=0, skiprows=[1])
        
        # Validate timesteps
        max_timestep = len(depnod_df) - 1
        if timestep >= max_timestep or target_timestep > max_timestep:
            raise ValueError(f"Timestep out of range. Max timestep: {max_timestep}")
        
        # Get depnod values for current and target timesteps
        current_depnod = depnod_df.iloc[timestep]
        target_depnod = depnod_df.iloc[target_timestep]
        
        # Initialize heterogeneous graph
        data = HeteroData()
        
        # Create node features and targets by type
        node_id_to_idx = {}  # For edge indexing
        
        for original_type in self.nodes['node_type'].unique():
            if original_type == 'node_type':  # Skip header
                continue
            
            # Map to model node type
            node_type = self.node_type_mapping.get(original_type, original_type.lower())
            
            # Get nodes of this type
            nodes_of_type = self.nodes[self.nodes['node_type'] == original_type]
            
            if len(nodes_of_type) == 0:
                continue
            
            node_features = []
            node_targets = []
            
            for idx, (_, node) in enumerate(nodes_of_type.iterrows()):
                node_id = node['node_id']
                
                # Store index mapping for edges
                node_id_to_idx[f"{node_type}_{node_id}"] = idx
                
                # Extract node features based on type
                if node_type == 'manhole':
                    features = [
                        float(node.get('ground_level', 0.0)),
                        float(node.get('chamber_area', 1.0)),
                        float(node.get('shaft_area', 1.0)),
                        float(node.get('x', 0.0)) / 1000.0,  # Normalized
                        float(node.get('y', 0.0)) / 1000.0,  # Normalized
                        float(current_depnod.get(node_id, 0.0)) if node_id in current_depnod else 0.0
                    ]
                    # Target is next timestep depth
                    target = float(target_depnod.get(node_id, 0.0)) if node_id in target_depnod else 0.0
                    
                elif node_type == 'outfall':
                    features = [
                        float(node.get('ground_level', 0.0)),
                        float(node.get('x', 0.0)) / 1000.0,
                        float(node.get('y', 0.0)) / 1000.0,
                    ]
                    target = 0.0  # Outfalls don't have depth targets
                    
                else:  # Other node types
                    features = [
                        float(node.get('ground_level', 0.0)),
                        float(node.get('x', 0.0)) / 1000.0,
                        float(node.get('y', 0.0)) / 1000.0,
                        float(node.get('chamber_area', 1.0))
                    ]
                    target = 0.0
                
                node_features.append(features)
                node_targets.append(target)
            
            # Add to HeteroData
            data[node_type].x = torch.tensor(node_features, dtype=torch.float32)
            data[node_type].y = torch.tensor(node_targets, dtype=torch.float32).unsqueeze(1)
        
        # Create edges with asset type as relation (model maps asset→behavior internally)
        for asset_type, edges_df in self.edges_by_type.items():
            behavior = self.asset_to_behavior.get(asset_type, 'CONVEYANCE')
            
            # Group edges by (src_type, asset_type, dst_type)
            edge_groups = {}
            
            for _, edge in edges_df.iterrows():
                us_node_id = edge.get('us_node_id') or edge.get('from_node')
                ds_node_id = edge.get('ds_node_id') or edge.get('to_node')
                
                if pd.isna(us_node_id) or pd.isna(ds_node_id):
                    continue
                
                # Get node types
                us_type = self._get_node_type(us_node_id)
                ds_type = self._get_node_type(ds_node_id)
                
                if not us_type or not ds_type:
                    continue
                
                # Get indices
                us_idx = self._get_node_index(us_node_id, us_type, node_id_to_idx)
                ds_idx = self._get_node_index(ds_node_id, ds_type, node_id_to_idx)
                
                if us_idx is None or ds_idx is None:
                    continue
                
                # Edge type tuple - use asset_type so model can find the right conv layer
                edge_type = (us_type, asset_type, ds_type)
                
                if edge_type not in edge_groups:
                    edge_groups[edge_type] = {'indices': [], 'features': []}
                
                edge_groups[edge_type]['indices'].append([us_idx, ds_idx])
                edge_groups[edge_type]['features'].append(
                    self._extract_edge_features(edge, asset_type, behavior)
                )
            
            # Add edge groups to HeteroData
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
                      max_simulations: int = 10,
                      timesteps_per_simulation: int = 10,
                      prediction_horizon: int = 1) -> List[HeteroData]:
        """
        Create a complete dataset from preprocessed depnod files.

        Args:
            max_simulations: Maximum number of simulation files to process
            timesteps_per_simulation: Number of timesteps to extract from each simulation
            prediction_horizon: Number of timesteps ahead to predict

        Returns:
            List of HeteroData graphs ready for training
        """
        dataset = []

        files_to_process = self.depnod_files[:max_simulations]

        if self.verbose:
            data_type = "preprocessed" if self.use_preprocessed else "raw"
            print(f"\nCreating dataset from {len(files_to_process)} {data_type} simulations...")

        for file_idx, depnod_file in enumerate(files_to_process):
            try:
                # Note: We don't need to load the full file here since create_heterogeneous_graph
                # will load it properly. Just validate it exists and get timestep count.
                if self.use_preprocessed:
                    depnod_df = pd.read_csv(depnod_file)
                else:
                    # Load with correct headers for raw files
                    depnod_df = pd.read_csv(depnod_file, encoding='latin-1', header=0, skiprows=[1])

                max_timestep = len(depnod_df) - prediction_horizon - 1

                # Create graphs for this simulation
                for t in range(0, min(max_timestep, timesteps_per_simulation)):
                    graph = self.create_heterogeneous_graph(
                        depnod_file,
                        timestep=t,
                        target_timestep=t + prediction_horizon
                    )
                    dataset.append(graph)

                if self.verbose and (file_idx + 1) % 5 == 0:
                    print(f"   Processed {file_idx + 1}/{len(files_to_process)} simulations...")

            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Error processing {depnod_file.name}: {e}")
                continue

        if self.verbose:
            print(f"Created dataset with {len(dataset)} graphs")

        return dataset


if __name__ == "__main__":
    # Example usage
    print("=" * 80)
    print("CompleteHeusdenLoader - Example Usage")
    print("=" * 80)
    
    # Replace with actual paths
    topology_dir = "/path/to/Detailed_heusden_topology"
    data_dir = "/path/to/run660_results"
    
    try:
        loader = CompleteHeusdenLoader(topology_dir, data_dir, verbose=True)
        
        print("\nNetwork Summary:")
        print(f"   Node types: {loader.get_node_types()}")
        print(f"   Edge types: {len(loader.get_edge_types())} unique combinations")
        
        # Create sample dataset
        # dataset = loader.create_dataset(max_simulations=3, timesteps_per_simulation=5)
        # print(f"\nDataset created with {len(dataset)} samples")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("   (This is expected if paths don't exist)")


