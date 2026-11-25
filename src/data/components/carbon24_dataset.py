"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import os
import warnings
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from src.data.components.preprocessing_utils import preprocess

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)


class Carbon24(InMemoryDataset):
    """The Carbon-24 dataset, as a PyG InMemoryDataset.

    In order to create a torch_geometric.data.InMemoryDataset, you need to implement four fundamental methods:
    - InMemoryDataset.raw_file_names(): A list of files in the raw_dir which needs to be found in order to skip the download.
    - InMemoryDataset.processed_file_names(): A list of files in the processed_dir which needs to be found in order to skip the processing.
    - InMemoryDataset.download(): Downloads raw data into raw_dir.
    - InMemoryDataset.process(): Processes raw data and saves it into the processed_dir.

    Args:
        root (str): Root directory where the dataset should be saved.
        split (str): Dataset split to use. Options: 'train', 'val', 'test', or 'all' (default: 'all').
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        root: str,
        split: str = "all",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        if os.path.exists(self.processed_paths[0]):
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw file names based on split."""
        if self.split == "all":
            return ["train.csv", "val.csv", "test.csv"]
        else:
            return [f"{self.split}.csv"]

    @property
    def processed_file_names(self) -> List[str]:
        """Return processed file name based on split."""
        return [f"carbon24_{self.split}.pt"]

    def download(self) -> None:
        """Download is not needed as data is already in the directory."""
        pass

    def process(self) -> None:
        """Process the raw CSV files and convert to PyG Data objects."""
        # Determine which CSV files to process
        if self.split == "all":
            csv_files = ["train.csv", "val.csv", "test.csv"]
        else:
            csv_files = [f"{self.split}.csv"]

        # Check if cached processed data exists
        cached_file = os.path.join(self.root, "raw", f"carbon24_{self.split}.pt")
        if os.path.exists(cached_file):
            cached_data = torch.load(cached_file)
        else:
            # Read and combine CSV files
            dfs = []
            for csv_file in csv_files:
                csv_path = os.path.join(self.root, csv_file)
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    dfs.append(df)
                else:
                    print(f"Warning: {csv_path} not found, skipping...")

            if not dfs:
                raise FileNotFoundError(f"No CSV files found for split {self.split}")

            # Combine all dataframes
            combined_df = pd.concat(dfs, ignore_index=True)

            # Save combined CSV temporarily for preprocessing
            temp_csv = os.path.join(self.root, "raw", f"temp_{self.split}.csv")
            os.makedirs(os.path.join(self.root, "raw"), exist_ok=True)
            combined_df.to_csv(temp_csv, index=False)

            # Preprocess the data
            cached_data = preprocess(
                temp_csv,
                num_workers=32,
                niggli=True,
                primitive=False,
                graph_method="crystalnn",
                prop_list=["energy_per_atom"],
                use_space_group=True,
                tol=0.1,
            )

            # Save cached data
            torch.save(cached_data, cached_file)

            # Clean up temporary CSV
            if os.path.exists(temp_csv):
                os.remove(temp_csv)

        data_list = []
        for data_dict in cached_data:
            # extract attributes from data_dict
            graph_arrays = data_dict["graph_arrays"]
            atom_types = graph_arrays["atom_types"]
            frac_coords = graph_arrays["frac_coords"]
            cell = graph_arrays["cell"]
            lattices = graph_arrays["lattices"]
            lengths = graph_arrays["lengths"]
            angles = graph_arrays["angles"]
            num_atoms = graph_arrays["num_atoms"]

            # normalize the lengths of lattice vectors, which makes
            # lengths for materials of different sizes at same scale
            _lengths = lengths / float(num_atoms) ** (1 / 3)
            # convert angles of lattice vectors to be in radians
            _angles = np.radians(angles)
            # add scaled lengths and angles to graph arrays
            graph_arrays["length_scaled"] = _lengths
            graph_arrays["angles_radians"] = _angles
            graph_arrays["lattices_scaled"] = np.concatenate([_lengths, _angles])

            data = Data(
                id=data_dict["mp_id"],
                atom_types=torch.LongTensor(atom_types),
                frac_coords=torch.Tensor(frac_coords),
                cell=torch.Tensor(cell).unsqueeze(0),
                lattices=torch.Tensor(lattices).unsqueeze(0),
                lattices_scaled=torch.Tensor(graph_arrays["lattices_scaled"]).unsqueeze(0),
                lengths=torch.Tensor(lengths).view(1, -1),
                lengths_scaled=torch.Tensor(graph_arrays["length_scaled"]).view(1, -1),
                angles=torch.Tensor(angles).view(1, -1),
                angles_radians=torch.Tensor(graph_arrays["angles_radians"]).view(1, -1),
                num_atoms=torch.LongTensor([num_atoms]),
                num_nodes=torch.LongTensor([num_atoms]),  # special attribute used for PyG batching
                token_idx=torch.arange(num_atoms),
                dataset_idx=torch.tensor(
                    [0], dtype=torch.long
                ),  # 0 --> indicates periodic/crystal
            )
            # 3D coordinates (NOTE do not zero-center prior to graph construction)
            data.pos = torch.einsum(
                "bi,bij->bj",
                data.frac_coords,
                torch.repeat_interleave(data.cell, data.num_atoms, dim=0),
            )
            # space group number
            data.spacegroup = torch.LongTensor([data_dict["spacegroup"]])

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        self.save(data_list, self.processed_paths[0])
