"""从 CIF/XYZ 文件加载为 PyG Data，供 VAE/LDM 编码使用。"""
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


_PDB_ELEMENT_TO_Z = {
    "H": 1,
    "HE": 2,
    "LI": 3,
    "BE": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "NE": 10,
    "NA": 11,
    "MG": 12,
    "AL": 13,
    "SI": 14,
    "P": 15,
    "S": 16,
    "CL": 17,
    "AR": 18,
    "K": 19,
    "CA": 20,
    "BR": 35,
    "I": 53,
}


def _load_pdb_molecule(path: Path):
    species = []
    coords = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            element = line[76:78].strip() or line[12:16].strip().lstrip("0123456789")
            element = element.upper()
            if element not in _PDB_ELEMENT_TO_Z:
                raise ValueError(f"PDB 中存在未支持元素: {element!r} ({path})")
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            species.append(_PDB_ELEMENT_TO_Z[element])
            coords.append([x, y, z])
    if not species:
        raise ValueError(f"PDB 文件中未找到原子记录: {path}")
    return species, np.array(coords, dtype=np.float32)


def load_structure_file_to_data(path: str, dataset_idx: int) -> Data:
    """从单个 CIF 或 XYZ 文件加载结构，转为与 JointDataModule 一致的 PyG Data。

    :param path: 文件路径，支持 .cif、.xyz 或 .pdb
    :param dataset_idx: 0=mp20/crystal, 1=qm9/molecule, 3=carbon24, 4=perov5
    :return: Data(atom_types, pos, frac_coords, cell, lengths, angles, ...)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()

    if suffix == ".cif":
        from pymatgen.core.structure import Structure
        struct = Structure.from_file(str(path))
        species = [s.specie.Z for s in struct]
        frac_coords = np.array(struct.frac_coords, dtype=np.float32)
        lattice = struct.lattice
        cell = np.array(lattice.matrix, dtype=np.float32)
        lengths = np.array(lattice.parameters[:3], dtype=np.float32)
        angles = np.array(lattice.parameters[3:], dtype=np.float32)
        num_atoms = len(species)
        # pos = frac @ cell
        pos = frac_coords @ cell
        lengths_scaled = lengths / (num_atoms ** (1 / 3))
        angles_radians = np.radians(angles)
        lattices_scaled = np.concatenate([lengths_scaled, angles_radians])

        data = Data(
            atom_types=torch.LongTensor(species),
            frac_coords=torch.tensor(frac_coords, dtype=torch.float32),
            cell=torch.tensor(cell, dtype=torch.float32).unsqueeze(0),
            lattices=torch.tensor(np.concatenate([lengths, angles]), dtype=torch.float32).unsqueeze(0),
            lattices_scaled=torch.tensor(lattices_scaled, dtype=torch.float32).unsqueeze(0),
            lengths=torch.tensor(lengths, dtype=torch.float32).view(1, -1),
            lengths_scaled=torch.tensor(lengths_scaled, dtype=torch.float32).view(1, -1),
            angles=torch.tensor(angles, dtype=torch.float32).view(1, -1),
            angles_radians=torch.tensor(angles_radians, dtype=torch.float32).view(1, -1),
            num_atoms=torch.LongTensor([num_atoms]),
            num_nodes=torch.LongTensor([num_atoms]),
            token_idx=torch.arange(num_atoms),
            dataset_idx=torch.tensor([dataset_idx], dtype=torch.long),
        )
        data.pos = torch.tensor(pos, dtype=torch.float32)
        data.spacegroup = torch.LongTensor([struct.get_space_group_info()[1] or 0])

    elif suffix in {".xyz", ".pdb"}:
        if suffix == ".pdb":
            species, coords = _load_pdb_molecule(path)
        else:
            from pymatgen.core import Molecule

            mol = Molecule.from_file(str(path))
            species = list(mol.atomic_numbers)
            coords = np.array(mol.cart_coords, dtype=np.float32)
        num_atoms = len(species)
        data = Data(
            atom_types=torch.LongTensor(species),
            pos=torch.tensor(coords, dtype=torch.float32),
            frac_coords=torch.zeros_like(torch.tensor(coords)),
            cell=torch.zeros(1, 3, 3),
            lattices=torch.zeros(1, 6),
            lattices_scaled=torch.zeros(1, 6),
            lengths=torch.zeros(1, 3),
            lengths_scaled=torch.zeros(1, 3),
            angles=torch.zeros(1, 3),
            angles_radians=torch.zeros(1, 3),
            num_atoms=torch.LongTensor([num_atoms]),
            num_nodes=torch.LongTensor([num_atoms]),
            token_idx=torch.arange(num_atoms),
            dataset_idx=torch.tensor([dataset_idx], dtype=torch.long),
            spacegroup=torch.zeros(1, dtype=torch.long),
        )
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")

    return data
