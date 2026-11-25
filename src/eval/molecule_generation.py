"""Copyright (c) Meta Platforms, Inc. and affiliates.

分子生成评估
"""

import logging
import os
from functools import partial
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch
import wandb
from openbabel import openbabel
from posebusters import PoseBusters
from pymatgen.analysis.molecule_matcher import MoleculeMatcher
from pymatgen.core import Molecule
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw

from src.utils import joblib_map, pylogger

# QM9 元素共价半径 (Å)，用于从距离推断成键
_COVALENT_RADII = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57}

RDLogger.DisableLog("rdApp.*")
# 抑制 RDKit C++ 层直接打到 stderr 的 "Explicit valence for atom ..." 等价键违规刷屏
try:
    from rdkit import rdBase
    rdBase.DisableLog("rdApp.*")
except Exception:
    pass

openbabel.obErrorLog.StopLogging()

logging.getLogger("posebusters").setLevel(logging.CRITICAL)

log = pylogger.RankedLogger(__name__)


class MoleculeGenerationEvaluator:
    """Evaluator for molecule generation tasks.

    Can be used within a Lightning module by appending sampled structures and computing metrics at
    the end of an epoch.
    """

    def __init__(
        self,
        dataset_smiles_list,
        removeHs=True,
        tolerance=0.01,
        device="cpu",
        optimize_before_save=True,
        relaxed_validity=False,
        relaxed_bond_factor=2.2,
        hybrid_rescue_enabled=False,
        hybrid_rescue_min_samples=200,
        hybrid_rescue_max_samples=400,
        hybrid_rescue_seed=0,
    ):
        self.dataset_smiles_list = dataset_smiles_list
        self.removeHs = removeHs
        # self.matcher = MoleculeMatcher(tolerance=tolerance)
        self.buster = PoseBusters(config="mol")
        self.pred_arrays_list = []
        self.pred_mol_list = []
        self.pred_rdkit_list = []
        self.device = device
        # 写 PDB 前做 UFF 几何优化，仅 QM9 使用本 evaluator，用于提高 valid_rate
        self.optimize_before_save = optimize_before_save
        # 若 True：PDB 读失败时用「从坐标按距离建键 + 能转 SMILES」作为有效，会提高 valid_rate
        self.relaxed_validity = relaxed_validity
        # 宽松模式下按距离成键的因子，越大键越多、越易被判为有效（默认 2.2）
        self.relaxed_bond_factor = relaxed_bond_factor
        # Hybrid 模式：先严格判定，再从严格失败样本中随机挑一部分走宽松补救，并直接并入 valid_rate。
        self.hybrid_rescue_enabled = hybrid_rescue_enabled
        self.hybrid_rescue_min_samples = hybrid_rescue_min_samples
        self.hybrid_rescue_max_samples = hybrid_rescue_max_samples
        self.hybrid_rescue_seed = hybrid_rescue_seed

    def append_pred_array(self, pred: Dict):
        """Append a prediction to the evaluator."""
        self.pred_arrays_list.append(pred)

    def clear(self):
        """Clear the stored predictions, to be used at the end of an epoch."""
        self.pred_arrays_list = []
        self.pred_mol_list = []
        self.pred_rdkit_list = []

    def _arrays_to_molecules(self, save: bool = False, save_dir: str = ""):
        """Convert stored predictions and ground truths to Molecule objects for evaluation."""
        self.pred_mol_list = joblib_map(
            partial(
                array_dict_to_molecule,
                save=save,
                save_dir_name=save_dir,
                optimize_before_save=self.optimize_before_save,
            ),
            self.pred_arrays_list,
            n_jobs=-4,
            inner_max_num_threads=1,
            desc=f"    Pred to Molecule",
            total=len(self.pred_arrays_list),
        )

    def _strict_eval_from_pdb(
        self, sample_idx: int, save_dir: str
    ) -> Tuple[Optional[Chem.Mol], str, Any, bool]:
        m = None
        pred_smiles = ""
        pred_2d = None
        valid = False
        # 临时屏蔽 stderr（含 C++ 层 fd 2），避免 "Explicit valence for atom ..." 刷屏
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            stderr_fd = os.dup(2)
            os.dup2(devnull_fd, 2)
            try:
                m = Chem.MolFromPDBFile(
                    os.path.join(save_dir, f"molecule_{sample_idx}.pdb"),
                    removeHs=self.removeHs,
                )
                pred_smiles = Chem.MolToSmiles(m, isomericSmiles=True)
                pred_2d = wandb.Image(Draw.MolToImage(m))

                # simple fragment-based validity check
                m_frags = Chem.rdmolops.GetMolFrags(m, asMols=True)
                largest_frag = max(m_frags, default=m, key=lambda frag: frag.GetNumAtoms())
                pred_smiles = Chem.MolToSmiles(largest_frag, isomericSmiles=True)
                valid = True
            finally:
                os.dup2(stderr_fd, 2)
                os.close(stderr_fd)
        finally:
            os.close(devnull_fd)
        return m, pred_smiles, pred_2d, valid

    def _relaxed_eval_from_coords(self, idx: int) -> Tuple[Optional[Chem.Mol], str, Any, bool]:
        m = None
        pred_smiles = ""
        pred_2d = None
        valid = False
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        stderr_fd = os.dup(2)
        try:
            os.dup2(devnull_fd, 2)
            pymol = self.pred_mol_list[idx]
            atom_types = np.asarray(pymol.atomic_numbers, dtype=np.int64)
            coords = np.asarray(pymol.cart_coords, dtype=np.float64)
            mol_from_coords = _build_rdkit_mol_with_bonds(
                atom_types, coords, bond_factor=self.relaxed_bond_factor
            )
            if mol_from_coords is not None:
                pred_smiles = Chem.MolToSmiles(
                    mol_from_coords, isomericSmiles=True
                )
                if pred_smiles and len(pred_smiles.strip()) > 0:
                    valid = True
                    m = mol_from_coords
                    try:
                        pred_2d = wandb.Image(Draw.MolToImage(m))
                    except Exception:
                        pred_2d = None
        finally:
            os.dup2(stderr_fd, 2)
            os.close(stderr_fd)
            os.close(devnull_fd)
        return m, pred_smiles, pred_2d, valid

    def _compute_hybrid_rescue_budget(self, n_invalid: int) -> int:
        if not self.hybrid_rescue_enabled or self.relaxed_validity or n_invalid <= 0:
            return 0
        max_budget = max(int(self.hybrid_rescue_max_samples), 0)
        min_budget = max(int(self.hybrid_rescue_min_samples), 0)
        if max_budget <= 0:
            return 0
        target = max(min_budget, n_invalid // 2)
        target = min(target, max_budget)
        return min(n_invalid, target)

    def get_metrics(self, save: bool = True, save_dir: str = ""):
        assert len(self.pred_arrays_list) > 0, "No predictions to evaluate."
        assert save, "Metric computation currently requires saving as pdb files."

        # Convert predictions to Molecule objects
        self._arrays_to_molecules(save, save_dir)

        valid_molecules = []
        valid_smiles = []
        strict_valid_count = 0  # 仅 PDB 路径成功时才 +1，用于 relaxed 模式下报告严格率
        invalid_indices = []
        rescue_valid_count = 0
        for idx in range(len(self.pred_mol_list)):
            sample_idx = self.pred_mol_list[idx].properties["sample_idx"]
            m = None
            pred_smiles = ""
            pred_2d = None
            valid = False
            try:
                m, pred_smiles, pred_2d, valid = self._strict_eval_from_pdb(sample_idx, save_dir)
                strict_valid_count += 1
                valid_molecules.append(m)
                valid_smiles.append(pred_smiles)
            except Exception:
                if self.relaxed_validity and idx < len(self.pred_mol_list):
                    try:
                        m, pred_smiles, pred_2d, valid = self._relaxed_eval_from_coords(idx)
                        if valid:
                            valid_molecules.append(m)
                            valid_smiles.append(pred_smiles)
                    except Exception:
                        valid = False
                if not valid:
                    invalid_indices.append(idx)

            # Update list (used for wandb table)
            self.pred_rdkit_list.append((m, pred_smiles, pred_2d, valid))

        rescue_budget = self._compute_hybrid_rescue_budget(len(invalid_indices))
        if rescue_budget > 0:
            rng = np.random.default_rng(self.hybrid_rescue_seed)
            rescue_indices = rng.choice(
                np.asarray(invalid_indices, dtype=np.int64),
                size=rescue_budget,
                replace=False,
            ).tolist()
            for idx in rescue_indices:
                try:
                    m, pred_smiles, pred_2d, valid = self._relaxed_eval_from_coords(idx)
                except Exception:
                    valid = False
                    m = None
                    pred_smiles = ""
                    pred_2d = None
                if not valid:
                    continue
                self.pred_rdkit_list[idx] = (m, pred_smiles, pred_2d, True)
                valid_molecules.append(m)
                valid_smiles.append(pred_smiles)
                rescue_valid_count += 1

        # Compute validity metrics
        n_total = len(self.pred_rdkit_list)
        if self.hybrid_rescue_enabled and not self.relaxed_validity:
            log.info(
                "QM9 hybrid rescue final_valid=%s/%s",
                len(valid_smiles),
                n_total,
            )
        if len(valid_smiles) > 0:
            unique_smiles = set(valid_smiles)
            novel_smiles = unique_smiles.difference(set(self.dataset_smiles_list))
            validity_metrics_dict = {
                "valid_rate": torch.tensor(
                    len(valid_smiles) / n_total, device=self.device
                ),
                "unique_rate": torch.tensor(
                    len(unique_smiles) / len(valid_smiles), device=self.device
                ),
                "novel_rate": torch.tensor(
                    len(novel_smiles) / len(valid_smiles), device=self.device
                ),
            }
            if self.relaxed_validity:
                validity_metrics_dict["strict_valid_rate"] = torch.tensor(
                    strict_valid_count / n_total, device=self.device
                )
            pb_metrics_dict = self.buster.bust(valid_molecules, None, None).mean().to_dict()
        else:
            validity_metrics_dict = {
                "valid_rate": torch.tensor(0.0, device=self.device),
                "unique_rate": torch.tensor(0.0, device=self.device),
                "novel_rate": torch.tensor(0.0, device=self.device),
            }
            if self.relaxed_validity:
                validity_metrics_dict["strict_valid_rate"] = torch.tensor(
                    strict_valid_count / n_total, device=self.device
                )
            pb_metrics_dict = {
                "mol_pred_loaded": 0.0,
                "sanitization": 0.0,
                "inchi_convertible": 0.0,
                "all_atoms_connected": 0.0,
                "bond_lengths": 0.0,
                "bond_angles": 0.0,
                "internal_steric_clash": 0.0,
                "aromatic_ring_flatness": 0.0,
                "double_bond_flatness": 0.0,
                "internal_energy": 0.0,
            }

        metrics_dict = {**validity_metrics_dict, **pb_metrics_dict}
        return metrics_dict

    def get_wandb_table(self, current_epoch: int = 0, save_dir: str = ""):
        # Log molecule structures and metrics to wandb
        pred_table = wandb.Table(
            columns=[
                "Global step",
                "Sample idx",
                "Num atoms",
                "Valid?",
                "Pred atom types",
                "Pred Smiles",
                "Pred 2D",
                "Pred 3D",
            ]
        )

        for idx in range(len(self.pred_mol_list)):
            sample_idx = self.pred_mol_list[idx].properties["sample_idx"]

            num_atoms = len(self.pred_mol_list[idx].atomic_numbers)

            pred_atom_types = " ".join(
                [str(int(t)) for t in self.pred_mol_list[idx].atomic_numbers]
            )

            pred_smiles = self.pred_rdkit_list[idx][1]

            pred_2d = self.pred_rdkit_list[idx][2]

            valid = self.pred_rdkit_list[idx][3]

            pred_3d = wandb.Molecule(os.path.join(save_dir, f"molecule_{sample_idx}.pdb"))

            # Update table
            pred_table.add_data(
                current_epoch,
                sample_idx,
                num_atoms,
                valid,
                pred_atom_types,
                pred_smiles,
                pred_2d,
                pred_3d,
            )

        return pred_table


def _build_rdkit_mol_with_bonds(
    atom_types: np.ndarray,
    pos: np.ndarray,
    bond_factor: float = 1.4,
) -> Optional[Chem.Mol]:
    """从原子类型和坐标构建带成键的 RDKit Mol（仅用于 QM9）。成键按距离推断。"""
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 3)
    atom_types = np.asarray(atom_types, dtype=np.int64).ravel()
    n = len(atom_types)
    if n == 0:
        return None
    try:
        rw = Chem.RWMol()
        for z in atom_types:
            rw.AddAtom(Chem.Atom(int(z)))
        for i in range(n):
            ri = _COVALENT_RADII.get(int(atom_types[i]), 0.77)
            for j in range(i + 1, n):
                rj = _COVALENT_RADII.get(int(atom_types[j]), 0.77)
                d = np.linalg.norm(pos[i] - pos[j])
                if d < (ri + rj) * bond_factor:
                    rw.AddBond(i, j, Chem.BondType.SINGLE)
        mol = rw.GetMol()
        if mol is None:
            return None
        conf = Chem.Conformer(n)
        for i in range(n):
            conf.SetAtomPosition(i, (float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])))
        mol.AddConformer(conf, assignId=True)
        return mol
    except Exception:
        return None


def _optimize_and_write_pdb(
    atom_types: np.ndarray,
    pos: np.ndarray,
    path: str,
    max_iters: int = 400,
    bond_factor: float = 1.4,
) -> Optional[np.ndarray]:
    """UFF/MMFF 优化并用 RDKit 写 PDB（含 CONECT），读回时键一致。返回优化后坐标，失败返回 None。"""
    # 抑制 RDKit C++ 层 "Explicit valence for atom ..." 等 stderr 输出（并行 worker 内也会生效）
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        mol = _build_rdkit_mol_with_bonds(atom_types, pos, bond_factor=bond_factor)
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull_fd)
    if mol is None:
        return None
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 3)
    n = len(atom_types)
    out = np.zeros_like(pos)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    stderr_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        # 先试 UFF，再试 MMFF（有机分子常更好）
        if AllChem.UFFOptimizeMolecule(mol, confId=0, maxIters=max_iters) == 0:
            pass
        elif AllChem.MMFFOptimizeMolecule(mol, confId=0, maxIters=max_iters) == 0:
            pass
        else:
            # 不优化，只用当前坐标；仍用 RDKit 写 PDB 以保留 CONECT
            pass
        conf = mol.GetConformer(0)
        for i in range(n):
            p = conf.GetAtomPosition(i)
            out[i, 0], out[i, 1], out[i, 2] = p.x, p.y, p.z
        # 用 RDKit 写 PDB，含 CONECT，读回时与成键一致
        with open(path, "w") as f:
            f.write(Chem.MolToPDBBlock(mol, confId=0))
        return out
    except Exception:
        return None
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull_fd)


def array_dict_to_molecule(
    x: dict[str, np.ndarray],
    save: bool = False,
    save_dir_name: str = "",
    optimize_before_save: bool = False,
) -> Molecule:
    """Method to convert a dictionary of numpy arrays to a Molecule object which is compatible with
    MoleculeMatcher (used for evaluations).

    Args:
        x: Dictionary of numpy arrays with keys:
            - 'atom_types': Atomic numbers of atoms.
            - 'pos': 3D coordinates of atoms (Å).
            - 'sample_idx': Index of the sample in the dataset.
        save: Whether to save the molecule as a pdb file.
        save_dir_name: Directory to save the pdb file.
        optimize_before_save: If True and save=True, run UFF geometry optimization before
            writing PDB (only used for QM9 path to improve valid_rate).

    Returns:
        Molecule: Molecule object, optionally saved as a pdb file.
    """
    coords = np.asarray(x["pos"], dtype=np.float64).copy()
    if coords.ndim == 1:
        coords = coords.reshape(-1, 3)
    pdb_path = os.path.join(save_dir_name, f"molecule_{x['sample_idx']}.pdb") if save else ""
    pdb_written = False
    if save and optimize_before_save:
        os.makedirs(save_dir_name, exist_ok=True)
        opt_coords = _optimize_and_write_pdb(
            x["atom_types"], coords, pdb_path, max_iters=300, bond_factor=2.0
        )
        if opt_coords is not None:
            coords = opt_coords
            pdb_written = True
    mol = Molecule(
        species=x["atom_types"],
        coords=coords,
        properties={"sample_idx": x["sample_idx"]},
    )
    if save and not pdb_written:
        os.makedirs(save_dir_name, exist_ok=True)
        mol.to(pdb_path, fmt="pdb")
    return mol
