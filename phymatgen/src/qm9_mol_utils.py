#!/usr/bin/env python3
"""Utilities for QM9-style 3D molecule validation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import math

try:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Geometry import Point3D
except ModuleNotFoundError as exc:  # pragma: no cover - runtime environment guard
    raise SystemExit(
        "RDKit is required. Install it in the active Python environment and rerun the script."
    ) from exc


QM9_ALLOWED_VALENCES = {
    "H": {1},
    "C": {4},
    "N": {3},
    "O": {2},
    "F": {1},
}

SUPPORTED_EXTENSIONS = {".pdb", ".xyz"}

COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
}


@dataclass(frozen=True)
class Atom3D:
    symbol: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class Bond3D:
    begin: int
    end: int
    order: int
    distance: float


@dataclass
class MoleculeEval:
    atoms: list[Atom3D]
    bonds: list[Bond3D]
    bond_source: str
    stable_atoms: int
    atom_stable_flags: list[bool]
    mol_stable: bool
    rdkit_valid: bool
    no_atomic_overlap: bool
    canonical_smiles: str
    error: str


_BOND_TYPE_BY_ORDER = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}


# Bond-length thresholds in Angstrom, adapted for QM9 atoms from common
# EDM-style distance-to-bond-order heuristics. A small margin is applied by
# infer_bond_order to tolerate coordinate noise.
_SINGLE_MAX = {
    ("H", "H"): 0.74,
    ("C", "H"): 1.09,
    ("H", "N"): 1.01,
    ("H", "O"): 0.96,
    ("F", "H"): 0.92,
    ("C", "C"): 1.54,
    ("C", "N"): 1.47,
    ("C", "O"): 1.43,
    ("C", "F"): 1.35,
    ("N", "N"): 1.45,
    ("N", "O"): 1.40,
    ("F", "N"): 1.36,
    ("O", "O"): 1.48,
    ("F", "O"): 1.42,
    ("F", "F"): 1.42,
}

_DOUBLE_MAX = {
    ("C", "C"): 1.34,
    ("C", "N"): 1.29,
    ("C", "O"): 1.20,
    ("N", "N"): 1.25,
    ("N", "O"): 1.21,
    ("O", "O"): 1.21,
}

_TRIPLE_MAX = {
    ("C", "C"): 1.20,
    ("C", "N"): 1.16,
    ("C", "O"): 1.13,
    ("N", "N"): 1.10,
}


def iter_molecule_files(input_dir: Path) -> list[Path]:
    return sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def parse_molecule_file(path: Path) -> list[Atom3D]:
    atoms, _ = parse_molecule_file_with_bonds(path)
    return atoms


def parse_molecule_file_with_bonds(path: Path) -> tuple[list[Atom3D], list[Bond3D]]:
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return _parse_pdb_with_bonds(path)
    if suffix == ".xyz":
        return _parse_xyz(path), []
    raise ValueError(f"Unsupported molecule file extension: {path.suffix}")


def _parse_pdb(path: Path) -> list[Atom3D]:
    atoms, _ = _parse_pdb_with_bonds(path)
    return atoms


def _parse_pdb_with_bonds(path: Path) -> tuple[list[Atom3D], list[Bond3D]]:
    atoms: list[Atom3D] = []
    conect_counts: dict[tuple[int, int], int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "CONECT":
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        begin = int(parts[1]) - 1
                        for raw_end in parts[2:]:
                            end = int(raw_end) - 1
                            if begin == end:
                                continue
                            pair = tuple(sorted((begin, end)))
                            conect_counts[pair] = conect_counts.get(pair, 0) + 1
                    except ValueError:
                        continue
                continue
            if record not in {"ATOM", "HETATM"}:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                parts = line.split()
                x, y, z = map(float, parts[6:9])
            symbol = line[76:78].strip() if len(line) >= 78 else ""
            if not symbol:
                symbol = "".join(ch for ch in line[12:16].strip() if ch.isalpha())
            atoms.append(Atom3D(_normalize_symbol(symbol), x, y, z))
    if not atoms:
        raise ValueError("No atoms found in PDB file")

    bonds: list[Bond3D] = []
    for (begin, end), count in sorted(conect_counts.items()):
        if begin < 0 or end < 0 or begin >= len(atoms) or end >= len(atoms):
            continue
        # Duplicate CONECT entries are used by this project to preserve bond
        # order in generated PDBs. External PDBs with one entry per direction
        # still decode as single bonds.
        order = max(1, min(3, (count + 1) // 2))
        bonds.append(Bond3D(begin, end, order, distance(atoms[begin], atoms[end])))
    return atoms, bonds


def _parse_xyz(path: Path) -> list[Atom3D]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("Empty XYZ file")
    try:
        atom_count = int(lines[0].strip())
        atom_lines = lines[2 : 2 + atom_count]
    except ValueError:
        atom_lines = lines

    atoms: list[Atom3D] = []
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            continue
        atoms.append(
            Atom3D(
                _normalize_symbol(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
            )
        )
    if not atoms:
        raise ValueError("No atoms found in XYZ file")
    return atoms


def _normalize_symbol(symbol: str) -> str:
    symbol = symbol.strip()
    if not symbol:
        raise ValueError("Missing atom symbol")
    if len(symbol) == 1:
        return symbol.upper()
    return symbol[0].upper() + symbol[1:].lower()


def infer_bonds(atoms: list[Atom3D], margin: float = 0.05) -> list[Bond3D]:
    bonds: list[Bond3D] = []
    for i, j in combinations(range(len(atoms)), 2):
        dist = distance(atoms[i], atoms[j])
        order = infer_bond_order(atoms[i].symbol, atoms[j].symbol, dist, margin)
        if order > 0:
            bonds.append(Bond3D(i, j, order, dist))
    return bonds


def infer_bond_order(symbol_a: str, symbol_b: str, dist: float, margin: float) -> int:
    pair = tuple(sorted((_normalize_symbol(symbol_a), _normalize_symbol(symbol_b))))
    if dist <= _TRIPLE_MAX.get(pair, -1.0) + margin:
        return 3
    if dist <= _DOUBLE_MAX.get(pair, -1.0) + margin:
        return 2
    if dist <= _SINGLE_MAX.get(pair, -1.0) + margin:
        return 1
    return 0


def distance(a: Atom3D, b: Atom3D) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def check_atomic_overlap(
    atoms: list[Atom3D], tolerance: float = 0.3
) -> tuple[bool, str]:
    """True if no pair is closer than the covalent-radius overlap threshold."""
    try:
        for i, atom_i in enumerate(atoms):
            r_i = COVALENT_RADII[atom_i.symbol]
            for j in range(i + 1, len(atoms)):
                atom_j = atoms[j]
                r_j = COVALENT_RADII[atom_j.symbol]
                dist = distance(atom_i, atom_j)
                threshold = (1.0 - tolerance) * (r_i + r_j)
                if dist < threshold:
                    return (
                        False,
                        f"overlap:{atom_i.symbol}{i}-{atom_j.symbol}{j}:{dist:.3f}A",
                    )
        return True, ""
    except Exception as exc:
        return False, f"overlap_error:{exc}"


def evaluate_molecule(
    path: Path, margin: float = 0.05, overlap_tolerance: float = 0.3
) -> MoleculeEval:
    error_parts: list[str] = []
    try:
        atoms, file_bonds = parse_molecule_file_with_bonds(path)
    except Exception as exc:
        return MoleculeEval([], [], "none", 0, [], False, False, False, "", str(exc))

    unsupported = sorted({a.symbol for a in atoms} - set(QM9_ALLOWED_VALENCES))
    if unsupported:
        error_parts.append(f"unsupported QM9 atom types: {' '.join(unsupported)}")

    no_atomic_overlap, overlap_error = check_atomic_overlap(
        atoms, tolerance=overlap_tolerance
    )
    if overlap_error:
        error_parts.append(overlap_error)

    bonds = file_bonds if file_bonds else infer_bonds(atoms, margin=margin)
    bond_source = "pdb_conect" if file_bonds else "distance_threshold_fallback"
    atom_stable_flags = stable_flags_from_bonds(atoms, bonds)
    rdkit_valid = False
    canonical_smiles = ""
    if file_bonds:
        try:
            mol = build_rdkit_mol_from_bonds(atoms, bonds)
            Chem.SanitizeMol(mol)
            atom_stable_flags = stable_flags_from_rdkit_mol(mol)
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            rdkit_valid = bool(canonical_smiles)
        except Exception as exc:
            error_parts.append(f"pdb_conect: {exc}")
    else:
        try:
            mol = build_rdkit_mol_from_coordinates(atoms)
            Chem.SanitizeMol(mol)
            bonds = extract_rdkit_bonds(atoms, mol)
            bond_source = "rdkit_determine_bonds"
            atom_stable_flags = stable_flags_from_rdkit_mol(mol)
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            rdkit_valid = bool(canonical_smiles)
        except Exception as exc:
            rdkit_error = f"rdkit: {exc}"
            try:
                mol = build_rdkit_mol_from_bonds(atoms, bonds)
                Chem.SanitizeMol(mol)
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                rdkit_valid = bool(canonical_smiles)
            except Exception as fallback_exc:
                error_parts.append(f"{rdkit_error}; fallback: {fallback_exc}")

    stable_atoms = sum(atom_stable_flags)
    mol_stable = len(atoms) > 0 and stable_atoms == len(atoms)

    return MoleculeEval(
        atoms=atoms,
        bonds=bonds,
        bond_source=bond_source,
        stable_atoms=stable_atoms,
        atom_stable_flags=atom_stable_flags,
        mol_stable=mol_stable,
        rdkit_valid=rdkit_valid,
        no_atomic_overlap=no_atomic_overlap,
        canonical_smiles=canonical_smiles,
        error=" | ".join(error_parts),
    )


def stable_flags_from_bonds(atoms: list[Atom3D], bonds: Iterable[Bond3D]) -> list[bool]:
    valences = [0] * len(atoms)
    for bond in bonds:
        valences[bond.begin] += bond.order
        valences[bond.end] += bond.order
    return [
        valence in QM9_ALLOWED_VALENCES.get(atom.symbol, set())
        for atom, valence in zip(atoms, valences)
    ]


def stable_flags_from_rdkit_mol(mol) -> list[bool]:
    mol.UpdatePropertyCache(strict=False)
    return [
        atom.GetExplicitValence() in QM9_ALLOWED_VALENCES.get(atom.GetSymbol(), set())
        for atom in mol.GetAtoms()
    ]


def extract_rdkit_bonds(atoms: list[Atom3D], mol) -> list[Bond3D]:
    bonds: list[Bond3D] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        order = int(round(bond.GetBondTypeAsDouble()))
        if order not in _BOND_TYPE_BY_ORDER:
            order = 1
        bonds.append(Bond3D(begin, end, order, distance(atoms[begin], atoms[end])))
    return bonds


def build_rdkit_mol_from_coordinates(atoms: list[Atom3D]):
    mol = Chem.RWMol()
    conformer = Chem.Conformer(len(atoms))
    for idx, atom in enumerate(atoms):
        mol.AddAtom(Chem.Atom(atom.symbol))
        conformer.SetAtomPosition(idx, Point3D(atom.x, atom.y, atom.z))
    mol = mol.GetMol()
    mol.AddConformer(conformer)
    rdDetermineBonds.DetermineBonds(mol, charge=0)
    return mol


def build_rdkit_mol_from_bonds(atoms: list[Atom3D], bonds: Iterable[Bond3D]):
    mol = Chem.RWMol()
    for atom in atoms:
        mol.AddAtom(Chem.Atom(atom.symbol))
    for bond in bonds:
        mol.AddBond(bond.begin, bond.end, _BOND_TYPE_BY_ORDER[bond.order])
    conformer = Chem.Conformer(len(atoms))
    for idx, atom in enumerate(atoms):
        conformer.SetAtomPosition(idx, Point3D(atom.x, atom.y, atom.z))
    out = mol.GetMol()
    out.AddConformer(conformer)
    return out
