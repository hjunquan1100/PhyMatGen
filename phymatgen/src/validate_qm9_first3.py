#!/usr/bin/env python3
"""Validate QM9 generated molecules with Atom Sta, Mol Sta, and Validity."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from tqdm import tqdm

from phymatgen_correction.src.qm9_mol_utils import evaluate_molecule, iter_molecule_files


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "validation" / "qm9"
DEFAULT_INPUT_DIR = DEFAULT_OUTPUT_DIR / "generated"
DEFAULT_TG_DIR = DEFAULT_OUTPUT_DIR / "tg"
DEFAULT_NTG_DIR = DEFAULT_OUTPUT_DIR / "ntg"
DEFAULT_CSV_OUT = DEFAULT_OUTPUT_DIR / "qm9_first3_results.csv"
DEFAULT_SUMMARY_OUT = DEFAULT_OUTPUT_DIR / "qm9_first3_summary.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate QM9 molecules. Files passing Mol Sta, RDKit Validity, "
            "and atomic-overlap validity are copied to tg; all others are "
            "copied to ntg."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--tg-dir", type=Path, default=DEFAULT_TG_DIR)
    parser.add_argument("--ntg-dir", type=Path, default=DEFAULT_NTG_DIR)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument(
        "--bond-margin",
        type=float,
        default=0.05,
        help="Distance margin in Angstrom for QM9 bond-order inference.",
    )
    parser.add_argument(
        "--overlap-tolerance",
        type=float,
        default=0.5,
        help="Covalent-radius tolerance for atomic-overlap checking.",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Remove old molecule files from tg/ntg before copying new results.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.tg_dir.mkdir(parents=True, exist_ok=True)
    args.ntg_dir.mkdir(parents=True, exist_ok=True)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)

    if args.clear_output:
        clear_molecule_files(args.tg_dir)
        clear_molecule_files(args.ntg_dir)

    files = iter_molecule_files(args.input_dir)
    rows: list[dict[str, object]] = []

    total_atoms = 0
    total_stable_atoms = 0
    stable_molecules = 0
    valid_molecules = 0
    passed_molecules = 0

    for path in tqdm(files, desc="QM9 first-3 validation"):
        result = evaluate_molecule(
            path,
            margin=args.bond_margin,
            overlap_tolerance=args.overlap_tolerance,
        )
        atom_count = len(result.atoms)
        unstable_atoms = atom_count - result.stable_atoms
        valid = result.rdkit_valid and result.no_atomic_overlap
        passed = result.mol_stable and valid
        dest_dir = args.tg_dir if passed else args.ntg_dir
        dest_path = dest_dir / path.name
        shutil.copy2(path, dest_path)

        total_atoms += atom_count
        total_stable_atoms += result.stable_atoms
        stable_molecules += int(result.mol_stable)
        valid_molecules += int(valid)
        passed_molecules += int(passed)

        rows.append(
            {
                "file": path.name,
                "source_path": str(path),
                "copied_to": str(dest_path),
                "num_atoms": atom_count,
                "stable_atoms": result.stable_atoms,
                "unstable_atoms": unstable_atoms,
                "atom_stable_percent": percent(result.stable_atoms, atom_count),
                "mol_stable": result.mol_stable,
                "rdkit_valid": result.rdkit_valid,
                "no_atomic_overlap": result.no_atomic_overlap,
                "valid": valid,
                "passed_to_tg": passed,
                "num_inferred_bonds": len(result.bonds),
                "bond_source": result.bond_source,
                "canonical_smiles": result.canonical_smiles,
                "error": result.error,
            }
        )

    write_results_csv(args.csv_out, rows)

    summary = {
        "total_molecules": len(files),
        "total_atoms": total_atoms,
        "stable_atoms": total_stable_atoms,
        "stable_molecules": stable_molecules,
        "valid_molecules": valid_molecules,
        "passed_molecules": passed_molecules,
        "atom_sta_percent": percent(total_stable_atoms, total_atoms),
        "mol_sta_percent": percent(stable_molecules, len(files)),
        "validity_percent": percent(valid_molecules, len(files)),
    }
    write_summary_csv(args.summary_out, summary)

    print("QM9 first-3 validation complete")
    print(f"Input molecules: {len(files)}")
    print(f"Atom Sta (%): {summary['atom_sta_percent']:.4f}")
    print(f"Mol Sta (%): {summary['mol_sta_percent']:.4f}")
    print(f"Validity (%): {summary['validity_percent']:.4f}")
    print(f"TG: {args.tg_dir}")
    print(f"NTG: {args.ntg_dir}")
    print(f"CSV: {args.csv_out}")
    print(f"Summary CSV: {args.summary_out}")
    return 0


def clear_molecule_files(directory: Path) -> None:
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in {".pdb", ".xyz"}:
            path.unlink()


def percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def write_results_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "file",
        "source_path",
        "copied_to",
        "num_atoms",
        "stable_atoms",
        "unstable_atoms",
        "atom_stable_percent",
        "mol_stable",
        "rdkit_valid",
        "no_atomic_overlap",
        "valid",
        "passed_to_tg",
        "num_inferred_bonds",
        "bond_source",
        "canonical_smiles",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, summary: dict[str, object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


if __name__ == "__main__":
    raise SystemExit(main())
