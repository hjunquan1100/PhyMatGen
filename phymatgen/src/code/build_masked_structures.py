#!/usr/bin/env python
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = PROJECT_ROOT / "data" / "masked" / "source"
OUTPUT_DIR = SOURCE_DIR / "Canque"

COLORS = {
    "H": "#e7e7e7",
    "C": "#222222",
    "N": "#2455c3",
    "O": "#d12c2c",
    "F": "#47a447",
    "S": "#d9b321",
    "Cl": "#35a853",
    "La": "#5e83d8",
    "Eu": "#d66bbb",
    "Sr": "#43a8a5",
    "Ti": "#8c8f96",
    "Ba": "#3a78c2",
    "Zr": "#9b9b9b",
    "Pb": "#59606b",
    "I": "#7d4fa3",
}

COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "S": 1.05,
    "Cl": 1.02,
    "La": 1.69,
    "Eu": 1.98,
    "Sr": 1.95,
    "Ti": 1.60,
    "Ba": 2.15,
    "Zr": 1.75,
    "Pb": 1.46,
    "I": 1.39,
}


@dataclass
class AtomSet:
    elements: list[str]
    coords: list[tuple[float, float, float]]


def parse_pdb(path: Path) -> tuple[list[str], list[str], list[int]]:
    lines = path.read_text().splitlines()
    atom_indices = [i for i, line in enumerate(lines) if line.startswith(("ATOM", "HETATM"))]
    elements = []
    for i in atom_indices:
        line = lines[i]
        element = line[76:78].strip() if len(line) >= 78 else ""
        if not element:
            element = "".join(ch for ch in line[12:16].strip() if ch.isalpha())
            element = element[:2].capitalize()
        elements.append(element)
    return lines, elements, atom_indices


def pdb_atomset(path: Path) -> AtomSet:
    lines, elements, atom_indices = parse_pdb(path)
    coords = []
    for i in atom_indices:
        line = lines[i]
        coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return AtomSet(elements, coords)


def write_defective_pdb(src: Path, dst: Path, remove_indices: set[int]) -> None:
    lines, _, atom_line_indices = parse_pdb(src)
    atom_index_by_line = {line_index: n for n, line_index in enumerate(atom_line_indices)}
    new_lines = []
    serial = 1
    for line_index, line in enumerate(lines):
        atom_index = atom_index_by_line.get(line_index)
        if atom_index is not None:
            if atom_index in remove_indices:
                continue
            new_lines.append(f"{line[:6]}{serial:5d}{line[11:]}")
            serial += 1
        else:
            new_lines.append(line)
    dst.write_text("\n".join(new_lines) + "\n")


def structure_atomset(structure: Structure) -> AtomSet:
    return AtomSet(
        [site.specie.symbol for site in structure],
        [tuple(map(float, xyz)) for xyz in structure.cart_coords],
    )


def select_removed_indices(elements: list[str]) -> list[int]:
    n_atoms = len(elements)
    if n_atoms <= 1:
        return []
    n_remove = max(1, min(4, int(round(n_atoms * 0.25))))
    n_remove = min(n_remove, n_atoms - 1)

    counts: dict[str, int] = {}
    for element in elements:
        counts[element] = counts.get(element, 0) + 1

    protected = set()
    if len(counts) > 1:
        for element in counts:
            protected.add(next(i for i, value in enumerate(elements) if value == element))

    preferred = [
        i
        for i, element in sorted(enumerate(elements), key=lambda item: (counts[item[1]], item[0]), reverse=True)
        if i not in protected
    ]
    fallback = [i for i in reversed(range(n_atoms)) if i not in preferred]
    return (preferred + fallback)[:n_remove]


def axis_limits(atomsets: list[AtomSet]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    coords = [coord for atomset in atomsets for coord in atomset.coords]
    xs, ys, zs = zip(*coords)
    ranges = [(min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))]
    centers = [(lo + hi) / 2 for lo, hi in ranges]
    span = max(max(hi - lo for lo, hi in ranges), 1.5)
    pad = span * 0.12
    half = span / 2 + pad
    return tuple((center - half, center + half) for center in centers)  # type: ignore[return-value]


def infer_bonds(atomset: AtomSet) -> list[tuple[int, int]]:
    bonds = []
    for i, (element_i, coord_i) in enumerate(zip(atomset.elements, atomset.coords)):
        for j in range(i + 1, len(atomset.coords)):
            element_j = atomset.elements[j]
            coord_j = atomset.coords[j]
            distance = math.dist(coord_i, coord_j)
            cutoff = COVALENT_RADII.get(element_i, 1.0) + COVALENT_RADII.get(element_j, 1.0) + 0.45
            if 0.35 < distance <= cutoff:
                bonds.append((i, j))
    return bonds


def draw_atomset(ax, atomset: AtomSet, title: str, removed_note: str, limits) -> None:
    for i, element in enumerate(atomset.elements):
        x, y, z = atomset.coords[i]
        ax.scatter(
            [x],
            [y],
            [z],
            s=150 if element != "H" else 70,
            c=COLORS.get(element, "#8f8f8f"),
            edgecolors="#1a1a1a",
            linewidths=0.7,
            depthshade=True,
        )
        ax.text(x, y, z, f" {element}", fontsize=8)

    segments = []
    for i, j in infer_bonds(atomset):
        segments.append([atomset.coords[i], atomset.coords[j]])
    if segments:
        ax.add_collection3d(Line3DCollection(segments, colors="#666666", linewidths=1.0, alpha=0.65))

    ax.set_title(title, fontsize=13, pad=10)
    ax.text2D(0.02, 0.96, removed_note, transform=ax.transAxes, fontsize=9)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=38)
    ax.set_axis_off()


def render_comparison(original: AtomSet, defective: AtomSet, image_path: Path, removed_note: str) -> None:
    limits = axis_limits([original, defective])
    fig = plt.figure(figsize=(10, 5), dpi=180)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    draw_atomset(ax1, original, "Original", f"atoms: {len(original.elements)}", limits)
    draw_atomset(ax2, defective, "Canque", f"atoms: {len(defective.elements)}; removed: {removed_note}", limits)
    fig.suptitle(image_path.stem.replace("_comparison", ""), fontsize=14)
    fig.tight_layout()
    fig.savefig(image_path, bbox_inches="tight")
    plt.close(fig)


def process_cif(path: Path) -> dict[str, str]:
    structure = Structure.from_file(path)
    original = structure_atomset(structure)
    removed = select_removed_indices(original.elements)
    defective_structure = structure.copy()
    defective_structure.remove_sites(removed)
    defective = structure_atomset(defective_structure)

    out_path = OUTPUT_DIR / f"{path.stem}_canque.cif"
    CifWriter(defective_structure).write_file(out_path)
    removed_note = ", ".join(f"{i}:{original.elements[i]}" for i in removed)
    image_path = OUTPUT_DIR / f"{path.stem}_comparison.png"
    render_comparison(original, defective, image_path, removed_note)
    return {
        "source": path.name,
        "canque": out_path.name,
        "comparison_png": image_path.name,
        "original_atoms": str(len(original.elements)),
        "canque_atoms": str(len(defective.elements)),
        "removed_indices_elements": removed_note,
    }


def process_pdb(path: Path) -> dict[str, str]:
    original = pdb_atomset(path)
    removed = select_removed_indices(original.elements)
    out_path = OUTPUT_DIR / f"{path.stem}_canque.pdb"
    write_defective_pdb(path, out_path, set(removed))
    defective = pdb_atomset(out_path)
    removed_note = ", ".join(f"{i}:{original.elements[i]}" for i in removed)
    image_path = OUTPUT_DIR / f"{path.stem}_comparison.png"
    render_comparison(original, defective, image_path, removed_note)
    return {
        "source": path.name,
        "canque": out_path.name,
        "comparison_png": image_path.name,
        "original_atoms": str(len(original.elements)),
        "canque_atoms": str(len(defective.elements)),
        "removed_indices_elements": removed_note,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(SOURCE_DIR.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".cif":
            rows.append(process_cif(path))
        elif path.suffix.lower() == ".pdb":
            rows.append(process_pdb(path))

    summary_path = OUTPUT_DIR / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "canque",
                "comparison_png",
                "original_atoms",
                "canque_atoms",
                "removed_indices_elements",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} defective structures to {OUTPUT_DIR}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
