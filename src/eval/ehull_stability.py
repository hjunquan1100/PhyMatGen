"""Copyright (c) Meta Platforms, Inc. and affiliates.

热力学稳定性（Ehull）评估流程：

1. CHGNet 预弛豫：ML 势对生成结构做预弛豫；
2. DFT 弛豫：将弛豫后 CIF 交由外部 DFT（如 VASP）计算，得到 formation energy；
3. Ehull：用 Matbench Discovery 兼容的 convex hull（pymatgen PhaseDiagram）计算
   energy above hull；
4. 统计热力学稳定 / 亚稳比例。

依赖（可选）：chgnet（预弛豫）, matbench-discovery（MP 参考、WBM 数据，可选）

================================================================================
使用流程（详细）
================================================================================

一、环境
  - 建议使用 conda 环境 tomvae：conda activate tomvae
  - 必需：pymatgen（PhaseDiagram、PDEntry、Structure、Composition）
  - 预弛豫：pip install chgnet（可选）
  - MP 参考能量等：pip install matbench-discovery（可选，用于形成能参考）

二、完整流程（CHGNet 预弛豫 → DFT → Ehull → 稳定/亚稳比例）

  Step 1：CHGNet 预弛豫，并保存 CIF 供 DFT 使用

    from pymatgen.core import Structure
    from src.eval.ehull_stability import run_chgnet_relax_and_save_cifs

    structures = [Structure.from_file(p) for p in your_cif_paths]  # 或从生成模型输出转成 Structure
    relaxed, cif_paths = run_chgnet_relax_and_save_cifs(
        structures,
        "cifs_for_dft",       # 输出目录
        prefix="relaxed",
        n_jobs=-2,
    )
    # relaxed[i] 为弛豫后 Structure，失败则为 None；cif_paths 为实际写出的 CIF 路径列表

  Step 2：外部 DFT 弛豫（本模块不执行）

    用 VASP / ABINIT / QE 等对 cifs_for_dft/*.cif 做结构弛豫，得到每个结构的
    formation energy (eV/atom)。形成能定义需与 hull 一致（如 MP 参考 + MP2020 修正）。
    若已安装 matbench-discovery，可用 get_mp_ref_energies_if_available() 获取参考能量。

  Step 3：准备 convex hull 的 PDEntry 列表（hull_entries）

    hull 由 PDEntry(composition, total_energy) 构建，其中 energy 为形成能*原子数
    （即 total_energy = e_form_per_atom * composition.num_atoms）。需覆盖你关心
    的化学空间（含终端元素、稳定相等）。与 Matbench Discovery 对齐时，使用 MP 系
    hull / MP2020 修正等。

    从字典构造示例：

      from pymatgen.analysis.phase_diagram import PDEntry
      from pymatgen.core import Composition
      from src.eval.ehull_stability import load_hull_entries_from_dicts

      entries = [
          {"formula": "Li", "energy_per_atom": 0.0},
          {"formula": "Fe", "energy_per_atom": 0.0},
          {"formula": "O", "energy_per_atom": 0.0},
          {"formula": "Li2O", "energy": -2.0 * 3},  # 或 energy_per_atom
          # ... 更多稳定相
      ]
      hull_entries = load_hull_entries_from_dicts(entries)

  Step 4：计算 Ehull 与稳定/亚稳比例

    from src.eval.ehull_stability import run_ehull_evaluation, pipeline_summary_str

    # structures：弛豫后结构（与 e_form 一一对应）；e_form 为 DFT 得到的 eV/atom
    metrics = run_ehull_evaluation(
        relaxed,               # list[Structure]，过滤掉 None 后与 e_form 对齐
        e_form_per_atom_list,
        hull_entries,
        thr_stable=0.02,       # 稳定：Ehull < 0.02
        thr_meta=0.10,         # 亚稳：0.02 <= Ehull < 0.10
    )
    print(pipeline_summary_str(metrics))

    metrics 包含：ehull、stable_mask、metastable_mask、stable_ratio、
    metastable_ratio、n_stable、n_metastable、n_unstable、n_valid、n_total 等。

三、仅做 Ehull（不做 CHGNet 预弛豫）

    若已有弛豫结构和形成能，可跳过 Step 1–2，直接准备 hull_entries，然后：

      metrics = run_ehull_evaluation(structures, e_form_per_atom_list, hull_entries)
      print(pipeline_summary_str(metrics))

    或对单个成分：

      from src.eval.ehull_stability import get_ehull_and_decomp
      ehull, decomp = get_ehull_and_decomp(composition, e_form_per_atom, hull_entries)

四、其他接口

  - chgnet_relax(s)：单结构 CHGNet 弛豫
  - chgnet_relax_batch(structures, n_jobs, ...)：批量弛豫
  - save_structures_as_cifs(structures, out_dir, prefix, skip_none)：保存 CIF
  - compute_ehull_metrics(compositions, e_form_list, hull_entries, ...)：仅算 Ehull 与比例
  - get_mp_ref_energies_if_available()：若装了 matbench-discovery，返回 MP 元素参考能量
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure

log = logging.getLogger(__name__)

# 常用 Ehull 阈值（eV/atom），与 Matbench Discovery 等基准一致
DEFAULT_EHULL_STABLE_THRESHOLD = 0.02  # 稳定
DEFAULT_EHULL_METASTABLE_THRESHOLD = 0.10  # 亚稳上界


def _ensure_chgnet():
    try:
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import StructOptimizer
        return CHGNet, StructOptimizer
    except ImportError as e:
        raise ImportError(
            "CHGNet 预弛豫需要安装 chgnet: pip install chgnet"
        ) from e


def chgnet_relax(structure: Structure, verbose: bool = False) -> Structure:
    """使用 CHGNet 对结构进行预弛豫。

    Args:
        structure: pymatgen Structure
        verbose: 是否打印优化过程

    Returns:
        弛豫后的 Structure
    """
    CHGNet, StructOptimizer = _ensure_chgnet()
    model = CHGNet.load()
    optimizer = StructOptimizer(model)
    relaxed = optimizer.relax(structure, verbose=verbose)
    return relaxed


def chgnet_relax_batch(
    structures: list[Structure],
    n_jobs: int = -2,
    verbose: bool = False,
) -> list[Structure | None]:
    """对多结构做 CHGNet 预弛豫；失败则对应位置为 None。"""
    from src.utils import joblib_map

    def _relax(s: Structure) -> Structure | None:
        try:
            return chgnet_relax(s, verbose=verbose)
        except Exception as e:
            log.warning("CHGNet relax failed for %s: %s", s.composition.reduced_formula, e)
            return None

    return list(
        joblib_map(
            _relax,
            structures,
            n_jobs=n_jobs,
            desc="CHGNet relax",
            total=len(structures),
        )
    )


def save_structures_as_cifs(
    structures: list[Structure],
    out_dir: str | Path,
    *,
    prefix: str = "crystal",
    skip_none: bool = True,
) -> list[str]:
    """将结构保存为 CIF，供外部 DFT 弛豫使用。

    Args:
        structures: 结构列表（可为 None，若 skip_none 则跳过）
        out_dir: 输出目录
        prefix: 文件名前缀，实际为 {prefix}_{idx}.cif
        skip_none: 是否跳过 None

    Returns:
        已保存的 CIF 路径列表
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, s in enumerate(structures):
        if skip_none and s is None:
            continue
        p = out / f"{prefix}_{i}.cif"
        s.to(str(p))
        paths.append(str(p))
    return paths


def load_hull_entries_from_dicts(
    entries: list[dict[str, Any]],
) -> list[PDEntry]:
    """从字典列表构造 PDEntry（用于 PhaseDiagram 的 hull）。

    每个 dict 需包含 'composition'（或 'formula'）和 'energy'（总能量）或
    'energy_per_atom'。也可为 ComputedEntry.from_dict 兼容格式。
    """
    out = []
    for d in entries:
        if "composition" in d:
            comp = Composition(d["composition"])
        elif "formula" in d:
            comp = Composition(d["formula"])
        else:
            raise ValueError("entry must have 'composition' or 'formula'")
        energy = d.get("energy")
        if energy is None and "energy_per_atom" in d:
            energy = float(d["energy_per_atom"]) * comp.num_atoms
        if energy is None:
            raise ValueError("entry must have 'energy' or 'energy_per_atom'")
        out.append(PDEntry(comp, float(energy)))
    return out


def get_ehull_and_decomp(
    composition: Composition,
    e_form_per_atom: float,
    hull_entries: list[PDEntry],
) -> tuple[float, dict[Composition, float] | None]:
    """对给定成分与形成能，计算 Ehull 及分解相。

    Args:
        composition: 成分
        e_form_per_atom: 形成能 (eV/atom)
        hull_entries: 构建 convex hull 的 PDEntry 列表（需包含终端元素等）

    Returns:
        (ehull eV/atom, 分解相 dict；若出错则为 (NaN, None))
    """
    try:
        pd = PhaseDiagram(hull_entries)
        n = composition.num_atoms
        entry = PDEntry(composition, e_form_per_atom * n)
        decomp, ehull = pd.get_decomp_and_e_above_hull(entry)
        return (float(ehull), decomp)
    except Exception as e:
        log.debug("get_ehull_and_decomp failed for %s: %s", composition.reduced_formula, e)
        return (float("nan"), None)


def compute_ehull_metrics(
    compositions: list[Composition],
    e_form_per_atom_list: list[float],
    hull_entries: list[PDEntry],
    *,
    thr_stable: float = DEFAULT_EHULL_STABLE_THRESHOLD,
    thr_meta: float = DEFAULT_EHULL_METASTABLE_THRESHOLD,
) -> dict[str, Any]:
    """计算 Ehull 及热力学稳定/亚稳比例。

    - 稳定：Ehull < thr_stable
    - 亚稳：thr_stable <= Ehull < thr_meta
    - 不稳定：Ehull >= thr_meta（或无效）

    Args:
        compositions: 成分列表
        e_form_per_atom_list: 对应形成能 (eV/atom)
        hull_entries: convex hull 的 PDEntry 列表
        thr_stable: 稳定阈值
        thr_meta: 亚稳上界阈值

    Returns:
        {
            "ehull": [...],
            "stable_mask": [...],
            "metastable_mask": [...],
            "stable_ratio": float,
            "metastable_ratio": float,
            "stable_or_metastable_ratio": float,
            "n_stable": int,
            "n_metastable": int,
            "n_unstable": int,
            "n_valid": int,
        }
    """
    assert len(compositions) == len(e_form_per_atom_list)
    ehulls = []
    stable_mask = []
    metastable_mask = []
    for comp, e_form in zip(compositions, e_form_per_atom_list):
        ehull, _ = get_ehull_and_decomp(comp, e_form, hull_entries)
        ehulls.append(ehull)
        valid = not np.isnan(ehull)
        stable_mask.append(valid and ehull < thr_stable)
        metastable_mask.append(valid and thr_stable <= ehull < thr_meta)

    ehulls = np.array(ehulls, dtype=float)
    stable_mask = np.array(stable_mask)
    metastable_mask = np.array(metastable_mask)
    valid_mask = ~np.isnan(ehulls)
    n_valid = int(np.sum(valid_mask))
    n_stable = int(np.sum(stable_mask))
    n_metastable = int(np.sum(metastable_mask))
    n_tot = len(ehulls)
    n_unstable = n_valid - n_stable - n_metastable

    stable_ratio = n_stable / n_tot if n_tot else 0.0
    metastable_ratio = n_metastable / n_tot if n_tot else 0.0
    stable_or_metastable_ratio = (n_stable + n_metastable) / n_tot if n_tot else 0.0

    return {
        "ehull": ehulls,
        "stable_mask": stable_mask,
        "metastable_mask": metastable_mask,
        "stable_ratio": stable_ratio,
        "metastable_ratio": metastable_ratio,
        "stable_or_metastable_ratio": stable_or_metastable_ratio,
        "n_stable": n_stable,
        "n_metastable": n_metastable,
        "n_unstable": n_unstable,
        "n_valid": n_valid,
        "n_total": n_tot,
    }


def run_chgnet_relax_and_save_cifs(
    structures: list[Structure],
    out_dir: str | Path,
    *,
    prefix: str = "relaxed",
    n_jobs: int = -2,
    verbose: bool = False,
) -> tuple[list[Structure | None], list[str]]:
    """CHGNet 预弛豫并保存 CIF，供后续 DFT 弛豫。

    Returns:
        (弛豫后的结构列表, 已保存的 CIF 路径列表)
    """
    relaxed = chgnet_relax_batch(structures, n_jobs=n_jobs, verbose=verbose)
    paths = save_structures_as_cifs(relaxed, out_dir, prefix=prefix, skip_none=True)
    return relaxed, paths


def run_ehull_evaluation(
    structures: list[Structure],
    e_form_per_atom_list: list[float],
    hull_entries: list[PDEntry],
    *,
    thr_stable: float = DEFAULT_EHULL_STABLE_THRESHOLD,
    thr_meta: float = DEFAULT_EHULL_METASTABLE_THRESHOLD,
) -> dict[str, Any]:
    """对已 DFT 弛豫、且有形成能的结构，对照 hull 算 Ehull 并统计稳定/亚稳比例。

    流程中「DFT 弛豫」在外部完成；此处仅做 Ehull 与比例统计。

    Args:
        structures: 弛豫后的结构（与 e_form 一一对应）
        e_form_per_atom_list: 形成能 (eV/atom)，与 Matbench Discovery 等基准兼容的
            定义（如 MP ref + MP2020 correction）
        hull_entries: Matbench Discovery 兼容的 convex hull 用 PDEntry 列表
        thr_stable: 稳定阈值
        thr_meta: 亚稳上界阈值

    Returns:
        compute_ehull_metrics 的返回值
    """
    comps = [s.composition for s in structures]
    return compute_ehull_metrics(
        comps,
        e_form_per_atom_list,
        hull_entries,
        thr_stable=thr_stable,
        thr_meta=thr_meta,
    )


def pipeline_summary_str(metrics: dict[str, Any]) -> str:
    """将 run_ehull_evaluation / compute_ehull_metrics 结果格式化为可读摘要。"""
    return (
        f"Ehull 热力学稳定性统计:\n"
        f"  稳定 (Ehull < {DEFAULT_EHULL_STABLE_THRESHOLD:.2f} eV/atom): "
        f"{metrics['n_stable']} / {metrics['n_total']} ({100 * metrics['stable_ratio']:.1f}%)\n"
        f"  亚稳 (0.02–{DEFAULT_EHULL_METASTABLE_THRESHOLD:.2f} eV/atom): "
        f"{metrics['n_metastable']} / {metrics['n_total']} ({100 * metrics['metastable_ratio']:.1f}%)\n"
        f"  稳定+亚稳: {100 * metrics['stable_or_metastable_ratio']:.1f}%\n"
        f"  无效/不稳定: {metrics['n_unstable']} (valid total: {metrics['n_valid']})"
    )


def get_mp_ref_energies_if_available() -> dict[str, float] | None:
    """若已安装 matbench-discovery，返回 MP 元素参考能量 (eV/atom)，用于形成能计算。"""
    try:
        from matbench_discovery.energy import mp_elemental_ref_energies
        return dict(mp_elemental_ref_energies)
    except Exception:
        return None


# 扩展元素参考能量 (eV/atom)，覆盖 relixue 等数据集中常见元素，MP 兼容近似值
# 用于形成能 E_f = (E_total - Σ n_i μ_i) / Σ n_i；无 matbench-discovery 时的后备
EXTENDED_REF_ENERGIES: dict[str, float] = {
    "H": -3.37, "He": 0.0, "Li": -1.90, "Be": -3.75, "B": -6.78, "C": -9.22, "N": -8.29,
    "O": -4.94, "F": -1.59, "Ne": 0.0, "Na": -1.12, "Mg": -1.54, "Al": -3.74, "Si": -5.43,
    "P": -5.41, "S": -4.13, "Cl": -1.60, "Ar": 0.0, "K": -1.08, "Ca": -1.97, "Sc": -6.54,
    "Ti": -7.91, "V": -8.89, "Cr": -9.46, "Mn": -9.03, "Fe": -8.29, "Co": -7.06, "Ni": -5.85,
    "Cu": -3.75, "Zn": -1.35, "Ga": -3.00, "Ge": -4.63, "As": -4.73, "Se": -3.53,
    "Br": -1.22, "Kr": 0.0, "Rb": -0.94, "Sr": -1.69, "Y": -6.51, "Zr": -8.89,
    "Nb": -10.08, "Mo": -10.83, "Tc": -10.37, "Ru": -8.60, "Rh": -5.75, "Pd": -3.94,
    "Ag": -1.17, "Cd": -0.82, "In": -2.52, "Sn": -3.99, "Sb": -3.98, "Te": -3.12, "I": -1.07,
    "Xe": 0.0, "Cs": -0.78, "Ba": -1.90, "La": -4.47, "Ce": -5.47, "Pr": -5.22, "Nd": -5.19,
    "Pm": -4.97, "Sm": -4.90, "Eu": -1.84, "Gd": -4.00, "Tb": -4.05, "Dy": -4.05,
    "Ho": -4.08, "Er": -4.11, "Tm": -4.14, "Yb": -1.66, "Lu": -4.37,
    "Hf": -9.02, "Ta": -11.15, "W": -12.95, "Re": -8.03, "Os": -8.17, "Ir": -6.93,
    "Pt": -5.85, "Au": -1.29, "Hg": -0.35, "Tl": -1.83, "Pb": -3.75, "Bi": -4.14,
    "Po": -3.10, "At": 0.0, "Rn": 0.0, "Fr": 0.0, "Ra": 0.0,
    "Ac": -5.17, "Th": -6.30, "Pa": -5.30, "U": -5.89, "Np": -6.27, "Pu": -5.96,
}


def get_extended_ref_energies() -> dict[str, float]:
    """获取扩展元素参考能量：优先 matbench-discovery MP 数据，否则用 EXTENDED_REF_ENERGIES。"""
    mp = get_mp_ref_energies_if_available()
    base = dict(EXTENDED_REF_ENERGIES)
    if mp:
        base.update(mp)
    return base


def get_ehull_subspace(
    composition: Composition,
    e_form_per_atom: float,
) -> tuple[float, dict[Composition, float] | None]:
    """对给定成分，仅用其包含的元素构建凸包计算 Ehull，避免高维凸包失败。

    适用于成分种类很多（如 60+ 元素）时，全空间 PhaseDiagram 易失败。
    此处 hull 仅为终端元素，Ehull 表示相对单质形成能的稳定性。

    当化合物比单质更稳定（E_form < 0）时，pymatgen 会抛出 ValueError
    "No valid decomposition found"，或返回 nan。此时回退为 Ehull = max(0, E_form)，
    即负形成能视为相对单质稳定（Ehull=0）。

    Returns:
        (ehull eV/atom, decomp 或 None)
    """
    e_form = float(e_form_per_atom)
    # 仅当输入形成能为 nan 时才允许返回 nan
    if not np.isfinite(e_form):
        return (float("nan"), None)
    fallback_ehull = max(0.0, e_form)

    try:
        elements = list(composition.elements)
        hull_entries = [PDEntry(Composition({el.symbol: 1}), 0.0) for el in elements]
        ehull, decomp = get_ehull_and_decomp(composition, e_form_per_atom, hull_entries)
        if np.isnan(ehull) or not np.isfinite(ehull):
            ehull = fallback_ehull
        return (ehull, decomp)
    except Exception:
        return (fallback_ehull, None)


def compute_ehull_metrics_subspace(
    compositions: list[Composition],
    e_form_per_atom_list: list[float],
    *,
    thr_stable: float = DEFAULT_EHULL_STABLE_THRESHOLD,
    thr_meta: float = DEFAULT_EHULL_METASTABLE_THRESHOLD,
) -> dict[str, Any]:
    """用成分子空间凸包计算 Ehull 及稳定/亚稳比例，适用于多元素数据集（避免高维 hull 失败）。"""
    assert len(compositions) == len(e_form_per_atom_list)
    ehulls = []
    stable_mask = []
    metastable_mask = []
    for comp, e_form in zip(compositions, e_form_per_atom_list):
        ehull, _ = get_ehull_subspace(comp, e_form)
        ehulls.append(ehull)
        valid = not np.isnan(ehull)
        stable_mask.append(valid and ehull < thr_stable)
        metastable_mask.append(valid and thr_stable <= ehull < thr_meta)

    ehulls = np.array(ehulls, dtype=float)
    stable_mask = np.array(stable_mask)
    metastable_mask = np.array(metastable_mask)
    n_valid = int(np.sum(~np.isnan(ehulls)))
    n_stable = int(np.sum(stable_mask))
    n_metastable = int(np.sum(metastable_mask))
    n_tot = len(ehulls)
    n_unstable = n_valid - n_stable - n_metastable

    return {
        "ehull": ehulls,
        "stable_mask": stable_mask,
        "metastable_mask": metastable_mask,
        "stable_ratio": n_stable / n_tot if n_tot else 0.0,
        "metastable_ratio": n_metastable / n_tot if n_tot else 0.0,
        "stable_or_metastable_ratio": (n_stable + n_metastable) / n_tot if n_tot else 0.0,
        "n_stable": n_stable,
        "n_metastable": n_metastable,
        "n_unstable": n_unstable,
        "n_valid": n_valid,
        "n_total": n_tot,
    }
