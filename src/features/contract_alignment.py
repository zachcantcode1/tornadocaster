from __future__ import annotations

from typing import Iterable

import numpy as np
import re


def align_feature_matrix(
    feature_arrays_by_name: dict[str, np.ndarray],
    ordered_feature_names: Iterable[str],
    grid_size: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    """
    Build a deterministic feature matrix with columns ordered by ordered_feature_names.
    Missing features are zero-filled and returned for reporting.
    """
    ordered = list(ordered_feature_names)
    cols = []
    missing = []
    for name in ordered:
        arr = feature_arrays_by_name.get(name)
        if arr is None:
            cols.append(np.zeros((grid_size,), dtype=np.float32))
            missing.append(name)
            continue
        cols.append(arr.astype(np.float32).reshape(-1))
    if not cols:
        return np.zeros((grid_size, 0), dtype=np.float32), ordered, missing
    matrix = np.stack(cols, axis=1)
    return matrix, ordered, missing


def build_feature_coverage_summary(
    implemented_feature_names: Iterable[str],
    ordered_feature_names: Iterable[str],
    missing_feature_names: Iterable[str],
) -> dict:
    implemented = list(implemented_feature_names)
    ordered = list(ordered_feature_names)
    missing = list(missing_feature_names)
    total = len(ordered)
    implemented_count = total - len(missing)
    pct = (implemented_count / total) if total else 0.0
    return {
        "total_features": total,
        "implemented_features": implemented_count,
        "missing_features": len(missing),
        "coverage_fraction": pct,
        "implemented_feature_names": implemented,
        "missing_feature_names_sample": missing[:50],
    }


def add_first_pass_legacy_aliases(
    feature_arrays_by_name: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Add pragmatic first-pass mappings from current Python feature names to
    legacy Nadocast contract names. This increases contract coverage while full
    parity math is still being implemented.
    """
    out = dict(feature_arrays_by_name)
    alias_map = {
        "CAPE:surface:hour fcst:wt ens mean": "raw",
        "CAPE:surface:hour fcst:wt ens mean:25mi mean": "25mi_mean",
        "CAPE:surface:hour fcst:wt ens mean:50mi mean": "50mi_mean",
        "CAPE:surface:hour fcst:wt ens mean:100mi mean": "100mi_mean",
        # Directional proxy mapping until true legacy kernels are ported.
        "CAPE:surface:hour fcst:wt ens mean:50mi forward grad": "50mi_gradient_x",
        "CAPE:surface:hour fcst:wt ens mean:50mi leftward grad": "50mi_gradient_y",
        "CAPE:surface:hour fcst:wt ens mean:100mi forward grad": "100mi_gradient_x",
        "CAPE:surface:hour fcst:wt ens mean:100mi leftward grad": "100mi_gradient_y",
        "CAPE:surface:hour fcst:wt ens mean:100mi linestraddling grad": "100mi_gradient_mag",
    }
    for legacy_name, internal_name in alias_map.items():
        arr = feature_arrays_by_name.get(internal_name)
        if arr is not None:
            out[legacy_name] = arr
    # Multi-field aliases: map <field>__<block> to first-pass legacy contract names.
    field_base_map = {
        "cape_surface": "CAPE:surface:hour fcst:wt ens mean",
        "cape_ml": "CAPE:90-0 mb above ground:hour fcst:wt ens mean",
        "cape_mu": "CAPE:180-0 mb above ground:hour fcst:wt ens mean",
        "cin_surface": "CIN:surface:hour fcst:wt ens mean",
        "cin_ml": "CIN:90-0 mb above ground:hour fcst:wt ens mean",
        "cin_mu": "CIN:180-0 mb above ground:hour fcst:wt ens mean",
    }
    block_suffix_map = {
        "raw": "",
        "25mi_mean": ":25mi mean",
        "50mi_mean": ":50mi mean",
        "100mi_mean": ":100mi mean",
        "50mi_gradient_x": ":50mi forward grad",
        "50mi_gradient_y": ":50mi leftward grad",
        "100mi_gradient_x": ":100mi forward grad",
        "100mi_gradient_y": ":100mi leftward grad",
        "100mi_gradient_mag": ":100mi linestraddling grad",
    }
    for field_name, base in field_base_map.items():
        for block, suffix in block_suffix_map.items():
            internal_name = f"{field_name}__{block}"
            arr = feature_arrays_by_name.get(internal_name)
            if arr is not None:
                out[f"{base}{suffix}"] = arr
    return out


def add_dynamic_legacy_aliases(
    feature_arrays_by_name: dict[str, np.ndarray],
    field_to_legacy_base: dict[str, str],
) -> dict[str, np.ndarray]:
    """
    Generic alias mapper:
      <field>__raw -> <legacy_base>
      <field>__25mi_mean -> <legacy_base>:25mi mean
      <field>__50mi_mean -> <legacy_base>:50mi mean
      <field>__100mi_mean -> <legacy_base>:100mi mean
      <field>__50mi_gradient_x -> <legacy_base>:50mi forward grad
      <field>__50mi_gradient_y -> <legacy_base>:50mi leftward grad
      <field>__100mi_gradient_x -> <legacy_base>:100mi forward grad
      <field>__100mi_gradient_y -> <legacy_base>:100mi leftward grad
      <field>__100mi_gradient_mag -> <legacy_base>:100mi linestraddling grad
    """
    out = dict(feature_arrays_by_name)
    block_suffix_map = {
        "raw": "",
        "25mi_mean": ":25mi mean",
        "50mi_mean": ":50mi mean",
        "100mi_mean": ":100mi mean",
        "50mi_gradient_x": ":50mi forward grad",
        "50mi_gradient_y": ":50mi leftward grad",
        "100mi_gradient_x": ":100mi forward grad",
        "100mi_gradient_y": ":100mi leftward grad",
        "100mi_gradient_mag": ":100mi linestraddling grad",
    }
    for field_name, legacy_base in field_to_legacy_base.items():
        for block, suffix in block_suffix_map.items():
            internal_key = f"{field_name}__{block}"
            arr = feature_arrays_by_name.get(internal_key)
            if arr is not None:
                out[f"{legacy_base}{suffix}"] = arr
    return out


def add_temporal_proxy_aliases(
    feature_arrays_by_name: dict[str, np.ndarray],
    ordered_feature_names: Iterable[str],
) -> dict[str, np.ndarray]:
    """
    Add first-pass temporal proxies for names that include forecast-time transforms.
    This is a temporary bridge until true multi-hour windows are implemented.
    """
    out = dict(feature_arrays_by_name)
    ordered = list(ordered_feature_names)

    temporal_suffixes = [
        ":-1hr",
        ":+1hr",
        ":3hr mean",
        ":3hr min",
        ":3hr max",
        ":3hr delta",
    ]

    def strip_once(name: str) -> str:
        for suffix in temporal_suffixes:
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return name

    for name in ordered:
        if name in out:
            continue
        candidate = name
        # Some names stack suffixes (e.g. "...:25mi mean:+1hr"), so strip repeatedly.
        for _ in range(3):
            stripped = strip_once(candidate)
            if stripped == candidate:
                break
            candidate = stripped
            if candidate in out:
                out[name] = out[candidate]
                break

    return out


def account_for_ordered_features(
    feature_arrays_by_name: dict[str, np.ndarray],
    ordered_feature_names: Iterable[str],
    grid_size: int,
) -> tuple[np.ndarray, list[str], dict]:
    """
    Ensure every ordered feature is accounted for by one of:
      - direct: exact key exists
      - proxy: heuristic mapping to an existing key
      - zero_fill: no heuristic match, fill zeros
    """
    ordered = list(ordered_feature_names)
    arrays = dict(feature_arrays_by_name)
    cols = []
    direct = []
    proxy = []
    zero_fill = []
    proxy_source = {}

    temporal_suffixes = [
        ":-1hr",
        ":+1hr",
        ":3hr mean",
        ":3hr min",
        ":3hr max",
        ":3hr delta",
    ]

    def temporal_strip(name: str) -> str:
        cur = name
        for _ in range(4):
            changed = False
            for suffix in temporal_suffixes:
                if cur.endswith(suffix):
                    cur = cur[: -len(suffix)]
                    changed = True
                    break
            if not changed:
                break
        return cur

    def candidate_sources(name: str) -> list[str]:
        out = []
        out.append(name)
        n = temporal_strip(name)
        out.append(n)
        # Probabilistic features often correlate with their "estimated from probs" counterpart.
        if ":prob " in n:
            out.append(re.sub(r":prob [^:]+", ":estimated from probs", n))
        if ":estimated from probs" in n:
            out.append(n.replace(":estimated from probs", ""))
        # Gradient fallbacks to nearby scales/components.
        out.append(n.replace(":100mi linestraddling grad", ":100mi mean"))
        out.append(n.replace(":100mi leftward grad", ":100mi mean"))
        out.append(n.replace(":100mi forward grad", ":100mi mean"))
        out.append(n.replace(":50mi leftward grad", ":50mi mean"))
        out.append(n.replace(":50mi forward grad", ":50mi mean"))
        out.append(n.replace(":25mi leftward grad", ":25mi mean"))
        out.append(n.replace(":25mi forward grad", ":25mi mean"))
        # Strip trailing colon artifact from calculated names.
        if n.endswith(":"):
            out.append(n[:-1])
        # De-duplicate while preserving order.
        seen = set()
        uniq = []
        for c in out:
            if c and c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq

    for name in ordered:
        if name in arrays:
            cols.append(arrays[name].astype(np.float32).reshape(-1))
            direct.append(name)
            continue

        mapped = None
        for c in candidate_sources(name):
            if c in arrays:
                mapped = c
                break
        if mapped is not None:
            cols.append(arrays[mapped].astype(np.float32).reshape(-1))
            proxy.append(name)
            proxy_source[name] = mapped
        else:
            cols.append(np.zeros((grid_size,), dtype=np.float32))
            zero_fill.append(name)

    matrix = np.stack(cols, axis=1) if cols else np.zeros((grid_size, 0), dtype=np.float32)
    accounting = {
        "total": len(ordered),
        "direct_count": len(direct),
        "proxy_count": len(proxy),
        "zero_fill_count": len(zero_fill),
        "direct_features": direct,
        "proxy_features_sample": proxy[:100],
        "proxy_source_sample": {k: proxy_source[k] for k in list(proxy_source.keys())[:100]},
        "zero_fill_features_sample": zero_fill[:100],
    }
    return matrix, ordered, accounting
