"""Physics-aware construction of the conditional Forward Detector response view.

For one generated hadron the truth state is

    x = (p_gen, theta_gen, phi_gen, s_gen),

and, for a successfully reconstructed FD particle, the response target is

    Delta = (Delta p, Delta theta, Delta phi, Delta beta)
          = (p_rec - p_gen,
             theta_rec - theta_gen,
             wrap(phi_rec - phi_gen)).

This module constructs samples for only the conditional factor

    P(Delta, s_rec | x, T=1, C=FD, fiducial selection),

where T is the event trigger-electron proxy and C is the reconstruction-region
outcome.  The FD-cuts input has conditioned away T=0 and C!=FD, so this module
must never be used to estimate trigger or reconstruction efficiency.
"""

from __future__ import annotations

import glob
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


CONTINUOUS_FEATURES = (
    "log1p_gen_p",
    "gen_theta",
    "sin_gen_phi",
    "cos_gen_phi",
    "beta_gen",
)


TARGET_COLUMNS = ("delta_p", "delta_theta", "delta_phi", "delta_beta")
DELTA_P_ASINH_SCALE_GEV = 0.020
TARGET_TRANSFORM = {"name": "asinh_delta_p", "scale_gev": DELTA_P_ASINH_SCALE_GEV}
SPECIES = (-211, 211, 2212)
GEN_MASS_GEV = {
    -211: 0.13957039,   # pi-
     211: 0.13957039,   # pi+
    2212: 0.93827209,   # proton
}
REQUIRED_COLUMNS = {
    "source_file_id",
    "event_id",
    "mcindex",
    "gen_pid",
    "gen_p",
    "gen_theta",
    "gen_phi",
    "gen_vz",
    "rec_theta",
    "rec_pid",
    "rec_beta",
    "rec_detector_region",
    "match_reciprocal",
    "usable_for_hadron_response_training",
    "delta_p",
    "delta_theta",
    "delta_phi",
}


@dataclass(frozen=True)
class Standardizer:
    """Training-only affine coordinates z_j=(x_j-mu_j)/sigma_j.

    Scaling is a numerical coordinate change, not a physics correction.  The
    inverse map restores GeV/radian units before any closure statement.
    """
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = np.asarray(values, dtype=np.float64).mean(axis=0)
        scale = np.asarray(values, dtype=np.float64).std(axis=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean

    def as_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, list[float]]) -> "Standardizer":
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float32),
            scale=np.asarray(data["scale"], dtype=np.float32),
        )


def transform_targets_for_model(values: np.ndarray) -> np.ndarray:
    """Transform physical response targets into model coordinates."""
    transformed = np.asarray(values, dtype=np.float32).copy()
    transformed[:, 0] = np.arcsinh(
        transformed[:, 0] / DELTA_P_ASINH_SCALE_GEV
    )
    return transformed


def inverse_targets_from_model(values: np.ndarray) -> np.ndarray:
    """Undo the nonlinear delta-p transformation."""
    physical = np.asarray(values, dtype=np.float32).copy()
    physical[:, 0] = (
        DELTA_P_ASINH_SCALE_GEV * np.sinh(physical[:, 0])
    )
    return physical


@dataclass
class PreparedSplit:
    name: str
    event_keys: np.ndarray
    continuous: np.ndarray
    species_index: np.ndarray
    targets: np.ndarray
    rec_pid_index: np.ndarray
    raw_species: np.ndarray
    feature_names: tuple[str, ...] = CONTINUOUS_FEATURES

    def __len__(self) -> int:
        return len(self.targets)


def _quote_sql_string(value: str) -> str:
    return value.replace("'", "''")


def connect(parquet_glob: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    quoted = _quote_sql_string(parquet_glob)
    con.execute(f"CREATE OR REPLACE VIEW particles AS SELECT * FROM '{quoted}'")
    return con


def assert_schema(con: duckdb.DuckDBPyConnection) -> list[str]:
    columns = [row[0] for row in con.execute("DESCRIBE particles").fetchall()]
    missing = sorted(REQUIRED_COLUMNS.difference(columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return columns


def fiducial_sql(config: dict[str, Any]) -> str:
    """Encode the initial reconstructed-FD fiducial phase-space definition.

    In probability notation this restricts the response density to

        C = FD,  theta_rec < 33 degrees,
        -5.5 cm < z_vertex,gen < -0.5 cm,  T = 1.

    `usable_for_hadron_response_training` supplies the T=1 hadron conditioning.
    These are not efficiency cuts: failures are absent from this conditional
    response sample and belong in the later P(T|x_e) and P(C|x,T) heads.
    """
    selection = config["data"]["selection"]
    terms = [
        "rec_detector_region = 'FD'",
        f"rec_theta < radians({float(selection['theta_max_deg'])})",
        f"gen_vz > {float(selection['vz_min_cm'])}",
        f"gen_vz < {float(selection['vz_max_cm'])}",
        "usable_for_hadron_response_training",
        "gen_pid IN (-211, 211, 2212)",
        "delta_p IS NOT NULL AND delta_theta IS NOT NULL AND delta_phi IS NOT NULL",
        "isfinite(delta_p) AND isfinite(delta_theta) AND isfinite(delta_phi)",
    ]
    return "\n      AND ".join(terms)


def selection_sql(config: dict[str, Any]) -> str:
    """Add an explicit residual-density quality policy to the fiducial view.

    Reciprocal matching approximates a well-defined truth<->reconstruction
    pairing.  Sentinel and |Delta p| rules protect likelihood normalization
    from known pathological records.  They are recorded as modeling choices;
    they do not alter the immutable Parquet teacher sample.
    """
    selection = config["data"]["selection"]
    terms = [fiducial_sql(config)]
    if selection["require_reciprocal_match"]:
        terms.append("match_reciprocal")
    if selection["reject_rec_pid_zero"]:
        terms.append("rec_pid <> 0")
    if selection["reject_beta_sentinel"]:
        terms.append("rec_beta > -99")
    max_abs_dp = selection.get("max_abs_delta_p_gev")
    if max_abs_dp is not None:
        terms.append(f"abs(delta_p) <= {float(max_abs_dp)}")
    return "\n      AND ".join(terms)


def split_predicate(split: str, config: dict[str, Any]) -> str:
    """Split by the true event E=(source_file_id,event_id), never by row.

    All particles from the same physical Monte Carlo event therefore share a
    partition.  This enforces E_train intersection E_test = empty and prevents
    correlated four-particle event information from leaking into validation.
    """
    data = config["data"]
    modulus = int(data["split_modulus"])
    train_boundary = int(data["train_boundary"])
    validation_boundary = int(data["validation_boundary"])
    seed = int(data.get("split_seed", config["project"]["seed"]))
    bucket = f"hash(source_file_id, event_id, {seed}) % {modulus}"
    if split == "train":
        return f"{bucket} < {train_boundary}"
    if split == "validation":
        return f"{bucket} >= {train_boundary} AND {bucket} < {validation_boundary}"
    if split == "test":
        return f"{bucket} >= {validation_boundary}"
    raise ValueError(f"Unknown split: {split}")


def _load_frame(
    con: duckdb.DuckDBPyConnection,
    split: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    configured_limit = config["data"]["max_rows_per_species"][split]
    limit = int(configured_limit) if configured_limit is not None else None
    seed = int(config["data"].get("order_seed", config["data"].get("split_seed", config["project"]["seed"])))
    # `null` means no development subsampling: retain the complete physical
    # population in this event partition. Keeping this separate from a very
    # large Top-N also avoids DuckDB's finite arg_min/arg_max N limit.
    qualify = ""
    if limit is not None:
        qualify = f"""
    QUALIFY row_number() OVER (
      PARTITION BY gen_pid
      ORDER BY hash(source_file_id, event_id, mcindex, {seed + 17})
    ) <= {limit}
        """
    query = f"""
    SELECT source_file_id, event_id, mcindex, gen_pid,
           gen_p, gen_theta, gen_phi, rec_beta,
           delta_p, delta_theta, delta_phi, rec_pid
    FROM particles
    WHERE {selection_sql(config)}
      AND {split_predicate(split, config)}
    {qualify}
    ORDER BY gen_pid, hash(source_file_id, event_id, mcindex, {seed + 29})
    """
    frame = con.execute(query).fetch_df()

    frame["delta_beta"] = frame["rec_beta"] - (
       frame["gen_p"] / np.sqrt(
        frame["gen_p"]**2
        + frame["gen_pid"].map(GEN_MASS_GEV)**2
      )
    )

    if frame.empty:
        raise ValueError(f"Selection produced no rows for split {split!r}")
    return frame


def continuous_feature_names(config: dict[str, Any]) -> tuple[str, ...]:
    """Append raw p_gen/GeV to the original 5 coordinates when requested."""
    if config["data"].get("include_raw_generated_momentum", False):
        return (*CONTINUOUS_FEATURES, "gen_p")
    return CONTINUOUS_FEATURES


def build_features(
    frame: pd.DataFrame, feature_names: tuple[str, ...] = CONTINUOUS_FEATURES
) -> np.ndarray:
    """Map truth kinematics to numerically smooth model coordinates.

    Momentum uses log(1+p/GeV) to compress its dynamic range. Azimuth lives on
    the circle S^1, so (sin(phi), cos(phi)) makes phi=-pi and phi=+pi adjacent.

    beta_gen is calculated from generated momentum and generated particle mass.

    Generated species s_gen enters separately through a learned embedding.
    """
    p = frame["gen_p"].to_numpy(dtype=np.float64)
    phi = frame["gen_phi"].to_numpy(dtype=np.float64)
    pid = frame["gen_pid"].to_numpy()

    if not np.all(np.isfinite(p) & (p > 0)):
        raise ValueError("Generated momentum must be finite and positive")
    if not np.isin(pid, SPECIES).all():
        raise ValueError(f"Supported generated PIDs are {SPECIES}")
    if feature_names not in (CONTINUOUS_FEATURES, (*CONTINUOUS_FEATURES, "gen_p")):
        raise ValueError(f"Unsupported feature order: {feature_names}")
    mass = np.array(
        [GEN_MASS_GEV[int(x)] for x in pid],
        dtype=np.float64,
    )

    beta_gen = p / np.sqrt(p**2 + mass**2)

    values = np.column_stack(
        [
            np.log1p(p),
            frame["gen_theta"].to_numpy(dtype=np.float64),
            np.sin(phi),
            np.cos(phi),
            beta_gen,
        ]
    )

    if "gen_p" in feature_names:
        values = np.column_stack([values, p])
    if not np.isfinite(values).all():
        raise ValueError("Generated features must be finite")
    return values.astype(np.float32)


def _event_keys(frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [f"{source}:{event}" for source, event in zip(frame.source_file_id, frame.event_id)],
        dtype=object,
    )


def discover_rec_pid_vocabulary(frame: pd.DataFrame) -> list[int]:
    return sorted(int(value) for value in frame["rec_pid"].unique())


def prepare_split(
    frame: pd.DataFrame,
    split: str,
    feature_scaler: Standardizer,
    target_scaler: Standardizer,
    rec_pid_vocabulary: list[int],
    feature_names: tuple[str, ...] = CONTINUOUS_FEATURES,
) -> PreparedSplit:
    """Encode x, Delta, generated species, and reconstructed PID labels.

    `rec_pid_index` represents the categorical stochastic outcome

        P(s_rec | x, T=1, C=FD),

    so a reconstructed PID different from `gen_pid` is retained as detector
    contamination rather than treated as corrupt data.
    """
    species_to_index = {pid: index for index, pid in enumerate(SPECIES)}
    rec_pid_to_index = {pid: index for index, pid in enumerate(rec_pid_vocabulary)}
    unknown_index = len(rec_pid_vocabulary)
    raw_species = frame["gen_pid"].to_numpy(dtype=np.int64)
    species_index = np.asarray([species_to_index[int(pid)] for pid in raw_species], dtype=np.int64)
    raw_targets = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32)
    model_targets = transform_targets_for_model(raw_targets)

    rec_pid_index = np.asarray(
        [rec_pid_to_index.get(int(pid), unknown_index) for pid in frame["rec_pid"]],
        dtype=np.int64,
    )
    return PreparedSplit(
        name=split,
        event_keys=_event_keys(frame),
        continuous=feature_scaler.transform(build_features(frame, feature_names)),
        species_index=species_index,
        targets=target_scaler.transform(model_targets),
        rec_pid_index=rec_pid_index,
        raw_species=raw_species,
        feature_names=feature_names,
    )


def load_all_splits(
    config: dict[str, Any],
) -> tuple[dict[str, PreparedSplit], Standardizer, Standardizer, list[int], dict[str, Any]]:
    parquet_glob = config["data"]["parquet_glob"]
    con = connect(parquet_glob)
    columns = assert_schema(con)
    frames = {name: _load_frame(con, name, config) for name in ("train", "validation", "test")}

    feature_names = continuous_feature_names(config)
    train_features = build_features(frames["train"], feature_names)
    train_targets = frames["train"].loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32)
    train_model_targets = transform_targets_for_model(train_targets)
    if any(not np.isfinite(frame.loc[:, TARGET_COLUMNS].to_numpy()).all() for frame in frames.values()):
        raise ValueError("The selected response targets contain nonfinite values")
    feature_scaler = Standardizer.fit(train_features)
    target_scaler = Standardizer.fit(train_model_targets)
    rec_pid_vocabulary = discover_rec_pid_vocabulary(frames["train"])
    splits = {
        name: prepare_split(frame, name, feature_scaler, target_scaler, rec_pid_vocabulary, feature_names)
        for name, frame in frames.items()
    }
    assert_event_disjoint(splits)
    audit = build_audit(con, frames, columns, config)
    audit.update({
        "feature_names": list(feature_names),
        "target_names": list(TARGET_COLUMNS),
        "target_transform": TARGET_TRANSFORM,
        "split_seed": int(config["data"].get("split_seed", config["project"]["seed"])),
        "order_seed": int(config["data"].get("order_seed", config["data"].get("split_seed", config["project"]["seed"]))),
        "particle_order_sha256": {
            name: hashlib.sha256(pd.util.hash_pandas_object(
                frame[["source_file_id", "event_id", "mcindex", "gen_pid"]], index=False
            ).to_numpy().tobytes()).hexdigest()
            for name, frame in frames.items()
        },
    })
    con.close()
    return splits, feature_scaler, target_scaler, rec_pid_vocabulary, audit


def assert_event_disjoint(splits: dict[str, PreparedSplit]) -> None:
    """Require E_train intersect E_val = E_train intersect E_test = empty."""
    key_sets = {name: set(split.event_keys.tolist()) for name, split in splits.items()}
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for left, right in pairs:
        overlap = key_sets[left].intersection(key_sets[right])
        if overlap:
            example = next(iter(overlap))
            raise AssertionError(f"Event leakage between {left} and {right}; example {example}")


def build_audit(
    con: duckdb.DuckDBPyConnection,
    frames: dict[str, pd.DataFrame],
    columns: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    selected = selection_sql(config)
    fiducial = fiducial_sql(config)
    counts = con.execute(
        f"""
        SELECT gen_pid,
               count(*) AS rows,
               count(DISTINCT (source_file_id, event_id)) AS events,
               count(*) FILTER (WHERE abs(delta_p) > 10) AS abs_dp_gt_10,
               min(delta_phi) AS min_delta_phi,
               max(delta_phi) AS max_delta_phi
        FROM particles
        WHERE {selected}
        GROUP BY gen_pid ORDER BY gen_pid
        """
    ).fetch_df()
    duplicate_rows = con.execute(
        f"""
        SELECT count(*) FROM (
          SELECT source_file_id, event_id, mcindex, count(*) AS n
          FROM particles WHERE {selected}
          GROUP BY 1, 2, 3 HAVING count(*) > 1
        )
        """
    ).fetchone()[0]
    phi_violation_count = con.execute(
        f"SELECT count(*) FROM particles WHERE {selected} AND NOT (delta_phi >= -pi() AND delta_phi <= pi())"
    ).fetchone()[0]
    quality_cutflow = con.execute(
        f"""
        SELECT gen_pid,
               count(*) AS fiducial_rows,
               count(*) FILTER (WHERE NOT match_reciprocal) AS non_reciprocal,
               count(*) FILTER (WHERE rec_pid = 0 OR rec_beta <= -99) AS pid_or_beta_sentinel,
               count(*) FILTER (WHERE abs(delta_p) > 10) AS abs_dp_gt_10,
               count(*) FILTER (WHERE {selected}) AS final_quality_rows
        FROM particles
        WHERE {fiducial}
        GROUP BY gen_pid ORDER BY gen_pid
        """
    ).fetch_df()

    files = sorted(glob.glob(config["data"]["parquet_glob"]))
    file_records = [
        {"name": Path(path).name, "bytes": Path(path).stat().st_size}
        for path in files
    ]
    fingerprint_payload = json.dumps(file_records, sort_keys=True).encode("utf-8")
    fingerprint = hashlib.sha256(fingerprint_payload).hexdigest()

    sampled_counts: dict[str, dict[str, int]] = {}
    for split, frame in frames.items():
        sampled_counts[split] = {
            str(pid): int(count)
            for pid, count in frame.groupby("gen_pid").size().items()
        }

    return {
        "dataset_glob": config["data"]["parquet_glob"],
        "dataset_file_count": len(files),
        "dataset_total_bytes": sum(item["bytes"] for item in file_records),
        "dataset_metadata_sha256": fingerprint,
        "schema_column_count": len(columns),
        "schema_columns": columns,
        "fiducial_sql": fiducial,
        "selection_sql": selected,
        "selected_population": counts.to_dict(orient="records"),
        "quality_cutflow": quality_cutflow.to_dict(orient="records"),
        "sampled_counts": sampled_counts,
        "duplicate_composite_particle_keys": int(duplicate_rows),
        "delta_phi_range_violation_count": int(phi_violation_count),
        "event_split_overlap_count": 0,
    }
