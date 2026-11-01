"""Unweighted event-level context and correction checks for exclusive rho MC.

Run on immutable native Parquet exports; select complete events, never rows.
No FM event kernel is implied: reconstructed plots compare COATJAVA momenta
with Alex's XGBoost postprocessing for exactly the same selected events.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import duckdb
import numpy as np


BEAM_ENERGY_GEV = 10.6041
ELECTRON_MASS_GEV = 0.00051099895
PROTON_MASS_GEV = 0.93827209
PION_MASS_GEV = 0.13957039


def event_table(path: Path, max_events: int, seed: int, reconstructed: bool):
    """Compute four-vectors for a deterministic complete-event subsample.

    With metric (+---), E_i=sqrt(|p_i|^2+m_i^2) and
    M_pipi^2=(E_pi+ + E_pi-)^2-|p_pi+ + p_pi-|^2.
    Q_miss=P_beam+P_target-sum_i P_i, M_X^2=Q_miss^2.
    Truth species assign all masses, retaining reconstructed PID mistakes.
    """
    con = duckdb.connect()
    try:
        con.execute("SET threads=4")
        con.execute("SET memory_limit='2GB'")
        con.read_parquet(str(path)).create_view("src")
        columns = {row[0] for row in con.execute("DESCRIBE src").fetchall()}
        if reconstructed and not {"p_rec_uncorrected", "px_rec_uncorrected", "py_rec_uncorrected", "pz_rec_uncorrected"} <= columns:
            raise ValueError("Rho correction comparison requires all raw COATJAVA momentum backups")
        con.execute(f"""
            CREATE TEMP TABLE event_keys AS
            SELECT file_id,event FROM src WHERE pid_gen=11
            GROUP BY file_id,event
            ORDER BY hash(file_id,event,{int(seed)}) LIMIT {int(max_events)}
        """)
        con.execute(f"""
            CREATE TEMP TABLE sampled AS
            SELECT p.*, CASE pid_gen WHEN 11 THEN {ELECTRON_MASS_GEV}
                WHEN 2212 THEN {PROTON_MASS_GEV} ELSE {PION_MASS_GEV} END AS mass
            FROM src p JOIN event_keys USING (file_id,event)
        """)
        terms = [
            "file_id", "event", "count(*) AS n_particles", "count(DISTINCT pid_gen) AS n_species",
            "bool_and(pid_gen IN (11,-211,211,2212)) AS known_species",
            "bool_or(pid_gen=11 AND is_trigger_particle=1 AND detector=1) AS electron_fd_trigger",
        ]
        modes = {"gen": ("px_gen", "py_gen", "pz_gen")}
        if reconstructed:
            modes.update({"raw": ("px_rec_uncorrected", "py_rec_uncorrected", "pz_rec_uncorrected"),
                          "corrected": ("px_rec", "py_rec", "pz_rec")})
            terms.extend([
                "bool_and(reconstructed=1) AS all_reconstructed",
                "bool_and(abs(status_rec)>=2000 AND abs(status_rec)<4000) AS all_fd",
                "bool_and(isfinite(p_rec_uncorrected) AND p_rec_uncorrected>0 AND isfinite(p_rec) AND p_rec>0) AS positive_momenta",
                "bool_and(abs(CAST(p_rec_uncorrected AS DOUBLE)-p_gen)<=10) AS raw_tail_cut",
                "sum(CAST(was_corrected AS INT)) AS corrected_tracks",
            ])
        for mode, coordinates in modes.items():
            px, py, pz = [f"CAST({value} AS DOUBLE)" for value in coordinates]
            energy = f"sqrt({px}*{px}+{py}*{py}+{pz}*{pz}+mass*mass)"
            terms.extend([
                f"bool_and(isfinite({px}) AND isfinite({py}) AND isfinite({pz})) AS {mode}_finite",
                f"sum({energy}) AS {mode}_energy",
                f"sum({energy}) FILTER(WHERE abs(pid_gen)=211) AS {mode}_pion_energy",
            ])
            for axis, expression in zip(("px", "py", "pz"), (px, py, pz)):
                terms.extend([f"sum({expression}) AS {mode}_{axis}",
                    f"sum({expression}) FILTER(WHERE abs(pid_gen)=211) AS {mode}_pion_{axis}"])
        frame = con.execute("SELECT " + ",".join(terms) + " FROM sampled GROUP BY file_id,event").fetch_df()
        beam_p = np.sqrt(BEAM_ENERGY_GEV**2 - ELECTRON_MASS_GEV**2)
        for mode in modes:
            pion_mass2 = frame[f"{mode}_pion_energy"]**2 - sum(
                frame[f"{mode}_pion_{axis}"]**2 for axis in ("px", "py", "pz")
            )
            frame[f"{mode}_m_pipi"] = np.sqrt(np.maximum(0, pion_mass2))
            frame[f"{mode}_missing_mass2"] = (
                BEAM_ENERGY_GEV + PROTON_MASS_GEV - frame[f"{mode}_energy"]
            )**2 - frame[f"{mode}_px"]**2 - frame[f"{mode}_py"]**2 - (
                beam_p-frame[f"{mode}_pz"]
            )**2
        return frame
    finally:
        con.close()


def distribution_stats(name: str, variable: str, values: np.ndarray, edges: np.ndarray):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    stats = {"sample": name, "observable": variable, "n": len(values),
        "nonfinite": int(len(values)-len(finite)), "display_min": float(edges[0]),
        "display_max": float(edges[-1]), "underflow": int(np.count_nonzero(finite<edges[0])),
        "overflow": int(np.count_nonzero(finite>edges[-1]))}
    stats.update({key: float(value) for key, value in zip(
        ("q16", "median", "q84"), np.quantile(finite, [.16, .5, .84]))})
    stats.update(mean=float(finite.mean()), std=float(finite.std()))
    return stats


def draw_distribution(ax, values, edges, label, color, marker=None):
    """Normalize by all finite entries, including displayed under/overflow.

    Error bars use marginal binomial variance n_k(1-n_k/N)/N^2/dx^2.
    Covariances between normalized bins are not drawn.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    density = counts / (len(values)*widths)
    errors = np.sqrt(counts*(1-counts/len(values))) / (len(values)*widths)
    if marker:
        centers = (edges[1:]+edges[:-1])/2
        ax.errorbar(centers, density, yerr=errors, color=color, marker=marker,
                    markersize=3, linestyle="none", linewidth=.8, label=label)
    else:
        ax.stairs(density, edges, color=color, linewidth=1.5, label=label)
        ax.fill_between(edges, np.r_[density-errors,(density-errors)[-1]],
                        np.r_[density+errors,(density+errors)[-1]],
                        step="post", color=color, alpha=.15, linewidth=0)


def note_outside_window(ax, samples, edges):
    fractions = []
    for label, values in samples:
        values = np.asarray(values, dtype=float)
        finite = values[np.isfinite(values)]
        outside = np.count_nonzero((finite<edges[0]) | (finite>edges[-1])) / len(finite)
        fractions.append((label, outside))
    if any(fraction>0 for _, fraction in fractions):
        text = "Outside window: " + "; ".join(f"{label} {fraction:.1%}" for label, fraction in fractions)
        ax.text(.03, .03, text, transform=ax.transAxes, fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho", type=Path, required=True)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-events", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--no-plots", action="store_true", help="Write numerical outputs without Matplotlib")
    args = parser.parse_args()
    if args.max_events < 1:
        parser.error("--max-events must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rho = event_table(args.rho, args.max_events, args.seed, reconstructed=True)
    phase = event_table(args.phase, args.max_events, args.seed, reconstructed=False)
    tables = {}
    for name, frame in (("rho", rho), ("phase", phase)):
        selected = frame.n_particles.eq(4) & frame.n_species.eq(4) & frame.known_species & frame.gen_finite
        tables[name] = frame.loc[selected].copy()
    valid = rho.n_particles.eq(4) & rho.n_species.eq(4) & rho.known_species & rho.gen_finite
    cutflow = {"sampled_events": len(rho), "complete_generated_topology": int(valid.sum())}
    for column in ("electron_fd_trigger", "all_reconstructed", "all_fd", "positive_momenta",
                   "raw_finite", "corrected_finite", "raw_tail_cut"):
        valid &= rho[column].fillna(False)
        cutflow[column] = int(valid.sum())
    common = rho.loc[valid].copy()
    if common.empty:
        raise ValueError("No complete reconstructed rho events survive the documented comparison selection")

    beam_p = np.sqrt(BEAM_ENERGY_GEV**2-ELECTRON_MASS_GEV**2)
    context = [
        ("m_pipi", np.linspace(0, 6, 121), r"$M_{\pi^+\pi^-}^{\rm gen}$ [GeV/$c^2$]", "gen_m_pipi"),
        ("sum_px", np.linspace(-8, 8, 129), r"$\sum_i p_{x,i}^{\rm gen}$ [GeV/$c]", "gen_px"),
        ("sum_pz_minus_beam", np.linspace(-12, 20, 129), r"$\sum_i p_{z,i}^{\rm gen}-p_{\rm beam}$ [GeV/$c]", "gen_pz"),
    ]
    records = []
    for variable, edges, _, column in context:
        for name, frame in tables.items():
            values = frame[column].to_numpy() - (beam_p if variable == "sum_pz_minus_beam" else 0)
            records.append(distribution_stats(name, variable, values, edges))
    correction_specs = [
        ("m_pipi", np.linspace(.2, 1.5, 79), r"$M_{\pi^+\pi^-}$ [GeV/$c^2$]"),
        ("missing_mass2", np.linspace(-.2, .2, 101), r"$M_X^2(ep\to e'p'\pi^+\pi^-X)$ [GeV$^2$/$c^4$]"),
    ]
    for variable, edges, _ in correction_specs:
        for mode in ("gen", "raw", "corrected"):
            records.append(distribution_stats(f"rho_common_{mode}", variable, common[f"{mode}_{variable}"].to_numpy(), edges))
    summary = {
        "study": "unweighted MC; generator weights are absent from native exports",
        "event_sample": {"max_events_per_source": args.max_events, "seed": args.seed,
                         "selection": "1 generated-electron event key, ordered by DuckDB hash(file_id,event,seed); keep every particle row"},
        "input_paths": {"rho": str(args.rho), "phase": str(args.phase)},
        "beam_energy_gev": BEAM_ENERGY_GEV,
        "mass_assignment": "generated species; no reconstructed PID agreement requirement",
        "correction_selection": "T_e=1; exactly e-,pi-,pi+,p; every track reciprocally reconstructed in FD; finite positive raw/corrected p; finite vectors; |p_raw-p_gen|<=10 GeV/c for every track",
        "correction_cutflow": cutflow,
        "common_events_with_any_correction": int((common.corrected_tracks>0).sum()),
        "common_corrected_particle_rows": int(common.corrected_tracks.sum()),
        "statistics": records,
        "interpretation": "These are generator-context and COATJAVA/XGBoost checks, not an FM-generated exclusive-event closure or a cross-section prediction.",
    }
    (args.output/"physics_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    with (args.output/"physics_statistics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    if args.no_plots:
        print(json.dumps({"cutflow": cutflow, "output": str(args.output)}, indent=2))
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 9,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True,
                         "axes.grid": False, "savefig.dpi": 180})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    for ax, (variable, edges, xlabel, column) in zip(axes, context):
        window_samples = []
        for name, color, label in (("rho", "#b2182b", r"Exclusive $\rho^0$"),
                                   ("phase", "#2166ac", "Phase space (Sep)")):
            values = tables[name][column].to_numpy() - (beam_p if variable=="sum_pz_minus_beam" else 0)
            draw_distribution(ax, values, edges, label, color)
            window_samples.append((name, values))
        ax.set(xlabel=xlabel, ylabel="Normalized entries / bin width", yscale="log")
        ax.legend(frameon=False)
        note_outside_window(ax, window_samples, edges)
    fig.suptitle("Generated event kinematics — unweighted MC", fontsize=14)
    for extension in ("png", "pdf"):
        fig.savefig(args.output/f"generated_event_context.{extension}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    for ax, (variable, edges, xlabel) in zip(axes, correction_specs):
        if variable=="m_pipi":
            draw_distribution(ax, common[f"gen_{variable}"], edges, "Generated (same events)", "#555555")
        else:
            ax.axvline(0, color="#555555", ls="--", lw=1, label="Exclusive expectation")
        draw_distribution(ax, common[f"raw_{variable}"], edges, "COATJAVA raw", "#2166ac")
        draw_distribution(ax, common[f"corrected_{variable}"], edges, "+ XGBoost correction", "#b2182b", marker="o")
        ax.set(xlabel=xlabel, ylabel="Normalized entries / bin width")
        ax.legend(frameon=False)
        note_outside_window(ax, [("raw", common[f"raw_{variable}"]),
                                ("corrected", common[f"corrected_{variable}"])], edges)
    fig.suptitle(f"Common reciprocal FD events: {len(common):,} — unweighted MC", fontsize=13)
    for extension in ("png", "pdf"):
        fig.savefig(args.output/f"exclusive_event_correction.{extension}")
    plt.close(fig)
    print(json.dumps({"cutflow": cutflow, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
