# Physics-informed beta-gen factorial study

All conditions use the same beta-valid teacher rows and event-disjoint split. The locked test checkpoint minimizes validation PID cross-entropy.

## Condition summary

| Condition | Macro TV | Correct-ID MAE | Test PID CE | Test PID accuracy |
|---|---:|---:|---:|---:|
| A_original | 0.03440 ± 0.00541 | 0.01851 ± 0.00405 | 1.00605 ± 0.00717 | 0.6768 ± 0.0005 |
| B_beta_input | 0.03413 ± 0.00445 | 0.01813 ± 0.00279 | 1.00572 ± 0.00623 | 0.6769 ± 0.0004 |
| C_beta_target | 0.03509 ± 0.00682 | 0.01880 ± 0.00307 | 1.00551 ± 0.00853 | 0.6770 ± 0.0005 |
| D_input_target_pid02 | 0.03593 ± 0.00738 | 0.01941 ± 0.00630 | 1.00603 ± 0.00776 | 0.6771 ± 0.0004 |
| A_original_pid1 | 0.02489 ± 0.00132 | 0.01294 ± 0.00114 | 0.99417 ± 0.00089 | 0.6775 ± 0.0003 |
| B_beta_input_pid1 | 0.02492 ± 0.00190 | 0.01283 ± 0.00224 | 0.99425 ± 0.00078 | 0.6775 ± 0.0003 |
| D_input_target_pid1 | 0.02418 ± 0.00160 | 0.01202 ± 0.00102 | 0.99293 ± 0.00136 | 0.6775 ± 0.0002 |

## Provenance controls

The analysis aborts unless every run agrees on these quantities:

| Quantity | Common value |
|---|---|
| dataset_metadata_sha256 | `6a7245cb0ec4125610b9dcd8c1635d70a7773eeb2b29d146dd80d5f149eb43ab` |
| selection_sql | `identical beta-valid FD selection; see provenance.csv` |
| sampled_counts | `{"test": {"-211": 45817, "211": 56774, "2212": 55891}, "train": {"-211": 364925, "211": 455246, "2212": 446432}, "validation": {"-211": 45893, "211": 57048, "2212": 56131}}` |
| data_split_seed / data_order_seed | `20260822 / 20260822` |
| generated_species | `[-211, 211, 2212]` |
| reconstructed_PID_vocabulary | `[-2212, -321, -211, -11, 11, 22, 45, 211, 321, 2112, 2212, "OTHER"]` |
| momentum_edges_GeV | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` |
| training_budget | `{"batch_size": 8192, "epochs_realized": 30, "gradient_clip_norm": 5.0, "learning_rate": 0.001, "weight_decay": 1e-05}` |
| model_constants | `{"dropout": 0.03, "hidden_layers": 4, "hidden_width": 256, "mixture_components": 8, "pid_embedding_dim": 16}` |
| checkpoint_rule / uses_test | `pid_cross_entropy / False` |

## Interpretation

The sequential Email-3 contrasts are $A\to B$ (add $\beta_{\rm gen}$ input) and $B\to D$ (also learn $\Delta\beta$). For macro TV:

- $A\to B$, $\lambda=0.2$: +0.00027 ([-0.00179, +0.00234]);
- $B\to D$, $\lambda=0.2$: -0.00180 ([-0.00639, +0.00278]);
- $A\to B$, $\lambda=1.0$: -0.00004 ([-0.00073, +0.00066]);
- $B\to D$, $\lambda=1.0$: +0.00074 ([-0.00120, +0.00268]).

The paired effect of increasing $\lambda_{\rm PID}:0.2\to1.0$ is:

- A: +0.00952 ([+0.00543, +0.01361]), 10/10 favorable pairs.
- B: +0.00921 ([+0.00513, +0.01328]), 9/10 favorable pairs.
- D: +0.01175 ([+0.00688, +0.01662]), 10/10 favorable pairs.

The smallest mean macro TV is D_input_target_pid1 at 0.02418. Claims about either beta coordinate use paired intervals above; test-set ranking is descriptive, not a checkpoint-selection rule.

## Paired primary contrasts

Positive improvement favors the treatment because both metrics are lower-is-better.

| Contrast | Metric | Mean improvement [95% CI] | Median | Favorable pairs | Exact p |
|---|---|---:|---:|---:|---:|
| beta_input_without_target | macro_weighted_bin_tv | 0.00027 [-0.00179, 0.00234] | -0.00007 | 5/10 | 0.771484 |
| beta_input_without_target | macro_correct_mae_unweighted | 0.00039 [-0.00191, 0.00268] | -0.00019 | 5/10 | 0.707031 |
| delta_beta_target_without_input | macro_weighted_bin_tv | -0.00068 [-0.00546, 0.00410] | 0.00022 | 6/10 | 0.748047 |
| delta_beta_target_without_input | macro_correct_mae_unweighted | -0.00029 [-0.00378, 0.00320] | -0.00108 | 4/10 | 0.853516 |
| combined_vs_original | macro_weighted_bin_tv | -0.00153 [-0.00583, 0.00277] | -0.00154 | 5/10 | 0.439453 |
| combined_vs_original | macro_correct_mae_unweighted | -0.00090 [-0.00539, 0.00360] | -0.00212 | 4/10 | 0.669922 |
| beta_input_with_target | macro_weighted_bin_tv | -0.00085 [-0.00490, 0.00321] | 0.00052 | 7/10 | 0.796875 |
| beta_input_with_target | macro_correct_mae_unweighted | -0.00061 [-0.00492, 0.00371] | 0.00173 | 6/10 | 0.921875 |
| delta_beta_target_with_input | macro_weighted_bin_tv | -0.00180 [-0.00639, 0.00278] | -0.00036 | 4/10 | 0.386719 |
| delta_beta_target_with_input | macro_correct_mae_unweighted | -0.00129 [-0.00654, 0.00397] | 0.00003 | 5/10 | 0.652344 |
| beta_input_pid1 | macro_weighted_bin_tv | -0.00004 [-0.00073, 0.00066] | 0.00027 | 6/10 | 0.904297 |
| beta_input_pid1 | macro_correct_mae_unweighted | 0.00011 [-0.00098, 0.00120] | 0.00042 | 7/10 | 0.859375 |
| delta_beta_target_pid1 | macro_weighted_bin_tv | 0.00074 [-0.00120, 0.00268] | 0.00070 | 5/10 | 0.398438 |
| delta_beta_target_pid1 | macro_correct_mae_unweighted | 0.00081 [-0.00084, 0.00245] | 0.00078 | 7/10 | 0.298828 |
| pid_weight_original | macro_weighted_bin_tv | 0.00952 [0.00543, 0.01361] | 0.00866 | 10/10 | 0.001953 |
| pid_weight_original | macro_correct_mae_unweighted | 0.00558 [0.00230, 0.00885] | 0.00385 | 10/10 | 0.001953 |
| pid_weight_beta_input | macro_weighted_bin_tv | 0.00921 [0.00513, 0.01328] | 0.00938 | 9/10 | 0.003906 |
| pid_weight_beta_input | macro_correct_mae_unweighted | 0.00530 [0.00273, 0.00787] | 0.00524 | 9/10 | 0.003906 |
| pid_weight_combined | macro_weighted_bin_tv | 0.01175 [0.00688, 0.01662] | 0.00976 | 10/10 | 0.001953 |
| pid_weight_combined | macro_correct_mae_unweighted | 0.00739 [0.00277, 0.01201] | 0.00738 | 9/10 | 0.003906 |
| factorial_interaction_beta_input_by_beta_target | macro_weighted_bin_tv | -0.00112 [-0.00593, 0.00369] | 0.00094 | 6/10 | 0.732422 |
| factorial_interaction_beta_input_by_beta_target | macro_correct_mae_unweighted | -0.00099 [-0.00667, 0.00468] | 0.00146 | 8/10 | 0.867188 |

## Dr. Joo-aligned correct-ID MAE

Values are unweighted momentum-bin MAE in percent, averaged over seeds.

| Species | A02 | B02 | C02 diagnostic | D02 | A10 | B10 | D10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| $\pi^-$ | 1.40 ± 0.45% | 1.56 ± 0.46% | 1.51 ± 0.36% | 1.52 ± 0.42% | 1.14 ± 0.20% | 1.19 ± 0.46% | 1.22 ± 0.24% |
| $\pi^+$ | 2.06 ± 0.65% | 1.92 ± 0.28% | 1.79 ± 0.37% | 1.85 ± 0.80% | 1.30 ± 0.16% | 1.30 ± 0.24% | 1.19 ± 0.25% |
| proton | 2.09 ± 0.59% | 1.96 ± 0.65% | 2.34 ± 0.60% | 2.46 ± 1.07% | 1.44 ± 0.38% | 1.35 ± 0.34% | 1.19 ± 0.28% |

## Continuous beta response

| Condition | Mean beta W1 across species |
|---|---:|
| C_beta_target | 0.00475 ± 0.00065 |
| D_input_target_pid02 | 0.00499 ± 0.00112 |
| D_input_target_pid1 | 0.00534 ± 0.00067 |

![Momentum-dependent correct-ID closure](pid_correct_id_vs_gen_p_factorial.png)

![Collaborator-style correct-ID MAE comparison](pid_closure_mae_comparison_our_10seed.png)

![Collaborator-style momentum comparison](pid_closure_beta_comparison_our_10seed.png)

![Seed-to-seed PID closure](pid_closure_across_conditions.png)

![Full PID-distribution closure](pid_total_variation_vs_gen_p_factorial.png)

![Selected PID migration channels](pid_migration_channels_factorial.png)

![Continuous beta-response closure](beta_response_vs_gen_p_factorial.png)

Machine-readable results: `per_run_metrics.csv`, `condition_summary.csv`, `paired_contrasts.csv`, and `provenance.csv`.
