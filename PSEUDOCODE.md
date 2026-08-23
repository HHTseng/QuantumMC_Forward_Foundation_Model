# Step-one pseudocode: seed of the CLAS12 Forward Foundation Model

## Scope decision

The supplied `load_phase_space_Aug17-26_FDcuts.py` view has already selected
reconstructed Forward Detector particles. Therefore this first neural network
implements only

`P(delta_p, delta_theta, delta_phi, rec_pid | generated hadron, triggered, selected FD)`.

It does **not** estimate trigger or reconstruction efficiency. Those require
the all-event views in `load_phase_space_Aug17-26.py` and become the next two
factorized components.

The current Parquet contains only raw residuals. The code trains on those raw
targets and records that choice; it does not invent an energy-loss or swum-back
phi correction.

## Pseudocode

```text
INPUTS
    canonical Aug17-26 Parquet shards
    FD fiducial parameters: rec_theta < 33 degrees and -5.5 < gen_vz < -0.5 cm
    model-quality policy and random seed

FREEZE DATA CONTRACT
    define true event key = (source_file_id, event_id)
    require generated truth, region, match, residual, and reconstructed-PID columns
    assert delta_phi lies in [-pi, pi]
    keep the source Parquet immutable

BUILD THE CONDITIONAL FD RESPONSE POPULATION
    select rows where rec_detector_region == FD
    select rec_theta < radians(33)
    select -5.5 < gen_vz < -0.5
    select usable_for_hadron_response_training == true
    retain generated PIDs pi-, pi+, and proton

    for this residual-model baseline only:
        require reciprocal truth match
        reject rec_pid == 0 and rec_beta == -99 sentinel rows
        reject |delta_p| > 10 GeV as an explicit, configurable pathology rule
    record every predicate in the model artifact

SPLIT WITHOUT EVENT LEAKAGE
    bucket = deterministic_hash(source_file_id, event_id, seed)
    train      = bucket in first 80 percent
    validation = bucket in next 10 percent
    test       = bucket in final 10 percent
    assert pairwise intersection of event keys is empty
    only after splitting, optionally take a deterministic species-balanced
    development sample; a null limit means use the complete split

TRANSFORM USING TRAINING DATA ONLY
    generated-particle inputs:
        log(1 + gen_p)
        gen_theta
        sin(gen_phi), cos(gen_phi)  # periodic representation
        learned embedding(gen_pid)
    targets:
        delta_p, delta_theta, wrapped delta_phi
    fit feature and target mean/scale on train only
    serialize both transformations with the checkpoint

TRAIN A STOCHASTIC SHARED BACKBONE
    shared_embedding = MLP(concatenate(continuous_inputs, PID_embedding))

    residual head outputs K sets of:
        mixture probability
        mean vector for (delta_p, delta_theta, delta_phi)
        diagonal scale vector
    residual loss = negative log likelihood under the K-component mixture

    PID head outputs categorical probabilities over observed rec_pid values
    PID loss = unweighted cross entropy
        # unweighted so predicted probabilities preserve the physical class measure

    total loss = residual_NLL + lambda_PID * PID_cross_entropy
    optimize with AdamW and gradient clipping
    select the checkpoint with the lowest event-disjoint validation loss
    stop early when validation loss no longer improves

VALIDATE ON THE UNTOUCHED TEST SPLIT
    sample one residual vector and one rec_pid per test particle
    for every generated species and residual coordinate:
        compare observed versus sampled mean and standard deviation
        compare 1%, 5%, 50%, 95%, and 99% quantiles
        compute one-dimensional Wasserstein distance
        save overlay histograms
    compare observed PID fractions with mean predicted probabilities
    report NLL, PID cross entropy, PID accuracy, throughput, and all closure tables

PACKAGE FOR REPRODUCIBILITY
    save:
        model weights and architecture
        input/target transformations
        PID vocabularies
        exact SQL selection
        composite-event split seed
        data/schema audit and dataset metadata fingerprint
        resolved configuration
        environment versions
        model card and plots

NEXT FACTORIZED STEPS (using the full all-event loader, not FDcuts)
    train P(trigger | generated electron) on one gen_pid=11 row from every event
    train P(unreconstructed/FT/FD/CD | generated particle, trigger) on triggered events
    execute trigger -> outcome -> FD response in a sampler
    add corrected targets only after correction version/sign/coordinates are frozen
    then expand conditions, event context, species, and generative architecture
```

## Exit criterion for this step

This step is successful when it reproducibly creates an event-disjoint
checkpoint and held-out stochastic closure artifacts. It is not a physics
release until the collaboration defines numerical closure tolerances and the
remaining trigger/outcome factors are implemented.
