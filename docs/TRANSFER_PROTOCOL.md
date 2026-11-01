# Exclusive rho transfer protocol

1. **Data contract.** A native row is one generated particle. The event key is
   `(file_id,event)`. Reconstruction means mutual MC-to-REC-to-same-MC association.
   The electron trigger is the eventwise existence of a generated electron
   associated with the first negative-status REC row in FD. Reconstructed PID
   need not equal 11. Preserve all generated rows, including failures.
2. **Coordinates.** Convert degrees to radians and recompute REC−GEN residuals.
   Use Alex's pre-XGBoost backups for the raw response. The September export
   lacks raw backups; its native momentum must not be described as uncorrected.
   `detector=0` includes matched FT/other outcomes as well as failures.
3. **Inputs.** Select native whole events with `hash(file_id,event)%10=0`.
   Retain all four particles. The August control is the existing held-out
   event partition. Save exact input SHA-256 hashes. Copies on both platforms
   must match before execution. Neither native export contains generator weights.
4. **Response.** Freeze the 3 existing 5-input split/asinh checkpoints. Use their
   training scalers, feature vocabulary and selected FD/quality population.
   Compare raw and corrected rho on identical particles selected using raw
   momentum. No fitting or checkpoint selection uses rho. The August comparison
   also changes skim/trigger conventions; it is not a causal generator ablation.
5. **Efficiency.** Reuse the categorical machinery from the hadron-efficiency
   branch. Fit 3 species gates using September phase-space data only, with
   300,000 train, 50,000 validation and up to 100,000 test events/species.
   Fix event splits independently of network seeds. Fit scalers on training
   examples only. Do not cut the denominator on reconstruction, PID, beta,
   residuals or FD membership. Targets are failure/FD/FT/CD/other.
6. **Replication.** Use 3 network seeds per platform, 30 epochs maximum,
   phase-validation early stopping and identical inputs/hyperparameters.
   Record hardware, package versions, source hashes, selected event hashes,
   checkpoints and logs. Hardware/software differences preclude bitwise
   training equivalence. Frozen-response inference uses a shared CPU RNG.
7. **Metrics.** Conditional PID closure uses mean softmax probabilities,
   not argmax accuracy. Report correct-ID MAE and full-vocabulary total
   variation in fixed 1-GeV bins with at least 200 particles. Report both
   full-sample and common-bin comparisons. Momentum response uses central
   Gaussian fits, quantiles and explicit tails. Efficiency uses probability
   calibration, Brier score, and momentum/angular closure. Count-weighted bin
   MAE is not generator-weighted physics. Report coordinate-box support
   diagnostics; a bounding box does not prove learned-density coverage.
8. **Physics plots.** Show generated pion-pair mass and momentum conservation,
   then raw/corrected pion-pair and missing-mass distributions for the same
   fully matched FD events with truth species mass assignments. These are
   MC correction diagnostics, not an end-to-end FM event prediction.
9. **Scope.** Without HEPGen weights, all figures concern unweighted MC counts.
   Weighted physical yields/distributions need LUND weights and an explicit
   source-event mapping. Independent per-particle gates and selected response
   heads do not yet form a validated, correlated exclusive-event simulator.

The current work starts at `main` (`56c5e19`) and incorporates gate machinery
from `feature/trigger-conditioned-hadron-efficiency` (`4674686`). Existing
training/sampling paths remain available. New data, fitted gate weights and
event predictions remain outside Git publication; aggregate results and
reproduction commands are versioned.
