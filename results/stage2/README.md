# Stage-2 HPO results (8 images, t0 = 1.0)

Copies of `outputs/stage2_<model>[_rN]/results*.csv`.  `stage2_<model>` runs the candidates
chosen by `scripts/hpo_stage2.py` (Stage-1 top three, the manuscript's configuration and one
grid-edge extension per axis); `_r2`, `_r3`, ... are follow-up rounds from
`scripts/hpo_stage2_round.py`, each pushing one step further on every axis where the winner
still sat on the edge of everything tried.  The final benchmark selects winners among the
candidates inside the Stage-1 grid only; the extension rounds are reported in the appendix of
`docs/best_hyperparameters.md`.
