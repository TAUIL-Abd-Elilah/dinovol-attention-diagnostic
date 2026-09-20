# Opt-in helper replication plan, frozen before helper runs

Use the standalone `sdpa_padding.py` with its conservative capability checks, explicit enabled=True, and unchanged defaults. Record exact helper hash in every report. The benchmark alone intercepts SDPA calls; a module-local proxy retains the saved native callable so dispatch cannot recurse. This is not the production integration mechanism.

Each arm is a fresh process, two warmups, five synchronized complete similarity calls and a separate instrumented profile. Same model, reference, source, precision, windowing and captures as FULL_COMPARISON_PLAN.md. Baseline is measured again with the same helper runner (enabled path unused), so driver overhead is matched. No concurrent GPU jobs.

Fixed run order: original cube A helper, A unchanged; preselected adjacent cube B unchanged, B helper; preselected adjacent cube C helper, C unchanged. The two additional cube coordinates and source identity are frozen independently in ADDITIONAL_INPUT_PLAN.md before any corresponding inference outcomes. Report every arm, including failures. These neighboring cubes are a local robustness check, not independent scrolls or broad generalization.

Compare the original cube's helper output against the earlier raw-pad candidate to confirm the published helper executes the measured mechanism. Use exact operator counts to verify all 192 attention calls per profiler pass. Report full-map differences and strict >0.5 eligibility flip counts on all three fixtures. Production pseudo-label equivalence remains untested without the exact phase teachers; eligibility is not a pseudo-label.

The original cube's forced-math arm isolates the arithmetic effect of zero-padding; no need to repeat that mechanism control on every neighboring input. Keep the existing >=20% full-similarity time reduction screen. Publish per-cube results and the full range, not only the best speedup. Any broader model-quality/adoption claim requires separate evidence.
