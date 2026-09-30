# BNN development

One research repository: base BNN, two-scale inference, scalar elasticity,
documentation, and paper replication. Maintain full paper attribution.

Keep exact rank weights, actual integer scales, stable ties, and independent
subset/resampling/deletion oracles. Do not silently truncate weights or replace
exact neighbors. Run numerical checks on a compute host with bounded threads
and accelerator memory. Keep fleet orchestration and credentials outside this
repository. Timings must specify workloads, environments, warmup, repeats,
baselines, and synchronization. Preserve paper-specific replication settings.

The repository and research website are public releases. Keep internal proof
reviews and professional-positioning notes outside public source. Treat changes
to published theory statements as substantive revisions with explicit sources.
