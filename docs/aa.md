Yes. For the verified PhaseForge vs. `pf_centroid_random` runs, both methods use:

| Component | PhaseForge | `pf_centroid_random` |
|---|---|---|
| Stage-1 encoder | Loaded, frozen | Loaded, frozen |
| Stage-1 ActionHead | Loaded, frozen, copied into experts only in PhaseForge | Loaded, frozen, then effectively unused |
| Stage-1 phase head | Loaded, frozen | Loaded, frozen |
| Router architecture | Same Top-K router | Same Top-K router |
| Router initialization | Same centroid initialization | Same centroid initialization |
| MoE experts | Warm-started from ActionHead + jitter | Fresh random initialization |
| Stage-2 training | Router + experts | Router + experts |

So yes, the router setup is the same for the compared experiment. However, they are separate model instances: they do not share router weights during training. Each router starts from the same initialization procedure and then trains independently.

The critical difference is the experts:

```text
PhaseForge:
Stage-1 ActionHead → copy into every expert → train MoE

pf_centroid_random:
Stage-1 ActionHead → discard for Stage 2
randomly initialize experts → train MoE
```

Also, the current YAML files may show newer V2 defaults such as eight experts or soft mapping. For this comparison, we should trust the resolved historical run configs, which recorded six experts and only `expert_init` as the treatment difference.