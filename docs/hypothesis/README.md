# Research hypothesis documents

This directory contains the professor-facing technical hypothesis and evidence
record for the focused router-initialization ablation.

- [Router-initialization hypothesis and evidence](router_initialization_hypothesis.md)
- [MoE routing stability & control decoupling report](moe_routing_stability_and_decoupling_report.md)

The documents are based on the recorded Can/Square ablation outputs, resolved
Stage 2 configurations, runner ledger, 5-task benchmark matrix, and commanded-
action discontinuity analyses. They distinguish observed structural routing
effects from closed-loop task performance. The later Square repair sweep is
explicitly treated as non-equivalent to the original PhaseForge protocol
because it disabled the original margin objective; no further cloud runs of
that repair manifest are recommended.
