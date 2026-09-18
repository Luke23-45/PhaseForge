# 2. Related Work

This paper sits at the intersection of three lines of work: mixture-of-experts policies for behavioral decomposition, trajectory segmentation as a source of behavioral structure, and the role of initialization in shaping learned modular architectures. We organize the discussion around how each line bears on the specific question this paper addresses — whether the kinematic regime structure of demonstrations can be used to initialize expert routing, and whether the resulting organization predicts closed-loop performance.


## 2.1 Mixture-of-experts in policy learning

Mixture-of-experts architectures partition a model's computation among specialized sub-networks, with a gating function determining which expert processes each input [Jacobs et al., 1991; Jordan and Jacobs, 1994]. In large-scale language modeling, sparse MoE layers with learned gating have become a standard mechanism for scaling model capacity without proportional increases in per-sample compute [Shazeer et al., 2017; Fedus et al., 2022; Lepikhin et al., 2021]. The routing dynamics in these models — load balancing, expert collapse, and the sensitivity of specialization to initialization — are active research topics [Zoph et al., 2022; Zhou et al., 2022].

In robot learning, the MoE structure has a different motivation: behavioral regimes within a single task or across tasks may require qualitatively different state-to-action mappings, and a modular architecture can assign each regime to a dedicated expert. Prior work has used mixture formulations for multi-modal action distributions [Zhao et al., 2023], hierarchical skill decomposition [Kipf et al., 2019; Shankar et al., 2020], and option-conditioned policies [Bacon et al., 2017; Zhang et al., 2019]. In most of these approaches, the expert partition emerges jointly with the policy through end-to-end training. The question of whether the initial partition — not just the final one — matters for downstream behavior has received comparatively little attention in the manipulation setting.


## 2.2 Trajectory segmentation and behavioral phases

Decomposing continuous trajectories into discrete behavioral segments has a long history in robotics. Dynamic movement primitives [Ijspeert et al., 2013] and probabilistic movement primitives [Paraschos et al., 2018] represent trajectories as sequences of parameterized motion segments. Changepoint detection methods — including Bayesian online detection [Adams and MacKay, 2007], kernel-based tests [Harchaoui and Cappé, 2007], and penalized cost approaches [Killick et al., 2012; Truong et al., 2020] — identify temporal boundaries where the generating process changes.

In imitation learning, trajectory segmentation has been used to discover sub-goals [Niekum et al., 2012; Konidaris et al., 2012], to define options for hierarchical policies [Krishnan et al., 2017; Fox et al., 2017], and to provide auxiliary supervision for representation learning [Shiarlis et al., 2018]. These methods produce behavioral labels — segments, primitives, or phase indices — but the labels are typically consumed as training targets or as inputs to a separate planning layer. The possibility of using them to *initialize* the geometry of a routing partition, rather than as targets or planning abstractions, is the connection this paper explores.


## 2.3 Initialization, prototypes, and routing geometry

The effect of initialization on neural network training is well-studied in the supervised learning literature [Glorot and Bengio, 2010; He et al., 2015; Mishkin and Matas, 2016], and recent work has examined how initialization interacts with the loss landscape geometry of deep networks [Li et al., 2018; Fort et al., 2019]. In the MoE context specifically, the initial placement of routing parameters can determine which experts receive gradient signal early in training, creating a path dependence that persists through optimization [Lewis et al., 2021; Roller et al., 2021].

Prototype-based classification — where predictions are made by proximity to learned class representatives in an embedding space — provides the routing mechanism used in this work. Prototypical networks [Snell et al., 2017] and related metric-learning methods [Vinyals et al., 2016; Sung et al., 2018] learn prototypes jointly with representations. The connection to Voronoi partitions of the latent space is explicit: each prototype defines a cell, and nearest-prototype assignment partitions inputs into regions. Whether the initial locations of these prototypes — derived from an external source of structure rather than learned from scratch — affect the properties of the converged partition is the specific empirical question this paper tests.


## 2.4 The gap this paper addresses

The three lines above converge on a specific intersection. Mixture-of-experts policies provide modular architectures whose routing can, in principle, reflect the phase structure of a task. Trajectory segmentation methods can discover that phase structure from demonstrations. Initialization is known to affect neural network optimization and expert specialization in MoE models. To the best of our knowledge, the specific question of whether demonstration-derived kinematic regime structure changes the organization of an MoE routing partition when used as an initialization prior, and whether that change predicts task performance, has not been directly evaluated in the setting studied here. This statement is contingent on the scope of the cited literature rather than a claim that no related study exists.

This paper provides that test, with a controlled comparison that separates the structural effect (routing organization) from the performance effect (closed-loop success) under matched training conditions.
