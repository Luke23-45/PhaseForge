# 2. Related Work

This work connects mixture-of-experts routing, trajectory segmentation in robot learning, and prototype-based initialization. The central question is whether kinematic structure extracted from demonstrations can initialize an MoE routing partition, and whether the resulting routing organization predicts closed-loop control quality.

## 2.1 Mixture-of-experts in policy learning

Mixture-of-experts architectures divide computation among specialized subnetworks and use a gating function to select or weight expert outputs [Jacobs et al., 1991; Jordan and Jacobs, 1994]. In large-scale models, sparse routing enables increased capacity without evaluating every expert for every input [Shazeer et al., 2017; Lepikhin et al., 2021; Fedus et al., 2022]. This literature also studies expert utilization, load balancing, and collapse, all of which affect whether experts develop distinct roles [Zoph et al., 2022; Zhou et al., 2022]. Recent work on mixture-of-experts models in other domains has likewise distinguished router-level organization from end-task performance, indicating that allocation or routing metrics alone need not determine downstream quality [Roller et al., 2021; Qiu et al., 2025; Nguyen et al., 2026]. Although the architectures, objectives, and data differ from robot manipulation, this evidence motivates evaluating routing structure and closed-loop control as separate outcomes.

In robot learning, modular decompositions address a different problem: a task may require distinct local control laws across approach, contact, transport, and placement. Generative formulations have been used for multimodal action prediction [Zhao et al., 2023; Chi et al., 2023], while hierarchical and option-based methods divide behavior into skills or temporally extended decisions [Bacon et al., 2017; Krishnan et al., 2017a; Kipf et al., 2019; Shankar et al., 2020; Zhang et al., 2019]. These approaches differ in architecture and supervision, but each must determine how observations or trajectory segments are assigned to specialized computation.

Most such assignments are learned jointly with the policy. This paper instead studies the initial geometry of a hard prototype router: whether initializing its partition from demonstration-derived regimes changes the routing structure that remains after joint fine-tuning.

## 2.2 Trajectory segmentation and behavioral phases

Trajectory segmentation identifies intervals with distinct motion or task-variable statistics. Movement-representation methods, including dynamic and probabilistic movement primitives, model trajectories through structured temporal components [Ijspeert et al., 2013; Paraschos et al., 2018]. Change-point methods identify boundaries at which the generating process changes, using online Bayesian inference, kernel-based tests, or penalized cost objectives [Adams and MacKay, 2007; Harchaoui and Cappé, 2007; Killick et al., 2012; Truong et al., 2020].

In robot learning, segmentation has supported subgoal discovery, option construction, primitive learning, and auxiliary representation objectives [Niekum et al., 2012; Konidaris et al., 2012; Krishnan et al., 2017b; Shiarlis et al., 2018]. These methods commonly use segments as skills, planning abstractions, or supervisory targets.

Our use of segmentation is narrower. Trajectory-derived regime labels group Stage-1 latent representations when constructing the initial routing prototypes. They do not define a fixed skill sequence or replace the separate rule-derived phase labels used for representation supervision. The intervention is therefore the initialization of routing geometry from trajectory-derived regimes.

## 2.3 Initialization, prototypes, and routing geometry

Initialization affects optimization by determining the parameter region from which learning begins [Glorot and Bengio, 2010; He et al., 2015; Mishkin and Matas, 2016]. Analyses of neural-network geometry further show that training trajectories can depend on the initial parameterization [Li et al., 2018; Fort et al., 2019]. In an MoE, early routing assignments influence which expert parameters receive updates [Dai et al., 2022]. Initial router geometry can therefore influence later specialization without fixing the final partition [Nguyen et al., 2026].

Prototype-based methods provide the routing mechanism used here. Prototypical networks and related metric-learning methods represent classes or groups by vectors in an embedding space and assign examples according to proximity [Vinyals et al., 2016; Snell et al., 2017; Sung et al., 2018]. Nearest-prototype assignment induces a Voronoi partition: each prototype defines the region routed to its associated expert. This paper adapts that geometry to hard expert dispatch in a control policy. Rather than learning prototype locations from an arbitrary initial state, it initializes them from latent centroids grouped by trajectory-derived regimes and then allows the encoder, prototypes, and experts to adapt jointly.

## 2.4 Position of This Work

The relevant gap is not whether trajectory segmentation, phase supervision, or MoE routing can each support robot learning; prior work establishes each independently. The question studied here is whether trajectory-derived regime structure can initialize a prototype-routing partition in a way that changes its learned organization after fine-tuning. To our knowledge, this question has not been directly evaluated in the robot-manipulation setting studied here.

We examine this question through a matched comparison of trajectory-derived, rule-based, and random prototype initialization on Can and Square. Routing alignment and switch rate are measured on validation demonstrations, while success is measured in closed-loop rollouts. This design separates the structural effect of initialization on routing organization from its effect on task performance.