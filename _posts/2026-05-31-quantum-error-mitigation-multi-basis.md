---
layout: post
title: "When Noise Correction Makes Things Worse: A Deep Dive into Quantum Error Mitigation"
date: 2026-05-31
description: We show that the standard trick of training error correctors on Z-basis measurements actively harms Hamiltonian energy prediction, and build a fix that beats ZNE at one-third the cost.
---

**Author:** Hesham Youssef

---

## 1. Introduction

Quantum computers today are noisy. Every gate misfires a little; every qubit loses coherence a bit faster than we'd like. For small problems on a handful of qubits, this is tolerable, but if we want to use quantum processors to compute molecular energies or simulate physical systems, we need those answers to be *accurate*, not just approximately right.

Full quantum error correction exists, but it is expensive: it demands hundreds of physical qubits per logical qubit and has not yet reached the scale needed for useful chemistry. The near-term alternative is **quantum error mitigation (QEM)**, a set of classical post-processing tricks that squeeze more accurate expectation values out of noisy raw data, at the cost of more circuit executions rather than more qubits. Throughout, we quote that overhead as a multiplier: **N×** means N circuit executions per estimate, relative to running the bare circuit once, so the uncorrected baseline is 1×.

Two methods dominate the literature:

- **Zero-Noise Extrapolation (ZNE)**: run the same circuit at several deliberately amplified noise levels (by "folding" gates), then extrapolate the measurements back to zero noise. Simple in principle, 3× the circuit cost.
- **Clifford Data Regression (CDR)** exploits a quirk: a special family of gates (the *Clifford* gates: H, S, CNOT) is classically simulable, because their states need only $O(n^2)$ bits rather than $2^n$ amplitudes; the hard part of any circuit is the *non-Clifford* gates, the "off-grid" rotations whose angle isn't a multiple of $\pi/2$. CDR snaps each off-grid rotation to the nearest $\pi/2$ multiple, giving a *near-Clifford* circuit that is almost the original but now cheap to simulate exactly. Simulation provides a clean reference label; the same circuit run on hardware provides the noisy value. Fit a noisy-to-exact corrector across those $k$ circuit pairs and apply it to the original. Typically 11× the circuit cost, growing with $k$.

More recently, **machine-learning approaches** have emerged: train a single global model on a pool of calibration circuits, then apply it to every new circuit without any extra quantum overhead. The state-of-the-art version from Strikis et al. and Lowe et al. uses "physics features" (a compact $O(n^2)$-dimensional summary of the measurement statistics) as inputs to a ridge regression model.

The catch is that **every existing benchmark uses a single $Z$-basis observable**, typically $\langle Z_0 \rangle$. Real quantum chemistry is different. A molecular Hamiltonian like the H₂ molecule looks like

$$H = g_0 + g_1 Z_1 + g_2 Z_0 + g_3 Z_0 Z_1 + g_4 X_0 X_1,$$

where you need to measure in both the $Z$ and $X$ bases to evaluate the energy. We asked: what happens when you take ML-based mitigation (designed and tested on $Z$-only observables) and apply it to a real Hamiltonian?

The answer, it turns out, is that it can make things dramatically *worse*. And fixing it turns out to be surprisingly cheap.

---

## 2. Background: Physics Features for Error Mitigation

The core idea of ML-based mitigation is ordinary feature engineering: replace the $2^n$-dimensional probability vector (one entry per measured bitstring) with a compact, hand-crafted summary that still captures the noise structure. For a 4-qubit circuit measured in the $Z$ basis, the feature vector is:

$$\phi(\mathbf{p}) = \bigl[\{P(q_i = 1)\}_i,\ \{\langle Z_i Z_j \rangle\}_{i<j},\ \{P(\text{Hamming weight} = k)\}_k\bigr]$$

For $n = 4$ this gives a 15-dimensional vector (4 single-qubit marginals, 6 two-qubit correlators, 5 Hamming weight probabilities), much more tractable than 16 bitstring probabilities, and it scales as $O(n^2)$ instead of $O(2^n)$.

A ridge regression model is then trained on a pool of calibration circuits:
$$w^* = \arg\min_w \sum_i \bigl(\langle Z_0 \rangle^{(i)}_\text{ideal} - w^\top \phi(\mathbf{p}^{(i)}_\text{noisy})\bigr)^2 + \alpha \|w\|^2.$$

Once trained, correcting a new circuit costs nothing extra: just extract features from the noisy measurement and apply the weights.

This works beautifully for $\langle Z_0 \rangle$. Our question was whether it generalizes.

---

## 3. What We Tried First: Quantum Reservoir Computing

Before arriving at the physics-feature approach described above, we explored a different direction entirely: **quantum reservoir computing (QRC)**. The idea is appealing: use quantum circuits themselves as the feature extractor, exploiting the exponentially large state space (the $2^n$-dimensional Hilbert space) to capture correlations that a classical model might miss. Think of it as a quantum analogue of a random-projection or reservoir-computing layer.

We built three generations of reservoir computers:

**Classical ESN (echo state network)**. The baseline: a random recurrent network with tanh nonlinearity processes the measurement bitstring. Each noisy circuit's output is encoded as a fixed input signal, and the reservoir drives $n_\text{steps}$ iterations. We tried 25, 100, and 500 nodes.

It failed to help. Under simple depolarizing noise, the noisy-to-ideal mapping is approximately *linear*: the noise scales expectations proportionally. Random nonlinear features from a tanh reservoir are pure noise on a linear task. The ESN barely matched the raw noisy baseline (MAE 0.046–0.057 across sizes vs raw 0.059) and never came close to the classical polynomial kernel (0.031); worse, performance degraded monotonically as the reservoir grew, from 0.046 at 25 nodes to 0.057 at 500. Adding capacity actively hurt. Classical random projections were no better.

**Random quantum circuits (hybrid QRC)**. Next: replace the classical reservoir with 20 random 3-qubit probe circuits (and separately, 10 random 8-qubit circuits). For each calibration circuit, run the probe circuits initialized from the main circuit's output state and concatenate the resulting probability vectors into a feature representation. Expressive, but at enormous cost.

| Design | Cost | Simple MAE | Realistic MAE |
|---|---|---|---|
| Hybrid QRC (20 × 3q circuits) | 1× + 20×3q | 0.0476 | 0.0562 |
| Fully QRC (10 × 8q circuits) | 1× + 10×8q | 0.0490 | 0.050 |
| Classical poly kernel d=3 | 1× | 0.0306 | 0.0318 |

Random quantum circuits are worse than the classical polynomial kernel despite costing orders of magnitude more. The problem: random circuits project into random directions of the Hilbert space, most of which carry no information about the noise structure.

**Targeted QRC (structured probe circuits)**. The key insight was that a 2-qubit circuit with $R_y(\theta_1) \otimes R_y(\theta_2) + \text{CZ}$ produces outputs proportional to cross-terms like $P(00) \cdot P(11)$, exactly the kind of polynomial interaction that a polynomial kernel computes. So we built a structured reservoir of 20 such 2-qubit probe circuits, each computing a different polynomial cross-term.

This worked much better:

| Design | Cost | Simple MAE | Realistic MAE |
|---|---|---|---|
| Targeted QRC (20 × 2q, 1 CZ layer) | 1× + 20×2q | 0.0314 | 0.0358 |
| Classical poly kernel d=3 | 1× | **0.0306** | **0.0318** |

The targeted QRC gets within 3% of the classical polynomial kernel on simple noise, but the kernel beats it, costs far less, and is deterministic. We had arrived at the quantum approach to what classical kernel methods already solve exactly.

**The 8-qubit wall**. On 8 qubits, the targeted QRC collapses completely: MAE 0.356 on simple noise, *worse than raw* by 6×. The culprit: the random rotation matrices chosen to initialize the reservoir are misaligned with the physics feature space at higher qubit counts. The structured 2-qubit probe circuits that worked well at 4 qubits no longer capture the right directions when the state space is 256-dimensional.

The lesson was not about quantum vs. classical models; it was about the **choice of features**. The targeted QRC worked at 4 qubits because its probe circuits happened to approximate polynomial features of the probability vector. At 8 qubits that approximation broke down, and the whole approach fell apart.

That pushed us toward the question we should have started with: *what features of the measurement statistics actually carry noise-correction information?* The answer was sitting in the data we were already collecting, from every basis we were measuring.

---

## 4. The Hidden Failure: Z-Only Features Are Actively Harmful

To measure $\langle H \rangle$ for a Hamiltonian with both $Z$ and $X$ terms, you run the circuit twice: once measuring in the $Z$ basis (for the $ZZ$ terms), once with a Hadamard gate appended to each qubit before measurement (rotating the $X$ basis into the $Z$ basis for measurement). These two executions give you the $Z$-basis correlators and the $X$-basis expectation values, respectively.

The old approach extracts physics features only from the $Z$-basis measurement. The $\langle ZZ \rangle$ correlator (the correlation between two qubits' $Z$-readouts) is all the model "sees." But here is the problem: in the transverse-field Ising model $H = -J\sum_i Z_i Z_{i+1} - h\sum_i X_i$ (a standard toy physics system; read it as "energy is a weighted sum of $Z$-correlations and $X$-readouts"), the $\langle ZZ \rangle$ correlator is *anticorrelated* with the $X$-field contribution $\langle H_X \rangle = -h \sum_i \langle X_i \rangle$. The model learns a correction for $\langle ZZ \rangle$ that systematically over-corrects the total energy, introducing sign errors in the $X$ contribution.

![Z-only failure: bar chart showing Ridge Z-only MAE of 0.640 vs raw noisy 0.181 on Ising 4q, and 0.076 vs 0.020 on H₂](/assets/img/fig1_z_only_failure.png)

*Figure 1. Z-only ridge regression (red bars, hatched) on the 4-qubit Ising chain (left) and H₂ molecule (right). The dotted line marks the raw noisy baseline. Z-only features actively increase error by 3.5× on Ising and 3.8× on H₂ compared to doing nothing.*

The numbers in Table 1 tell the story clearly:

| Method | Cost | Ising MAE | H₂ MAE |
|---|---|---|---|
| Raw noisy | 1× | 0.181 | 0.020 |
| **Ridge Z-only** | **1×** | **0.640** | **0.076** |
| Poly kernel Z-only | 1× | 0.602 | 0.080 |
| **Per-group ridge (ours)** | **1×** | **0.083** | **0.007** |
| ZNE [1,3,5] | 3× | 0.106 | 0.014 |
| Ridge MS | 3× | **0.038** | **0.004** |
| CDR k=5 | 11× | 0.038 | 0.025 |

*Table 1. Hamiltonian benchmark results (simple noise). Z-only methods make things worse; multi-basis methods beat ZNE at lower cost.*

---

## 5. Our Approach: Multi-Basis Physics Features

Once you see the problem, the fix is obvious. When you measure in multiple bases to evaluate $\langle H \rangle$, you already have the measurement statistics from each basis. Extract physics features from *all* of them:

$$\Phi(\mathbf{p}) = [\phi_1(\mathbf{p}_{B_1}),\ \phi_2(\mathbf{p}_{B_2}),\ \ldots,\ \phi_M(\mathbf{p}_{B_M})].$$

For the 4-qubit Ising Hamiltonian with two measurement groups (all-$Z$ and all-$X$ bases), this concatenates two 15-dimensional feature vectors into a 30-dimensional input. For the 2-qubit H₂ Hamiltonian, each basis yields 6 features (2 marginals + 1 correlator + 3 Hamming weights), giving a 12-dimensional total. The key property: **zero additional circuit cost**. These measurements were already required to compute $\langle H \rangle_\text{noisy}$ in the first place.

**Per-group ridge** takes this one step further. Instead of training a single model to predict the total $\langle H \rangle$, we train a separate ridge model for each commuting Pauli group:
- Model 1: predict $\langle H_{ZZ} \rangle$ from $Z$-basis features
- Model 2: predict $\langle H_X \rangle$ from $X$-basis features
- Final answer: sum the two predictions

This lets the model learn that $Z$-basis measurements have different noise rates than $X$-basis measurements, which is physically true, since readout errors primarily affect $Z$-basis marginals while thermal relaxation ($T_1$) dominates $X$-basis expectations.

This same observation explains a result we will see later: *why scalar CDR underperforms on Hamiltonians.* Standard CDR fits a single global slope, $\langle H \rangle_\text{ideal} \approx a\,\langle H \rangle_\text{noisy} + b$. But the $Z$ and $X$ components of $\langle H \rangle$ experience genuinely different effective noise rates, and one slope $a$ can only average over them, leaving a systematic residual no matter how many near-Clifford samples you spend. Fitting the correction *per commuting group*, whether in ridge or in CDR, is what matches the actual noise structure. The multi-basis idea is not specific to ridge regression; it is the right unit of correction for any data-driven mitigator applied to a multi-Pauli observable.

**Multi-scale ridge** extends this further by running at three noise-amplification levels ($\lambda \in \{1, 3, 5\}$) and concatenating features from all levels:
$$\Phi^\text{MS} = [\Phi(\lambda=1),\ \Phi(\lambda=3),\ \Phi(\lambda=5)].$$

This matches ZNE's 3× circuit cost but gives the model explicit information about how each observable changes with noise, letting the regressor learn an implicit Richardson extrapolation (fitting the noise-vs-signal trend and projecting it back to zero noise) rather than having ZNE impose a fixed extrapolation formula.

### The methods, at a glance

| Name | Cost | What it does |
|---|---|---|
| Raw noisy | 1× | No correction; the baseline every other method must beat. |
| Ridge $Z$-only | 1× | The prior approach: ridge regression on physics features from the $Z$ basis alone. |
| Per-group ridge | 1× | Ours: a separate ridge model per commuting group, each fed features from its own basis; predictions summed. |
| Ridge MS | 3× | "Multi-scale": per-group ridge with features gathered at three noise levels ($\lambda \in \{1,3,5\}$), matching ZNE's budget. |
| ZNE | 3× | Zero-noise extrapolation. |
| CDR ($k$) | 4–11× | Clifford data regression with $k$ near-Clifford reference samples; cost grows with $k$. |

These are the canonical names used in every table below.

---

## 6. Results

### Simulation benchmarks

The results on our simulated 4-qubit system (Table 1) show three things clearly:

1. **Per-group ridge at 1× cost beats ZNE at 3× cost** on both Hamiltonians. On H₂ with simple noise, per-group ridge achieves MAE 0.007 vs ZNE's 0.014, a 50% improvement at one-third the circuit budget.

2. **Ridge MS at 3× cost matches or beats CDR at 11× cost** in all four settings. On H₂ under simple noise: Ridge MS 0.004 vs CDR 0.025, a 6× improvement at less than one-third the cost.

3. **The cross-Hamiltonian transfer is lossless**. We trained a single circuit pool on Ising circuits and applied the learned weights to H₂ labels without retraining, with zero degradation. One calibration run covers any Hamiltonian on the same device.

We tested under two noise models: a simple uniform depolarizing model (every qubit is randomized with a fixed probability, the "white noise" of quantum hardware), and a realistic model that adds the messy effects of real devices: thermal relaxation (qubits decaying toward 0 over a characteristic time $T_1 = 60\,\mu\text{s}$, and losing phase coherence over $T_2 = 40\,\mu\text{s}$), coherent over-rotation (every gate consistently turning a little too far, $\varepsilon = 0.04$ rad), nearest-neighbor $ZZ$ crosstalk (idle qubits nudging their neighbors), and asymmetric readout errors (misreading a 1 as 0 more often than the reverse). The per-group model degrades gracefully under realistic noise; Ridge MS holds up well for small systems.

The scalability story is more nuanced. At 8 qubits, the Ridge MS feature vector is 270-dimensional, but we only have 224 training circuits, leaving it underdetermined. Ridge MS exceeds the raw noisy baseline at 8 qubits. The per-group model (90-dimensional at 8 qubits, ratio 2.5×) remains well-determined at all sizes and consistently beats ZNE. **Practical rule**: use Ridge MS only when you have at least 3× more training circuits than feature dimensions.

### ZNE has a low-noise failure mode

One of the more surprising findings: ZNE can make things *worse* even in simulation. Figure 2 shows what happens as we vary circuit depth (and thus intrinsic noise level).

![Depth crossover: MAE vs circuit depth for all methods under simple noise (left) and realistic noise (right)](/assets/img/fig2_depth_crossover.png)

*Figure 2. MAE vs circuit depth for H₂ under simple noise (left) and realistic noise (right). Left: ZNE is fractionally worse than raw at depth 2 (shaded region) but improves from depth 4 onward. Ridge and CDR are net-positive at every depth.*

At very shallow circuits (depth 2), raw MAE is already as low as 0.014. ZNE amplifies the noise three ways and extrapolates, but when the original signal is already almost clean, the amplified variants are dominated by statistical fluctuation. The Richardson extrapolation variance exceeds the bias it removes, and ZNE undershoots. The crossover point lies in the band raw MAE $\in [0.014, 0.019]$.

Under simple depolarizing noise, ridge and CDR have no such threshold: they are net-positive at every depth tested. Under realistic noise the picture is more complex (ridge per-group slightly exceeds raw at depth 32, and CDR fails at depth 64 when the linear noise model breaks down), but neither has the low-noise reversal that afflicts ZNE.

### IBM hardware validation

We validated on the IBM `ibm_marrakesh` Heron-r2 processor (156 qubits, accessed via the Qiskit Runtime open plan). The target was the H₂ Hamiltonian at bond length $R = 0.74$ Å, decomposed into two commuting groups. Total QPU time: 275 seconds.

The hardware results confirmed the simulation predictions, and added some surprises:

| Method | Cost | Sim MAE | HW@1024 shots | HW@4096 shots |
|---|---|---|---|---|
| Raw noisy | 1× | 0.0233 | 0.0253 | 0.0243 |
| Ridge sim-trained | 1× | 0.0082 | 0.0265 | 0.0235 |
| Ridge HW-calibrated | 1× | — | 0.0262 | 0.0230 |
| ZNE [1,3,5] | 3× | 0.0155 | 0.0331 | — |
| CDR k=3 | 4× | 0.0248 | 0.0295 | — |
| **CDR k=5** | **6×** | **0.0240** | **0.0193** | — |

*Table 2. IBM `ibm_marrakesh` results for H₂ at $R = 0.74$ Å. ZNE and CDR were run at 1024 shots only; ridge methods were run at both shot counts.*

**Finding 1**: Simulator-trained ridge weights transfer to the QPU with only 1–2% degradation (1.1% at 1024 shots; 2.2% at 4096 shots, comparing sim-trained vs HW-calibrated ridge). The on-device calibration we thought was mandatory turns out to be unnecessary *for this 2-qubit Hamiltonian*. The qualifier matters: in simulation, training Ridge MS on the simple noise model and applying it to the realistic one degrades 2-qubit H₂ by under 1%, but the 4-qubit Ising chain by **128%** (and 232% in the reverse direction). Coherent errors and $ZZ$ crosstalk reshape the $X$-basis feature–label relationship in a way that simple-noise calibration cannot anticipate, and the effect is far more severe for the larger 4-qubit system. The practical rule that emerges: for $n \geq 4$ qubits with $X$/$Y$ terms, calibrate on the actual device, or fall back to CDR, which adapts per-circuit regardless of the noise model.

**Finding 2**: ZNE is 31% *worse* than uncorrected hardware. The hardware raw MAE of 0.024 sits above the simulator crossover threshold, but the actual ZNE failure is more severe than the simple-noise model predicts: correlated errors, coherent noise, and inter-scale drift all inflate the Richardson extrapolation variance beyond what depolarizing noise alone would produce.

**Finding 3**: CDR's performance is entirely budget-dependent. At $k=3$ near-Clifford samples, CDR was 17% worse than raw. A direct rerun at $k=5$ reversed the result completely: 24% better than raw and 35% better than the same algorithm at $k=3$. Increasing $k$ from 3 to 5 converts CDR from the worst method to the best.

We then traced the full H₂ potential energy surface at five bond lengths ($R = 0.5, 0.74, 1.0, 1.5, 2.0$ Å):

![H₂ binding curve: MAE vs bond length R for raw, ridge HW, ridge sim, and CDR k=5](/assets/img/fig3_pes.png)

*Figure 3. H₂ binding curve on `ibm_marrakesh`. CDR k=5 delivers a uniform ~58% MAE reduction at every bond length; ridge sim-trained is within 14% of hardware-calibrated ridge across the full curve.*

CDR k=5 achieves a remarkably uniform 58–59% improvement at every geometry, and the correction works regardless of the molecular configuration, which matters because a potential energy surface is only useful if the errors are consistent across it.

Finally, a calibration-set size sweep showed that hardware ridge MAE plateaus by $N \approx 40$ circuits and does not improve further with more data:

![Hardware learning curve: ridge MAE vs number of calibration circuits](/assets/img/fig4_hw_lc.png)

*Figure 4. Hardware calibration learning curve on `ibm_marrakesh`. Red shading: ridge worse than raw; green shading: better. The plateau at $N \approx 40$ indicates a model-capacity ceiling, not a data shortage.*

This is a **ridge-capacity ceiling**: the linear physics-feature model has extracted all the information these calibration circuits can provide about `ibm_marrakesh`'s noise. Further improvement would need nonlinear models or more hardware-specific calibration circuits.

### Beyond H₂: 4-qubit Ising on hardware

Everything above used the 2-qubit H₂ Hamiltonian. To check that the method scales past a single small chemistry problem, we ran the same protocol on the 4-qubit transverse-field Ising chain at three field ratios $h/J \in \{0.5, 1.0, 2.0\}$ (240 circuits, 68 s QPU, sharing one circuit pool across all three).

| $h/J$ | Raw | Ridge HW-cal | Ridge sim | CDR k=5 |
|---|---|---|---|---|
| 0.5 (ZZ-dominated) | 0.120 | **0.069** | 0.086 | 0.072 |
| 1.0 | 0.148 | 0.118 | 0.119 | **0.097** |
| 2.0 (X-dominated) | 0.217 | 0.223 ⚠️ | 0.201 | **0.159** |
| **Mean** | 0.162 | 0.137 | 0.135 | **0.109** |

*Table 3. 4-qubit Ising on `ibm_marrakesh` (1024 shots, 30-circuit calibration). Bold marks the best method per row; ⚠️ marks a method worse than raw.*

The Ising results are messier than H₂, and that is informative:

- **Ridge has a genuine hardware failure mode.** At $h/J = 2.0$, where the $X$-field dominates, HW-calibrated ridge (0.223) is actually *worse than doing nothing* (0.217). This is the first hardware confirmation of the X-dominated failure the realistic-noise depth sweep had predicted, and notably the simpler sim-trained weights (0.201) are *more robust* here than the over-fit 30-circuit hardware calibration.
- **But ridge can also win outright.** At $h/J = 0.5$, where the ground state is $ZZ$-dominated and the noisy-to-ideal map stays nearly linear, ridge HW-cal (0.069) beats CDR k=5 (0.072), the first hardware case where a learned model is the single best method.
- **CDR k=5 is the most robust overall** (mean 0.109, 32% better than raw), though its advantage here is smaller than on H₂ (58%), because the 4-qubit problem has a larger Hamiltonian range and deeper state-prep circuits.

The ridge-vs-CDR choice is **problem-dependent, not universal**. Ridge wins when the noise map is approximately linear and the calibration set samples it well; CDR wins when the linear range breaks down or calibration data is scarce. And the per-group ridge failure mode is not a simulator artifact; it shows up on real hardware, in exactly the regime the simulator flagged.

---

## 7. Discussion

No mitigation method is universally net-positive. A practical decision guide:

| Setting | Budget | Best method |
|---|---|---|
| Any Hamiltonian, 2–4 qubits | 1× | Per-group ridge |
| Any Hamiltonian, 2–4 qubits | 3× | Ridge MS |
| Any Hamiltonian, ≥6 qubits | 3× | Ridge MS (if $n_\text{train} \gtrsim 3 \times \text{dim}$), else ZNE |
| Any Hamiltonian, ≥6 qubits | 11× | CDR per-group |
| Unknown noise model | any | CDR per-group $k \geq 5$ |
| Low-noise device (raw MAE < 0.02) | 1× | Per-group ridge (ZNE hurts) |
| X/Y-dominated Hamiltonian, ≥4 qubits | any | CDR per-group $k=5$ (ridge can fail) |
| Real hardware, no device calibration | 6× | CDR per-group $k=5$ |
| Shots < 512 | any | Per-group ridge (CDR unreliable) |

A few things worth keeping in mind if you are implementing any of this:

- **Never use Z-only features for a Hamiltonian with X or Y terms**. The multi-basis feature vector is free (same circuits, same measurements) and is not merely better; Z-only features actively harm the correction.

- **ZNE can backfire on low-noise devices**. If your device is already clean (raw MAE ≲ 0.02), the Richardson extrapolation variance can dominate, and ZNE makes things worse. Ridge methods have no such threshold.

- **CDR k matters enormously**. The difference between k=3 and k=5 is not marginal; it is the difference between the worst and best method on our hardware. If you are using CDR, check that your budget covers at least k=5.

- **Ridge vs. CDR is problem-dependent, not universal**. Ridge wins when the noise map is approximately linear and your calibration set samples it well; it can be net-*negative* on hardware in $X$-dominated regimes. CDR adapts per-circuit and is the more robust default when the noise model is unknown or calibration data is scarce.

---

## 8. Conclusion

The main conceptual contribution is identifying that the ML-QEM approach of Strikis et al. and Lowe et al. (training a regressor on physics features) carries a silent assumption: *all observables are Z-basis*. Their inputs were $Z$-basis probability vectors, which works for the $\langle Z_0\rangle$ benchmarks the field has standardized on. For the multi-Pauli Hamiltonians that appear in actual quantum chemistry, that same assumption causes systematic sign errors and can increase error by 3–4× relative to the raw noisy value. The fix, a multi-basis feature vector that matches the measurement structure of the target observable, is what extends their framework from a single $Z$-observable to real molecular energies.

Because those extra-basis measurements were already being taken to evaluate $\langle H \rangle$ in the first place, the correction costs nothing and eliminates the failure mode entirely. Per-group ridge at 1× circuit cost consistently outperforms ZNE at 3× cost; Ridge MS (multi-scale, multi-basis) at 3× cost matches or beats CDR at 11× cost.

Hardware validation on the IBM `ibm_marrakesh` Heron-r2 processor, on both the 2-qubit H₂ molecule and a 4-qubit Ising chain, confirmed the simulation predictions and added important nuance: ZNE fails in practice for reasons beyond the simple-noise model (correlated errors inflate variance more than depolarizing noise would predict), CDR's failures are budget-induced rather than fundamental, and simulator-trained ridge weights transfer to real hardware with under 3% degradation on H₂. But that transfer is not free everywhere: on the 4-qubit chain, cross-noise-model calibration degrades sharply and ridge can even fall behind the raw baseline in $X$-dominated regimes, making the ridge-vs-CDR choice problem-dependent rather than universal.

The broader message is that mitigation choice depends sharply on the noise regime: there is no universally best algorithm, and testing only on Z-basis observables is not sufficient to characterize a method's behavior on real quantum chemistry problems.

---

## References

1. K. Temme, S. Bravyi, J. M. Gambetta, "Error mitigation for short-depth quantum circuits," *Physical Review Letters* 119, 180509 (2017).
2. Y. Li, S. C. Benjamin, "Efficient variational quantum simulator incorporating active error minimization," *Physical Review X* 7, 021050 (2017).
3. P. Czarnik, A. Arrasmith, P. J. Coles, L. Cincio, "Error mitigation with Clifford quantum-circuit data," *Quantum* 5, 592 (2021).
4. A. Strikis, D. Qin, Y. Chen, S. C. Benjamin, Y. Li, "Learning-based quantum error mitigation," *PRX Quantum* 2, 040330 (2021).
5. A. Lowe et al., "Unified approach to data-driven quantum error mitigation," *Physical Review Research* 3, 033098 (2021).
6. Qiskit contributors, "Qiskit: An Open-source Framework for Quantum Computing" (2023).
7. Q. Sun et al., "PySCF: the Python-based simulations of chemistry framework," *WIREs Computational Molecular Science* 8, e1340 (2018).
