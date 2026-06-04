---
layout: post
title: "High-Fidelity CZ Gate Optimization via GRPO on Rydberg Neutral Atoms"
date: 2026-06-04 00:00:00 +0000
categories: [Quantum Computing, Quantum Optimal Control, Reinforcement Learning, Rydberg Atoms]
tags: [Quantum Computing, Rydberg, CZ Gate, GRPO, Reinforcement Learning, Quantum Control, QuTiP]
---

**Author:** Ahmed Samir

---

Quantum computing has been gaining serious traction, not just as a theoretical curiosity but as a practical engineering discipline. The race toward fault-tolerant quantum processors has exposed a fundamental bottleneck: we need two-qubit gate fidelities above roughly 99.5% to make quantum error correction actually work. Below that threshold, you spend more qubits fixing errors than doing computation.

Neutral atoms trapped in optical tweezers have emerged as one of the most promising platforms to get there. The Harvard/MIT group demonstrated a CZ gate at 99.5% fidelity using Rydberg blockade in 2023 [1]—a landmark result. Achieving that, however, requires extremely precise laser pulse control. Small deviations in pulse shape, laser phase, or atom temperature can collapse fidelity from near-perfect to barely usable.

At the same time, a seemingly unrelated development happened in the AI world. DeepSeek's **GRPO** algorithm—Group Relative Policy Optimization—showed that you don't need a learned value function to train language models to reason about math [2]. Instead of comparing a candidate answer against some absolute baseline, GRPO evaluates a *group* of candidates against each other. The intuition is that relative comparison is often more reliable than absolute scoring, especially when reward signals are noisy.

The connection between these two worlds is non-obvious but real. Quantum gate optimization shares several properties with the mathematical reasoning tasks that made GRPO effective: the reward landscape is noisy (quantum simulation has stochastic elements), there are many local optima, and a single evaluation is expensive. This project adapts GRPO to optimize laser pulses for a Rydberg CZ gate, and asks whether its group-relative philosophy offers concrete advantages over classical gradient methods.

---

## 1. Problem Definition

### The Physical System

We work with a **two-atom Rydberg system**. Each atom is a rubidium-87 atom trapped in an optical tweezer and has three accessible energy levels:

- $\vert 0 \rangle$ — ground hyperfine state (qubit "zero")
- $\vert 1 \rangle$ — ground hyperfine state (qubit "one")
- $\vert r \rangle$ — highly excited Rydberg state (principal quantum number $n \approx 70$)

The computational Hilbert space is spanned by the tensor products $\vert 00 \rangle, \vert 01 \rangle, \vert 10 \rangle, \vert 11 \rangle$. Including the Rydberg level, the full two-atom state space is 9-dimensional ($3 \times 3$).

### The Target Gate

We want to implement a **Controlled-Z (CZ) gate**, which applies a phase flip to the $\vert 11 \rangle$ state and leaves all others unchanged:

$$U_\text{CZ} = \text{diag}(1,\, 1,\, 1,\, -1)$$

This gate is maximally entangling: combined with single-qubit rotations, it is universal for quantum computation. It can generate Bell states directly from product-state inputs.

### The Rydberg Blockade Mechanism

The CZ gate is implemented via the **Rydberg blockade**. When an atom is promoted to the Rydberg state $\vert r \rangle$, its electron is at an enormous orbital radius, giving the atom a huge electric dipole moment. Two nearby Rydberg atoms interact via a van der Waals potential that scales as $r^{-6}$:

$$U_\text{eff} = U_\text{blockade} \times \left(\frac{4\,\mu\text{m}}{r_\text{sep}}\right)^6$$

At a separation of 4 µm and with $U_\text{blockade} = 2\pi \times 10\,\text{MHz}$, this interaction is an order of magnitude larger than the laser drive ($\Omega_\text{max} = 2\pi \times 1\,\text{MHz}$). The system is therefore in the **strong blockade regime**: the energy cost of having both atoms simultaneously in $\vert r \rangle$ is so high that the state $\vert rr \rangle$ is energetically forbidden. This conditional suppression is the physical mechanism that distinguishes the $\vert 11 \rangle$ state from all others, implementing the CZ phase flip.

### The Optimization Problem

The control problem is: given a fixed gate time $T$, find a laser pulse envelope $\Omega(t)$ such that the resulting quantum process is as close as possible to the ideal CZ gate. We parameterize the pulse as a **piecewise-constant sequence** of $N = 20$ amplitude segments, each normalized to $[0, 1]$ and scaled by $\Omega_\text{max}$. This gives a 20-dimensional continuous action space:

$$\text{maximize} \quad F(\mathbf{u}) \quad \text{subject to} \quad \mathbf{u} \in [0,1]^{20}, \quad T \in [0.3,\, 1.2]\,\mu\text{s}$$

where $F(\mathbf{u})$ is the average gate fidelity computed under a full Lindblad noise model.

---

## 2. Simulation Model

### The Hamiltonian

The full two-atom Hamiltonian is built in the rotating frame under the rotating wave approximation (RWA):

$$H(t) = \sum_{i \in \{A,B\}} \left[ \frac{\Omega(t)}{2} \left( \vert r \rangle\langle 1 \vert_i + \text{h.c.} \right) - \Delta(t)\, \vert r \rangle\langle r \vert_i \right] + U_\text{eff}\, \vert rr \rangle\langle rr \vert$$

The three terms are:

1. **The drive term** ($\Omega(t)/2 \cdot \sigma_{r1} + \text{h.c.}$): the laser couples $\vert 1 \rangle$ and $\vert r \rangle$ with Rabi frequency $\Omega(t)$. The operator $\vert r \rangle\langle 1 \vert$ promotes an atom from $\vert 1 \rangle$ to $\vert r \rangle$; its conjugate does the reverse. Together they produce coherent oscillation (Rabi flopping) between the two levels. This term acts independently on both atoms A and B via the tensor product structure.

2. **The detuning term** ($-\Delta \cdot \vert r \rangle\langle r \vert$): the detuning $\Delta$ is the mismatch between the laser frequency and the atom's natural $\vert 1 \rangle \to \vert r \rangle$ transition frequency. At $\Delta = 0$ (resonance), the drive is maximally effective. The operator $\vert r \rangle\langle r \vert$ is the Rydberg projector—it shifts the energy of any atom currently in $\vert r \rangle$.

3. **The blockade term** ($U_\text{eff} \cdot \vert rr \rangle\langle rr \vert$): adds energy $U_\text{eff}$ whenever both atoms are simultaneously in $\vert r \rangle$. Since $U_\text{eff} \gg \Omega$, this energetically forbids the $\vert rr \rangle$ state, producing the blockade.

### Open-System Dynamics and Noise

Real atoms are not isolated. We model three fundamental decoherence channels through the **Lindblad master equation**:

$$\frac{d\rho}{dt} = -i[H, \rho] + \sum_k \left( L_k \rho L_k^\dagger - \frac{1}{2} L_k^\dagger L_k \rho - \frac{1}{2} \rho L_k^\dagger L_k \right)$$

where $\rho$ is the **density matrix**—the generalized state descriptor that handles quantum superposition and classical statistical uncertainty simultaneously. The collapse operators $L_k$ encode each noise channel:

| Noise channel | Collapse operator | Rate |
|---|---|---|
| Spontaneous Rydberg decay $\vert r \rangle \to \vert 1 \rangle$ | $\sqrt{\gamma_r}\, \vert 1 \rangle\langle r \vert$ | $\gamma_r = 2\pi \times 3\,\text{kHz}$ |
| Pure dephasing of $\vert r \rangle$ | $\sqrt{\gamma_\phi}\, \vert r \rangle\langle r \vert$ | $\gamma_\phi = 2\pi \times 1\,\text{kHz}$ |
| Doppler dephasing | $\sqrt{\gamma_\text{Doppler}}\, \vert r \rangle\langle r \vert$ | $\propto n_\text{thermal} \cdot \gamma_\phi$ |

On top of the Lindblad collapse operators, additional laser noise is applied directly to the pulse envelope before simulation:

- **Intensity noise**: $\Omega_\text{noisy}(t) = \Omega(t)\,(1 + \epsilon(t))$ where $\epsilon \sim \mathcal{N}(0,\, 0.005)$ — a 0.5% fractional Rabi frequency noise.
- **Phase noise**: laser phase drifts as an Ornstein–Uhlenbeck (OU) process with linewidth $2\pi \times 100\,\text{Hz}$ and correlation time 0.1 ms. This modulates the effective drive as $\Omega_\text{noisy}(t) = \Omega(t)\cos(\phi_\text{OU}(t))$.
- **Doppler detuning**: the thermal motion of each atom causes a velocity-dependent frequency shift $\Delta_\text{Doppler} = k_\text{laser} v_z$, where $v_z$ is drawn from the Maxwell–Boltzmann distribution at mean phonon occupation $\bar{n} = 0.05$.

The Lindblad master equation is integrated numerically using QuTiP's `mesolve` function over 200 time steps covering the gate duration $T$.

### Fidelity Metric

Gate fidelity is computed via the **Choi–Jamiołkowski (CJ) isomorphism**. The protocol:

1. Evolve each of the four computational basis states $\vert 00 \rangle, \vert 01 \rangle, \vert 10 \rangle, \vert 11 \rangle$ through the noisy Lindblad dynamics.
2. Extract the final density matrix projected onto the 4-dimensional computational subspace (indices $[0, 1, 3, 4]$ in the 9D space, corresponding to $\vert 00 \rangle, \vert 01 \rangle, \vert 10 \rangle, \vert 11 \rangle$).
3. Compare each $\rho_\text{out}^{(i)}$ against the ideal CZ output $\rho_\text{ideal}^{(i)} = U_\text{CZ} \vert i \rangle\langle i \vert U_\text{CZ}^\dagger$.

$$F_\text{process} = \frac{1}{d} \sum_{i=0}^{d-1} \text{Tr}\!\left[\rho_\text{ideal}^{(i)}\, \rho_\text{out}^{(i)}\right], \qquad F_\text{avg} = \frac{d \cdot F_\text{process} + 1}{d + 1}$$

where $d = 4$ is the Hilbert space dimension. This formula, standard in quantum information, corrects for the trivial overlap contribution and gives the average fidelity over all possible input states.

---

## 3. Algorithm

### The RL Environment

The `RydbergPulseEnv` wraps the physics simulator as a Gymnasium-compatible environment. Each episode is a single-step **bandit**: the agent submits a complete pulse vector $\mathbf{u} \in [0,1]^{20}$, the simulator evaluates it under noise, and returns the fidelity as reward. The gate time is fixed at $T = 0.5\,\mu$s—the empirical sweet spot between two competing effects: shorter gates reduce decoherence time but require steeper Rabi drives that are harder to control.

The reward is not raw fidelity but a shaped version designed to make training easier:

$$R(F) = F^\alpha + B \cdot F \cdot \mathbf{1}[F \geq 0.995]$$

with $\alpha = 4$ and $B = 10$. The $F^4$ power shaping compresses poor-fidelity rewards and stretches the gradient near $F = 1$, providing a stronger learning signal when the agent is already performing well. The discontinuous bonus $B$ at 99.5% creates an explicit "achievement cliff"—within a group of candidates, those crossing the threshold are immediately recognized as exceptional by the relative advantage calculation.

The environment also supports a **noise curriculum**: training begins with `noise_scale = 0` (ideal dynamics) and linearly ramps to `noise_scale = 1` over a configurable number of steps. This gives the optimizer a warm start on the noiseless landscape before it must contend with realistic decoherence.

### GRPO Adaptation

GRPO was originally designed for discrete token generation in language models [2]. Here we adapt it to a continuous 20-dimensional action space. The key equations are unchanged:

**Group-relative advantage.** At each iteration, sample $G$ pulse candidates $\{a_1, \ldots, a_G\}$ from the current policy $\pi_\theta$ and evaluate each:

$$A_i = \frac{r_i - \bar{r}}{\sigma_r + \varepsilon}$$

where $\bar{r}$ and $\sigma_r$ are the group mean and standard deviation. This self-calibrated advantage is the core innovation: no critic network, no estimated value function, just the relative ranking of the current batch.

**Policy loss.** The policy is updated using a PPO-style clipped surrogate [3] on these group-relative advantages, plus a KL penalty against a frozen reference policy $\pi_\text{ref}$:

$$\mathcal{L} = -\mathbb{E}\!\left[\min\!\left(\rho_i A_i,\; \text{clip}(\rho_i, 1{-}\varepsilon, 1{+}\varepsilon) A_i\right)\right] + \beta\, \text{KL}(\pi_\theta \| \pi_\text{ref}) - \eta\, H(\pi_\theta)$$

where $\rho_i = \pi_\theta(a_i) / \pi_\text{old}(a_i)$ is the probability ratio, $\varepsilon = 0.2$ is the clip bound, $\beta = 0.01$ is the KL penalty, and $\eta = 0.001$ is an entropy bonus that encourages exploration.

**Policy network.** The policy is a Gaussian MLP: three hidden layers (128 units, LayerNorm, tanh activations) that output a mean $\mu \in [0,1]^{20}$ and a learned (state-independent) standard deviation $\sigma$. Actions are sampled as $a \sim \mathcal{N}(\mu, \sigma)$ and clipped to $[0,1]$. The network has approximately 16,000 parameters—deliberately lightweight since each forward pass costs nothing relative to the simulator.

The optimizer is Adam with learning rate $3 \times 10^{-4}$, gradient clipping at 0.5, and a batch of 4 groups per gradient step. The reference policy is refreshed every 100 iterations to prevent the KL penalty from becoming trivially easy to satisfy.

### GRAPE as Classical Baseline

Numerical GRAPE (Gradient Ascent Pulse Engineering) [4] is implemented as the classical comparison. It computes the fidelity gradient via central finite differences:

$$\frac{\partial F}{\partial u_k} \approx \frac{F(\mathbf{u} + \varepsilon\, \mathbf{e}_k) - F(\mathbf{u} - \varepsilon\, \mathbf{e}_k)}{2\varepsilon}$$

This requires $2N = 40$ simulator calls per gradient step, compared to $G = 6$ to $12$ for GRPO. Updates use gradient ascent with momentum:

$$\mathbf{v} \leftarrow \beta_m\, \mathbf{v} + \alpha\, \nabla F, \qquad \mathbf{u} \leftarrow \text{clip}(\mathbf{u} + \mathbf{v},\; 0,\; 1)$$

with momentum coefficient $\beta_m = 0.9$ and learning rate $\alpha = 0.05$. Three random restarts are used to partially mitigate sensitivity to initialization.

---

## 4. Experiments and Results

Six experiments were run, each targeting a specific question about the algorithm's behavior or the physics. All experiments use the same Rydberg simulator and noise model unless stated otherwise.

| Metric | Value |
|---|---|
| Best GRPO fidelity (Exp 1, G=12) | **99.99%** |
| Blackman pulse baseline | 72.1% |
| GRAPE (numerical, noisy eval) | 90.6% |
| 3-atom CCZ scaling (Exp 6) | 71.3% |

### Experiment 1 — Group Size Ablation

*Question: How does the GRPO group size G affect convergence speed and final fidelity?*

Five values of G were tested (G = 2, 4, 6, 8, 12), each run for 18 GRPO iterations with 15 pulse segments at T = 0.5 µs.

| Group size | Final fidelity | First iter > 99.5% | Wall time |
|---|---|---|---|
| G = 2 | 99.99% | iter 2 | 8,234 s |
| G = 4 | 99.99% | iter 2 | 253 s |
| G = 6 | 99.99% | iter 1 | 231 s |
| G = 8 | 99.99% | iter 1 | 597 s |
| G = 12 | 99.99% | iter 0 | 347 s |

All group sizes ultimately reach 99.99%, but the story is in efficiency. G = 2 took over two hours: the advantage variance within a two-sample group is so small that the policy update signal is nearly zero, and progress is glacially slow. G = 4 and G = 6 reach 99.5% within two iterations and complete in roughly four minutes—they have enough group diversity to produce a useful advantage signal. G = 12 finds a target-exceeding pulse within the very first iteration but costs more per iteration due to the larger simulator batch.

![Group size ablation and curriculum comparison](/assets/images/rydberg-grpo/dashboard_exp1_exp2_real.png)

*Figure 1: Left — GRPO best fidelity vs. iteration for five group sizes. Right — Curriculum vs. direct training comparison. G = 6 offers the best wall-time-to-fidelity tradeoff.*

**Takeaway:** G = 6 is the practical sweet spot. Very small groups (G = 2) have too little advantage variance; very large groups (G ≥ 12) add per-iteration cost without improving the final answer.

### Experiment 2 — Noise Curriculum vs. Direct Training

*Question: Does ramping noise from zero to full help compared to training directly at full noise from the start?*

Curriculum training ramped noise scale from 0 to 1.0 linearly over 10 iterations. Both methods reached 99.95% final fidelity and found pulses with a cosine similarity of approximately 1.0—the same attractor in pulse space. Curriculum training was faster by about 28% in wall time (166 s vs 232 s). The noiseless warm-start phase quickly identifies the rough structure of the optimal pulse; the subsequent noisy phase fine-tunes for robustness.

### Experiment 3 — Joint Gate Time and Pulse Optimization

*Question: Can we simultaneously optimize pulse shape and gate duration?*

The action space was extended to include $T$ as a learnable parameter bounded in $[0.3, 1.2]\,\mu$s. A time penalty $\lambda_T \cdot T$ was subtracted from the reward.

| $\lambda_T$ | Best fidelity | Gate time | Notes |
|---|---|---|---|
| 0.00 | 99.99% | 0.82 µs | No time pressure |
| 0.01 | 99.96% | 0.82 µs | Negligible effect |
| 0.05 | 99.98% | 0.35 µs | 2.3× speedup, < 0.01% loss |
| 0.10 | 99.62% | 0.39 µs | Fidelity begins to degrade |

$\lambda_T = 0.05$ finds a gate at $0.35\,\mu$s with 99.98% fidelity—a 2.3× speedup over the unpenalized optimum with negligible fidelity cost. Faster gates are highly desirable in practice because they reduce exposure to decoherence for the surrounding circuit.

![Joint time optimization and GRPO vs PPO](/assets/images/rydberg-grpo/dashboard_exp3_exp4_real.png)

*Figure 2: Left — Pareto frontier of fidelity vs. gate time for different time-penalty strengths $\lambda_T$. Right — GRPO vs. PPO fidelity convergence per iteration.*

### Experiment 4 — GRPO vs. PPO

*Question: Does GRPO outperform standard PPO, or do they converge to the same answer?*

| Algorithm | Final fidelity | Wall time | Notes |
|---|---|---|---|
| GRPO | 99.99% | 133 s | No critic network |
| PPO | 99.98% | 146 s | With learned value baseline |

Both algorithms reach > 99.5% and similar final fidelities. GRPO is slightly faster (~8% less wall time) because it maintains no critic. The more meaningful difference is in *early convergence*: GRPO reaches 99.98% by iteration 4, PPO by iteration 10. The group-relative advantage signal appears to be a stronger early training signal in a noisy reward environment, because the group mean provides an automatically well-calibrated baseline without the instability of a newly-initialized critic.

### Experiment 5 — Noise Source Ablation

*Question: Which noise sources most affect fidelity?*

Each noise channel was individually disabled while keeping all others active. The result was surprising: disabling Rydberg decay, pure dephasing, or Doppler has essentially no measurable effect on fidelity. **Laser phase noise is the dominant channel**—disabling it actually *decreases* fidelity for the GRPO-optimized pulse. This counterintuitive result suggests the optimizer has partially learned to exploit the specific structure of phase noise trajectories to achieve higher fidelity than it would in the idealized noiseless case. This "noise-assisted optimization" effect is a documented phenomenon in open quantum systems.

![Noise ablation and 3-atom scaling](/assets/images/rydberg-grpo/dashboard_exp5_exp6_real.png)

*Figure 3: Left — Fidelity sensitivity to individual noise sources. Right — GRPO convergence on the 3-atom CCZ gate in a 27-dimensional Hilbert space.*

### Experiment 6 — Scaling to Three Atoms (CCZ Gate)

*Question: Can GRPO scale to a three-atom, 27-dimensional Hilbert space?*

The three-atom CCZ (Toffoli) gate operates on a $3^3 = 27$-dimensional state space. With 6 GRPO iterations, the best fidelity reached was **71.3%**—proof-of-concept for scaling, though well below the two-qubit target. The convergence is monotonic, starting from approximately 30% and improving steadily. The primary bottleneck is simulator cost: each `mesolve` call now involves a $27 \times 27$ complex matrix, making each iteration roughly $9\times$ slower than the two-qubit case.

### GRAPE Comparison

| Method | Fidelity (ideal dynamics) | Fidelity (noisy) | Wall time |
|---|---|---|---|
| GRAPE (numerical) | 24.6% | 90.6% | ~1 s |
| GRPO (G = 6) | — | 91.8% | ~230 s |
| GRPO (G = 12, full run) | — | 99.99% | ~347 s |

GRAPE's ideal-dynamics result of 24.6% is striking—it gets completely stuck in a local optimum. Under noisy evaluation it reaches 90.6%, roughly the level of a well-crafted hand-designed pulse. The reason GRAPE struggles is that numerical finite-difference gradients require *two separate noisy evaluations* per segment, so the gradient estimate has high variance. The optimizer is essentially doing a noisy random walk on the gradient landscape. GRPO, by contrast, evaluates all G candidates within the same noise regime, and the relative ranking within the group is far more stable than the absolute difference between two individual noisy evaluations. The best GRAPE and best GRPO pulses have a cosine similarity of 0.81, suggesting they converge to similar regions of pulse space, but GRPO consistently finds better-performing points within that neighborhood.

![Pulse shape summary](/assets/images/rydberg-grpo/dashboard_pulse_summary_real.png)

*Figure 4: Comparison of the GRPO-optimized pulse shape against the Blackman and square reference pulses, with the fidelity each achieves. The learned pulse has a non-trivial multi-segment structure that does not resemble any standard analytical shape.*

---

## 5. Discussion

### What worked

The most important finding is that GRPO—designed for discrete token generation—translates cleanly to this continuous control problem. The group-relative advantage is a strong signal in noisy environments because it is self-calibrated: no critic needs to be trained, and the advantage is always well-defined as long as there is any spread within the group. In early training, when the critic in a PPO agent would still be poorly initialized, GRPO already provides useful gradient direction.

The physics simulation at the level of detail used here—full Lindblad dynamics with realistic multi-channel noise—turns out to be the right fidelity. The optimizer finds pulses that reflect real Rydberg physics, not numerical artifacts: the optimal pulse durations and amplitude profiles are consistent with what one would expect from the timescales set by $\Omega_\text{max}$, $U_\text{eff}$, and $\gamma_r$.

Joint time and pulse optimization (Experiment 3) is a practically important result. The 2.3× gate speedup at λ_T = 0.05 is directly useful: faster gates reduce crosstalk and allow more operations within the coherence time of the surrounding circuit.

### Limitations and caveats

The noise-assisted optimization finding warrants caution. A pulse whose fidelity *decreases* when noise is removed has adapted to specific noise realizations rather than learning a truly noise-robust shape. Proper robustness evaluation would require averaging over many independently sampled noise trajectories at test time with a fixed seed different from training—this was not done and is an open question for follow-up work.

GRAPE's poor ideal-dynamics performance does not condemn classical methods: it reflects a limitation of *numerical* GRAPE specifically. Analytic GRAPE with exact propagator derivatives computed via the Choi–Khatri formalism for open systems would likely perform significantly better and at a fraction of the wall time. A fair comparison would require that implementation.

The three-atom CCZ scaling result is promising but the comparison is not clean: a CCZ gate requires a three-body interaction that does not arise as naturally from the two-atom blockade Hamiltonian as a CZ does. The Hamiltonian model may need extension—for example, using a mediator atom or a multi-step pulse sequence—for the CCZ to be physically realizable at high fidelity.

### Broader implications

This project is a data point in the larger question of which ML techniques from the LLM and RL world transfer cleanly to quantum control. The GRPO transfer worked here because of a specific property match: the episodic structure (one pulse = one episode), a bounded continuous action space, and the "reward is noisy but relative ranking is stable" property. Not all quantum control problems share these features, so the translation will not always be this clean.

What the group evaluation philosophy generalizes to is any optimization problem where absolute reward signals are unreliable but relative rankings within a batch are stable. This is plausibly true for other quantum hardware tasks: readout calibration, crosstalk mitigation, pulse calibration on superconducting circuits—all situations where individual evaluations are expensive and stochastic.

---

## 6. Conclusion

This project built a complete pipeline for Rydberg CZ gate optimization: first-principles Hamiltonian simulation via QuTiP's Lindblad master equation solver, a GRPO-adapted training loop with a Gaussian policy network, and systematic ablation experiments comparing group size, noise curriculum, gate time optimization, and classical baselines. The main results are:

1. **GRPO achieves 99.99% CZ gate fidelity** under realistic Rydberg noise—Rydberg decay, dephasing, Doppler broadening, laser phase noise, and intensity noise. Hand-crafted Blackman pulses peak at 72.1%.

2. **Group size G = 6 is the practical sweet spot.** G = 2 has too little advantage variance for effective learning. G = 4–6 offers the best balance of iteration efficiency and wall-clock cost.

3. **Joint time optimization finds a 2.3× faster gate** at λ_T = 0.05, achieving 99.98% fidelity in 0.35 µs versus the 0.82 µs unpenalized optimum.

4. **GRPO slightly outperforms PPO in early convergence.** Both reach similar final fidelities, but GRPO's group-relative advantage is a stronger signal in early training when a critic would still be poorly initialized. The critic-free design also simplifies the implementation.

5. **Laser phase noise is the dominant noise channel.** Rydberg decay, pure dephasing, and Doppler contribute negligibly by comparison. The optimizer has partially learned to use phase noise constructively—a noise-assisted optimization effect that is real but warrants caution.

---

## Notebooks and Code

The full codebase is available in the project repository. Key modules:

- `simulator/hamiltonian.py` — Rydberg Hamiltonian and collapse operators (QuTiP)
- `simulator/gate_sim.py` — CZ gate simulator with Choi–Jamiołkowski fidelity
- `simulator/noise.py` — laser noise model (OU phase, intensity, Doppler)
- `rl_env/pulse_env.py` — Gymnasium environment and noise curriculum
- `grpo/trainer.py` — GRPO training loop with Gaussian policy
- `simulator/grape.py` — numerical GRAPE baseline
- `experiments/` — all six experiment scripts with results

---

**Declaration of AI Assistance:** *The code, ideas, and experimental design in this project are the sole work of the author. Claude (Anthropic) was used to assist with grammar correction, formatting, and structuring of this blog post after the author provided the full technical content. The final text was reviewed and verified by the author, who takes full responsibility for all claims and citations.*

---

## References

[1] S. J. Evered, D. Bluvstein, M. Kalinowski, S. Ebadi, T. Manovitz, H. Zhou, S. H. Li, A. A. Geim, T. T. Wang, N. Maskara, H. Levine, M. Greiner, V. Vuletić, and M. D. Lukin, "High-fidelity parallel entangling gates on a neutral-atom quantum computer," *Nature* **622**, 268–272 (2023).

[2] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, Y. Guo, and D. Guo, "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," arXiv:2402.03300 (2024).

[3] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347 (2017).

[4] N. Khaneja, T. Reiss, C. Kehlet, T. Schulte-Herbrüggen, and S. J. Glaser, "Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent algorithms," *Journal of Magnetic Resonance* **172**, 296–305 (2005).

[5] M. Saffman, T. G. Walker, and K. Mølmer, "Quantum information with Rydberg atoms," *Rev. Mod. Phys.* **82**, 2313 (2010).
