---
layout: post
title: "High-Fidelity CZ Gate Optimization via GRPO on Rydberg Neutral Atoms"
date: 2026-05-30 00:00:00 +0000
categories: [Quantum Computing, Quantum Optimal Control, Reinforcement Learning, Rydberg Atoms]
tags: [Quantum Computing, Rydberg, CZ Gate, GRPO, Reinforcement Learning, Quantum Control, QuTiP]
---

**Author:** Ahmed Samir

---

Quantum computing has been gaining serious traction, not just as a theoretical curiosity but as a practical engineering discipline. The race toward fault-tolerant quantum processors has exposed a fundamental bottleneck: we need two-qubit gate fidelities above roughly 99.5% to make quantum error correction actually work. Below that threshold, you spend more qubits fixing errors than doing computation.

Neutral atoms trapped in optical tweezers have emerged as one of the most promising platforms to get there. The Harvard/MIT group demonstrated a CZ gate at 99.5% fidelity using Rydberg blockade in 2023 [1]—a landmark result. Achieving that, however, requires extremely precise laser pulse control. Small deviations in pulse shape, laser phase, or atom temperature can collapse fidelity from near-perfect to barely usable.
![Neutral atom processor](/assets/images/rydberg-grpo/neutral_atom_processor.png)

At the same time, a seemingly unrelated development happened in the AI world. DeepSeek's **GRPO** algorithm—Group Relative Policy Optimization—showed that you don't need a learned value function to train language models to reason about math [2]. Instead of comparing a candidate answer against some absolute baseline, GRPO evaluates a *group* of candidates against each other. The intuition is that relative comparison is often more reliable than absolute scoring, especially when reward signals are noisy.

The connection between these two worlds is non-obvious but real. Quantum gate optimization shares several properties with the mathematical reasoning tasks that made GRPO effective: the reward landscape is noisy (quantum simulation has stochastic elements), there are many local optima, and a single evaluation is expensive. This project adapts GRPO to optimize laser pulses for a Rydberg CZ gate, and asks whether its group-relative philosophy offers concrete advantages over classical gradient methods.

---

## 1. Problem Definition

### The Physical System

We work with a **two-atom Rydberg system**. Each atom is trapped in an optical tweezer and has three accessible energy levels:

- $\vert 0 \rangle$ — ground hyperfine state (qubit "zero")
- $\vert 1 \rangle$ — ground hyperfine state (qubit "one")
- $\vert r \rangle$ — highly excited Rydberg state (principal quantum number $n \approx 70$)

To understand why the $\vert r \rangle$ state is special, consider what happens physically. In the ground states $\vert 0 \rangle$ and $\vert 1 \rangle$, the atom's valence electron sits close to the nucleus in a low-energy, highly stable orbit. When a precisely tuned laser kicks that electron into the Rydberg state, it travels to an orbital radius roughly $n^2 \approx 5000$ times larger than the ground state. The atom balloons to a size comparable to a biological cell. At that scale, the far-separated positive nucleus and negative electron create a huge electric dipole moment—and that dipole is the key to everything that follows.

![Rydberg state](/assets/images/rydberg-grpo/rydberg_state.png)

### The Target Gate

We want to implement a **Controlled-Z (CZ) gate**, which applies a phase flip to the $\vert 11 \rangle$ state and leaves all others unchanged:

$$U_\text{CZ} = \text{diag}(1,\, 1,\, 1,\, -1)$$

This gate is maximally entangling: combined with single-qubit rotations, it is universal for quantum computation. It can generate Bell states directly from product-state inputs.

### The Rydberg Blockade

The CZ gate is implemented via the **Rydberg blockade**. When one atom enters $\vert r \rangle$, its giant electric dipole shifts the energy levels of any neighboring atom via a van der Waals interaction that scales as $r^{-6}$:

$$U_\text{eff} = U_\text{blockade} \times \left(\frac{4\,\mu\text{m}}{r_\text{sep}}\right)^6$$

At a separation of 4 µm with $U_\text{blockade} = 2\pi \times 10\,\text{MHz}$, this shift is an order of magnitude larger than the laser drive ($\Omega_\text{max} = 2\pi \times 1\,\text{MHz}$). The consequence is decisive: if atom A is already in $\vert r \rangle$, the transition frequency of atom B has been pushed so far off-resonance that the same laser can no longer excite it. The state $\vert rr \rangle$ is effectively forbidden. This conditional suppression is the **Rydberg blockade**, and it is the entire physical basis for the two-qubit interaction.

### The Three-Pulse CZ Protocol

The blockade turns into a CZ gate through a specific sequence of three laser pulses. The following derivation tracks every computational basis state through the protocol exactly, including the phase compensation step required to reach the standard CZ form. A detailed treatment of this protocol is given in the Bloqade.jl documentation [6].

#### Laser pulse rules

Two types of pulses are used, both tuned to drive the $\vert 1 \rangle \leftrightarrow \vert r \rangle$ transition. Lasers are targeted: a pulse aimed at one atom has no effect on the other. Crucially, lasers only couple to $\vert 1 \rangle$—an atom in $\vert 0 \rangle$ is always unaffected.

- **$\pi$ pulse on atom $i$**: drives the atom halfway through a Rabi cycle, $\vert 1 \rangle \rightarrow \vert r \rangle$ or $\vert r \rangle \rightarrow \vert 1 \rangle$. No phase is accumulated.
- **$2\pi$ pulse on atom $i$**: drives the atom through a *complete* Rabi cycle, $\vert 1 \rangle \rightarrow \vert r \rangle \rightarrow \vert 1 \rangle$. Completing a full loop injects a phase factor of $e^{i\pi} = -1$, so the state picks up a global minus sign:

$$\vert 1 \rangle \xrightarrow{2\pi} -\vert 1 \rangle$$

#### The three pulses

**Pulse 1 — $\pi$ on control:** If the control atom is in $\vert 1 \rangle$, excite it to $\vert r \rangle$. If it is in $\vert 0 \rangle$, do nothing.

**Pulse 2 — $2\pi$ on target:** Attempt to drive the target atom through a full $\vert 1 \rangle \rightarrow \vert r \rangle \rightarrow \vert 1 \rangle$ cycle. Whether this succeeds depends on the control atom's state.

**Pulse 3 — $\pi$ on control:** If the control atom is in $\vert r \rangle$, return it to $\vert 1 \rangle$. If it is in $\vert 0 \rangle$, do nothing.

#### State-by-state tracking

**State $\vert 00 \rangle$:**  
Pulse 1: Control is $\vert 0 \rangle$ — ignored. State: $\vert 00 \rangle$.  
Pulse 2: Target is $\vert 0 \rangle$ — ignored. State: $\vert 00 \rangle$.  
Pulse 3: Control is $\vert 0 \rangle$ — ignored.  

$$\vert 00 \rangle \;\longrightarrow\; +\vert 00 \rangle$$

**State $\vert 01 \rangle$ (control $= 0$, target $= 1$):**  
Pulse 1: Control is $\vert 0 \rangle$ — ignored. Blockade is inactive. State: $\vert 01 \rangle$.  
Pulse 2: Target is $\vert 1 \rangle$. No blockade. Full $2\pi$ cycle completes. Phase $-1$ is acquired. State: $-\vert 01 \rangle$.  
Pulse 3: Control is $\vert 0 \rangle$ — ignored.  

$$\vert 01 \rangle \;\longrightarrow\; -\vert 01 \rangle$$

**State $\vert 10 \rangle$ (control $= 1$, target $= 0$):**  
Pulse 1: Control is $\vert 1 \rangle$. $\pi$ pulse moves it to $\vert r \rangle$. Blockade now active. State: $\vert r0 \rangle$.  
Pulse 2: Target is $\vert 0 \rangle$ — ignored regardless of blockade. State: $\vert r0 \rangle$.  
Pulse 3: Control is in $\vert r \rangle$. $\pi$ pulse returns it to $\vert 1 \rangle$. The control atom has completed half a loop in Pulse 1 and half a loop in Pulse 3—one full $2\pi$ loop in total—so it acquires the $-1$ phase.  

$$\vert 10 \rangle \;\longrightarrow\; -\vert 10 \rangle$$

**State $\vert 11 \rangle$ (control $= 1$, target $= 1$):**  
Pulse 1: Control is $\vert 1 \rangle$. $\pi$ pulse moves it to $\vert r \rangle$. Blockade now active. State: $\vert r1 \rangle$.  
Pulse 2: Target is $\vert 1 \rangle$ and the laser fires, but the blockade has shifted the target atom's $\vert 1 \rangle \rightarrow \vert r \rangle$ transition off-resonance. The laser cannot excite it. The target atom stays in $\vert 1 \rangle$ and acquires **no phase**. State: $\vert r1 \rangle$.  
Pulse 3: Control returns from $\vert r \rangle$ to $\vert 1 \rangle$, completing its own full loop and picking up $-1$.  

$$\vert 11 \rangle \;\longrightarrow\; -\vert 11 \rangle$$

#### The raw output and phase compensation

After the three pulses, the raw transformation is:

$$\vert 00 \rangle \rightarrow +\vert 00 \rangle, \quad \vert 01 \rangle \rightarrow -\vert 01 \rangle, \quad \vert 10 \rangle \rightarrow -\vert 10 \rangle, \quad \vert 11 \rangle \rightarrow -\vert 11 \rangle$$

This is *not* a CZ gate yet. The $\vert 01 \rangle$ and $\vert 10 \rangle$ states carry unwanted minus signs. To remove them, we apply a **single-qubit Z-rotation** to each atom independently. A Z-rotation flips the sign of an atom if and only if it is in $\vert 1 \rangle$:

| Raw state | Correction applied | Final state |
|---|---|---|
| $+\vert 00 \rangle$ | Neither atom is $\vert 1 \rangle$; no flip | $+\vert 00 \rangle$ |
| $-\vert 01 \rangle$ | Target is $\vert 1 \rangle$; one flip: $(-1)(-1) = +1$ | $+\vert 01 \rangle$ |
| $-\vert 10 \rangle$ | Control is $\vert 1 \rangle$; one flip: $(-1)(-1) = +1$ | $+\vert 10 \rangle$ |
| $-\vert 11 \rangle$ | Both are $\vert 1 \rangle$; two flips: $(-1)(-1)(-1) = -1$ | $-\vert 11 \rangle$ |

The double flip on $\vert 11 \rangle$ cancels, preserving its negative sign. After compensation, the transformation is exactly the CZ gate:

$$\vert 00 \rangle \rightarrow \vert 00 \rangle, \quad \vert 01 \rangle \rightarrow \vert 01 \rangle, \quad \vert 10 \rangle \rightarrow \vert 10 \rangle, \quad \vert 11 \rangle \rightarrow -\vert 11 \rangle$$



### The Optimization Problem

The analytical three-pulse protocol described above achieves a CZ gate under idealized conditions. In practice, real laser pulses are not perfect square waves—they have finite rise times, fluctuating intensities, and phase noise—and the atoms sit in a noisy thermal environment. The three-pulse decomposition is therefore a starting point, not the final answer. The goal of this project is to learn a single continuous pulse envelope $\Omega(t)$ that implements the same transformation more robustly, without being constrained to the discrete three-step structure.

The control problem is: given a fixed gate time $T$, find a laser pulse envelope $\Omega(t)$ such that the resulting quantum process is as close as possible to the ideal CZ gate. We parameterize the pulse as a **piecewise-constant sequence** of $N = 20$ amplitude segments, each normalized to $[0, 1]$ and scaled by $\Omega_\text{max}$. This gives a 20-dimensional continuous action space:

$$\text{maximize} \quad F(\mathbf{u}) \quad \text{subject to} \quad \mathbf{u} \in [0,1]^{20}, \quad T \in [0.3,\, 1.2]\,\mu\text{s}$$

where $F(\mathbf{u})$ is the average gate fidelity computed under a full Lindblad noise model.

---

## 2. Simulation Model

### The Hamiltonian

The full two-atom Hamiltonian is built in the rotating frame under the rotating wave approximation (RWA):

$$H(t) = \sum_{i \in \{A,B\}} \left[ \frac{\Omega(t)}{2} \left( \vert r \rangle\langle 1 \vert_i + \text{h.c.} \right) - \Delta(t)\,   \right] + U_\text{eff}\, \vert rr \rangle\langle rr \vert$$

The three terms are:

1. **The drive term** ($\Omega(t)/2 \cdot \sigma_{r1} + \text{h.c.}$): the laser couples $\vert 1 \rangle$ and $\vert r \rangle$ with Rabi frequency $\Omega(t)$. The operator $\vert r \rangle\langle 1 \vert$ promotes an atom from $\vert 1 \rangle$ to $\vert r \rangle$; its conjugate does the reverse. Together they produce coherent oscillation (Rabi flopping) between the two levels. This term acts independently on both atoms A and B via the tensor product structure.

2. **The detuning term** ($-\Delta$): the detuning $\Delta$ is the mismatch between the laser frequency and the atom's natural $\vert 1 \rangle \to \vert r \rangle$ transition frequency. At $\Delta = 0$ (resonance), the drive is maximally effective. 

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

The Lindblad master equation is integrated numerically using QuTiP's `mesolve` function over 200 time steps covering the gate duration $T$.

### Fidelity Metric

Gate fidelity is computed via the **Choi–Jamiołkowski (CJ) isomorphism**. 

<!-- The protocol:

1. Evolve each of the four computational basis states $\vert 00 \rangle, \vert 01 \rangle, \vert 10 \rangle, \vert 11 \rangle$ through the noisy Lindblad dynamics.
2. Extract the final density matrix projected onto the 4-dimensional computational subspace (indices $[0, 1, 3, 4]$ in the 9D space, corresponding to $\vert 00 \rangle, \vert 01 \rangle, \vert 10 \rangle, \vert 11 \rangle$).
3. Compare each $\rho_\text{out}^{(i)}$ against the ideal CZ output $\rho_\text{ideal}^{(i)} = U_\text{CZ} \vert i \rangle\langle i \vert U_\text{CZ}^\dagger$.

$$F_\text{process} = \frac{1}{d} \sum_{i=0}^{d-1} \text{Tr}\!\left[\rho_\text{ideal}^{(i)}\, \rho_\text{out}^{(i)}\right], \qquad F_\text{avg} = \frac{d \cdot F_\text{process} + 1}{d + 1}$$

where $d = 4$ is the Hilbert space dimension. This formula, standard in quantum information, corrects for the trivial overlap contribution and gives the average fidelity over all possible input states. -->

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

This requires $2N = 40$ simulator calls per gradient step, compared to GRPO. Updates use gradient ascent with momentum:

$$\mathbf{v} \leftarrow \beta_m\, \mathbf{v} + \alpha\, \nabla F, \qquad \mathbf{u} \leftarrow \text{clip}(\mathbf{u} + \mathbf{v},\; 0,\; 1)$$

with momentum coefficient $\beta_m = 0.9$ and learning rate $\alpha = 0.05$. 

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

Five values of G were tested (G = 2, 4, 6, 8, 12), each run for 18 GRPO iterations with 20 pulse segments at T = 0.5 µs.

| Group size | Final fidelity | First iter > 99.5% | Wall time |
|---|---|---|---|
| G = 2 | 99.99% | iter 2 | 21,734 s |
| G = 4 | 99.99% | iter 2 | 4,153 s |
| G = 6 | 99.99% | iter 1 | 4,231 s |
| G = 8 | 99.99% | iter 1 | 1,597 s |
| G = 12 | 99.99% | iter 0 | 1,347 s |

All group sizes ultimately reach 99.99%, but the story is in efficiency. G = 2 took over 6 hours: the advantage variance within a two-sample group is so small that the policy update signal is nearly zero, and progress is glacially slow. G = 4 and G = 6 reach 99.5% within two iterations and complete in around one hour—they have enough group diversity to produce a useful advantage signal. G = 12 finds a target-exceeding pulse within the very first iteration but costs more per iteration due to the larger simulator batch.

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
| GRPO | 99.99% | 2,133 s | No critic network |
| PPO | 99.98% | 3,146 s | With learned value baseline |

Both algorithms reach > 99.5% and similar final fidelities. GRPO is slightly faster (~14% less wall time) because it maintains no critic. The more meaningful difference is in *early convergence*: GRPO reaches 99.98% by iteration 4, PPO by iteration 10. The group-relative advantage signal appears to be a stronger early training signal in a noisy reward environment, because the group mean provides an automatically well-calibrated baseline without the instability of a newly-initialized critic.

### Experiment 5 — Noise Source Ablation

*Question: Which noise sources most affect fidelity?*

Each noise channel was individually disabled while keeping all others active. The result was surprising: disabling Rydberg decay, pure dephasing, or Doppler has essentially no measurable effect on fidelity. **Laser phase noise is the dominant channel**—disabling it actually *decreases* fidelity for the GRPO-optimized pulse. This counterintuitive result suggests the optimizer has partially learned to exploit the specific structure of phase noise trajectories to achieve higher fidelity than it would in the idealized noiseless case. This "noise-assisted optimization" effect is a documented phenomenon in open quantum systems.

![Noise ablation and 3-atom scaling](/assets/images/rydberg-grpo/dashboard_exp5_exp6_real.png)

*Figure 3: Left — Fidelity sensitivity to individual noise sources. Right — GRPO convergence on the 3-atom CCZ gate in a 27-dimensional Hilbert space.*

### Experiment 6 — Scaling to Three Atoms (CCZ Gate)

*Question: Can GRPO scale to a three-atom, 27-dimensional Hilbert space?*

With 6 GRPO iterations, the best fidelity reached was **71.3%**—proof-of-concept for scaling, though well below the two-qubit target. The convergence is monotonic, starting from approximately 30% and improving steadily. The primary bottleneck is simulator cost: each `mesolve` call, making each iteration roughly $9\times$ slower than the two-qubit case.

### GRAPE Comparison

| Method | Fidelity (ideal dynamics) | Fidelity (noisy) | Wall time |
|---|---|---|---|
| GRAPE (numerical) | 24.6% | 90.6% | ~500 s |
| GRPO (G = 6) | — | 91.8% | ~4,130 s |
| GRPO (G = 12, full run) | — | 99.99% | ~1,347 s |

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

---

## 6. Conclusion

This project built a complete pipeline for Rydberg CZ gate optimization: first-principles Hamiltonian simulation via QuTiP's Lindblad master equation solver, a GRPO-adapted training loop with a Gaussian policy network, and systematic ablation experiments comparing group size, noise curriculum, gate time optimization, and classical baselines. The main results are:

1. **GRPO achieves 99.99% CZ gate fidelity** under realistic Rydberg noise—Rydberg decay, dephasing, Doppler broadening, laser phase noise, and intensity noise. Hand-crafted Blackman pulses peak at 72.1%.

2. **Group size G = 6 is the practical sweet spot.** G = 2 has too little advantage variance for effective learning. G = 4–6 offers the best balance of iteration efficiency and wall-clock cost.

3. **Joint time optimization finds a 2.3× faster gate** at λ_T = 0.05, achieving 99.98% fidelity in 0.35 µs versus the 0.82 µs unpenalized optimum.

4. **GRPO slightly outperforms PPO in early convergence.** Both reach similar final fidelities, but GRPO's group-relative advantage is a stronger signal in early training when a critic would still be poorly initialized. The critic-free design also simplifies the implementation.

---

## Appendix

### A. The Full Optimization Cost Function

In the quantum optimal control (QOC) framing adopted in this project and in related DRL-based Rydberg gate work [7], the pulse design problem is cast as minimizing a composite cost function $\mathcal{J}$ that simultaneously targets gate quality, physical decay, and gate speed:

$$\min_{\vec{\Omega}(t),\, T_{\text{gate}}} \mathcal{J} = \mathcal{I}_{\text{gate}} + \eta_p \,\Gamma_p(t) + \eta_r \,\Gamma_r(t) + \lambda\, T_{\text{gate}}$$

Each term encodes a distinct physical objective. The total cost $\mathcal{J}$ is what the agent implicitly minimizes when it maximizes the shaped reward $R = 1 - \mathcal{J}$.

#### A.1 Gate infidelity $\mathcal{I}_{\text{gate}}$

The primary term measures how far the simulated gate $U_{\text{sim}}$ deviates from the ideal CZ operation $U_{\text{CZ}}$:

$$\mathcal{I}_{\text{gate}} = 1 - \mathcal{F}_{\text{gate}}$$

where $\mathcal{F}_{\text{gate}}$ is the average gate fidelity. In the trace-overlap form it reads:

$$\mathcal{F}_{\text{gate}} = \frac{1}{d^2} \left\vert \operatorname{Tr}\!\left( U_{\text{CZ}}^\dagger \cdot U_{\text{sim}}(T_{\text{gate}}) \right) \right\vert^2, \qquad d = 4$$

This measures the normalized overlap between the ideal and simulated unitaries across all $d^2 = 16$ matrix elements. When the gate is perfect, $U_{\text{sim}} = U_{\text{CZ}}$ and the trace equals $d$, giving $\mathcal{F}_{\text{gate}} = 1$ and $\mathcal{I}_{\text{gate}} = 0$.

In the open-system (Lindblad) setting used in this project, $U_{\text{sim}}$ is replaced by the noisy process map $\mathcal{E}$, and the fidelity is computed as the Choi–Jamiołkowski process fidelity described in Appendix C. The two expressions are equivalent in the noiseless limit and are both conventions for the same quantity.

#### A.2 Decay penalties $\Gamma_p(t)$ and $\Gamma_r(t)$

Because atoms must pass through the short-lived Rydberg state $| r \rangle$ (and in a two-photon drive scheme, through an intermediate excited state $| p \rangle$) to interact, the optimizer must also limit the time each atom spends in those fragile states. The decay penalties are computed as the time-integrated average population of each dangerous state:

$$\Gamma_{p,r}(t) = \gamma_{p,r} \int_0^{T_{\text{gate}}} \bar{n}_{p,r}(t)\, dt$$

Here $\bar{n}_{p,r}(t)$ is the ensemble-averaged population of state $| p \rangle$ or $| r \rangle$ at time $t$, obtained from the density matrix $\rho(t)$ as $\bar{n}_r(t) = \operatorname{Tr}[| r\rangle\langle r| \otimes I\, \rho(t)] + \operatorname{Tr}[I \otimes | r\rangle\langle r|\, \rho(t)]$. The rates $\gamma_{p,r}$ are the physical spontaneous emission rates of those states; $\eta_p$ and $\eta_r$ are dimensionless regularization weights that control how strongly the optimizer avoids them.

The physical intuition is that spontaneous emission from $| r \rangle$ is the dominant decoherence channel in neutral-atom processors. A pulse that achieves the correct unitary but keeps the atoms in $| r \rangle$ for a long time will still suffer high decoherence in practice. By penalizing $\Gamma_r$ directly, the optimizer learns to find pulses that dart in and out of the Rydberg manifold as quickly as possible—the key insight behind the IU-DRL framework of Cai et al. [7], which autonomously discovers an early-cutoff policy by suppressing unnecessary Rydberg-state population.

In the present project, this penalty is implemented implicitly: the Lindblad collapse operators $L_k = \sqrt{\gamma_r}\,| 1\rangle\langle r|$ already subtract probability from $| r \rangle$ at rate $\gamma_r$ during every integration step of `mesolve`, so the fidelity automatically drops whenever the atom spends excess time in $| r \rangle$. The explicit $\Gamma_r$ term makes this dependence transparent and allows weighting it separately from gate infidelity.

#### A.3 Gate time penalty $\lambda\, T_{\text{gate}}$

The simplest term: a linear penalty proportional to the total gate duration. A larger $\lambda$ forces the optimizer toward shorter gates, at the cost of potentially sacrificing some fidelity. This is the same penalty explored in Experiment 3, where $\lambda_T = 0.05$ produced a 2.3× speedup with negligible fidelity loss.

The physical motivation is twofold. First, shorter gates reduce the total exposure time to all decoherence channels, not just Rydberg decay. Second, faster gates allow more operations within the coherence time of the surrounding quantum circuit.

#### A.4 Connection to the GRPO reward

In the GRPO framework, the agent maximizes a shaped reward rather than minimizing a cost. The reward used in this project,

$$R(F) = F^\alpha + B \cdot F \cdot \mathbf{1}[F \geq 0.995]$$

maps onto $\mathcal{J}$ via $R \approx 1 - \mathcal{J}$ at high fidelity. The $F^4$ shaping amplifies the gradient near $F = 1$ (where $1 - \mathcal{I}_{\text{gate}}$ is large) and compresses it near $F = 0$ (where most of the loss is irrecoverable anyway). The bonus term creates a hard cliff at the fault-tolerance threshold, providing the same role as the regularization weights $\eta_{p,r}$ and $\lambda$: it tells the agent which region of the solution space is physically acceptable.

---

### B. The Lindblad Master Equation

The Lindblad master equation is the standard framework for describing the time evolution of a quantum system that is weakly coupled to an external environment (a "bath"). It generalizes the Schrödinger equation to open systems, where energy and quantum coherence can leak out.

#### B.1 Why a density matrix instead of a state vector

For a perfectly isolated quantum system, the state is a vector $| \psi \rangle$ in Hilbert space. For a system that interacts with an environment, the state of the system alone cannot be represented as a pure vector—it is a statistical mixture of possible quantum states. The **density matrix** $\rho$ captures both cases:

$$\rho = \sum_i p_i | \psi_i \rangle\langle \psi_i |, \qquad \operatorname{Tr}(\rho) = 1, \qquad \rho \geq 0$$

The diagonal entries $\rho_{ii}$ give the probability of finding the system in basis state $| i \rangle$. The off-diagonal entries $\rho_{ij}$ ($i \neq j$) are **coherences**: they encode quantum superposition. When the environment disturbs the system, coherences decay—a process called **decoherence**. The density matrix tracks this decay explicitly; a pure state vector cannot.

#### B.2 The equation

The Lindblad master equation governs how $\rho(t)$ evolves in time:

$$\frac{d\rho}{dt} = -i[H, \rho] + \sum_k \left( L_k \rho L_k^\dagger - \frac{1}{2} L_k^\dagger L_k \rho - \frac{1}{2} \rho L_k^\dagger L_k \right)$$

The two parts have distinct physical meanings.

**The coherent part** $-i[H, \rho]$: this is the commutator of the Hamiltonian with the density matrix. It is the density-matrix version of the Schrödinger equation—it describes the reversible, unitary evolution driven by the laser pulse and the atomic energy structure. If this were the only term, $\rho$ would stay pure forever.

**The dissipator** $\sum_k \mathcal{D}[L_k]\rho$: each term in the sum is a Lindblad dissipator, parameterized by a **collapse operator** $L_k$. Each collapse operator describes one physical noise channel. For a Rydberg atom:

The structure of the dissipator $L_k \rho L_k^\dagger - \frac{1}{2}L_k^\dagger L_k \rho - \frac{1}{2}\rho L_k^\dagger L_k$ is not arbitrary—it is the unique form that guarantees $\rho(t)$ remains a valid density matrix for all $t$ (positive semidefinite, trace-preserving). The first term $L_k \rho L_k^\dagger$ adds population at the destination of the jump; the two subtracted terms remove it from the source and enforce norm conservation.

#### B.3 Solution in this project

QuTiP's `mesolve` function integrates this equation numerically over 200 time steps for each of the four computational basis states $| 00\rangle, | 01\rangle, | 10\rangle, | 11\rangle$. The result is four density matrices $\rho_{\text{out}}^{(i)}(T_{\text{gate}})$, one per input state, which are then fed into the fidelity calculation described in Appendix C.

---

### C. The Choi–Jamiołkowski Fidelity

#### C.1 The problem with testing one state

Measuring a quantum gate by running it on a single input state and checking the output is insufficient. A gate that perfectly maps $| 00\rangle \to | 00\rangle$ might still be completely broken for inputs like $| ++\rangle = (| 00\rangle + | 01\rangle + | 10\rangle + | 11\rangle)/2$ (a superposition). To certify a gate, you need to know how it performs on *all* possible inputs simultaneously.

#### C.2 The Choi–Jamiołkowski isomorphism

The Choi–Jamiołkowski (CJ) isomorphism is an exact mathematical correspondence between a quantum operation (a map from density matrices to density matrices) and a single density matrix called the **Choi state**. It works by running the quantum process on one half of a maximally entangled state:

$$| \Phi^+ \rangle = \frac{1}{d} \sum_{i=0}^{d-1} | i \rangle \otimes | i \rangle$$

The Choi state of a process $\mathcal{E}$ is $\chi_\mathcal{E} = (\mathcal{I} \otimes \mathcal{E})|\Phi^+\rangle\langle\Phi^+|$—you apply the process to one half of the entangled pair and leave the other half untouched. The result encodes the full behavior of $\mathcal{E}$ on all inputs at once, because the entangled pair "probes" all input states simultaneously.

#### C.3 Process fidelity and average gate fidelity

The **process fidelity** between the ideal gate $U_{\text{CZ}}$ and the simulated noisy process $\mathcal{E}$ is:

$$F_{\text{process}} = \langle \Phi^+_{\text{CZ}} | \chi_\mathcal{E} | \Phi^+_{\text{CZ}} \rangle$$

where $|\Phi^+_{\text{CZ}}\rangle$ is the Choi state of the ideal unitary. In practice this is computed as:

$$F_{\text{process}} = \frac{1}{d} \sum_{i=0}^{d-1} \operatorname{Tr}\!\left[ \rho_{\text{ideal}}^{(i)} \cdot \rho_{\text{out}}^{(i)} \right]$$

where $\rho_{\text{ideal}}^{(i)} = U_{\text{CZ}} | i\rangle\langle i| U_{\text{CZ}}^\dagger$ is what the ideal gate would produce from basis state $| i\rangle$, and $\rho_{\text{out}}^{(i)}$ is what the noisy simulation actually produces. The sum over all $d = 4$ basis states is what makes this a full process characterization rather than a single-input test.

The quantity $\operatorname{Tr}[\rho_{\text{ideal}} \cdot \rho_{\text{out}}]$ is the **Hilbert–Schmidt inner product** between two density matrices. It equals 1 when they are identical and 0 when they are orthogonal (completely different quantum states).

The standard conversion between process fidelity and **average gate fidelity** (averaged uniformly over all input states, not just the four basis states) is:

$$F_{\text{avg}} = \frac{d \cdot F_{\text{process}} + 1}{d + 1}$$

The $+1/(d+1)$ correction accounts for the fact that even a completely depolarizing channel (which destroys all quantum information) has a nonzero overlap with any target state due to the identity component. This formula ensures $F_{\text{avg}} = 1$ only for the perfect gate and $F_{\text{avg}} = 1/(d+1)$ for the fully depolarizing channel.

#### C.4 Why this is the right metric

The CJ fidelity is the correct metric for benchmarking a quantum gate for three reasons. First, it is basis-independent—it does not depend on which four input states you choose to test, as long as they span the Hilbert space. Second, it averages over all possible inputs, including superpositions that are not computational basis states. Third, it is directly connected to the physical error rate under randomized benchmarking protocols, which is the standard experimental tool for characterizing gate quality on real hardware.

---

**Declaration of AI Assistance:** *The code, ideas, and experimental design in this project are the sole work of the author. Claude (Anthropic) was used to assist with grammar correction, formatting, and structuring of this blog post after the author provided the full technical content. The final text was reviewed and verified by the author, who takes full responsibility for all claims and citations.*

---

## References

[1] S. J. Evered, D. Bluvstein, M. Kalinowski, S. Ebadi, T. Manovitz, H. Zhou, S. H. Li, A. A. Geim, T. T. Wang, N. Maskara, H. Levine, M. Greiner, V. Vuletić, and M. D. Lukin, "High-fidelity parallel entangling gates on a neutral-atom quantum computer," *Nature* **622**, 268–272 (2023).

[2] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, Y. Guo, and D. Guo, "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," arXiv:2402.03300 (2024).

[3] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347 (2017).

[4] N. Khaneja, T. Reiss, C. Kehlet, T. Schulte-Herbrüggen, and S. J. Glaser, "Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent algorithms," *Journal of Magnetic Resonance* **172**, 296–305 (2005).

[5] M. Saffman, T. G. Walker, and K. Mølmer, "Quantum information with Rydberg atoms," *Rev. Mod. Phys.* **82**, 2313 (2010).

[6] QuEra Computing, "Pulse-level CZ gate via Rydberg blockade," *Bloqade.jl Documentation*, [https://queracomputing.github.io/Bloqade.jl/dev/3-level/#pulse-CZ-gate](https://queracomputing.github.io/Bloqade.jl/dev/3-level/#pulse-CZ-gate) (accessed June 2026).

[7] Y. Cai, H. Zhang, K. Zhang, and J. Qian, "Intelligent Optimal Control of Rydberg Gates with Incremental-Update Deep Reinforcement Learning," arXiv:2605.04628 (2026).