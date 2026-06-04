---
layout: post
title: "High-Fidelity CZ Gate Optimization via GRPO on Rydberg Neutral Atoms"
date: 2026-06-01 00:00:00 +0000
categories: [Quantum Computing, Quantum Optimal Control, Reinforcement Learning, Rydberg Atoms]
tags: [Quantum Computing, Rydberg, CZ Gate, GRPO, Reinforcement Learning, Quantum Control, QuTiP]
---

**Author:** Ahmed Samir

---

## 1. Problem Definition

### The Physical System

Neutral Atom Processors is a an advanced quantum computing platform that uses array of atoms trapped in optical tweezers as qubits.
The interaction between this qubits is operate when exciting the atoms to Rydberg states by applying laser pulses that match the energy required to transition from the ground state to the excited Rydberg state. 
The Rydberg blockade is a phenomenon occurs when on atom is excited to Rydberg state, it shifts the energy levels of nearby atoms, preventing them from being excited to the Rydberg state. This allows for the implementation of two-qubit gates, such as the CZ gate, by controlling the excitation of one atom based on the state of another.
![Neutral atom processor](/assets/images/rydberg-grpo/neutral_atom_processor.png)
![Optical tweezers](/assets/images/rydberg-grpo/optical_tw.png)


We work with a two qubit Rydberg system. Each atom whould  has three energy states:
    
- $\vert 0 \rangle$ — ground state (qubit "zero")
- $\vert 1 \rangle$ — ground state (qubit "one")
- $\vert r \rangle$ — excited Rydberg state

In the ground states $\vert 0 \rangle$ and $\vert 1 \rangle$, the atom's outer most electron sits close to the nucleus in a low energy, highly stable orbit. When a laser with specific power that matches the gap between the electron and the atom hits that atom, it make the atom excited and the electron is kicked, to an orbital radius roughly $n^2 \approx 5000$ times larger than the ground state. This create a huge electric dipole moment and that dipole is the key to everything that follows.

![Rydberg state](/assets/images/rydberg-grpo/rydberg_state.png)

### The Target Gate


I decide to choose to implement a Controlled-Z (CZ) gate. The CZ gate is a fundamental two-qubit gate that applies a phase flip to the |11⟩ state while leaving the other states unchanged.

$$U_\text{CZ} = \text{diag}(1,\, 1,\, 1,\, -1)$$



### The Rydberg Blockade

The CZ gate is implemented via the Rydberg blockade. When one atom enters $\vert r \rangle$, its giant electric dipole shifts the energy levels of any neighboring atom to a very higher degree : 

$$E_\text{eff} = E_\text{Laser} + E_\text{blockade}$$

So, if the laser need to excite the atom to $\vert r \rangle$ is $E_\text{Laser}$, it will need $E_\text{Laser} + E_\text{blockade}$ to excite the same atom if its neighbor is already in $\vert r \rangle$. Since $E_\text{blockade}$ is much larger than the laser linewidth, so that the state $\vert rr \rangle$ is effectively forbidden. This conditional suppression is the Rydberg blockade, and it is the entire physical basis for the two qubit interaction.

### The Three Pulse CZ Protocol

The blockade turns into a CZ gate through a sequence of three laser pulses. The following derivation tracks every basis state through the protocol, including the phase compensation step required to reach the standard CZ form. A detailed explaination of this protocol is given in the Bloqade.jl documentation [6].

#### Laser pulse rules

Two types of pulses are used, both tuned to drive the $\vert 1 \rangle \leftrightarrow \vert r \rangle$ transition. Lasers are targeted: a pulse aimed at one atom has no effect on the other. Crucially, lasers only couple to $\vert 1 \rangle$, that is an atom in $\vert 0 \rangle$ is always unaffected.

- $\pi$ pulse on atom $i$: drives the atom halfway through a Rabi cycle, $\vert 1 \rangle \rightarrow \vert r \rangle$ or $\vert r \rangle \rightarrow \vert 1 \rangle$. No phase is accumulated.
- $2\pi$ pulse on atom $i$: drives the atom through a *complete* Rabi cycle, $\vert 1 \rangle \rightarrow \vert r \rangle \rightarrow \vert 1 \rangle$. Completing a full loop injects a phase factor of $e^{i\pi} = -1$, so the state picks up a global minus sign:

$$\vert 1 \rangle \xrightarrow{2\pi} -\vert 1 \rangle$$


#### State-by-state tracking

State $\vert 00 \rangle$:  
Pulse 1: Control is $\vert 0 \rangle$:  ignored. State: $\vert 00 \rangle$.  
Pulse 2: Target is $\vert 0 \rangle$:  ignored. State: $\vert 00 \rangle$.  
Pulse 3: Control is $\vert 0 \rangle$:  ignored.  

$$\vert 00 \rangle \;\longrightarrow\; +\vert 00 \rangle$$

State $\vert 01 \rangle$ (control $= 0$, target $= 1$):  
Pulse 1: Control is $\vert 0 \rangle$: ignored. Blockade is inactive. State: $\vert 01 \rangle$.  
Pulse 2: Target is $\vert 1 \rangle$: No blockade. Full $2\pi$ cycle completes. Phase $-1$ is acquired. State: $-\vert 01 \rangle$.  
Pulse 3: Control is $\vert 0 \rangle$: ignored.  

$$\vert 01 \rangle \;\longrightarrow\; -\vert 01 \rangle$$

State $\vert 10 \rangle$ (control $= 1$, target $= 0$):  
Pulse 1: Control is $\vert 1 \rangle$: $\pi$ pulse moves it to $\vert r \rangle$. Blockade now active. State: $\vert r0 \rangle$.  
Pulse 2: Target is $\vert 0 \rangle$: ignored regardless of blockade. State: $\vert r0 \rangle$.  
Pulse 3: Control is in $\vert r \rangle$: $\pi$ pulse returns it to $\vert 1 \rangle$. The control atom has completed half a loop in Pulse 1 and half a loop in Pulse 3—one full $2\pi$ loop in total—so it acquires the $-1$ phase.  

$$\vert 10 \rangle \;\longrightarrow\; -\vert 10 \rangle$$

State $\vert 11 \rangle$ (control $= 1$, target $= 1$):  
Pulse 1: Control is $\vert 1 \rangle$: $\pi$ pulse moves it to $\vert r \rangle$. Blockade now active. State: $\vert r1 \rangle$.  
Pulse 2: Target is $\vert 1 \rangle$ and the laser fires, but the blockade has shifted the target atom's $\vert 1 \rangle \rightarrow \vert r \rangle$ transition off-resonance. The laser cannot excite it. The target atom stays in $\vert 1 \rangle$ and acquires no phase. State: $\vert r1 \rangle$.  
Pulse 3: Control returns from $\vert r \rangle$ to $\vert 1 \rangle$, completing its own full loop and picking up $-1$.  

$$\vert 11 \rangle \;\longrightarrow\; -\vert 11 \rangle$$

#### The raw output and phase compensation

After the three pulses, the raw transformation is:

$$\vert 00 \rangle \rightarrow +\vert 00 \rangle, \quad \vert 01 \rangle \rightarrow -\vert 01 \rangle, \quad \vert 10 \rangle \rightarrow -\vert 10 \rangle, \quad \vert 11 \rangle \rightarrow -\vert 11 \rangle$$

This is not a CZ gate yet. The $\vert 01 \rangle$ and $\vert 10 \rangle$ states carry unwanted minus signs. To remove them, we apply a local qubit Z-rotation to each atom independently. A Z-rotation flips the sign of an atom if and only if it is in $\vert 1 \rangle$:

| Raw state | Correction applied | Final state |
|---|---|---|
| $+\vert 00 \rangle$ | Neither atom is $\vert 1 \rangle$; no flip | $+\vert 00 \rangle$ |
| $-\vert 01 \rangle$ | Target is $\vert 1 \rangle$; one flip: $(-1)(-1) = +1$ | $+\vert 01 \rangle$ |
| $-\vert 10 \rangle$ | Control is $\vert 1 \rangle$; one flip: $(-1)(-1) = +1$ | $+\vert 10 \rangle$ |
| $-\vert 11 \rangle$ | Both are $\vert 1 \rangle$; two flips: $(-1)(-1)(-1) = -1$ | $-\vert 11 \rangle$ |

After compensation, the transformation is exactly the CZ gate:

$$\vert 00 \rangle \rightarrow \vert 00 \rangle, \quad \vert 01 \rangle \rightarrow \vert 01 \rangle, \quad \vert 10 \rangle \rightarrow \vert 10 \rangle, \quad \vert 11 \rangle \rightarrow -\vert 11 \rangle$$



### The Optimization Problem

The analytical three pulse protocol described above achieves a CZ gate under ideal conditions. In practice, real laser pulses are not perfect square waves they have finite rise times, fluctuating intensities, and phase noise. Atoms sit in a noisy thermal environment. Therefore the goal of this project is to learn a single continuous laser pulse $\Omega(t)$ that implements the same gate robustly, without being constrained to the discrete three step structure.

The control problem is: for a fixed gate time $T$, find a laser pulse envelope $\Omega(t)$ such that the resulting quantum process is as close as possible to the ideal CZ gate. As the laser pulse is a continous signal, it is parameterized as a piecewise constant sequence of $N = 20$ amplitude segments, each normalized to $[0, 1]$ and scaled by $\Omega_\text{max}$. This gives a 20-dimensional continuous action space:

$$\mathbf{u} = (u_i)_{i=1}^{20} \quad \text{where} \quad 0 \le u_i \le 1 \text{ for all } i \in \{1, 2, \dots, 20\}$$

So that the final amplitude at time $t$ is $\Omega(t) = \Omega_\text{max} \cdot u_i$ for $t \in [(i-1)\frac{T}{N}, i\frac{T}{N})$. 

The optimization objective is to maximize the average gate fidelity $F(\mathbf{u})$ under this noisy environment, where the fidelity is the overlap between the noisy quantum process generated by $\Omega(t)$ and the ideal CZ unitary.

$$\text{maximize} \quad F(\mathbf{u}) \quad \text{subject to} \quad \mathbf{u}$$

---

## 2. Simulation Model



### The Hamiltonian


We can model the two atom system with the following time dependent Hamiltonian in the rotating frame of the laser:

$$H(t) = \sum_{i \in \{A,B\}} \left[ \frac{\Omega(t)}{2} \left( \vert r \rangle\langle 1 \vert_i + \vert 1 \rangle\langle r \vert_i \right) - \Delta(t)  \right] + U_\text{eff} \vert rr \rangle\langle rr \vert$$

The three terms are:

1. **The drive term** ($\Omega(t)/2 \cdot \sigma_{r1} + \text{h.c.}$): the laser couples $\vert 1 \rangle$ and $\vert r \rangle$ with Rabi frequency $\Omega(t)$. The operator $\vert r \rangle\langle 1 \vert$ promotes an atom from $\vert 1 \rangle$ to $\vert r \rangle$ and its conjugate does the reverse. Together they produce coherent oscillation (Rabi flopping) between the two levels. This term acts independently on both atoms A and B.

2. **The detuning term** ($-\Delta$): the detuning $\Delta$ is the mismatch between the laser frequency and the atom's natural $\vert 1 \rangle \to \vert r \rangle$ transition frequency. So if the laser is perfectly on resonance, $\Delta = 0$ and this term gone. If the laser is off-resonance, $\Delta \neq 0$ so that this term adds an energy penalty to the $\vert r \rangle$ state, effectively suppressing excitation. In this project, we set $\Delta(t) = 0$ for simplicity, but it could be made a second control parameter.

3. **The blockade term** ($U_\text{eff} \cdot \vert rr \rangle\langle rr \vert$): adds energy $U_\text{eff}$ whenever both atoms are simultaneously in $\vert r \rangle$. Since $U_\text{eff} \gg \Omega$, this energetically forbids the $\vert rr \rangle$ state, producing the blockade.

### Open-System Dynamics and Noise


For the ideal and isolated quantum system, the system described by a state vector $\lvert\psi\rangle$ that evolves according to the Schrödinger equation:
$$\frac{d}{dt} \lvert\psi(t)\rangle = -i H(t) \lvert\psi(t)\rangle$$

But real atoms are not isolated. We model three fundamental decoherence channels through the Lindblad master equation:

$$\frac{d\rho}{dt} = -i[H, \rho] + \sum_k \left( L_k \rho L_k^\dagger - \frac{1}{2} L_k^\dagger L_k \rho - \frac{1}{2} \rho L_k^\dagger L_k \right)$$

where $\rho$ is the density matrix, the generalized state descriptor that handles quantum superposition and classical statistical uncertainty simultaneously. The collapse operators $L_k$ encode each noise channel.

This equation is integrated numerically using QuTiP's `mesolve` function over 200 time steps covering the gate duration $T$. This make it possible to simulate the noisy evolution of the quantum state under any candidate pulse $\Omega(t)$ and compute the resulting gate fidelity.


### The noise channels 

| Noise channel | Collapse operator | Rate |
|---|---|---|
| Spontaneous Rydberg decay $\vert r \rangle \to \vert 1 \rangle$ | $\sqrt{\gamma_r}\, \vert 1 \rangle\langle r \vert$ | $\gamma_r = 2\pi \times 3\,\text{kHz}$ |
| Pure dephasing of $\vert r \rangle$ | $\sqrt{\gamma_\phi}\, \vert r \rangle\langle r \vert$ | $\gamma_\phi = 2\pi \times 1\,\text{kHz}$ |
| Doppler dephasing | $\sqrt{\gamma_\text{Doppler}}\, \vert r \rangle\langle r \vert$ | $\gamma_\text{Doppler} = 2\pi \times 0.5\,\text{kHz}$ |

The Lindblad master equation is integrated numerically using QuTiP's `mesolve` function over 200 time steps covering the gate duration $T$.

### Fidelity Metric

Gate fidelity is computed as the overlap between the noisy quantum process generated by the candidate pulse and the ideal CZ unitary. This is done by applying the candidate pulse to a complete set of input states, getting the resulting quantum process, and comparing it to the ideal process using the standard fidelity formula for quantum channels.

We need to optimize the pulse \$\Omega(t)$ to maximize this fidelity, which serves as the reward signal for the optimization algorithm.


---

## 3. Algorithm

## ML techniques to optimize the pulse shape

We need to implement an optimization algorithm to find the pulse shape $\Omega(t)$ that maximizes the fidelity.But at each evaluation of the fidelity requires a costly numerical simulation of the quantum dynamics, and the reward/loss landscape is noisy due to the stochastic nature of the environment.
The ML model should generate candidate pulse shapes and learn from the fidelity feedback to improve over time. 
We choose to implement the Group Relative Policy Optimization (GRPO) algorithm, which is a policy gradient method that uses a group relative advantage signal to stabilize training in noisy environments.

### The RL Environment

I implement the physics simulator as a Gymnasium "python library for RL" compatible environment. The action space is a 20-dimensional box representing the pulse segments. Each time the agent "model" generates a complete pulse vector $\mathbf{u} \in [0,1]^{20}$, the simulator evaluates it under noise, and returns the fidelity as reward. The gate time is fixed at $T = 0.5\,\mu$s the empirical sweet spot between two competing effects, shorter gates reduce decoherence time but require steeper Rabi drives that are harder to control.

For the optimization of the model, the reward is not raw fidelity but a shaped version designed to make training easier:

$$R(F) = F^\alpha + B \cdot F \cdot \mathbf{1}[F \geq 0.995]$$

with $\alpha = 4$ and $B = 10$. The $F^4$ power shaping compresses poor-fidelity rewards and stretches the gradient near $F = 1$, providing a stronger learning signal when the agent is already performing well. 
As the main achievement is to cross the 99.5% threshold, the discontinuous bonus $B$ at 99.5% creates an explicit "achievement cliff" within a group of candidates, those crossing the threshold are immediately recognized as exceptional by the relative advantage calculation.

Also implemented in the environment is a noise curriculum where the  training begins with noise_scale = 0 (ideal dynamics) and linearly ramps to noise_scale = 1 over a configurable number of steps. This gives the optimizer a warm start on the noiseless landscape before it must contend with realistic decoherence. When the noise scale become 1, all noise channels are active at their full strength.

### GRPO Adaptation

The following section describes the specific implementation of GRPO for this problem, which include the advantage calculation and the policy update rule.

GRPO was originally designed for discrete token generation in language models [2]. Here we adapt it to a continuous 20 dimensional action space. The key equations are unchanged:

Group-relative advantage: At each iteration, sample group $G$ pulse candidates $\{a_1, \ldots, a_G\}$ from the current policy model $\pi_\theta$ and evaluate each candidate's reward $r_i = R(F(a_i))$. Compute the group mean $\bar{r}$ and standard deviation $\sigma_r$, and calculate the advantage for each candidate as:

$$A_i = \frac{r_i - \bar{r}}{\sigma_r + \varepsilon}$$

This advantage is the core component of the grpo which make it takes decisive action upon the relative performance of the group, which make the model choose the best pulse inside the group and update the network, so that we don't need a critic network to judge how the agent is performing, nor estimated value function, just the relative ranking of the current group.

Policy loss "loss function": The policy is updated using a PPO-style clipped surrogate [3] on these group relative advantages, plus a KL penalty against a frozen reference policy $\pi_\text{ref}$:

$$\mathcal{L} = -\mathbb{E}\!\left[\min\!\left(\rho_i A_i,\; \text{clip}(\rho_i, 1{-}\varepsilon, 1{+}\varepsilon) A_i\right)\right] + \beta\, \text{KL}(\pi_\theta \| \pi_\text{ref}) - \eta\, H(\pi_\theta)$$

where $\rho_i = \pi_\theta(a_i) / \pi_\text{old}(a_i)$ is the probability ratio, $\varepsilon = 0.2$ is the clip bound, $\beta = 0.01$ is the KL penalty, and $\eta = 0.001$ is an entropy bonus that encourages exploration.

Policy network: The policy is a Gaussian MLP: three hidden layers (128 units, LayerNorm, tanh activations) that output a mean $\mu \in [0,1]^{20}$. The network has approximately 16,000 parameters deliberately lightweight since each forward pass costs nothing relative to the simulator.

The optimizer is Adam with learning rate $3 \times 10^{-4}$, gradient clipping at 0.5, and a batch of 4 groups per gradient step. The reference policy is refreshed every 100 iterations to prevent the KL penalty from becoming trivially easy to satisfy.

### GRAPE as Classical Baseline

We also implement a classical gradient-based optimization method for comparison: GRAPE (Gradient Ascent Pulse Engineering) [4]. GRAPE computes the fidelity gradient with respect to the pulse parameters using finite differences, which requires multiple simulator evaluations per gradient step. 

Numerical GRAPE is implemented as the classical comparison. It computes the fidelity gradient via central finite differences:

$$\frac{\partial F}{\partial u_k} \approx \frac{F(\mathbf{u} + \varepsilon\, \mathbf{e}_k) - F(\mathbf{u} - \varepsilon\, \mathbf{e}_k)}{2\varepsilon}$$

This requires $2N = 40$ simulator calls per gradient step, compared to GRPO. Updates use gradient ascent with momentum:

$$\mathbf{v} \leftarrow \beta_m\, \mathbf{v} + \alpha\, \nabla F, \qquad \mathbf{u} \leftarrow \text{clip}(\mathbf{u} + \mathbf{v},\; 0,\; 1)$$

with momentum coefficient $\beta_m = 0.9$ and learning rate $\alpha = 0.05$. 

---

## 4. Experiments and Results
We use QuTiP's framework to implement the simulator and noise model, which allows us to easily simulate the open quantum dynamics and compute fidelities. 

Four experiments were run, each targeting a specific question about the algorithm's behavior or the physics. All experiments use the same Rydberg simulator and noise model unless stated otherwise.

| Metric | Value |
|---|---|
| Best GRPO fidelity (Exp 1, G=12) | **99.99%** |
| Blackman pulse baseline | 72.1% |
| GRAPE (numerical, noisy eval) | 90.6% |
| 3-atom CCZ scaling (Exp 4) | 71.3% |

### Experiment 1 — Group Size Ablation
For the grpo algorithm, we need to define the group size G, which is the number of candidate pulses sampled at each iteration to compute the group relative advantage. A larger G provides a more stable and informative advantage signal but increases the computational cost per iteration. A smaller G is cheaper but may produce a noisier signal that slows convergence.

Five values of G were tested (G = 2, 4, 6, 8, 12), each run for 18 GRPO iterations with 20 pulse segments at T = 0.5 µs.

| Group size | Final fidelity | First iter > 99.5% | Wall time |
|---|---|---|---|
| G = 2 | 99.99% | iter 2 | 21,734 s |
| G = 4 | 99.99% | iter 2 | 4,153 s |
| G = 6 | 99.99% | iter 1 | 4,231 s |
| G = 8 | 99.99% | iter 1 | 1,597 s |
| G = 12 | 99.99% | iter 0 | 1,347 s |

All group sizes ultimately reach 99.99%, but the story is in efficiency. G = 2 took over 6 hours: the advantage variance within a two sample group is so small that the policy update signal is nearly zero, and progress is very slow. G = 4 and G = 6 reach 99.5% within two iterations and complete in around one hour they have enough group diversity to produce a useful advantage signal. G = 12 finds a target exceeding pulse within the very first iteration but costs more per memory due to the larger simulator batch.

![Group size ablation and curriculum comparison](/assets/images/rydberg-grpo/dashboard_exp1_exp2_real.png)

*Figure 1: Left — GRPO best fidelity vs. iteration for five group sizes. Right — Curriculum vs. direct training comparison. G = 6 offers the best wall-time-to-fidelity tradeoff.*

**Takeaway:** G = 6 is the practical sweet spot. Very small groups (G = 2) have too little advantage variance; very large groups (G ≥ 12) add per-iteration cost without improving the final answer.

### Experiment 2 — Noise Curriculum vs. Direct Training

<!-- *Question: Does ramping noise from zero to full help compared to training directly at full noise from the start?* -->

We use circulum training to make the model learn the ideal dynamics first, then gradually introduce noise. The comparison is between two runs with G = 6: one with the noise curriculum and one trained directly at full noise.

Curriculum training ramped noise scale from 0 to 1.0 linearly over 10 iterations. Both methods reached 99.95% final fidelity and found pulses with a cosine similarity of approximately 1.0 the same attractor in pulse space, which means they converged to similar solutions.
Curriculum training was faster by about 28% in wall time, as the noiseless warm start phase quickly identifies the rough structure of the optimal pulse and the subsequent noisy phase fine tunes for robustness.


### Experiment 3 — Noise Source Ablation

We need to understand which noise channel we use is the most one affecting the fidelity, so that we can focus more while optimizing the pulse shape. 

Each noise channel was individually disabled while keeping all others active. The result was surprising: disabling Rydberg decay, pure dephasing, or Doppler has essentially no measurable effect on fidelity, which means the optimized pulse is not relying on any specific noise mitigation strategy for those channels.

This experiment run for G = 2 with only 6 iterations to save time, so that the results doesn't achieve the same final fidelity as the main runs, but the relative differences between noise ablations are still informative. 

![Noise ablation and 3-atom scaling](/assets/images/rydberg-grpo/dashboard_exp5_exp6_real.png)

*Figure 3: Left — Fidelity sensitivity to individual noise sources. Right — GRPO convergence on the 3-atom CCZ gate.*

### Experiment 4 — Scaling to Three Atoms (CCZ Gate)

We extend the problem to experince with a three atom system and a CCZ gate, which applies a phase flip to the |111⟩ state.  
With 6 GRPO iterations, the best fidelity reached was **71.3%** which is proof of concept (PoC) for scaling, though well below the two qubit target. The convergence is monotonic, starting from approximately 30% and improving steadily. The primary bottleneck is simulator cost: each `mesolve` call, making each iteration roughly $9\times$ slower than the two qubit case.

### GRAPE Comparison

| Method | Fidelity (noisy) | Wall time |
|---|---|---|
| GRAPE (numerical) | 90.6% | ~500 s |
| GRPO (G = 12, full run) | — | 99.99% | ~1,347 s |

In the noisy evaluation GRAPE reaches 90.6%, which can be achieved using a well crafted pulse design. GRAPE struggles due to is that numerical finite difference gradients require two separate noisy evaluations per segment, so the gradient estimate has high variance. On the other hand, GRPO evaluates all G candidates within the same noise regime, and the relative ranking within the group is far more stable than the absolute difference between two individual noisy evaluations. But surprisingly, the we found that the best GRAPE and best GRPO pulses have a cosine similarity of 0.81, suggesting they converge to similar regions of pulse space. 

![Pulse shape summary](/assets/images/rydberg-grpo/dashboard_pulse_summary_real.png)

*Figure 4: Comparison of the GRPO-optimized pulse shape against the Blackman and square reference pulses, with the fidelity each achieves.*

---

## 5. Discussion

### What worked

The most important finding is that GRPO— esigned for discrete token generation translates cleanly to this continuous control problem. The group relative advantage is a strong signal in noisy environments and the advantage is always well-defined as long as there is any spread within the group.

The physics simulation at the level of detail used here, full Lindblad dynamics with realistic multi channel noise turns out to be the right fidelity. The optimizer finds pulses that reflect real Rydberg physics, not numerical artifacts: the optimal pulse durations and amplitude profiles are consistent with what one would expect from the timescales set by $\Omega_\text{max}$, $U_\text{eff}$, and $\gamma_r$.


### Limitations 

The noise assisted optimization finding warrants caution. A pulse whose fidelity decreases (exp3) when noise is removed has adapted to specific noise realizations rather than learning a truly noise robust shape. Proper robustness evaluation would require averaging over many independently sampled noise trajectories at test time with a fixed seed different from training—this was not done and is an open question for follow-up work.

The three-atom CCZ scaling result is promising but the comparison is not clean: a CCZ gate requires a three body interaction that does not arise as naturally from the two-atom blockade Hamiltonian as a CZ does. The Hamiltonian model may need another formulation, may be using  multi-step pulse sequence for the CCZ to be physically realizable at high fidelity.

---

## 6. Conclusion

This project built a simple pipeline for Rydberg CZ gate optimization: first principles Hamiltonian simulation via QuTiP's Lindblad master equation solver, a GRPO adapted training loop with a Gaussian policy network, and systematic ablation experiments comparing group size, noise curriculum and classical baselines. This project explain how to integrate physics simulation with modern RL algorithms to solve a real-world quantum control problem.
## References

[1] S. J. Evered, D. Bluvstein, M. Kalinowski, S. Ebadi, T. Manovitz, H. Zhou, S. H. Li, A. A. Geim, T. T. Wang, N. Maskara, H. Levine, M. Greiner, V. Vuletić, and M. D. Lukin, "High-fidelity parallel entangling gates on a neutral-atom quantum computer," *Nature* **622**, 268–272 (2023).

[2] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, Y. Guo, and D. Guo, "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," arXiv:2402.03300 (2024).

[3] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347 (2017).

[4] N. Khaneja, T. Reiss, C. Kehlet, T. Schulte-Herbrüggen, and S. J. Glaser, "Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent algorithms," *Journal of Magnetic Resonance* **172**, 296–305 (2005).

[5] M. Saffman, T. G. Walker, and K. Mølmer, "Quantum information with Rydberg atoms," *Rev. Mod. Phys.* **82**, 2313 (2010).

[6] QuEra Computing, "Pulse-level CZ gate via Rydberg blockade," *Bloqade.jl Documentation*, [https://queracomputing.github.io/Bloqade.jl/dev/3-level/#pulse-CZ-gate](https://queracomputing.github.io/Bloqade.jl/dev/3-level/#pulse-CZ-gate) (accessed June 2026).

[7] Y. Cai, H. Zhang, K. Zhang, and J. Qian, "Intelligent Optimal Control of Rydberg Gates with Incremental-Update Deep Reinforcement Learning," arXiv:2605.04628 (2026).