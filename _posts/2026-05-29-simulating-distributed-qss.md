---
layout: post
title: "Simulating Distributed Quantum Secret Sharing: A Diskit Implementation of a ((3,5)) Threshold Scheme"
date: 2026-05-29 00:00:00 +0800
categories: [Quantum Computing, Quantum Secret Sharing, Qiskit, Diskit]
tags: [Quantum Computing, Quantum Secret Sharing, Qiskit, Diskit]
---

**Author:** Mariam Hossam

---

In an era of increasingly decentralized networks, secure information distribution is paramount. Classical secret sharing, first introduced by Shamir [1], allows a dealer to divide a secret among $n$ participants such that only a subset of $k$ participants (where $k \le n$) can reconstruct it. However, classical schemes are fundamentally vulnerable to eavesdropping and computational assumptions.

**Quantum Secret Sharing (QSS)** [2] elevates this concept by encoding the secret into quantum states. Relying on the principles of quantum mechanics—specifically entanglement and the no-cloning theorem—QSS guarantees unconditional security. Any attempt by an eavesdropper to intercept a share inherently disturbs the state, revealing the intrusion.

The motivation for this project stems from the deep, underlying relationship between QSS and Quantum Error-Correcting Codes (QECC). In essence, a QSS protocol can be viewed as a quantum error-correcting code where the "errors" are the missing shares of the uncooperative parties. Implementing QSS on near-term hardware is therefore not just a cryptographic exercise; it is a vital stepping stone toward validating the error-correcting capabilities of current quantum architectures. Robust QSS implementation proves that our hardware can successfully encode, distribute, and decode logical qubits across noisy environments.


## Background and Related Work
Current research has made significant strides in realizing QSS, but profound architectural gaps remain.

### Literature Review & The Research Gap
In their foundational paper, "Implementing Quantum Secret Sharing on Current Hardware" [3], the authors successfully demonstrated a working QSS protocol. Their contribution was pivotal in proving that current Noisy Intermediate-Scale Quantum (NISQ) devices possess the fidelity required to execute the necessary entangling gates for secret sharing.

However, their implementation is strictly limited to a single Quantum Processing Unit (QPU). This presents a fundamental limitation: real-world secret sharing inherently requires multi-party distribution across geographically separated nodes. A single-device execution fails to account for the complexities of network topology, entanglement distribution over quantum channels, and inter-QPU communication overhead. This gap in the literature raises a critical research question: **Would the state fidelity of an implemented ((3,5)) QSS scheme remain acceptable when distributed across multiple parties at separate, interconnected QPUs?**

### Distributed Simulation via Diskit
To address this gap and directly answer this question, this project investigates how QSS behaves on a distributed infrastructure. Rather than relying on a monolithic QPU, we utilize Diskit [4], an extension framework designed for distributed quantum computing. Diskit allows us to define distinct, interconnected quantum nodes, simulating the realistic routing of qubits and the necessary entanglement swapping required to share secrets across a network. This provides novel scientific insights into the practical limitations and topological requirements of deploying QSS in a real-world quantum internet.


### Prerequisite Concepts for Distributed QSS

To bridge the gap between theoretical multi-party secret sharing and practical implementation, several advanced quantum computational techniques must be understood. These concepts form the backbone of our simulation setup and optimization strategy.

#### State Verification via the Swap Test
In any quantum communication protocol, it is essential to objectively measure how perfectly the target state matches the source state. Because direct measurement collapses the quantum state, algorithms rely on the **Swap Test**. This fundamental primitive computes the squared overlap (fidelity) between two arbitrary quantum states by introducing an ancillary control qubit and applying a controlled-SWAP (Fredkin) gate. If the two states are identical, the control qubit is guaranteed to be measured in the ground state ($|0\rangle$). 

In our setup, we employ the Swap Test as our primary evaluation metric. By comparing the dealer's original secret against the final state recovered by the participants, the Swap Test provides a direct, measurable percentage of how successfully the quantum secret was reconstructed across the Diskit topology.

![QSS circuit](/assets/images/qss/swap_circuit.png)

*Figure 1:  Quantum SWAP test circuit diagram. [3]*

#### Mitigating Hardware Noise (M3)
Simulating distributed quantum networks inherently exposes the system to realistic hardware noise, most notably during readout operations where qubit states are classically recorded. Traditional error mitigation techniques combat this using tensored calibration matrices, but these scale exponentially and become computationally prohibitive for multi-qubit systems. **Matrix-free Measurement Mitigation (M3)** [5] resolves this by operating efficiently within a reduced subspace, bypassing the need for full matrix inversion. 

We integrated M3 directly into our Diskit simulation pipeline. By applying this mitigation to our final readouts, we strip away the simulated hardware measurement bias. This ensures that the fidelity degradation we observe is strictly a result of the network topology and entangling operations, isolating the algorithm's actual performance from baseline readout errors.

#### Circuit Depth: Mid-Circuit vs. Delayed Measurements
A crucial architectural decision in quantum algorithm design is managing when qubits are measured. The traditional **delayed measurement** principle dictates that all quantum operations must complete before any classical measurements occur. While conceptually elegant, preserving states until the end of a complex protocol requires a massive overhead of ancillary qubits and SWAP operations. Alternatively, **mid-circuit measurements** allow specific qubits to be measured mid-execution, collapsing their states and freeing up resources, while the rest of the circuit continues to run.


For a distributed $((3,5))$ threshold scheme, understanding this distinction is vital. In our setup, relying on delayed measurements for the participant nodes yielded an impractical circuit depth of 66,190. By optimizing our architecture to utilize mid-circuit measurements—collapsing participant shares as soon as they were logically processed—we reduced the depth to just 2,477 gates. This foundational optimization makes the deployment of complex multi-party protocols feasible on near-term hardware, as shallower circuits are significantly less susceptible to decoherence.

## Methods and Setup: The ((3,5)) Threshold Scheme
Our experiment implements a $((3,5))$ threshold scheme, meaning the quantum secret is divided into $5$ distinct shares, and a minimum of $3$ participants must collaborate to reconstruct the original state.

This specific configuration was chosen because it perfectly aligns with the well-known $[[5,1,3]]$ quantum error-correcting code. The $[[5,1,3]]$ code encodes one logical qubit into five physical qubits and can correct an arbitrary single-qubit error and 2 erasure errors—or, in our case, reconstruct the state if two shares are missing.

### Circuit Architecture
To establish our baseline, Figure 2 displays the full monolithic $((3,5))$ circuit structure.
![QSS circuit](/assets/images/qss/qss_circuit.png)
*Figure 2: The monolithic quantum circuit for the $((3,5))$ QSS scheme [3]. This serves as the theoretical baseline for our distributed implementation.*


**Design Explanation:**

1. State Preparation: The circuit initiates by entangling the dealer's secret qubit with four auxiliary qubits, generating the highly entangled logical state necessary for the $((3,5))$ threshold.
2. Distribution (Diskit Topology): Utilizing Diskit’s topology mapping, the 5 physical qubits are virtually routed across 5 distinct, simulated nodes. This shifts the execution from a single-QPU environment to a distributed infrastructure.
3. Reconstruction: To reconstruct the secret, participants apply inverse unitary transformations. Our design relies on stabilizer measurements to decode the state back to the target node, ensuring the secret is only recovered if the threshold of shares is met.

In an ideal scenario where all shares are available, the operators cancel out, yielding the original input state. However, when 2 qubits are lost—simulated in our experiment by swapping the lost shares with fresh, initialized qubits—the state is corrupted. To recover the secret, we apply a Pauli correction ($R_k$) to the fifth qubit based on the specific error syndrome identified. The required corrections are summarized below:
![Correction Table](/assets/images/qss/correction.png)
*Figure 3: This table details the unitary corrections ($R_k$) required on the fifth qubit to address arbitrary Pauli errors on the first and second qubits within the $((3,5))$ QSS scheme, mapped to each corresponding syndrome measurement [3]*


### Example Trace

To better understand the intuition behind the circuit design, we will trace a concrete numeric example and demonstrate how the decoded state matches the original secret.

The state vectors are written following the Qiskit little-endian convention:
$|q_4 q_3 q_2 q_1 q_0\rangle$.


Let 
$\theta = \pi$
and
$\phi = \pi / 2$
. Therefore, the secret state to share is configured as   $|\psi\rangle = i|1\rangle$   on the data qubit  $q_4$
, while all other qubits start in the ground state  $|0\rangle$.


$$|\Psi_{\text{input}}\rangle = |0\rangle_0 \otimes |0\rangle_1 \otimes |0\rangle_2 \otimes |0\rangle_3 \otimes (i|1\rangle)_4 = i|10000\rangle$$

#### Step 1: Secret State Preparation
At the first barrier, the state remains isolated on the data qubit:

$$|\Psi_1\rangle = i|10000\rangle$$

#### Step 2: Encoding 
The encoding network distributes the secret state across a highly entangled 5-qubit graph state. The single state shatters into an equal superposition of 16 basis states, each with an amplitude magnitude of $\frac{1}{4} = 0.25$:


$$\begin{aligned}
|\Psi_2\rangle = -0.25i \big( &|00001\rangle + |00010\rangle + |00100\rangle + |00111\rangle + |01000\rangle + |01110\rangle + |10000\rangle \\
&+ |10011\rangle + |11001\rangle + |11100\rangle \big) \\
+ 0.25i \big( &|01011\rangle + |01101\rangle + |10101\rangle + |10110\rangle + |11010\rangle + |11111\rangle \big)
\end{aligned}$$

#### Step 3: Erasure
Qubits $q_0$ and $q_1$ are lost to erasure and replaced with freshly initialized $|0\rangle$ qubits. This forces a collapse into a subspace where the rightmost two qubits ($q_1, q_0$) map directly back to $00$.

Normalizing the remaining 4 valid states yields an amplitude magnitude of $\frac{1}{\sqrt{4}} = 0.5$:

$$|\Psi_3\rangle = -0.5i \big( |00100\rangle + |01000\rangle + |10000\rangle + |11100\rangle \big)$$

#### Step 4: Decoding
The decoding operations run the inverse encoding sequence on the remaining components. This process successfully untangles the data qubit $q_4$ from the system, gathering the erasure errors onto the syndrome qubits ($q_1, q_0$):

$$|\Psi_4\rangle = 0.5i \big( |10000\rangle + |10001\rangle + |10010\rangle + |10011\rangle \big)$$

Notice that $q_4 = 1$ in all remaining states, meaning the secret phase information is successfully isolated back onto the data channel.

#### Step 5: Mid-Circuit Measurement
The circuit measures the syndrome qubits $q_3 q_2 q_1 q_0$. Suppose that after measurement, the state projects into a single computational outcome:

$$|\Psi_5\rangle = i|10010\rangle$$

This collapse yields the classical syndrome register value:

$$\text{syn} = q_3 q_2 q_1 q_0 = 0010_2$$

#### Step 6: Feedforward Corrections 
The conditional lookup logic checks the syndrome value. Because the state vector contains $q_4 = 1$, the secret state on the data qubit is already $i|1\rangle$.

Since no phase or bit-flip error altered the data qubit during this specific erasure path, the syndrome $0010$ maps to an Identity operation ($I$) according to Figure 3. No correction gate is triggered, keeping the state vector completely stable:

$$|\Psi_6\rangle = i|10010\rangle$$

#### Final Reconstructed State
To find the final state of our recovered secret, we isolate the data qubit $q_4$ from the inactive ancillas ($q_3 = 0, q_2 = 0, q_1 = 1, q_0 = 0$):

$$|\Psi_{\text{reconstructed}}\rangle = \text{State}(q_4) = i|1\rangle$$


$$|\Psi_{\text{reconstructed}}\rangle = |\Psi_{\text{input}}\rangle$$

The quantum secret sharing protocol has perfectly preserved and reconstructed the original state despite the complete erasure of shares 0 and 1.


### Implementation Details
Below are the core components of our Diskit implementation:

#### A. Initializing the Distributed Topology
In Diskit, we must first define the network structure. Here, we create 5 distinct nodes representing our participants. The fifth QPU is assigned three qubits to perform the swap test necessary for verifying reconstruction fidelity.
<!-- This configuration is specifically architected to support our protocol's logic: QPUs 1 and 2 are each allocated two qubits to handle local swap operations, while  -->

```python
# Initialize Diskit topology and map qubits to physical nodes
# The list [1, 1, 1, 1, 3] defines the distribution of qubits across the network
circuit_topo = Topology()
circuit_topo.create_qmap(5, [1, 1, 1, 1, 3], "sys_qss")

# Retrieve the registers and initialize the Quantum Circuit
qregs = circuit_topo.get_regs()
qc = QuantumCircuit(*qregs)

# Initialize the remapper to manage cross-node operations
remapper = CircuitRemapper(circuit_topo)
```

#### B. Circuit Initialization
We begin by initializing the secret state using random single-qubit rotation angles ($\theta, \phi$). This randomized input ensures our protocol is robust and not biased toward a specific basis.

```python
# Initialize random secret state
theta = np.random.uniform(0, np.pi)
phi = np.random.uniform(0, 2 * np.pi)

qc.rx(theta, 4)
qc.rz(phi, 4)

qc.rx(theta, 5)
qc.rz(phi, 5)
```

#### C. Encoding

The encoding phase uses a series of Hadamard, CNOT, and Controlled-Z gates to entangle the secret with four auxiliary qubits. This generates the highly entangled logical state required for the ((3,5)) threshold code, distributing the information across the simulated network.
```python
# Encoding gates
qc.h(0); qc.h(1); qc.h(2); qc.h(3); qc.z(4)
qc.cx(0, 4); qc.cx(1, 4); qc.cx(2, 4); qc.cx(3, 4)
qc.cz(0, 1); qc.cz(1, 2); qc.cz(2, 3); qc.cz(3, 4); qc.cz(0, 4)
```

#### C. Erasure
To simulate the loss of two shares, we perform a hard reset on the first two qubits. This tests the protocol's capability to reconstruct the original state even when a portion of the distributed system is inaccessible.
```python
# Simulate loss of two shares
qc.reset(0)
qc.reset(1)
```

#### D. Decoding
Here, we apply the inverse unitary transformations. This sequence must be the exact reverse of the encoding process to return the entangled state to its original form, effectively "re-concentrating" the distributed information into the target qubit.

```python
# Decoding (inverse operations)
qc.cz(0, 4); qc.cz(3, 4); qc.cz(2, 3); qc.cz(1, 2); qc.cz(0, 1)
qc.cx(3, 4); qc.cx(2, 4); qc.cx(1, 4); qc.cx(0, 4)
qc.z(4); qc.h(3); qc.h(2); qc.h(1); qc.h(0)
```

#### E. Mid-Circuit Measurement (MCM) & Pauli Corrections
We measure the four syndrome qubits to identify the error pattern. Based on the syndrome result, we apply dynamic Pauli corrections ($X, Y, Z$) to the fifth qubit to recover the intended state.

```python
# Measure syndrome
cr_syndrome = ClassicalRegister(4, "syn")
qc.add_register(cr_syndrome)
qc.measure([0, 1, 2, 3], cr_syndrome)

# Conditional Pauli corrections based on measurement syndrome
with qc.if_test((cr_syndrome, int('0101', 2))): qc.x(4)
with qc.if_test((cr_syndrome, int('1101', 2))): qc.y(4)
with qc.if_test((cr_syndrome, int('1010', 2))): qc.z(4)
# ... additional correction gates ...
```

#### F. Swap Test
Finally, we verify the fidelity of the reconstructed state by comparing it against the original state using an ancilla qubit.

```python
# Swap test for fidelity verification
qc.h(6)
qc.cswap(6, 4, 5) # 4 is reconstructed, 5 is original
qc.h(6)

cr_swap = ClassicalRegister(1, "swap_res")
qc.add_register(cr_swap)
qc.measure([6], cr_swap)
```

#### G. Circuit Remapping
After defining the logical circuit, we use Diskit to project it onto our distributed infrastructure. One of the primary advantages of this framework is its abstraction layer: it automatically handles the "invisible" work of quantum networking. Under the hood, Diskit manages the generation and distribution of EPR pairs, teleportation protocols, and inter-node routing required to realize the gates across separated QPUs.

```python
# Map the logical QSS circuit to the distributed Diskit topology
dist_circ = remapper.remap_circuit(qc, decompose=True)
```
You can inspect the resulting circuit, which includes the necessary distributed communication overhead, here: [[ Distributed Output Circuit](https://drive.google.com/file/d/1zLGTEjK79LCgCn9muC9pLsnwiDU28law/view?usp=sharing)].


#### H. Execution on Hardware-Realistic Backend
To assess how our ((3,5)) scheme performs under authentic noise conditions, we execute the circuit on the FakeBrisbane backend. 

```python
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane

# 1. Initialize the hardware-realistic simulator
backend = FakeBrisbane()
sim = AerSimulator.from_backend(backend)

# 2. Transpile for the target backend with optimization
compiled_circ = transpile(dist_circ, backend=sim, seed_transpiler=42)

# 3. Execute the simulation
shots = 20000
result = sim.run(compiled_circ, shots=shots).result()
counts = result.get_counts()
```

#### I. Measurement Error Mitigation (M3)
Finally, we apply M3 mitigation on the obtained measurements to effectively correct for readout bias.

```python
mit = m3.M3Mitigation(backend)

mappings  = m3.utils.final_measurement_mapping(compiled_circ)

mit.cals_from_system(mappings )

quasi = mit.apply_correction(
    counts,
    mappings 
)

mitigated_probs = quasi.nearest_probability_distribution()
```

## Results and Analysis

To rigorously evaluate our scientific claims, we simulated both the monolithic baseline and the distributed Diskit implementations of the ((3,5)) QSS scheme for 20,000 shots. The circuits were evaluated across all four Qiskit transpilation optimization levels: Level 0 (no optimization), Level 1 (light), Level 2 (medium), and Level 3 (heavy). 

To gain complete insight into how spatial distribution impacts protocol fidelity, we analyzed three primary metrics: **SWAP test success rate**, **circuit depth**, and **total gate count**.

### Performance Metrics & Overhead Analysis

* **Fidelity and Success Rate:** The monolithic QSS circuit consistently outperformed its distributed counterpart by a substantial margin across all optimization levels. In the distributed simulation, the SWAP test success rate hovered near **0.50**—the theoretical lower bound for the SWAP test, which represents a complete loss of state overlap (equivalent to random guessing). This stark drop strongly implies that the communication overhead of multi-QPU architectures introduces a prohibitive amount of noise for near-term, unmitigated hardware.
* **Circuit Depth Inflation:** This degradation in fidelity is directly explained by the structural metrics shown in Figures 5 and 6. Spatially distributing the ((3,5)) scheme incurs a massive routing penalty, resulting in an approximate **5x increase in circuit depth** compared to the monolithic baseline. This extended depth leaves the qubits highly vulnerable to environmental decoherence.
* **Gate Count Overhead:** Similarly, the number of multi-qubit operations and total gates experienced an approximate **6x increase**. This surge is caused by the underlying EPR pair generations and teleportation protocols that Diskit must inject to execute non-local operations across distinct QPUs. 

Even at optimization Level 3, the heavy gate and depth overhead introduces too much accumulated error for the underlying logical state to be reliably recovered on a noisy backend like FakeBrisbane.

---

### Experimental Visualizations

![Rate of Success of SWAP Test](/assets/images/qss/plot1_pass_rate.png)
***Figure 4:** SWAP test success rate for the ((3,5)) QSS scheme on the noisy IBM Fake Brisbane simulator across optimization levels (erasure set size = 2).*

![Circuit Depth Comparison](/assets/images/qss/plot2_circuit_depth.png)
***Figure 5:** Transpiled circuit depth comparison between monolithic and distributed Diskit implementations.*

![Total Gate Count Comparison](/assets/images/qss/plot3_gate_count.png)
***Figure 6:** Transpiled circuit gate count comparison highlighting the multi-qubit operation overhead.*


## Conclusion and Limitations

### Conclusion

This project set out to answer the question we raised at the beginning of the blog: **Would the state fidelity of an implemented ((3,5)) QSS scheme remain acceptable when distributed across multiple parties at separate, interconnected QPUs?**

Our empirical results provide the following answer for near-term quantum computing: **under unmitigated, noisy NISQ conditions, the fidelity of the distributed protocol drops to unacceptable levels.** While the monolithic baseline successfully preserved state information, shifting the protocol to a distributed Diskit topology triggered a massive hardware penalty:
1. **Structural Inflation:** Managing non-local gates and inter-node qubit routing forced an approximate **5x increase in circuit depth** and a **6x increase in total gate count**. 
2. **Fidelity Collapse:** This dramatic accumulation of gate operations and prolonged exposure to decoherence on the simulated FakeBrisbane backend caused the final SWAP test success rate to decrease to approximately **0.50**. 

Because a 0.50 success rate represents the absolute theoretical lower bound for a SWAP test (equivalent to random guessing or orthogonal states), our findings demonstrate that multi-QPU communication overhead currently acts as a prohibitive barrier to deploying unmitigated distributed cryptographic algorithms. 

### Limitations and Future Work

While our implementation provides a robust demonstration of distributed Quantum Secret Sharing (QSS) within the Diskit framework, several limitations and avenues for future research remain:

- **Simulation vs. Physical Hardware:** The primary limitation of this study is our reliance on the FakeBrisbane backend. While this provides a reliable, noise-aware simulation environment for validating the distributed logic of the $((3,5))$ scheme, it cannot fully capture the non-deterministic environment, time-varying noise profiles, crosstalk, and calibration drift inherent in physical quantum hardware. This limitation is underscored by prior work [3], which reported a notable discrepancy between results obtained on simulated noisy backends and those executed on real QPUs.

- **Advanced Error Mitigation:** While we successfully integrated M3 for measurement mitigation, the fidelity of our reconstructed states can likely be improved further by integrating active error mitigation techniques, such as Dynamical Decoupling (DD), as noted in [3], to suppress decoherence during the extended periods where logical qubits must be held across the network.

- **Generalization of Code Parameters:** This evaluation was conducted exclusively using the $[[5, 1, 3]]$ quantum error-correcting code to realize the $((3,5))$ threshold scheme. Future research should extend this distributed benchmarking framework to a wider variety of stabilizer and subsystem codes with varying parameters (such as larger code distances $d$ or higher dimensions) to assess the generalizability and consistency of the structural overhead trends observed in this work.


## Notebooks

[Qiskit implementation](https://colab.research.google.com/drive/16hznHUfkzKnGYoS5PCTCFfwuiZ7AImHB#scrollTo=wvxK63O-gnet)

[Diskit implementation](https://colab.research.google.com/drive/1Njs13-kn4NjyI5mCvpGDMu7vEVxkL1R-#scrollTo=iBQGNDvdVhnw)


--- 
**Declaration of AI Assistance:** *The code, ideas, and discussions in this project are the sole work of the author. We acknowledge Google Gemini for its assistance in editing and polishing the blog text after prompting it with: "Rephrase and correct the following text: ". The final text was entirely reviewed and verified by the authors, who take full responsibility for all provided claims and citations.*


## References
[1] A. Shamir, How to share a secret, Commun. ACM 22,
612–613 (1979).

[2]  R. Cleve, D. Gottesman, and H.-K. Lo, How to share a
quantum secret, Phys. Rev. Lett. 83, 648 (1999).

[3] Graves, Jay & Nelson, Mike & Chitambar, Eric. (2025). Implementing Quantum Secret Sharing on Current Hardware. Entropy. 27. 993. 10.3390/e27100993. 


[4] https://github.com/Interlin-q/diskit

[5] https://www.ibm.com/quantum/blog/mthree-qiskit-extension
