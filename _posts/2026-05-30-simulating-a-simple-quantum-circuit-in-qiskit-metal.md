---
layout: post
title: "Simulating a Simple Circuit in Qiskit Metal"
date: 2026-05-30
description: Designing and preparing a simple superconducting quantum circuit for simulation using Qiskit Metal.
---

Yahya Ahmed Azzam

---

## Introduction

The project intended to simulate a simple circuit in **Qiskit Metal**, but before explaining what was done, it is worth explaining what Qiskit Metal is and what its capabilities are.

> Qiskit Metal is a Python framework that can be used to design and analyze quantum computer physical elements. It offers a variety of components and a GUI that can be used instead of writing code directly. It is free, community-driven, and can run in either Jupyter notebooks or standard Python scripts.

---

## Circuit Objective

The main idea behind the simulation was to create a system containing a single qubit connected to a Hadamard gate, where the initial state

$$
|0\rangle
$$

would be transformed into

$$
\frac{|0\rangle + |1\rangle}{\sqrt{2}}
$$

The overall circuit is shown below.

![Circuit Overview](/assets/images/simulating-a-simple-quantum-circuit-in-qiskit-metal/fig1.png)

*Figure 1 — Complete circuit layout.*

---

## Components

### 1. Transmon Pocket Qubit

The **Transmon Pocket Qubit** is the main component of the circuit.

It is controlled using microwave pulses and contains two connection pads:

- **Drive Pad** (smaller pad)
  - Receives microwave control pulses.

- **Readout Pad** (longer pad)
  - Passes the qubit state to a resonator for measurement.

This qubit consists of two large metal islands that form the capacitor of the qubit.

For reference, superconducting qubits behave similarly to electrical oscillators, where energy continuously oscillates between electric and magnetic forms.

The inductor loop was intentionally left out and will be added later during circuit analysis and simulation.

![Transmon Pocket Qubit](/assets/images/simulating-a-simple-quantum-circuit-in-qiskit-metal/fig2.png)

*Figure 2 — Transmon Pocket Qubit.*

---

### 2. Launchpad Wirebond Driven

The **Launchpad Wirebond Driven** serves as the entry point of the system.

Its purpose is to receive microwave control pulses from an external generator (not shown in the figure) and deliver them into the quantum circuit.

The **Driven** design helps:

- Minimize signal reflections.
- Reduce impedance mismatches.
- Improve pulse transmission fidelity.

![Launchpad Wirebond Driven](/assets/images/simulating-a-simple-quantum-circuit-in-qiskit-metal/fig3.png)

*Figure 3 — Launchpad Wirebond Driven.*

---

### 3. Launchpad Wirebond Coupled

The **Launchpad Wirebond Coupled** acts as the output point of the circuit.

Its role is to receive the qubit state information and pass it outside the chip where measurement equipment can process it.

![Launchpad Wirebond Coupled](/assets/images/simulating-a-simple-quantum-circuit-in-qiskit-metal/fig4.png)

*Figure 4 — Launchpad Wirebond Coupled.*

---

### 4. Route Meanders

Unlike the previous components, **Route Meanders** are transmission structures rather than active circuit elements.

They serve two primary purposes:

#### Drive Line

Delivers microwave control pulses to the qubit.

#### Readout Resonator

Acts as the bridge between the qubit and the output measurement system.

By carefully tuning the resonator length, it becomes possible to transfer information from the qubit while minimizing photon absorption and preserving the fragile quantum state.

---

## Simulation Status

The next phase of the project was intended to:

1. Inject microwave control pulses.
2. Observe the system response.
3. Tune design parameters.
4. Validate the circuit operation through simulation.

Unfortunately, restrictions within the simulation environment (**Ansys**) required significant time to be spent investigating alternative approaches and workarounds.

Although progress was made, the simulation could not be fully completed within the available timeframe.

> The simulation remains a work in progress and will be revisited in the near future.

---

## Future Work

The primary objective moving forward is to:

- Resolve the remaining Ansys simulation issues.
- Complete the microwave pulse analysis.
- Validate the full circuit operation.
- Publish a detailed guide for future learners interested in superconducting quantum hardware design.

---

*Tools Used: Qiskit Metal, Python*

*To be Used: Ansys Desktop Student, KLayout*
