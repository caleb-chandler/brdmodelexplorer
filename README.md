# BRD Model Explorer

This application simulates the **Believer-Receiver-Disbeliever (BRD)** model, a stochastic variant of the SI (Susceptible-Infected) compartmental model. It visualizes how beliefs spread through various network topologies, accounting for initial adoption and subsequent conversion.

---

## How the Model Works

The model categorizes nodes into three distinct compartments:
* **Receivers**: Neutral nodes that have not yet been exposed to the belief.
* **Believers**: Nodes that have adopted and are actively spreading the belief.
* **Disbelievers**: Nodes that have been exposed but initially rejected the belief.

### The Stochastic Process
1.  **Initial Seed**: A random node is selected as the first Believer.
2.  **Exposure**: Believers attempt to spread the belief to their neighbors.
3.  **Adoption ($\lambda$)**: When a Receiver is exposed, they become a Believer with probability $p$; otherwise, they become a Disbeliever.
4.  **Conversion ($\eta$)**: In each timestep, existing Disbelievers have a probability $\eta$ of "converting" into Believers.
5.  **Saturation**: The simulation runs until all nodes are Believers or the maximum allowed timesteps are reached.

---

## Features

* **Network Topologies**: Support for Watts-Strogatz, Barabási-Albert, Erdős-Rényi, and Random Geometric graphs.
* **Interactive Parameters**: Adjust population size ($N$), Adoption Probability ($\lambda$), and Conversion Probability ($\eta$) via the sidebar.
* **Visual Evolution**: Real-time generation of animated GIFs to watch the spread across the network (available for $N \le 750$).
* **Sensitivity Analysis**: Automated batch simulations to analyze how "Time to Full Belief" scales with changes in size or probability.

---
