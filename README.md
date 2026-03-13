# Reinforcement Learning Traffic Signal Optimization (SUMO)

## Overview

This project implements a **Deep Reinforcement Learning system for adaptive traffic signal control** using:

* **SUMO (Simulation of Urban MObility)**
* **TraCI API**
* **PyTorch**

The system trains traffic signals to dynamically adapt their phases based on real-time traffic conditions in order to **reduce congestion and improve flow efficiency**.

The environment uses a **3×3 grid traffic network**, where vehicles are generated dynamically each episode. Traffic lights observe lane conditions and learn policies that minimize intersection pressure.

Two architectures are implemented through **separate Git branches**:

1. **Independent Intersection Models**
2. **Shared Multi-Agent Model**

Both approaches are designed to explore different reinforcement learning strategies for traffic signal control.

---

# Key Features

* Multi-agent reinforcement learning
* SUMO traffic simulation
* TraCI based signal control
* Deep Q-Network (DQN)
* Experience replay buffer
* Target network stabilization
* Epsilon-greedy exploration
* Automatic route generation
* Support for **multiple RL architectures via Git branches**
* Automatic **model checkpoint saving**
* Modular training and evaluation pipelines

---

# Branch Architecture

The repository contains **two main reinforcement learning designs**, each implemented in its own branch.

---

## 1️⃣ Independent Intersection Networks (Branch: `intersection_models`)

Each traffic light maintains **its own independent neural network**.

### Characteristics

* One **DQN per intersection**
* Models stored individually in the `models/` folder
* Agents learn **independently**
* No parameter sharing between intersections

### Advantages

* High specialization per intersection
* Flexible for heterogeneous networks

### Drawbacks

* Increased memory usage
* Slower training when scaling to large road networks

---

## 2️⃣ Shared Multi-Agent Network (Branch: `shared_state`)

Traffic lights **share neural networks based on intersection type**.

Two intersection categories are supported:

* **3-lane intersections**
* **4-lane intersections**

Each type maintains a **shared neural network**, meaning multiple traffic lights learn using the **same model parameters**.

### Characteristics

* One model for **3-lane intersections**
* One model for **4-lane intersections**
* Agents share replay buffers and training updates
* Faster learning through **experience aggregation**

### Advantages

* Much more scalable
* Faster training convergence
* Reduced model complexity

---

# Project Structure

```
Traffic_optimization/
│
├── models/                     # Automatically saved neural network weights
│
├── src/
│   ├── helper.py               # SUMO interaction utilities
│   ├── neural_net.py           # Neural networks, agents, replay buffer
│   ├── train_pipeline.py       # Reinforcement learning training pipeline
│   └── test_pipeline.py        # Evaluation pipeline for trained models
│
├── SUMO/
│   ├── grid_3.net.xml          # 3x3 grid road network
│   ├── routes.rou.xml          # Generated routes
│   ├── trips.trips.xml         # Generated trips
│   ├── sim_3.sumocfg           # SUMO simulation configuration
│
├── .venv/                      # Python virtual environment
├── .gitignore
└── README.md
```

---

# Model Storage

The project automatically manages the **`models/` directory**.

When training is executed:

```
python src/train_pipeline.py
```

the system:

1. Creates the `models/` directory if it does not exist.
2. Saves trained neural networks periodically.
3. Updates saved models when a **better reward is achieved**.

Example saved models:

```
models/
   traffic_model_three.pt
   traffic_model_four.pt
```

These models are then loaded during evaluation.

---

# Installation

## Install SUMO

Example for Ubuntu:

```
sudo apt install sumo sumo-tools sumo-doc
```

Verify installation:

```
sumo --version
```

---

# Environment Variables

Set the SUMO path:

```
export SUMO_HOME=/usr/share/sumo
export PATH=$PATH:$SUMO_HOME/bin
```

(Optional: add to `.bashrc`.)

---

# Python Setup

Create a virtual environment:

```
python -m venv .venv
```

Activate it:

Linux / Mac:

```
source .venv/bin/activate
```

Install dependencies:

```
pip install torch traci sumolib
```

---

# Generating the Road Network

The simulation uses a **3×3 grid network** generated with SUMO.

```
netgenerate \
--grid \
--grid.number 3 \
--tls.guess \
--tls.guess.threshold 10 \
--tls.unset A0,A2,C0,C2 \
-o grid_3.net.xml
```

Explanation:

| Parameter               | Description                        |
| ----------------------- | ---------------------------------- |
| `--grid`                | Generates a grid road network      |
| `--grid.number 3`       | Creates a **3×3 grid**             |
| `--tls.guess`           | Automatically adds traffic lights  |
| `--tls.guess.threshold` | Avoids lights on low-traffic nodes |
| `--tls.unset`           | Removes unnecessary corner lights  |

---

# Training the RL Agents

Run the training pipeline:

```
python src/train_pipeline.py
```

Training procedure:

1. Random traffic routes are generated
2. SUMO simulation starts
3. Agents observe intersection states
4. Agents choose actions (epsilon-greedy)
5. Traffic lights update phases
6. Rewards are computed
7. Transitions stored in replay buffers
8. Neural networks are trained
9. Best models are saved
10. Simulation resets for the next episode

---

# Running Evaluation

After training, run the evaluation pipeline:

```
python src/test_pipeline.py
```

Evaluation mode:

* Loads trained models from `models/`
* Disables exploration (`epsilon = 0`)
* Runs traffic lights using the learned policy

---

# Reinforcement Learning Design

## State Representation

Each agent observes:

* Vehicle counts on **incoming lanes**
* Vehicle counts on **outgoing lanes**
* Current **traffic signal phase**

Example state vector:

```
[
 incoming_lane_vehicle_counts,
 outgoing_lane_vehicle_counts,
 current_phase
]
```

---

# Action Space

Agents choose between **traffic signal phase changes**.

| Action | Meaning                           |
| ------ | --------------------------------- |
| 0      | Maintain current phase            |
| 1-3    | Switch to specific traffic phases |

The selected phase is applied using:

```
traci.trafficlight.setPhase()
```

---

# Reward Function

The reward is based on **intersection pressure**, encouraging balanced traffic flow.

Pressure is calculated using normalized vehicle density between incoming and outgoing lanes:

```
pressure = |incoming_density − outgoing_density|
```

The final reward:

```
reward = -pressure
```

Lower pressure means smoother traffic flow.

---

# Neural Network Architecture

Each agent uses a **Deep Q-Network (DQN)**.

Architecture:

```
Input Layer: state dimension

Hidden Layer: 128 neurons (ReLU)
Hidden Layer: 128 neurons (ReLU)

Output Layer: Q-values for actions
```

Implemented using **PyTorch**.

---

# Training Techniques

## Experience Replay

Transitions stored as:

```
(state, action, reward, next_state, done)
```

Mini-batches are randomly sampled during training.

---

## Target Network

A **target network** stabilizes Q-learning updates.

The target network is periodically synchronized:

```
target_model.load_state_dict(model.state_dict())
```

---

## Exploration Strategy

Agents use **epsilon-greedy exploration**:

```
epsilon start = 1.0
epsilon decay ≈ 0.99995
epsilon minimum = 0.1
```

This balances exploration and exploitation during training.

---

# Logging

Training logs periodically display:

```
Step: <time>
Average reward (3-lane intersections)
Average reward (4-lane intersections)
Loss
Replay buffer size
Epsilon
```

---

# Future Work

Potential improvements include:

* Coordinated **multi-intersection learning**
* Larger road networks (5×5 or real city maps)
* Graph Neural Networks for road networks
* Communication between intersections
* Emergency vehicle priority routing
* Green-wave signal coordination

---

# References

SUMO Documentation
https://sumo.dlr.de/docs/

TraCI Documentation
https://sumo.dlr.de/docs/TraCI.html

Deep Q-Learning (Nature)
https://www.nature.com/articles/nature14236
