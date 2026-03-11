# Reinforcement Learning Traffic Signal Optimization (SUMO)

## Overview

This project implements a **Deep Reinforcement Learning (DRL) system for adaptive traffic signal control** using:

- **SUMO (Simulation of Urban MObility)**
- **TraCI API**
- **PyTorch**

Each traffic light in the road network is controlled by an **independent Deep Q-Network (DQN) agent**. The agents observe local traffic conditions and learn when to **change or maintain the current signal phase** in order to **reduce congestion and queue lengths**.

The current simulation environment uses a **3×3 grid road network** where traffic signals are automatically generated at intersections. Vehicles are dynamically generated for each episode, allowing agents to learn under varying traffic conditions.

The system supports both:

- **Training mode** (learning traffic policies)
- **Evaluation mode** (running trained models)

---

# Features

- SUMO traffic simulation environment
- 3×3 grid road network
- Multi-agent reinforcement learning
- Independent agent per traffic signal
- Deep Q-Network (DQN) implementation
- Experience replay buffer
- Target network for stable Q-learning
- Epsilon-greedy exploration
- Dynamic vehicle generation each episode
- Training and testing pipelines
- Model checkpoint saving
- TraCI based traffic control

---

# Project Structure

```
Traffic_optimization/
│
├── models/                     # Saved neural network weights
│   ├── traffic_modelA1.pt
│   ├── traffic_modelB0.pt
│   ├── traffic_modelB1.pt
│   ├── traffic_modelB2.pt
│   └── traffic_modelC1.pt
│
├── src/
│   ├── helper.py               # SUMO interaction utilities
│   ├── neural_net.py           # Neural networks, agents, replay buffer
│   ├── train_pipeline.py       # RL training loop
│   └── test_pipeline.py        # Run trained agents
│
├── SUMO/
│   ├── grid_3.net.xml          # Road network
│   ├── routes.rou.xml          # Generated vehicle routes
│   ├── trips.trips.xml         # Generated trips
│   └── sim_3.sumocfg           # SUMO simulation configuration
│
├── .venv/                      # Python virtual environment
├── .gitignore
└── README.md
```

---

# Requirements

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

# Set Environment Variables

```
export SUMO_HOME=/usr/share/sumo
export PATH=$PATH:$SUMO_HOME/bin
```

(Optional: add to `.bashrc`.)

---

# Python Environment Setup

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

The simulation uses a **3×3 grid road network** with automatically generated traffic signals.

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

- `--grid` generates a grid road network
- `--grid.number 3` creates a **3×3 grid**
- `--tls.guess` automatically adds traffic lights
- `--tls.guess.threshold 10` avoids lights at trivial intersections
- `--tls.unset` removes unnecessary corner traffic lights

---

# Running Training

Run the training pipeline:

```
python src/train_pipeline.py
```

The training process:

1. Generate random vehicle routes
2. Start the SUMO simulation
3. Initialize an RL agent for each traffic light
4. Agents interact with the environment
5. Experience is stored in replay buffers
6. Neural networks are trained using sampled batches
7. Simulation restarts after each episode

Models are periodically saved to:

```
models/
```

---

# Running Evaluation

To run the simulation using trained models:

```
python src/test_pipeline.py
```

This will:

- Load trained traffic light models
- Disable exploration (`epsilon = 0`)
- Run the simulation using learned policies

---

# Reinforcement Learning Design

## Multi-Agent Setup

Each traffic light is controlled by **its own RL agent**.

Agents:

- Observe only **local traffic conditions**
- Learn independently
- Interact through the shared traffic environment

---

# State Representation

Each agent observes:

### 1. Halting vehicles per lane

```
traci.lane.getLastStepHaltingNumber(lane)
```

### 2. Downstream lane occupancy

Measures congestion in the lane **after the intersection**.

```
traci.lane.getLastStepOccupancy(lane)
```

### 3. Current traffic light phase

```
traci.trafficlight.getPhase(tl_id)
```

Final state vector:

```
[
 halting vehicles per lane,
 downstream occupancy per lane,
 current phase
]
```

---

# Action Space

Each agent has **two possible actions**.

| Action | Meaning |
|------|------|
| 0 | Keep current phase |
| 1 | Switch to next phase |

Phase switching:

```
traci.trafficlight.setPhase(tl, (phase + 1) % 4)
```

---

# Reward Function

The reward encourages **shorter vehicle queues**.

```
reward = - average halting vehicles per lane
```

Implementation:

```
reward = -total_queue / number_of_lanes
```

Agents therefore learn policies that **reduce congestion**.

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

# Training Mechanisms

## Experience Replay

Transitions are stored as:

```
(state, action, reward, next_state, done)
```

Mini-batches are sampled randomly during training.

---

## Target Network

A **target network** stabilizes learning.

Periodically updated using:

```
target_model.load_state_dict(model.state_dict())
```

---

## Exploration Strategy

Agents use **epsilon-greedy exploration**.

```
epsilon start = 1.0
epsilon decay = gradual
minimum epsilon = 0.1
```

This balances **exploration and exploitation**.

---

# Training Loop

Each episode performs:

1. Generate new vehicle routes
2. Start SUMO simulation
3. Initialize traffic states
4. Agents choose actions periodically
5. Traffic lights update phases
6. Rewards are computed
7. Transitions stored in replay buffers
8. Neural networks trained
9. Simulation ends when all vehicles exit
10. Next episode begins

---

# Logging

Training outputs periodic statistics:

```
Step: <time>
Average reward: <value>
Loss: <value>
Epsilon: <value>
Replay buffer size
```

Models are saved automatically during training.

---

# Future Work

Potential improvements include:

- Coordinated **multi-intersection learning**
- Larger road networks (5×5 grids or real maps)
- More advanced reward functions
- Comparison with **fixed-time traffic signals**
- **Green Wave traffic coordination** to synchronize signals along corridors for smoother traffic flow for emergency vehicles

---

# References

SUMO Documentation  
https://sumo.dlr.de/docs/

TraCI Documentation  
https://sumo.dlr.de/docs/TraCI.html

Deep Q-Learning (Nature)  
https://www.nature.com/articles/nature14236