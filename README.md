# Reinforcement Learning Traffic Signal Optimization (SUMO)

## Overview

This project implements a **Deep Reinforcement Learning system for traffic signal control** using the **SUMO (Simulation of Urban MObility)** traffic simulator and **PyTorch**.

The system trains **independent RL agents for each traffic light** in a road network. Each agent observes the local traffic conditions and learns when to change the signal phase to reduce congestion.

The current environment uses a **3×3 grid road network**, where traffic signals are placed automatically at intersections. Vehicles are generated dynamically for each episode, and agents learn through repeated interaction with the simulation.

The objective of the RL agents is to **minimize vehicle queue lengths at intersections**.

---

# Current Features

* SUMO traffic simulation environment
* 3×3 grid road network
* Multi-agent reinforcement learning (one agent per traffic light)
* Deep Q-Network (DQN) implementation using PyTorch
* Experience replay buffer
* Target network for stable learning
* Epsilon-greedy exploration
* Dynamic vehicle generation each episode
* Python control using TraCI

---

# Project Structure

```
Traffic_optimization/
│
├── main.py               # Training loop and SUMO simulation control
├── helper.py             # SUMO helper functions and state/reward logic
├── neural_net.py         # Neural network, replay buffer, and RL agent
│
├── sim_3.sumocfg         # SUMO simulation configuration
├── grid_3.net.xml        # Generated grid road network
├── routes.rou.xml        # Generated vehicle routes (auto-generated)
│
├── .venv/                # Python virtual environment
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

## Set Environment Variables

```
export SUMO_HOME=/usr/share/sumo
export PATH=$PATH:$SUMO_HOME/bin
```

(Optional: add to `.bashrc`.)

---

## Create Python Virtual Environment

```
python -m venv .venv
source .venv/bin/activate
```

Install required packages:

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

* `--grid` generates a grid road network
* `--grid.number 3` creates a **3×3 grid**
* `--tls.guess` automatically adds traffic lights
* `--tls.guess.threshold 10` avoids adding lights to trivial intersections
* `--tls.unset` removes unnecessary corner traffic lights

---

# Running Training

Run the training script:

```
python main.py
```

The script will:

1. Generate vehicle routes
2. Start the SUMO simulation
3. Create an RL agent for each traffic light
4. Train the agents using simulation feedback
5. Restart the simulation after each episode

---

# Reinforcement Learning Design

## Multi-Agent Setup

Each traffic light is controlled by its **own RL agent**. Agents learn independently but interact through the shared traffic environment.

---

# State Representation

Each agent observes:

1. **Halting vehicles per lane**

```
traci.lane.getLastStepHaltingNumber(lane)
```

2. **Downstream lane occupancy**

This measures congestion in the next lane after the intersection.

```
traci.lane.getLastStepOccupancy(lane)
```

3. **Current traffic light phase**

```
traci.trafficlight.getPhase(tl_id)
```

Final state vector:

```
[halting vehicles per lane,
 downstream occupancy per lane,
 current phase]
```

---

# Action Space

Each agent has **two possible actions**:

| Action | Meaning              |
| ------ | -------------------- |
| 0      | Keep current phase   |
| 1      | Change to next phase |

Phase switching:

```
traci.trafficlight.setPhase(tl, (phase + 1) % 4)
```

---

# Reward Function

The reward is designed to **minimize queue length**.

```
reward = - average halting vehicles per lane
```

Implementation:

```
reward = -total_queue / number_of_lanes
```

This encourages the agent to **reduce waiting vehicles**.

---

# Neural Network Architecture

Each agent uses a **Deep Q-Network (DQN)**.

Architecture:

```
Input layer: state dimension
Hidden layer: 128 neurons (ReLU)
Hidden layer: 128 neurons (ReLU)
Output layer: Q-values for actions
```

Implemented using **PyTorch**.

---

# Training Mechanisms

## Experience Replay

Agents store transitions in a replay buffer:

```
(state, action, reward, next_state, done)
```

Mini-batches are sampled randomly during training.

---

## Target Network

A separate **target network** stabilizes Q-learning.

The target network is updated periodically:

```
target_model.load_state_dict(model.state_dict())
```

---

## Exploration Strategy

The agents use **epsilon-greedy exploration**:

```
epsilon starts at 1.0
epsilon decays gradually
minimum epsilon = 0.1
```

This allows the agent to explore early and exploit learned behavior later.

---

# Training Loop

Each episode follows this sequence:

1. Generate new vehicle routes

2. Start SUMO simulation

3. Initialize states

4. Every fixed interval:

   * Agents select actions
   * Traffic lights update
   * Rewards are calculated
   * Transitions stored in replay buffer
   * Neural network training step performed

5. Simulation ends when all vehicles exit

6. Next episode begins

---

# Logging

Training prints periodic statistics such as:

```
Step: <time>
Average reward: <value>
Loss: <value>
Epsilon: <value>
Replay buffer size
```

Models are periodically saved to disk.

---

# Future Work

Possible improvements:

* Coordinated multi-intersection learning
* Larger traffic networks (5×5 or city maps)
* Advanced reward functions
* Graph neural networks for intersection coordination
* Emergency vehicle prioritization
* Evaluation against fixed-time traffic signals

---

# References

SUMO Documentation
https://sumo.dlr.de/docs/

TraCI Documentation
https://sumo.dlr.de/docs/TraCI.html

Deep Q-Learning
https://www.nature.com/articles/nature14236
