# Reinforcement Learning Traffic Signal Optimization (SUMO)

## Overview

This project builds a **reinforcement learning (RL) system to optimize traffic signals** using the **SUMO (Simulation of Urban MObility)** traffic simulator. The goal is to train an RL agent that dynamically adjusts traffic light phases based on real-time traffic conditions to **improve traffic flow, reduce congestion, and minimize vehicle waiting time**.

The simulation environment currently uses a **3×3 grid road network** where vehicles are generated automatically and move through intersections. Traffic signals will eventually be controlled by an RL model that observes lane congestion and learns optimal signal timings.

A planned extension of the system is **emergency vehicle prioritization**, allowing vehicles such as ambulances to receive traffic signal priority.

---

# Current Features

* SUMO traffic simulation environment
* Grid road network generation
* Random vehicle trip generation
* Python interface using TraCI
* Python-controlled simulation stepping
* Traffic light detection and inspection

---

# Project Structure

```
Traffic_optimization/
│
├── main.py               # Python script that runs the SUMO simulation
├── sim_3.sumocfg         # SUMO configuration file
├── grid_3.net.xml        # Generated road network
├── routes.rou.xml        # Vehicle routes
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
pip install traci sumolib
```

---

# Generating the Road Network

The simulation uses a **3×3 grid network**. Traffic lights are automatically placed at appropriate intersections using TLS guessing.

```
netgenerate \
--grid \
--grid.number 3 \
--tls.guess \
--tls.guess.threshold 10 \
-o grid_3.net.xml
```

Explanation:

* `--grid` generates a grid road network
* `--grid.number 3` creates a **3×3 grid**
* `--tls.guess` automatically adds traffic lights
* `--tls.guess.threshold 10` prevents simple intersections (like corners) from receiving unnecessary signals
* `--tls.unset A0,A2,C0,C2` hardcoded unnecessary signals at grid corners 

---

# Generating Vehicle Routes

Vehicle routes are generated using `randomTrips.py`.

```
python $SUMO_HOME/tools/randomTrips.py \
-n grid_3.net.xml \
-r routes.rou.xml \
-e 30000 \
--period 3 \
--validate
```

Explanation:

* `-n` network file
* `-r` output route file
* `-e 30000` vehicles spawn for **30,000 seconds**
* `--period 3` generates one vehicle roughly every **3 seconds**
* `--validate` ensures generated routes exist in the network

---

# Running the Simulation

Run the simulation using:

```
python main.py
```

The script will:

1. Launch SUMO GUI
2. Start the simulation automatically
3. Advance the simulation through TraCI
4. End once all vehicles have completed their routes

---

# Current Simulation Loop

The simulation is stepped from Python using TraCI:

```
while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()
```

This continues until there are **no vehicles left in the simulation or scheduled to depart**.

---

# Planned Reinforcement Learning System

The next development stage is building an RL agent that controls traffic signals.

## State Representation

The agent will observe traffic conditions such as:

* number of vehicles in each lane
* queue lengths
* waiting times

These will be collected via TraCI.

Example:

```
traci.lane.getLastStepVehicleNumber(lane_id)
```

---

## Action Space

The RL agent will control:

* traffic light phase switching
* signal timing decisions

Example action:

```
traci.trafficlight.setPhase(tls_id, phase)
```

---

## Reward Function

Possible reward definitions:

* minimize vehicle waiting time
* minimize queue length
* maximize vehicle throughput

---

# Future Work

Planned improvements:

* reinforcement learning traffic signal controller
* multi-intersection coordination
* emergency vehicle prioritization
* larger road networks (5×5 or real maps)
* performance comparison with fixed-time signals

---

# References

SUMO Documentation
https://sumo.dlr.de/docs/

TraCI Documentation
https://sumo.dlr.de/docs/TraCI.html
