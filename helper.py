import traci

import subprocess
import random
import os

#Sumo helpers start here

def generate_routes():
    sumo_home = os.environ["SUMO_HOME"]
    random_trips = os.path.join(sumo_home, "tools", "randomTrips.py")

    subprocess.run([
        "python", 
        random_trips,
        "-n", "grid_3.net.xml",
        "-r", "routes.rou.xml",
        "-e", "30000",
        "--period", "6",
        "--validate",
        "--seed", str(random.randint(0,100000))
    ])

def start_sumo():
    sudoCmd = [
        "sumo", 
         "-c", "sim_3.sumocfg",
         "--no-step-log",
        "--no-warnings",
        "--start"
    ]
    traci.start(sudoCmd)

def manual_configure_TL(traffic_lights):
    for tl in traffic_lights:
        traci.trafficlight.setPhaseDuration(tl, 30000)

def get_traffic_lights():
    traffic_lights = traci.trafficlight.getIDList()

    return traffic_lights

def get_vehicle_mapping(traffic_lights):
    vehicle_mapping = {}

    for tl in traffic_lights:
        vehicle_mapping[tl] = get_state(tl)

    return vehicle_mapping

def get_downstream_traffic(lane):
    count = 0

    links = traci.lane.getLinks(lane)

    required_lane = '_'
    for link in links:
        if link[6] == 's':
            required_lane = link[0]
            break 

    if required_lane != '_':
        count += traci.lane.getLastStepOccupancy(required_lane)

    return count 

#Neural helpers start here

def get_state(traffic_light):
    lanes = list(set(traci.trafficlight.getControlledLanes(traffic_light)))

    halting = []
    downstream_occupancy = []
    
    for lane in sorted(lanes):  
        halting.append(traci.lane.getLastStepHaltingNumber(lane))
        downstream_occupancy.append(get_downstream_traffic(lane))

    phase = traci.trafficlight.getPhase(traffic_light)
    
    return halting + downstream_occupancy + [phase]

def change_state(traffic_light, action):
    if(action == 1):
        phase = traci.trafficlight.getPhase(traffic_light)
        traci.trafficlight.setPhase(traffic_light, (phase + 1) % 4)
        traci.trafficlight.setPhaseDuration(traffic_light, 30000)

def get_reward(traffic_light):
    lanes = set(traci.trafficlight.getControlledLanes(traffic_light))
    
    total_queue = 0
    for lane in lanes:
        total_queue += traci.lane.getLastStepHaltingNumber(lane)
    
    return -total_queue / len(lanes)