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

def get_lane_direction(lane):
    shape = traci.lane.getShape(lane)

    x1, y1 = shape[0]
    x2, y2 = shape[-1]

    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) > abs(dy): 
        if dx > 0:
            return "east"
        else:
            return "west"

    else:
        if dy > 0:
            return "north"
        else:
            return "south"

def get_downstream_traffic(traffic_light):
    lanes = set(traci.trafficlight.getControlledLanes(traffic_light))

    lane_downstream_dir = {
        "north": 0.0,
        "south": 0.0,
        "east": 0.0,
        "west": 0.0
    }

    for lane in lanes:
        links = traci.lane.getLinks(lane)

        required_lane = '_'
        for link in links:
            if link[6] == 's':
                required_lane = link[0]
                break 

        if required_lane != '_':
            lane_downstream_dir[get_lane_direction(required_lane)] += traci.lane.getLastStepOccupancy(required_lane)

    return [
        lane_downstream_dir["north"],
        lane_downstream_dir["south"],
        lane_downstream_dir["east"],
        lane_downstream_dir["west"]
    ]

#Neural helpers start here

def get_state(traffic_light):
    lanes = set(traci.trafficlight.getControlledLanes(traffic_light))

    lane_waiting_dir = {
        "north": 0.0,
        "south": 0.0,
        "east": 0.0,
        "west": 0.0
    }
    lane_occupancy_dir = {
        "north": 0.0,
        "south": 0.0,
        "east": 0.0,
        "west": 0.0
    }

    vehicle_count = []
    lane_waiting_time = []

    for lane in lanes:
        direction = get_lane_direction(lane)

        lane_waiting_dir[direction] += traci.lane.getWaitingTime(lane)
        lane_occupancy_dir[direction] += traci.lane.getLastStepOccupancy(lane)  

    downstream_occupancy = get_downstream_traffic(traffic_light)

    phase = traci.trafficlight.getPhase(traffic_light)

    lane_occupancy = [
        lane_waiting_dir["north"],
        lane_waiting_dir["south"],
        lane_waiting_dir["east"],
        lane_waiting_dir["west"],

        lane_occupancy_dir["north"],
        lane_occupancy_dir["south"],
        lane_occupancy_dir["east"],
        lane_occupancy_dir["west"],
    ]

    return lane_occupancy + downstream_occupancy + [phase]

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

def show_reward(reward_sum, reward_count, time):
    avg_reward = reward_sum / reward_count
    print("Step:", time, "Average reward:", avg_reward)