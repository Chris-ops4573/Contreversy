import traci

import subprocess
import os
import math

#Sumo helpers start here

def generate_routes():
    sumo_home = os.environ["SUMO_HOME"]
    random_trips = os.path.join(sumo_home, "tools", "randomTrips.py")

    subprocess.run([
        "python", 
        random_trips,
        "-n", "SUMO/grid_4.net.xml",
        "-r", "SUMO/routes_4.rou.xml",
        "-o", "SUMO/trips_4.trips.xml",
        "-e", "30000",
        "--period", "3",
        "--validate"
    ])

def start_sumo():
    sudoCmd = [
        "sumo", 
        "-c", "SUMO/sim_4.sumocfg",
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

def build_structure(traffic_light):
    lanes = list(set(traci.trafficlight.getControlledLanes(traffic_light)))

    if len(lanes) == 3:
        return "three"
    else:
        return "four"

def get_lane_structure(traffic_light):
    links = traci.trafficlight.getControlledLinks(traffic_light)
    
    incoming = []
    outgoing = []
    
    for link_group in links:
        for link in link_group:
            inc = link[0]   
            out = link[1]   
            
            if inc and inc not in incoming:
                incoming.append(inc)
            if out and out not in outgoing:
                outgoing.append(out)
    
    return incoming, outgoing

def get_lane_capacity(lane):
    length = traci.lane.getLength(lane)
    return max(1, length / 7.5)

#Neural helpers start here

def get_state(traffic_light):
    incoming, outgoing = get_lane_structure(traffic_light)

    incoming_length = []
    outgoing_length = []

    for lane in incoming:
        incoming_length.append(traci.lane.getLastStepVehicleNumber(lane))

    for lane in outgoing:
        outgoing_length.append(traci.lane.getLastStepVehicleNumber(lane))

    phase = traci.trafficlight.getPhase(traffic_light)

    return incoming_length + outgoing_length + [phase]

def change_state(traffic_light, action):
        traci.trafficlight.setPhase(traffic_light, action)
        traci.trafficlight.setPhaseDuration(traffic_light, 30000)

def get_reward(traffic_light):
    incoming, outgoing = get_lane_structure(traffic_light)
    
    pressure = 0
    for inc in incoming:
        for out in outgoing:
            x_inc = traci.lane.getLastStepVehicleNumber(inc)
            x_out = traci.lane.getLastStepVehicleNumber(out)

            x_max_inc = get_lane_capacity(inc)
            x_max_out = get_lane_capacity(out)

            pressure += abs(x_inc/x_max_inc - x_out/x_max_out)
    
    return -pressure