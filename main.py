import traci
from traci.exceptions import FatalTraCIError

from helper import show_reward, get_reward, change_state, get_state, get_vehicle_mapping, get_traffic_lights, manual_configure_TL, generate_routes, start_sumo
from neural_net import target_model, model, optimizer, buffer, Agent

try:
    generate_routes()
    start_sumo()

    traffic_lights = get_traffic_lights()
    manual_configure_TL(traffic_lights)

    reward_sum = 0
    reward_count = 0

    agents = {}

    for tl in traffic_lights:
        agents[tl] = Agent(model, target_model, optimizer, buffer)

    if traci.isLoaded():
        traci.close()

    count = 0

    while True:
        generate_routes()
        start_sumo()

        traffic_lights = get_traffic_lights()
        manual_configure_TL(traffic_lights)

        reward_sum = 0
        reward_count = 0

        states = {tl: get_state(tl) for tl in traffic_lights}

        step = 10
        time = 0
        
        while traci.simulation.getMinExpectedNumber() > 0:    
            
            if time % step == 0:
                actions = {}

                for tl in traffic_lights:
                    actions[tl] = agents[tl].select_action(states[tl])

                for tl in traffic_lights:
                    change_state(tl, actions[tl])

                next_states = {tl: get_state(tl) for tl in traffic_lights}
                rewards = {tl: get_reward(tl) for tl in traffic_lights}

                reward_sum += sum(rewards.values())
                reward_count += len(rewards)

                if time % 1000 == 0:
                    show_reward(reward_sum, reward_count, time)
                    reward_sum = 0
                    reward_count = 0

                for tl in traffic_lights:
                    buffer.push(
                        states[tl],
                        actions[tl],
                        rewards[tl],
                        next_states[tl],
                        False if time != 30000 else True
                    )

                agents[traffic_lights[0]].train()

                states = next_states

            traci.simulationStep()
            time += 1

        if traci.isLoaded():
            traci.close()

        count += 1
        print("Episode ", count, " over, starting new episode...")

except FatalTraCIError:
    print("Sumo stopped by user")

finally:
    if traci.isLoaded():
        traci.close()
