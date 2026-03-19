import torch
import traci
from traci.exceptions import FatalTraCIError
import os

from helper import get_reward, change_state, get_state, build_structure, get_traffic_lights, manual_configure_TL, generate_routes, start_sumo
from neural_net import TrafficAgent, ReplayBuffer, Agent

try:
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/real", exist_ok=True)
    generate_routes()
    start_sumo()
    traffic_lights = get_traffic_lights()
    manual_configure_TL(traffic_lights)

    type_models = {}
    agents = {}

    best_rewards = {"three": float('-inf'), "four": float('-inf')}

    structure = {tl: build_structure(tl) for tl in traffic_lights}

    for tl in traffic_lights:
        itype = structure[tl]

        if itype not in type_models:
            state_dim = len(get_state(tl, 0))

            model = TrafficAgent(state_dim, 4)
            model_next = TrafficAgent(state_dim, 4)
            model_next.load_state_dict(model.state_dict())
            optimizer = torch.optim.Adam(model.parameters(), lr = 0.00005)
            buffer = ReplayBuffer()
            type_models[itype] = (model, model_next, optimizer, buffer)

        model, model_next, optimizer, buffer = type_models[itype]
        agents[tl] = Agent(model, model_next, optimizer, buffer, itype)

    if traci.isLoaded():
        traci.close()

    episodes = 1000

    while episodes:
        generate_routes()
        start_sumo()

        traffic_lights = get_traffic_lights()
        manual_configure_TL(traffic_lights)

        spent_duration = {tl: 0 for tl in traffic_lights}

        reward_sum_3 = 0
        reward_count_3 = 0
        reward_sum_4 = 0
        reward_count_4 = 0

        states = {tl: get_state(tl, spent_duration[tl]) for tl in traffic_lights}

        step = 10
        time = 0

        while traci.simulation.getMinExpectedNumber() > 0 and time < 30000:

            if time % step == 0:
                actions = {tl: agents[tl].select_action(states[tl]) for tl in traffic_lights}
                

                for tl in traffic_lights:
                    if actions[tl] == 0:
                        spent_duration[tl] = 0
                    else:
                        spent_duration[tl] += step

                for tl in traffic_lights:
                    change_state(tl, actions[tl])

                next_states = {tl: get_state(tl, spent_duration[tl]) for tl in traffic_lights}
                rewards = {tl: get_reward(tl) for tl in traffic_lights}

                done = traci.simulation.getMinExpectedNumber() == 0

                for tl in traffic_lights:
                    if structure[tl] == "three":
                        reward_count_3 += 1
                        reward_sum_3 += rewards[tl]
                    else:
                        reward_count_4 += 1
                        reward_sum_4 += rewards[tl] 

                if time % 1000 == 0 and time != 0:
                    avg_4 = reward_sum_4 / reward_count_4
                    avg_3 = reward_sum_3 / reward_count_3

                    print(f"Step: {time}, Average reward (4): {avg_4}, Average reward (3): {avg_3}")

                    if best_rewards["three"] < avg_3:
                        torch.save(type_models["three"][0].state_dict(), f"models/traffic_model_4tl_three.pt")
                        best_rewards["three"] = avg_3

                    if best_rewards["four"] < avg_4:
                        torch.save(type_models["four"][0].state_dict(), f"models/real/traffic_model_4tl_four.pt")
                        best_rewards["four"] = avg_4

                    reward_sum_4 = 0
                    reward_sum_3 = 0
                    reward_count_4 = 0
                    reward_count_3 = 0

                for tl in traffic_lights:
                    agents[tl].buffer.push(
                        states[tl],
                        actions[tl],
                        rewards[tl],
                        next_states[tl],
                        done
                    )

                trained_types = set()
                for tl in traffic_lights:
                    itype = structure[tl]

                    if itype not in trained_types:
                        agents[tl].train()
                        trained_types.add(itype)

                states = next_states

            traci.simulationStep()
            time += 1

        if traci.isLoaded():
            traci.close()

        episodes -= 1

        print(f"Episode {episodes} over, starting new episode...")

except KeyboardInterrupt:
    print("Sumo stopped by user")

except FatalTraCIError:
    print("Sumo stopped by user")

finally:
    try:
        if traci.isLoaded():
            traci.close()
    except Exception:
        pass