import torch
import traci
from traci.exceptions import FatalTraCIError

from helper import get_reward, get_state, change_state, get_traffic_lights, manual_configure_TL, generate_routes, start_sumo, build_structure
from neural_net import TrafficAgent, Agent, ReplayBuffer

def load_agents(traffic_lights, structures):
    type_models = {}
    agents = {}

    for tl in traffic_lights:
        itype = structures[tl]

        if itype not in type_models:
            state_dim = len(get_state(tl, 0))
            model = TrafficAgent(state_dim, 4)
            model.load_state_dict(torch.load(f"models/og_{itype}.pt"))
            model.eval()

            target = TrafficAgent(state_dim, 4)
            buffer = ReplayBuffer(1)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
            type_models[itype] = (model, target, optimizer, buffer)

        model, target, optimizer, buffer = type_models[itype]
        agent = Agent(model, target, optimizer, buffer, itype)
        agent.epsilon = 0
        agents[tl] = agent

    return agents

try:
    start_sumo()

    traffic_lights = get_traffic_lights()
    manual_configure_TL(traffic_lights)

    structures = {tl: build_structure(tl) for tl in traffic_lights}
    agents = load_agents(traffic_lights, structures)

    spent_duration = {tl: 0 for tl in traffic_lights}
    states = {tl: get_state(tl, spent_duration[tl]) for tl in traffic_lights}

    reward_sum_3 = 0
    reward_count_3 = 0
    reward_sum_4 = 0
    reward_count_4 = 0

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
                change_state(tl, actions[tl])

            next_states = {tl: get_state(tl, spent_duration[tl]) for tl in traffic_lights}
            rewards = {tl: get_reward(tl) for tl in traffic_lights}

            for tl in traffic_lights:
                if structures[tl] == "three":
                    reward_sum_3 += rewards[tl]
                    reward_count_3 += 1
                else:
                    reward_sum_4 += rewards[tl]
                    reward_count_4 += 1

            if time % 1000 == 0 and time > 0:
                avg_3 = reward_sum_3 / reward_count_3 if reward_count_3 > 0 else 0
                avg_4 = reward_sum_4 / reward_count_4 if reward_count_4 > 0 else 0
                print(f"Step: {time}, Average reward (4): {avg_4:.4f}, Average reward (3): {avg_3:.4f}")
                reward_sum_3 = reward_sum_4 = reward_count_3 = reward_count_4 = 0

            states = next_states

        traci.simulationStep()
        time += 1

except KeyboardInterrupt:
    print("Sumo stopped by user")

except FatalTraCIError:
    print("Sumo stopped")

finally:
    try:
        if traci.isLoaded():
            traci.close()
    except Exception:
        pass