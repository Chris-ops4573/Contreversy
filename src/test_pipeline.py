import torch
import traci
from traci.exceptions import FatalTraCIError

from helper import get_reward, get_state, change_state, get_traffic_lights, manual_configure_TL, generate_routes, start_sumo
from neural_net import TrafficAgent, Agent, ReplayBuffer

def load_agents(traffic_lights):
    agents = {}

    for tl in traffic_lights:
        state_dim = len(get_state(tl))
        model = TrafficAgent(state_dim, 2)
        model.load_state_dict(torch.load(f"models/traffic_model{tl}.pt"))
        model.eval()

        target = TrafficAgent(state_dim, 2)
        buffer = ReplayBuffer(1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
        agent = Agent(model, target, optimizer, buffer, tl)
        agent.epsilon = 0
        agents[tl] = agent

    return agents

try:
    generate_routes()
    start_sumo()

    traffic_lights = get_traffic_lights()
    manual_configure_TL(traffic_lights)
    agents = load_agents(traffic_lights)

    states = {tl: get_state(tl) for tl in traffic_lights}

    reward_sum = 0
    reward_count = 0
    step = 10
    time = 0

    while traci.simulation.getMinExpectedNumber() > 0:
        if time % step == 0:

            actions = {}
            for tl in traffic_lights:
                actions[tl] = agents[tl].select_action(states[tl])
                change_state(tl, actions[tl])
            
            next_states = {tl: get_state(tl) for tl in traffic_lights}
            rewards = {tl: get_reward(tl) for tl in traffic_lights}

            reward_sum += sum(rewards.values())
            reward_count += len(rewards)

            if time % 1000 == 0:
                print(f"Step: {time}, Average reward: {reward_sum / reward_count:.4f}")
                reward_sum = 0
                reward_count = 0

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