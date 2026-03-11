import torch
import traci
from traci.exceptions import FatalTraCIError
import os

from helper import get_reward, change_state, get_state, get_traffic_lights, manual_configure_TL, generate_routes, start_sumo
from neural_net import TrafficAgent, ReplayBuffer, Agent

try:
    os.makedirs("models", exist_ok=True)
    generate_routes()
    start_sumo()
    traffic_lights = get_traffic_lights()
    manual_configure_TL(traffic_lights)

    agents = {}
    for tl in traffic_lights:
        state_dim = len(get_state(tl))  
        action_dim = 2

        model = TrafficAgent(state_dim, action_dim)
        model_next = TrafficAgent(state_dim, action_dim)
        model_next.load_state_dict(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
        buffer = ReplayBuffer()
        agents[tl] = Agent(model, model_next, optimizer, buffer, tl)

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
                actions = {tl: agents[tl].select_action(states[tl]) for tl in traffic_lights}

                for tl in traffic_lights:
                    change_state(tl, actions[tl])

                next_states = {tl: get_state(tl) for tl in traffic_lights}
                rewards = {tl: get_reward(tl) for tl in traffic_lights}

                done = traci.simulation.getMinExpectedNumber() == 0

                reward_sum += sum(rewards.values())
                reward_count += len(rewards)

                if time % 1000 == 0:
                    avg = reward_sum / reward_count
                    print(f"Step: {time}, Average reward: {avg}")
                    
                    reward_sum = 0
                    reward_count = 0

                for tl in traffic_lights:
                    agents[tl].buffer.push(
                        states[tl],
                        actions[tl],
                        rewards[tl],
                        next_states[tl],
                        done
                    )
                    agents[tl].train()

                states = next_states

            traci.simulationStep()
            time += 1

        if traci.isLoaded():
            traci.close()

        count += 1
        print(f"Episode {count} over, starting new episode...")

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