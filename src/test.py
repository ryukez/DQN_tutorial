import argparse
import time
import re

import matplotlib.pyplot as plt
import gym
import torch

import config
import models
import utils
import agents


# extract rewards from log
def plot_progress(filename, label):
    N = 1000 # average num
    with open(filename, 'r') as f:
        log = f.read()

        pattern = re.compile(r'reward ([0-9.]+)')
        itr = pattern.finditer(log)
        rewards = [float(match.group(1)) for match in itr]
        averaged = [sum(rewards[i : i + N]) / N for i in range(len(rewards) - N)]

    plt.plot(averaged, label=label)


# demonstrate agent's play
def test(args):
    # setup
    env = gym.make('Breakout-v0')

    nAction = env.action_space.n

    Q = models.QNet(nAction)

    state_dict = None
    if config.isLocal:
        state_dict = torch.load(args.model_path, map_location='cpu')
    else:
        state_dict = torch.load(args.model_path)

    Q.load_state_dict(state_dict)
    Q.eval()

    if not config.isLocal:
        Q = Q.cuda()

    agent = agents.Agent(nAction, Q)

    t = 0
    action = env.action_space.sample()
    for episode in range(args.episode):
        print("episode: %d\n" % (episode + 1))

        observation = env.reset()
        state = torch.cat([utils.preprocess(observation)] * 4, 1)  # initial state
        sum_reward = 0

        # Exploration loop
        done = False
        while not done:
            if config.isLocal:
                env.render()

            action = agent.getAction(state, 0.0)

            # take action and calc next state
            observation, reward, done, _ = env.step(action)
            nextState = torch.cat([state.narrow(1, 1, 3), utils.preprocess(observation)], 1)

            state = nextState
            sum_reward += reward
            t += 1

            time.sleep(0.03)

        print("  reward %f\n" % sum_reward)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=20)
    parser.add_argument('--log_path', type=str, default='results/log.txt')
    parser.add_argument('--model_path', type=str, default='results/model.pth')

    args = parser.parse_args()

    # plotting
    plot_progress(args.log_path, '')

    plt.title('Training progress')
    plt.xlabel('episode')
    plt.ylabel('average reward (recent 100 samples)')
    plt.show()

    test(args)
