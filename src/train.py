import argparse
import time
import random

import gym
import torch
import torch.optim as optim

import config
import models
import utils
import agents
import trainers


# train model
def train(args):
    # setup
    env = gym.make('Breakout-v0')

    nAction = env.action_space.n
    buffer = utils.ReplayBuffer(args.buffer_size)

    Q = models.QNet(nAction)
    QTarget = models.QNet(nAction)

    if args.model_path is not None:
        state_dict = None
        if config.isLocal:
            state_dict = torch.load(args.model_path, map_location='cpu')
        else:
            state_dict = torch.load(args.model_path)

        Q.load_state_dict(state_dict)
        QTarget.load_state_dict(state_dict)

    Q.train()
    QTarget.eval()

    if not config.isLocal:
        Q = Q.cuda()
        QTarget = QTarget.cuda()

    opt = optim.Adam(Q.parameters(), lr=args.lr)

    agent = agents.Agent(nAction, Q)
    trainer = trainers.Trainer(Q, QTarget, opt, args.gamma)

    t = 0
    action = 0  # no op
    start_t = time.time()

    for episode in range(args.episode):
        print("episode: %d\n" % (episode + 1))

        observation = env.reset()
        state = torch.cat([utils.preprocess(observation)] * 4, 1)  # initial state
        sum_reward = 0
        initial_freeze = random.randint(0, args.initial_freeze_max)

        # Exploration loop
        done = False
        while not done:
            if config.isLocal:
                env.render()

            # replay start
            if t < args.replay_start:
                action = env.action_space.sample()
            # frame skip
            elif t % args.action_repeat == 0:
                alpha = t / args.exploration_steps
                eps = (1 - alpha) * args.initial_eps + alpha * args.final_eps
                eps = max(eps, args.final_eps)

                action = agent.getAction(state, eps)

            if initial_freeze > 0:
                action = 0  # no op
                initial_freeze -= 1

            # take action and calc next state
            observation, reward, done, _ = env.step(action)
            nextState = torch.cat([state.narrow(1, 1, 3), utils.preprocess(observation)], 1)
            buffer.push(utils.Step(state, action, reward, nextState, done))
            state = nextState
            sum_reward += reward
            t += 1

            # replay start
            if t < args.replay_start:
                continue

            # update model
            if t % args.train_freq == 0:
                batch = buffer.sample(args.batch)
                trainer.update(batch)

            # update target
            if t % args.target_update_freq == 0:
                QTarget.load_state_dict(Q.state_dict())

        print("  reward %.1f\n" % sum_reward)

        elapsed_minutes = (time.time() - start_t) / 60
        print("  elapsed %.1f min\n" % elapsed_minutes)
        print("  average %.2f min\n" % (elapsed_minutes / (episode + 1)))

        if episode % args.snapshot_freq == 0:
            torch.save(Q.state_dict(), "results/%d.pth" % episode)
            print("  model saved")

    torch.save(Q.state_dict(), "results/model.pth")
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=12000)
    parser.add_argument('--buffer_size', type=int, default=1000000)
    parser.add_argument('--train_freq', type=int, default=4)
    parser.add_argument('--replay_start', type=int, default=50000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--target_update_freq', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--action_repeat', type=int, default=4)
    parser.add_argument('--initial_freeze_max', type=int, default=30)
    parser.add_argument('--snapshot_freq', type=int, default=1000)
    parser.add_argument('--initial_eps', type=float, default=1.0)
    parser.add_argument('--final_eps', type=float, default=0.1)
    parser.add_argument('--exploration_steps', type=float, default=1000000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--model_path', type=str)

    args = parser.parse_args()
    train(args)