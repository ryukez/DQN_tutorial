import argparse

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

            # frame skip
            if t % args.frame_skip == 0:
                action = agent.getAction(state, args.eps)

            # take action and calc next state
            observation, reward, done, _ = env.step(action)
            nextState = torch.cat([state.narrow(1, 1, 3), utils.preprocess(observation)], 1)

            buffer.push(utils.Step(state, action, reward, nextState))
            state = nextState
            sum_reward += reward
            t += 1

            # initial waiting
            if t < args.initial_wait:
                continue

            # update model
            if t % args.train_freq == 0:
                batch = buffer.sample(args.batch)
                trainer.update(batch)

            # update target
            if t % args.target_update_freq == 0:
                QTarget.load_state_dict(Q.state_dict())

        print("  reward %f\n" % sum_reward)

        if episode % args.snapshot_freq == 0:
            torch.save(Q.state_dict(), "results/%d.pth" % episode)
            print("  model saved")

    torch.save(Q.state_dict(), "results/model.pth")
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=12000)
    parser.add_argument('--buffer_size', type=int, default=400000)
    parser.add_argument('--train_freq', type=int, default=4)
    parser.add_argument('--initial_wait', type=int, default=20000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--target_update_freq', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--frame_skip', type=int, default=4)
    parser.add_argument('--snapshot_freq', type=int, default=1000)
    parser.add_argument('--eps', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--model_path', type=str)

    args = parser.parse_args()
    train(args)

