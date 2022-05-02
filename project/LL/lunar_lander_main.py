import gym
from lunar_landerDQN import Agent
import pylab
import numpy as np
import torch as T
import sys, os


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 64, n_actions = 4,
                    input_dims = [8], lr = 0.003)

    if os.path.isfile('trained_model.pt'):
        agent.load_checkpoint()
        agent.epsilon = agent.eps_min

    scores, episodes = [], []
    episode = 100
    score = 0

    for i in range(episode):
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores)
            print('episode: ', i, 'score ', score,
                        'average score ', avg_score,
                        'epsilon ', agent.epsilon)
        if i % 50 == 0 and i > 0:
            agent.save_checkpoint()
            print('-==SAVING CHECKPOINT==- fn: \'trained_model.pt\'')
        score = 0
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(state, action, reward, next_state, done)
            agent.experience_replay()
            state = next_state
            if done:
                scores.append(score)
                episodes.append(i)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/ll_dqn.png")
                print("episode:", i, "  score:", score,"  epsilon:", agent.epsilon)
    agent.save_checkpoint()

