import gym
from DQN import Agent
import pylab
import numpy as np
import torch as T
import sys, os


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 64, n_actions = env.action_space.n,
                    input_dims = env.observation_space.shape, lr = 1e-4)

#Test
    if os.path.isfile('trained_model.pt'):
        agent.load_checkpoint()
        agent.epsilon = agent.eps_min

    scores, episodes = [], []
    episode = 100
    score = 0

    for i in range(episode):
        if i % 10 == 0 and i > 0:
            avg_score = np.mean(scores[-100:])
            print('episode: ', i, 'score ', score,
                        'average score %.3f' % avg_score,
                        'epsilon %.3f' % agent.epsilon)
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
            reward = reward if not done or score == 199 else -100
            score += reward
            agent.store_transition(state, action, reward, next_state, done)
            agent.experience_replay()
            state = next_state
            if done:
                score = score if score == 200 else score + 100
                scores.append(score)
                episodes.append(i)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/cartpole_dqn.png")
                print("episode:", i, "  score:", score,"  epsilon:", agent.epsilon)
    agent.save_checkpoint()