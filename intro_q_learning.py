"""
https://rubikscode.net/2019/06/24/introduction-to-q-learning-with-python-and-open-ai-gym/
"""
import numpy as np
import random
import gym

alpha = 0.1
gamma = 0.6
epsilon = 0.1


def main():
    env = gym.make("Taxi-v3").env
    env.render()

    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    num_of_episodes = 100000
    for episode in range(0, num_of_episodes):
        state = env.reset()
        terminated = False

        while not terminated:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, terminated, info = env.step(action)

            # Update Q-table
            q_value = q_table[state, action]
            max_value = np.max(q_table[next_state])

            # THE big q learning equation :)
            new_q_value = (1 - alpha) * q_value + alpha * (reward + gamma * max_value)

            # Update Q-table
            q_table[state, action] = new_q_value
            state = next_state

        if (episode + 1) % 100 == 0:
            # clear_output(wait=True)
            print("Episode: {}".format(episode + 1))
            env.render()

    total_epochs = 0
    total_penalties = 0
    num_of_episodes = 100

    for _ in range(num_of_episodes):
        state = env.reset()
        epochs = 0
        penalties = 0

        terminated = False

        while not terminated:
            action = np.argmax(q_table[state])
            state, reward, terminated, info = env.step(action)
            # print(state)

            if reward == -10:
                penalties += 1

            epochs += 1

        total_penalties += penalties
        total_epochs += epochs

    print("**********************************")
    print("Results")
    print("**********************************")
    print("Epochs per episode: {}".format(total_epochs / num_of_episodes))
    print("Penalties per episode: {}".format(total_penalties / num_of_episodes))


if __name__ == '__main__':
    main()
