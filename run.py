import os
import gymnasium as gym
import numpy as np

from optimizers.genetic_algorithm import GeneticOptimizer
from logger import ColoredLogger

log = ColoredLogger(os.path.basename(__file__)).get_logger()


def test(individual, env, num_episodes=5):
    """
    Test the performance of an individual in the given environment.

    :param individual: The individual to test.
    :param env: The environment in which to test the individual.
    :param num_episodes: The number of episodes to run the test.
    :return: Average reward achieved by the individual.
    """
    total_reward = 0.0
    for episode in range(num_episodes):
        individual.reset()
        observation, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            input_values = dict(zip(individual.input_node_ids, observation))
            output_nodes = individual.inference(input_values)
            action = np.argmax(list(output_nodes.values()))
            
            observation, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
            env.render()
            
        log.info(f"Episode {episode + 1} Reward: {episode_reward}")
        total_reward += episode_reward

    env.close()
    average_reward = total_reward / num_episodes
    log.info(f"Average Reward: {average_reward:.2f}")
    return average_reward


def main():
    experiment_name:str = "LunarLander-v2"
    experiment_name:str = "MountainCar-v0"
    experiment_name:str = "Acrobot-v1"
    experiment_name:str = "CartPole-v1"
    
    population_size:int = 20
    envs = gym.make_vec(experiment_name, num_envs=population_size)
    
    genalg = GeneticOptimizer(
        population_size=population_size,
        crossover_rate=0.4,
        mutation_rate=0.3,
        stateful=False,
        max_steps=6,
        envs=envs,
    )
    log.info("Starting optimization...")
    best_individual = genalg.optimize(10)
    best_individual.visualize()
    env = gym.make(experiment_name, render_mode="human")
    test(best_individual, env)
    
if __name__ == "__main__":
    main()
