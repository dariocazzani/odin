import random
import numpy as np
from odin.inference_engines.spherical import SphericalEngine
from odin.logger import ColoredLogger
from odin.optimizers.mutator import Mutator
from odin.interfaces.custom_types import float32

log = ColoredLogger("GeneticOptimizer").get_logger()

class GeneticOptimizer:
    def __init__(self,
                 population_size:int,
                 crossover_rate:float,
                 mutation_rate:float,
                 stateful:bool,
                 max_steps:int,
                 envs,
            ) -> None:

        self._population_size = population_size
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate
        self._stateful = stateful
        self._max_steps = max_steps
        self._envs = envs

        self._elitism_count:int = int(self._population_size * 0.2)

        self._input_size:int = envs.observation_space.shape[-1]
        self._output_size:int = envs.action_space[0].n
        self._input_node_ids = set(range(0, self._input_size))
        self._output_node_ids:set = set(range(self._input_size, self._input_size + self._output_size))

        self._population:list[SphericalEngine] = self._initialize_population()


    def _initialize_individual(self) -> SphericalEngine:
        return SphericalEngine.from_node_ids(self._input_node_ids, self._output_node_ids)


    def _initialize_population(self) -> list[SphericalEngine]:
        """
        Initializes a population of the given size using initialize_individual.
        Returns the initialized population.
        """
        log.info(f"Initializing population of size {self._population_size}")
        population = [self._initialize_individual() for _ in range(self._population_size)]
        return population


    def compute_fitness(self, population:list[SphericalEngine], num_episodes:int=5) -> list[float]:
        """
        Computes the fitness of all individuals in the population using a vectorized environment.
        The fitness is computed as the average reward over multiple episodes.
        """
        total_rewards = np.zeros(len(population), dtype=float)

        for episode in range(num_episodes):
            # Reset each individual at the beginning of each episode
            # NOTE: Do not reset during the episodes
            [individual.reset() for individual in population if self._stateful]

            observations, _ = self._envs.reset()
            episode_rewards = np.zeros(len(population), dtype=float)
            all_dones = np.array([False] * len(population))

            while not all(all_dones):
                actions = self.batch_inference(observations, population)
                new_observations, batch_rewards, dones, truncation, _ = self._envs.step(actions)
                all_dones = all_dones | dones | truncation

                episode_rewards[~all_dones] += batch_rewards[~all_dones]

                # Only update observations for active environments
                observations = new_observations

            total_rewards += episode_rewards

        # Average the rewards over the episodes.
        average_rewards = total_rewards / num_episodes

        return list(average_rewards)


    def batch_inference(self, observations: np.ndarray, population:list[SphericalEngine]) -> np.ndarray:
        """
        Obtain actions for a batch of observations. Assumes that the population is sorted in the same
        order as the observations.
        """
        actions = np.empty((len(observations),), dtype=int)
        for idx, (individual, observation) in enumerate(zip(population, observations)):
            input_values = dict(zip(self._input_node_ids, observation))
            output_nodes = individual.inference(input_values)
            actions[idx] = np.argmax(list(output_nodes.values()))
        return actions


    def select_parents(self, fitnesses:list[float]) -> tuple:
        parent1 = self.roulette_selection(fitnesses)[0]
        parent2 = self.roulette_selection(fitnesses)[0]
        while parent1 == parent2:  # Ensure distinct parents
            parent2 = self.roulette_selection(fitnesses)[0]
        return parent1, parent2


    def crossover(self, parent1, parent2) -> tuple:
        """
        Combines two parents to produce one or more offspring.
        """
        return ()


    def mutate(self, individual:SphericalEngine) -> SphericalEngine:
        biases = Mutator.modify_biases(individual._biases.copy())
        activations = Mutator.modify_activations(individual.activations.copy())
        adjacency_dict = Mutator.modify_weights({k: v.copy() for k, v in individual._adjacency_dict.items()})
        adjacency_dict, biases, activations = Mutator.add_node(adjacency_dict, biases, activations)
        adjacency_dict = Mutator.add_connection(adjacency_dict)
        adjacency_dict = Mutator.remove_connection(adjacency_dict)
        adjacency_dict, biases, activations = Mutator.remove_node(
            adjacency_dict, biases, activations, self._input_node_ids, self._output_node_ids
        )

        mutated_individual = SphericalEngine(
            adjacency_dict=adjacency_dict,
            activations=activations,
            biases=biases,
            output_node_ids=individual.output_node_ids,
            input_node_ids=individual.input_node_ids,
            stateful=individual._stateful,
            max_steps=individual._max_steps
        )

        return mutated_individual


    def replace_population(self, new_population:list[SphericalEngine], fitnesses:list[float]):
        """
        Replaces the old population with the new population.
        """

        combined = list(zip(new_population, fitnesses))
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
        sorted_population, _ = zip(*sorted_combined)

        elites = sorted_population[:self._elitism_count]
        final_population:list[SphericalEngine] = elites + sorted_population[self._elitism_count:self._population_size] #type: ignore
        self._population = final_population


    def roulette_selection(self, fitnesses:list[float]) -> list[SphericalEngine]:
        """
        Select individuals from the current population based on their fitnesses using roulette wheel selection.
        Returns the selected individuals.
        """
        selected_individuals = []

        total_fitness = sum(fitnesses)
        normalized_fitnesses = [f / total_fitness for f in fitnesses]

        for _ in range(self._population_size):
            pick = random.uniform(0, 1)
            current = float32(0.)
            for i, individual in enumerate(self._population):
                current += normalized_fitnesses[i]
                if current > pick:
                    selected_individuals.append(individual)
                    break

        return selected_individuals



    # def optimize(self, max_generations:int):
    #     """
    #     Main loop to run the genetic algorithm.
    #     """
    #     for generation in range(max_generations):

    #         # Calculate fitness of the current population
    #         fitness_values = self.compute_fitness()

    #         # This can be added for logging purposes:
    #         max_fitness = np.max(fitness_values)
    #         avg_fitness = np.mean(fitness_values)
    #         log.info(f"Generation {generation + 1}/{max_generations}: Max Fitness: {max_fitness:.2f}, Avg Fitness: {avg_fitness:.2f}")

    #         new_population = []
    #         while len(new_population) < self._population_size - self._elitism_count:  # Adjusted to keep space for elites
    #             mutant:SphericalEngine = random.choice(self._population)
    #             parent1, parent2 = self.select_parents()
    #             child1, child2 = self.crossover(parent1, parent2)
    #             self.mutate(child1)
    #             self.mutate(child2)
    #             new_population.extend([child1, child2])

    #         self.replace_population(new_population)


    def optimize(self, max_generations:int) -> SphericalEngine:
        best_individual:SphericalEngine = self._population[0]
        best_fitness = -np.inf
        for generation in range(max_generations):
            fitness_values = self.compute_fitness(self._population)
            max_fitness_gen = np.max(fitness_values)
            avg_fitness = np.mean(fitness_values)

            if max_fitness_gen > best_fitness:
                best_fitness = max_fitness_gen
                best_idx = np.argmax(fitness_values)
                best_individual = self._population[best_idx]

            log.info(f"Generation {generation + 1}/{max_generations}: Max Fitness: {max_fitness_gen:.2f}, Avg Fitness: {avg_fitness:.2f}")

            new_pop:list[SphericalEngine] = self.roulette_selection(fitness_values)

            for idx, individual in enumerate(new_pop):
                mutated_individual = self.mutate(individual)
                new_pop[idx] = mutated_individual

            total_population = self._population + new_pop
            total_fitness = fitness_values + self.compute_fitness(new_pop)
            sorted_indices = np.argsort(total_fitness)[::-1]
            self._population = [total_population[i] for i in sorted_indices[:self._population_size]]

        log.info(f"Optimization finished. Best fitness: {best_fitness:.2f}")
        return best_individual
