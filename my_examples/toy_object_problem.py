import random

import numpy as np
import torch

from evotorch import Problem
from evotorch import Solution
from evotorch.algorithms import GeneticAlgorithm
from evotorch.tools import ObjectArray
from evotorch.tools import as_tensor

random_seed = 420
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
rng = np.random.default_rng(random_seed)
population_size = 10
individual_mutation_rate = 0.20
gene_mutation_rate = 0.20
num_generations = 20
solution_1_length = 20
solution_2_length = 10
solution_1_mean = 0.00
solution_2_mean = 0.00
solution_1_stddev = 0.5
solution_2_stddev = 0.2
noise_stddev = 0.1


def generate_numbers(size, desired_mean, desired_std_dev):
    """
    Generates a list of numbers with a specified mean and standard deviation.

    Args:
        size (int): The number of elements in the list.
        desired_mean (float): The target mean for the list.
        desired_std_dev (float): The target standard deviation for the list.

    Returns:
        numpy.ndarray: A NumPy array containing numbers with the specified mean and std dev.
    """
    # 1. Generate numbers from a standard normal distribution (mean=0, std=1)
    numbers = np.random.normal(loc=0, scale=1, size=size)

    # 2. Scale to desired standard deviation
    numbers_scaled = numbers * desired_std_dev

    # 3. Shift to desired mean
    numbers_scaled_and_shifted = numbers_scaled + desired_mean

    return numbers_scaled_and_shifted.tolist()


# --- Problem that works with arbitrary Python objects -------------------------
class ToyObjectProblem(Problem):
    """
    Each solution is a small Python dict, e.g. {"x": [1,2], "y": [3,4,5]}.
    Goal: maximize sum(x) - sum(y)
    """

    def __init__(
        self,
        solution_1_length,
        solution_2_length,
        solution_1_mean,
        solution_2_mean,
        solution_1_stddev,
        solution_2_stddev,
    ):
        super().__init__(
            objective_sense="max",
            dtype=object,  # <— key: object solutions (uses ObjectArray internally)
        )
        self.solution_1_length = solution_1_length
        self.solution_2_length = solution_2_length
        self.solution_1_mean = solution_1_mean
        self.solution_2_mean = solution_2_mean
        self.solution_1_stddev = solution_1_stddev
        self.solution_2_stddev = solution_2_stddev

    # Generate initial object-shaped solutions (population seeding)
    def _fill(self, values: ObjectArray):
        population_size = len(values)

        values[:] = [
            {
                "x": generate_numbers(self.solution_1_length, self.solution_1_mean, self.solution_1_stddev),
                "y": generate_numbers(self.solution_2_length, self.solution_2_mean, self.solution_2_stddev),
            }
            for _ in range(population_size)
        ]

    # Evaluate one solution (you can also batch via _evaluate_batch)
    def _evaluate(self, solution: Solution):
        x = solution.values["x"]  # arbitrary Python access
        y = solution.values["y"]  # arbitrary Python access
        reward = sum(x) - sum(y)
        solution.set_evaluation(reward)


# Receives an ObjectArray of parent values and returns an ObjectArray of mutated offspring
def mutate(values: ObjectArray) -> ObjectArray:
    out = []

    # print(f"Population size: {len(values)}")
    for idx in range(len(values)):
        parent = values[idx]
        child = parent.clone()  # clone into a Python list of floats

        # Generate a random number between 0.0 and 1.0
        # If it is larger than individual_mutation_rate, do NOT mutate the individual
        # If it is smaller than equal to individual_mutation_rate, mutate the individual

        # If not selected for mutation, add the unchanged individual to the list
        random_number = random.random()
        if random_number > individual_mutation_rate:
            # print(
            #    f"Individual at idx {idx} NOT selected for mutation. {random_number} is greater than {individual_mutation_rate}"
            # )
            out.append(child)
            continue

        # print(f"Individual at idx {idx} SELECTED for mutation")

        # Pick a number of random indexes in the x weight vector to mutate
        # print(f"x vector BEFORE mutation: {child['x']}")
        len_x = len(child["x"])
        number_of_bits_to_mutate = int(len_x * gene_mutation_rate)
        idx_for_genes_to_mutate = rng.choice(np.arange(0, len_x), size=number_of_bits_to_mutate, replace=False)
        # print(f"Indexes to mutate genes: {idx_for_genes_to_mutate}")

        for idx_1 in idx_for_genes_to_mutate:
            # Add Gaussian noise to that weight
            child["x"][idx_1] += rng.normal(loc=0.0, scale=noise_stddev)
        # print(f"x vector AFTER mutation: {child['x']}")

        # Pick a number of random indexes in the y weight vector to mutate
        # print(f"y vector BEFORE mutation: {child['y']}")
        len_y = len(child["y"])
        number_of_bits_to_mutate = int(len_y * gene_mutation_rate)
        idx_for_genes_to_mutate = rng.choice(np.arange(0, len_y), size=number_of_bits_to_mutate, replace=False)
        # print(f"Indexes to mutate genes: {idx_for_genes_to_mutate}")

        for idx_2 in idx_for_genes_to_mutate:
            # Add Gaussian noise to that weight
            child["y"][idx_2] += rng.normal(loc=0.0, scale=noise_stddev)
        # print(f"y vector AFTER mutation: {child['y']}")

        out.append(child)

    # return the children wrapped as ObjectArray so EvoTorch can handle them
    return as_tensor(out, dtype=object)


# --- Wire it up ---------------------------------------------------------------
problem = ToyObjectProblem(
    solution_1_length=solution_1_length,
    solution_2_length=solution_2_length,
    solution_1_mean=solution_1_mean,
    solution_2_mean=solution_2_mean,
    solution_1_stddev=solution_1_stddev,
    solution_2_stddev=solution_2_stddev,
)

searcher = GeneticAlgorithm(
    problem,
    operators=[mutate],  # our custom object-aware operator
    popsize=population_size,
    elitist=True,
)

from evotorch.logging import StdOutLogger

_ = StdOutLogger(searcher)

for idx in range(num_generations):
    searcher.step()

    # Access the population's fitness values directly
    fitness_values = searcher.population.evals
    # print(f"Fitness of all individuals in the population in generation {(idx + 1)}:")
    # print(fitness_values)

    if idx % 10 == 0:
        # print(f"Generation {(idx + 1)}")
        population_values = searcher.population.values
        # print(population_values.numpy())

print("Best:", searcher.status["best"].values, searcher.status["best"].evaluation)
