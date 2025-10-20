import argparse
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import svd
from scipy.sparse import random
from scipy.sparse.linalg import svds
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from evotorch.algorithms import GeneticAlgorithm
from evotorch.core import Problem
from evotorch.core import Solution
from evotorch.logging import PandasLogger
from evotorch.logging import StdOutLogger
from evotorch.tools import ObjectArray
from evotorch.tools import as_tensor

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

population_size = 20
number_of_generations = 50
number_of_actors = 1
dtype = torch.float32
train_batch_size = 64
test_batch_size = 1000
# device = "cuda:0"
device = "cpu"

# -----------------------------
# Simple Neural Network
# -----------------------------

# In PyTorch, 'nn.Module' is the base class for all neural network models (and layers).
# Every custom model, or even built-in layers like nn.Linear, etc., inherits from nn.Module.

# When you create a model by subclassing nn.Module you
#    define layers (parameters) in __init__(), and
#    define computation (forward pass) in forward()

# By inheriting from nn.Module, your model automatically gets:
#     .parameters() → returns all weights for optimization.
#     .to(device) → move the whole model to GPU/CPU.
#     .eval() and .train() → switch between training and evaluation modes.
#     Easy saving/loading with torch.save and load_state_dict
# nn.Module supports nesting: you can compose modules inside other modules.
# E.g., nn.Sequential is itself a nn.Module that contains other modules.


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 256, bias=False)  # hidden layer with 256 neurons
        self.fc2 = nn.Linear(256, 10, bias=False)     # output layer for 10 classes

    def forward(self, x):
        # Go from (batch_size, 1, 28, 28)  to  (batch_size, 784)
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = SimpleNN().to(device)

# Loss function for classification problems.
# Input it expects:
#     Model outputs (logits): shape [batch_size, num_classes]
#         Example: for MNIST, [64, 10] if batch size = 64 and 10 digit classes.
#     Labels (targets): shape [batch_size]
#         containing class indices (0–9 for MNIST)
# What it computes: For each sample in the batch:
#    Applies softmax to logits to get predicted probability.
#    Computes -log(probability_of_correct_class).
#    Averages this over the batch to get one scalar loss.
# Lower loss = predictions closer to the true labels.
criterion = nn.CrossEntropyLoss()

# pylint: disable=too-many-positional-arguments, too-many-arguments
def generate_random_singular_vectors(
    n_rows: int,
    n_cols: int,
    num_singular_vals: int | None = None,
    density: float = 1.00,
    mean: float = 0.00,
    stddev: float = 1.00,
    seed: int | None = None,
):
    """
    Generate a (sparse) random rows x cols matrix and compute its SVD

    This function constructs a (sparse) random rows × cols matrix with entries drawn
    from a normal distribution with specific mean and standard deviation, and then
    computes its SVD.

    Parameters
    ----------
    n_rows : int
        The number of rows of the matrix.
    n_cols : int
        The number of columns of the matrix.
    num_singular_vals: int | None
        If int: compute sparse SVD. Must be >= 1 and smaller than min(n_rows, n_cols)
        If None, compute full SVD
    density: float
        The fraction of non-zero values in the matrix
    mean: float
        The mean of the normal distribution from which we sample the values for the matrix
    stddev: float
        The standard deviation of the normal distribution from which we sample the values for the matrix
    seed : int or None, optional
        Seed for the random number generator.
        - If an integer is provided, the results are reproducible.
        - If None (default), randomness varies each run.

    Returns dictionary with keys/values
    -------
    "U" :
        Left singular vectors of random matrix
    "S" :
        Singular values of random matrix
    "Vt": 
        Right singular vectors of random matrix

    Examples
    --------
    >>> U, S, VT, random_matrix = rotate_svd.generate_random_singular_vectors(5, 4, num_singular_vals=3)
    >>> U.shape
    (5, 3)
    >>> S
    array([1.7744256 , 1.86499659, 2.6570177 ])
    >>> S.shape
    (3,)
    >>> VT.shape
    (3, 4)
    >>> U, S, VT, random_matrix = rotate_svd.generate_random_singular_vectors(5, 4)
    >>> U.shape
    (5, 5)
    >>> S
    array([2.90737797, 2.35437921, 2.13745175, 0.36717192])
    >>> S.shape
    (4,)
    >>> VT.shape
    (4, 4)
    """

    # Return value dictionary
    svd_dict = {}

    if num_singular_vals is not None:
        if num_singular_vals < 1 or num_singular_vals >= min(n_rows, n_cols):
            raise ValueError(
                f"num_singular_vals {num_singular_vals} must be >= 1 \
                             and less than the smaller value of rows {n_rows} and cols {n_cols}"
            )

    # Creates a random number generator (RNG) using NumPy’s default_rng.
    # If you pass a seed (e.g., seed=42), the random numbers generated will be deterministic.
    # If seed=None, it will use system entropy, giving different results each run.
    rng = np.random.default_rng(seed)

    # Define custom random generator for nonzero values
    # flake8: noqa: E731
    # pylint: disable=unnecessary-lambda-assignment
    data_rvs = lambda n: rng.normal(loc=mean, scale=stddev, size=n)

    # Create size rows x cols matrix, with the specified density, where values are generated by data_rvs
    # "csr" stands for Compressed Sparse Row. It is a way to efficiently store sparse matrices (matrices with many
    # zeros). Instead of storing every element (including zeros), it stores only non-zero values along with metadata
    # to know where those values belong in the matrix
    a_random_matrix = random(
        n_rows,
        n_cols,
        density=density,
        format="csr",
        data_rvs=data_rvs,
        random_state=rng,
    )

    # Computes the Singular Value Decomposition of random_matrix
    if num_singular_vals is not None:
        left_singular_vectors, singular_values, right_singular_vectors_transposed = (
            svds(a_random_matrix, k=num_singular_vals)
        )
    else:
        left_singular_vectors, singular_values, right_singular_vectors_transposed = svd(
            a_random_matrix.toarray()
        )

    # return the left_singular_vectors, singular_values, and right_singular_vectors_transposed as a dictionary
    svd_dict["U"] = left_singular_vectors
    svd_dict["S"] = singular_values
    svd_dict["Vt"] = right_singular_vectors_transposed 
    return svd_dict


def rotate_singular_vector(
    singular_vector: np.ndarray, eps: float = 0.05, seed: int | None = None
) -> np.ndarray:
    """
    Return a small random rotation of the given orthonormal matrix.

    Orthonormal matrix is a square matrix where each column vector has a length of one
    and every distinct pair of column vectors is perpendicular to each other (their dot
    product is zero). A key property is that the inverse of an orthonormal matrix is equal
    to its transpose

    Parameters
    ----------
    singular_vector : (n, n) ndarray
        Orthonormal matrix (Q.T @ Q = Q @ Q.T = I).
    eps : float, default 0.05
        Size of the rotation (smaller -> closer to identity).
    seed : int | None
        RNG seed for reproducibility.

    Returns
    -------
    singular_vector_rot : (n, n) ndarray
        Orthonormal matrix obtained by applying a small rotation to Q.
    """
    n, m = singular_vector.shape
    if n != m:
        raise ValueError("singular_vector must be square (n x n).")

    # Optional: quick sanity check (tolerant to floating error)
    if not np.allclose(singular_vector.T @ singular_vector, np.eye(n), atol=1e-8):
        raise ValueError("Input singular_vector must be orthonormal (Q^T Q ≈ I).")

    rng = np.random.default_rng(seed)
    a_random_matrix = rng.standard_normal((n, n))

    # A skew-symmetric matrix A is a square matrix such that A.T = -A
    # Diagonal elements of A are 0, and for any element a_ij above the diagonal,
    # the corresponding element below the diagonal is its negative (a_ij = -a_ji).
    # Subtracting the transpose of a matrix from a matrix creates a skew-symmetric matrix.
    skew_symmetric = a_random_matrix - a_random_matrix.T  # skew-symmetric

    # A rotation in space is a linear transformation that preserves the origin, lengths, and angles.
    # If A is a skew-symmetric matrix, then the matrix exponential (e^A) produces an orthogonal
    # matrix (R) with determinant 1 (i.e., a rotation matrix).

    # The "plain" Cayley transformation is R = (I - K)(I + K)^-1
    # Input is a skew-symmetric matrix K, and output is an orthogonal matrix R
    # The "scaled" Cayley transformation is (I - (eps/2)K)^-1 (I + (eps/2) K)
    # Small values of eps produces small rotations
    identity = np.eye(n)
    first_term = identity - 0.5 * eps * skew_symmetric
    second_term = identity + 0.5 * eps * skew_symmetric

    # np.linalg.solve(M, N) is numerically better than explicitly
    # computing (I - (eps/2)K)^{-1} (I + (eps/2)K)
    rotation_matrix = np.linalg.solve(first_term, second_term)  # orthogonal and near I

    # Determinant of the rotation matrix should be 1.0
    det = np.linalg.det(rotation_matrix)
    print(f"det(rotation_matrix): {det}")

    # Apply the small rotation to the basis
    singular_vector_rot = singular_vector @ rotation_matrix

    # (Optional) tiny re-orthonormalization to clean numerical noise
    # Q_rot, _ = np.linalg.qr(Q_rot)

    return singular_vector_rot


def mutate_singular_values(
    singular_vals: np.ndarray,
    percentage: float,
    uniform_mutation_flag: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """
    Perturb the values of a singular values numpy array by adding or subtracting
    a percentage of each value to itself.

    Parameters
    ----------
    singular_vals : np.ndarray
        Input array.
    percentage : float
        Maximum percentage of each value to perturb (e.g., 0.1 = ±10%).
    uniform_mutation: bool
        If True, the same value will be added to (or subtracted from) each element in the array
        If False, the same percetnage will be added to (or subtracted from) each element in the array
    seed : int | None, optional
        Random seed for reproducibility. If None, randomness varies each run.

    Returns
    -------
    np.ndarray
        Perturbed array.
    """
    rng = np.random.default_rng(seed)

    if uniform_mutation_flag:
        factors = rng.uniform(low=-percentage, high=percentage, size=1)
    else:
        # Random factors between -percentage and +percentage
        factors = rng.uniform(
            low=-percentage, high=percentage, size=singular_vals.shape
        )

    # Perturbed array
    return singular_vals + singular_vals * factors


# After the evolution completes, save the best solution to file
def save_best_solution(searcher, output_dir, generation_counter, number_of_rows, number_of_columns, rank):
    print("Started save_best_solution")
    best_solution: Solution = searcher.status["best"].clone()

    U = best_solution.values["U"]
    S = best_solution.values["S"]
    Vt = best_solution.values["Vt"]

    # S is a 1D array of singular values, need to make it diagonal
    sigma = np.zeros((number_of_rows, number_of_columns))
    sigma[: len(S), : len(S)] = np.diag(S)

    # Re-construct rank k mattrix
    reconstructed_random_matrix = U[:,:rank] @ sigma[:rank, :rank] @ Vt[:rank,:]
    weights_matrix = torch.from_numpy(reconstructed_random_matrix).to(device)

    update_model(weights_matrix)

    torch.save(model.state_dict(), os.path.join(output_dir, "model_weights_" + str(generation_counter) + ".pth"))
    print("Finished save_best_solution")


# Calculate and return the accuracy and F1
# measure of the model for the dataset
def get_accuracy_f1_and_loss(dataloader):
    # print("Started get_accuracy_f1_and_loss")
    all_preds = []
    all_labels = []

    model.eval()
    for _, (images, labels) in enumerate(dataloader):
        images.to(device)
        labels.to(device)

        with torch.no_grad():
            outputs = model(images)
        predictions = outputs.argmax(dim=-1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")  # or 'macro', 'micro', 'binary'
    # Compute the loss (error) between predictions and true labels
    loss = criterion(outputs, labels)

    # print("Finished get_accuracy_f1_and_loss")
    return accuracy, f1, loss


# Make a copy of model_with_adapter's state_dict.
# Iterate through parameters that require training
# Update the values based on param_verctor
# Call load_state_dict() on model.
# This method updates only the model's the PEFT adapter
# weights based on the solution's parameter vector
def update_model(weights_matrix):
    new_state_dict = model.state_dict().copy()

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"name: {name}, numel : {param.numel()}, shape: {param.shape}")

            if "fc1.weight" in name:
                # Extract and reshape
                new_param = weights_matrix.view_as(param).to(param.dtype)
                new_state_dict[name] = new_param

    model.load_state_dict(new_state_dict)


def get_dataloader(random_seed):
    print("Started get_dataloader")

    # MNIST images are loaded as PIL images. transforms.ToTensor() converts
    # them into PyTorch tensors so they can be used by neural networks. After
    # conversion, pixel values are scaled from [0, 255] → [0.0, 1.0].
    transform = transforms.ToTensor()

    # Load the MNIST dataset. 'root' is the folder where the MNIST data will be
    # stored locally (e.g., ./root/MNIST). MNIST comes with two splits: training
    # set (60,000 images) and test set (10,000 images). 'transform' applies a
    # preprocessing transformation to every image. 'download=True' means if the
    # dataset isn’t already in the root folder, it will automatically download
    # it from the internet
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    # DataLoader wraps your train/test dataset into an iterator that you can loop over in training.
    # Each time you call it in a loop, it yields a batch of images and labels, randomly ordered for train.
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    return train_dataloader, test_dataloader


class EigenModel(Problem):
    """
    Each solution is a Python dict of the form:
    {
        "U": [...],   # Left singular vectors
        "S": [...],   # Singular values
        "Vt": [...],  # Right singular vectors
    }
    Where
        A = U @ np.diag(S) @ Vt
    And, possibly, a rank k reconstruction of A is
        Ak = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    Ak represents the weigths in a neural network layer
    """

    def __init__(
        self,
        number_of_rows,
        number_of_columns,
        rank,
        dist_mean,
        dist_stddev,
        random_seed,
    ):
        super().__init__(
            objective_sense="min",  # Goal is to minimize the model loss
            dtype=object,  # object solutions (uses ObjectArray internally)
        )
        self.number_of_rows = number_of_rows
        self.number_of_columns = number_of_columns
        self.rank = rank
        self.dist_mean = dist_mean
        self.dist_stddev = dist_stddev
        self.random_seed = random_seed
        self.train_dataloader, self.test_dataloader = get_dataloader(self.random_seed)

    # Generate initial object-shaped solutions (population seeding)
    def _fill(self, values: ObjectArray):
        print("Started _fill")
        population_size = len(values)

        values[:] = [
            generate_random_singular_vectors(self.number_of_rows, self.number_of_columns, None, 1.00, 0.00, 1.00, self.random_seed) 
            for _ in range(population_size)
        ]
        print("Finished _fill")

    # Evaluate one solution (you can also batch via _evaluate_batch)
    def _evaluate(self, solution: Solution):
        U = solution.values["U"]
        S = solution.values["S"]
        Vt = solution.values["Vt"]

        # S is a 1D array of singular values, need to make it diagonal
        sigma = np.zeros((self.number_of_rows, self.number_of_columns))
        sigma[: len(S), : len(S)] = np.diag(S)

        # Re-construct rank k mattrixn
        reconstructed_random_matrix = U[:,:self.rank] @ sigma[:self.rank, :self.rank] @ Vt[:self.rank,:]
        weights_matrix = torch.from_numpy(reconstructed_random_matrix).to(device)

        update_model(weights_matrix)
        accuracy, f1, loss = get_accuracy_f1_and_loss(self.train_dataloader)
        solution.set_evals(loss)
        print(f"loss: {loss}, accuracy: {accuracy}, f1: {f1}")


# Receives an ObjectArray of parent values and returns an ObjectArray of mutated offspring
def mutate(population: ObjectArray) -> ObjectArray:
    mutated_population = []

    # print(f"Population size: {len(values)}")
    for idx in range(len(population)):
        parent = population[idx]
        child = parent.clone()  # clone into a Python list of floats
        mutated_population.append(child)

    # return the children wrapped as ObjectArray so EvoTorch can handle them
    return as_tensor(mutated_population, dtype=object)


def evolve_peft_model(args: Dict):

    output_dir = args["output_dir"]
    number_of_rows = args["number_of_rows"]
    number_of_columns = args["number_of_columns"]
    rank = args["rank"]
    distribution_mean = args["distribution_mean"]
    distribution_stddev = args["distribution_stddev"]
    random_seed = args["random_seed"]

    problem = EigenModel(
        number_of_rows=number_of_rows,
        number_of_columns=number_of_columns,
        rank=rank,
        dist_mean=distribution_mean,
        dist_stddev=distribution_stddev,
        random_seed=random_seed,
    )

    searcher = GeneticAlgorithm(
        problem,
        operators=[mutate],  # our custom object-aware operator
        popsize=population_size,
        elitist=True,
    )

    _ = StdOutLogger(searcher, interval=1)
    pandas_logger = PandasLogger(searcher, interval=1)

    for idx in range(number_of_generations):
        print(f"************************************ Starting generation {(idx + 1)} ************************************")
        searcher.step()

        # Access the population's fitness values directly
        fitness_values = searcher.population.evals
        print(f"Fitness of all individuals in the population in generation {(idx + 1)}:")
        print(fitness_values)

        # Save the best solution
        save_best_solution(searcher, output_dir, idx, number_of_rows, number_of_columns, rank)
        print(f"************************************ Finishing generation {(idx + 1)} ************************************")

    # Reconstruct the model architecture
    """
    model_with_adapter.load_state_dict(torch.load(os.path.join(output_dir, "model_weights.pth")))
    accuracy, f1, loss = get_accuracy_f1_and_loss(problem.train_dataloader)
    print(f"Best Model -> Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, loss : {loss:.4f}")

    print("Save Pandas logger dataframe")
    pandas_logger.to_dataframe().to_csv(os.path.join(output_dir, "pandas_logger.csv"), index=False)
    """


if __name__ == "__main__":
    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument(
        "--output_dir", "-o", type=str, required=True, help="Directory to save the best model, logger file, etc."
    )
    argumentParser.add_argument(
        "--random_seed", "-s", type=int, required=True, help="Random seed for bootstrapping the training dataset"
    )
    argumentParser.add_argument(
        "--number_of_rows", "-r", type=int, required=True, help="Number of rows in the weight matrix to be learned"
    )
    argumentParser.add_argument(
        "--number_of_columns", "-c", type=int, required=True, help="Number of columns in the weight matrix to be learned"
    )
    argumentParser.add_argument(
        "--rank", "-k", type=int, required=True, help="Reconstruction rank for SVD of the weight matrix"
    )
    argumentParser.add_argument(
        "--distribution_mean", "-m", type=float, required=True, help="Mean of the distribution from which random weight matrixes are sampled from"
    )
    argumentParser.add_argument(
        "--distribution_stddev", "-d", type=float, required=True, help="Standard deviation of the distribution from which random weight matrixes are sampled from"
    )

    args = argumentParser.parse_args()

    # Set the random seed for bootstrapping the training dataset so the process is repeatable
    # Also, makes the shuffling of the datasets repeatable
    torch.manual_seed(args.random_seed)

    evolve_peft_model(vars(args))
