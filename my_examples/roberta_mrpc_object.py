import argparse
import os
import random

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig
from peft import PeftType
from peft import get_peft_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from evotorch.algorithms import GeneticAlgorithm
from evotorch.core import Problem
from evotorch.core import Solution
from evotorch.logging import PandasLogger
from evotorch.logging import StdOutLogger
from evotorch.operators import GaussianMutation
from evotorch.operators import OnePointCrossOver
from evotorch.tools import ObjectArray
from evotorch.tools import as_tensor

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

population_size = 10
number_of_generations = 5
number_of_actors = 1
dtype = torch.float32
batch_size = 32
model_name_or_path = "roberta-large"
dataset_name = "glue"  # General Language Understanding Evaluation (GLUE)
task_name = "mrpc"  # Microsoft Research Paraphrase Corpus (MRPC)
max_length = 128
peft_type = PeftType.LORA
device = "cuda:0"
# device = "cpu"
padding_side = "right"  # Right padding for an encoder model like RoBerta

lora_mean = 0.00
classifier_mean = 0.00
lora_stddev = 0.1
classifier_stddev = 0.2
individual_mutation_rate = 0.20
gene_mutation_rate = 0.005
noise_stddev = 0.05

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
model_with_adapter = get_peft_model(model, peft_config)
model_with_adapter.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# After the evolution completes, save the best solution to file
def save_best_solution(searcher, output_dir):
    best_solution: Solution = searcher.status["best"].clone()

    # Flattened parameters from EvoTorch (1D tensor)
    param_vector = best_solution.values

    update_model(param_vector)

    torch.save(model_with_adapter.state_dict(), os.path.join(output_dir, "model_weights.pth"))


# Calculate and return the accuracy and F1
# measure of the model for the dataset
def get_accuracy_f1_and_loss(dataloader):
    all_preds = []
    all_labels = []

    model_with_adapter.eval()
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model_with_adapter(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")  # or 'macro', 'micro', 'binary'

    return accuracy, f1, outputs.loss


# Make a copy of model_with_adapter's state_dict.
# Iterate through parameters that require training
# Update the values based on param_verctor
# Call load_state_dict() on model.
# This method updates only the model's the PEFT adapter
# weights based on the solution's parameter vector
def update_model(lora_weights, classifier_weights):
    new_state_dict = model_with_adapter.state_dict().copy()
    pointer_1 = 0
    pointer_2 = 0

    for name, param in model_with_adapter.named_parameters():
        if param.requires_grad:
            numel = param.numel()
            # print(f"name: {name}, numel : {numel}")

            if "lora_" in name:
                # Extract and reshape
                new_param = lora_weights[pointer_1 : pointer_1 + numel].view_as(param).to(param.dtype)
                new_state_dict[name] = new_param
                pointer_1 += numel
            if "classifier" in name:
                # Extract and reshape
                new_param = classifier_weights[pointer_2 : pointer_2 + numel].view_as(param).to(param.dtype)
                new_state_dict[name] = new_param
                pointer_2 += numel

    print("update_model ended")
    model_with_adapter.load_state_dict(new_state_dict)


def get_dataloader(random_seed):
    print("Loading mrpc dataset")

    # dataset features: ['sentence1', 'sentence2', 'label', 'idx']
    datasets = load_dataset(dataset_name, task_name)

    # Shuffle the dataset with a specific seed
    datasets = datasets.shuffle(seed=random_seed)

    # Concatenates 2 sentences with a special separator token: </s></s>
    # It also adds a start token <s> and an end token </s> to the overall sequence
    # If the length of the combined tokenized sequences (including the special tokens
    # like <s> at the start and </s> at the end and between sentences) is shorter than
    # the maximum length you've defined, the remaining space will be filled with the
    # padding token (<pad>) at the end of the sequence
    #
    # tokenizer.all_special_tokens: ['<s>', '</s>', '<unk>', '<pad>', '<mask>']
    # tokenizer.all_special_ids:    [0, 2, 3, 1, 50264]
    #
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["sentence1"],  # First sentence in the pair
            examples["sentence2"],  # Second sentence in the pair
            truncation=True,  # Truncates sequences longer than max_length
            max_length=max_length,  # Maximum sequence length
            padding="max_length",  # Pads all sequences to max_length, regardless of their actual length
            return_tensors="pt",  # Returns PyTorch tensors (vs. NumPy arrays or lists)
        )
        return outputs

    # Applies tokenize_function to each example in the dataset. Efficient and parallelizable — much better than a for loop
    # Processes multiple examples at once (as batches). Much faster than processing one-by-one.
    # Drops original input columns from the dataset after tokenization
    # tokenized dataset features: ['label', 'input_ids', 'attention_mask']
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )

    # We also rename the 'label' column to 'labels' which is the expected name
    # for labels by the models of the transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    # Converting your Hugging Face Dataset into a PyTorch-ready format
    # so it can be directly used with PyTorch models and DataLoaders
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    print(f"Number of rows in training set: {len(tokenized_datasets['train'])}")
    print(f"Number of rows in test set: {len(tokenized_datasets['test'])}")
    print(f"Number of rows in validation set: {len(tokenized_datasets['validation'])}")

    # Create a PyTorch DataLoaders for training/test/validation using Hugging Face tokenized datasets
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,  # Shuffles the data each epoch — improves generalization
        batch_size=batch_size,
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"],
        shuffle=False,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        shuffle=False,
        batch_size=batch_size,
    )

    print("get_dataloader ended")
    return train_dataloader, test_dataloader, eval_dataloader


def evaluate_model(model, dataloader, metric, device):
    model.to(device)
    model.eval()
    for step, batch in enumerate(dataloader):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    return eval_metric


def get_number_of_parameters(model):
    num_lora_params = 0
    num_classifier_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            numel = param.numel()
            if "lora_" in name:
                num_lora_params += numel
            if "classifier" in name:
                num_classifier_params += numel
    return num_lora_params, num_classifier_params


def generate_numbers(size, desired_mean, desired_std_dev):
    """
    Generates a list of numbers with a specified mean and standard deviation.

    Args:
        size (int): The number of elements in the list.
        desired_mean (float): The target mean for the list.
        desired_std_dev (float): The target standard deviation for the list.

    Returns:
        A list containing numbers with the specified mean and std dev.
    """
    # 1. Generate numbers from a standard normal distribution (mean=0, std=1)
    numbers = np.random.normal(loc=0, scale=1, size=size)

    # 2. Scale to desired standard deviation
    numbers_scaled = numbers * desired_std_dev

    # 3. Shift to desired mean
    numbers_scaled_and_shifted = numbers_scaled + desired_mean

    return numbers_scaled_and_shifted.tolist()


class PeftObjectModel(Problem):
    """
    Each solution is a Python dict of the form: {"lora": [...], "classifier": [...]}.
    Each vector represetns the weights/parameters for Lora adapter and the classifier
    in a Roberta model for fine tuning.
    """

    def __init__(
        self,
        lora_length,
        classifier_length,
        lora_mean,
        classifier_mean,
        lora_stddev,
        classifier_stddev,
        random_seed,
    ):
        super().__init__(
            objective_sense="min",  # Goal is to minimize the model loss
            dtype=object,  # object solutions (uses ObjectArray internally)
        )
        self.lora_length = lora_length
        self.classifier_length = classifier_length
        self.lora_mean = lora_mean
        self.classifier_mean = classifier_mean
        self.lora_stddev = lora_stddev
        self.classifier_stddev = classifier_stddev
        self.random_seed = random_seed
        self.train_dataloader, self.test_dataloader, self.eval_dataloader = get_dataloader(self.random_seed)

    # Generate initial object-shaped solutions (population seeding)
    def _fill(self, values: ObjectArray):
        population_size = len(values)

        values[:] = [
            {
                "lora": generate_numbers(self.lora_length, self.lora_mean, self.lora_stddev),
                "classifier": generate_numbers(self.classifier_length, self.classifier_mean, self.classifier_stddev),
            }
            for _ in range(population_size)
        ]
        print("_fill ended")

    # Evaluate one solution (you can also batch via _evaluate_batch)
    def _evaluate(self, solution: Solution):
        lora_weights = torch.tensor(list(solution.values["lora"]), dtype=dtype).to(device)
        classifier_weights = torch.tensor(list(solution.values["classifier"]), dtype=dtype).to(device)
        update_model(lora_weights, classifier_weights)
        accuracy, f1, loss = get_accuracy_f1_and_loss(self.train_dataloader)
        solution.set_evals(loss)
        print(f"loss: {loss}, accuracy: {accuracy}, f1: {f1}")


# Receives an ObjectArray of parent values and returns an ObjectArray of mutated offspring
def mutate(population: ObjectArray) -> ObjectArray:
    print("Mutate started")
    mutated_population = []

    # print(f"Population size: {len(values)}")
    for idx in range(len(population)):
        parent = population[idx]
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
            mutated_population.append(child)
            continue

        # print(f"Individual at idx {idx} SELECTED for mutation")

        # Pick a number of random indexes in the x weight vector to mutate
        # print(f"x vector BEFORE mutation: {child['x']}")
        len_x = len(child["lora"])
        number_of_bits_to_mutate = int(len_x * gene_mutation_rate)
        idx_for_genes_to_mutate = rng.choice(np.arange(0, len_x), size=number_of_bits_to_mutate, replace=False)
        # print(f"Indexes to mutate genes: {idx_for_genes_to_mutate}")

        for idx_1 in idx_for_genes_to_mutate:
            # Add Gaussian noise to that weight
            child["lora"][idx_1] += rng.normal(loc=0.0, scale=noise_stddev)
        # print(f"x vector AFTER mutation: {child['x']}")

        # Pick a number of random indexes in the y weight vector to mutate
        # print(f"y vector BEFORE mutation: {child['y']}")
        len_y = len(child["classifier"])
        number_of_bits_to_mutate = int(len_y * gene_mutation_rate)
        idx_for_genes_to_mutate = rng.choice(np.arange(0, len_y), size=number_of_bits_to_mutate, replace=False)
        # print(f"Indexes to mutate genes: {idx_for_genes_to_mutate}")

        for idx_2 in idx_for_genes_to_mutate:
            # Add Gaussian noise to that weight
            child["classifier"][idx_2] += rng.normal(loc=0.0, scale=noise_stddev)
        # print(f"y vector AFTER mutation: {child['y']}")

        mutated_population.append(child)

    # return the children wrapped as ObjectArray so EvoTorch can handle them
    print("Mutate ended")
    return as_tensor(mutated_population, dtype=object)


def evolve_peft_model(output_dir, random_seed):

    num_lora_params, num_classifier_params = get_number_of_parameters(model_with_adapter)
    print(f"Number of Lora parameters: {num_lora_params}")
    print(f"Number of classifier parameters: {num_classifier_params}")
    print(f"Total number of parameters: {num_lora_params + num_classifier_params}")

    problem = PeftObjectModel(
        lora_length=num_lora_params,
        classifier_length=num_classifier_params,
        lora_mean=lora_mean,
        classifier_mean=classifier_mean,
        lora_stddev=lora_stddev,
        classifier_stddev=classifier_stddev,
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
        print(f"Starting generation {(idx + 1)}")
        searcher.step()

        # Access the population's fitness values directly
        fitness_values = searcher.population.evals
        print(f"Fitness of all individuals in the population in generation {(idx + 1)}:")
        print(fitness_values)

        # Save the best solution
        save_best_solution(searcher, output_dir)
        print(f"Ending generation {(idx + 1)}")

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
        "-s", "--random_seed", type=int, help="Random seed for bootstrapping the training dataset", required=True
    )

    args = argumentParser.parse_args()

    # Set the random seed for bootstrapping the training dataset so the process is repeatable
    # Also, makes the shuffling of the datasets repeatable
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    global rng
    rng = np.random.default_rng(args.random_seed)

    evolve_peft_model(args.output_dir, args.random_seed)
