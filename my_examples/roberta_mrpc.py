import argparse
import math
import os

import evaluate
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, PeftType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evotorch.algorithms import SNES
from evotorch.core import Problem, Solution
from evotorch.logging import PandasLogger, StdOutLogger
from evotorch.operators import GaussianMutation, OnePointCrossOver

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

population_size = 5
number_of_generations = 500
number_of_actors = 1
dtype = torch.float32
batch_size = 32
model_name_or_path = "roberta-large"
dataset_name = "glue"  # General Language Understanding Evaluation (GLUE)
task_name = "mrpc"     # Microsoft Research Paraphrase Corpus (MRPC)
max_length = 128
peft_type = PeftType.LORA
device = "cuda:0"
# device = "cpu"
padding_side = "right"  # Right padding for an encoder model like RoBerta

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
    all_logits = []

    model_with_adapter.eval()
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model_with_adapter(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        all_logits.extend(outputs.logits.cpu())
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")  # or 'macro', 'micro', 'binary'

    # Instantiate BCELoss criterion
    ce_loss_criterion = nn.CrossEntropyLoss()

    # Calculate the loss. Lists must be converted to float tensors BCELoss
    all_labels_tensor = torch.tensor(all_labels).long()
    all_logits_stacked = torch.stack(all_logits)
    loss = ce_loss_criterion(all_logits_stacked, all_labels_tensor)

    print(f"loss: {loss}, accuracy: {accuracy}, f1: {f1}")
    return accuracy, f1, loss


# Make a copy of model_with_adapter's state_dict.
# Iterate through parameters that require training
# Update the values based on param_verctor
# Call load_state_dict() on model.
# This method updates only the model's the PEFT adapter
# weights based on the solution's parameter vector
def update_model(param_vector):
    new_state_dict = model_with_adapter.state_dict().copy()
    pointer = 0

    for name, param in model_with_adapter.named_parameters():
        if param.requires_grad:
            numel = param.numel()
            # print(f"name: {name}, numel : {numel}")

            # Extract and reshape
            new_param = param_vector[pointer : pointer + numel].view_as(param).to(param.dtype)
            new_state_dict[name] = new_param
            pointer += numel

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
    tokenized_datasets.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

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

    return train_dataloader, test_dataloader, eval_dataloader


def repeat_elements(input_tensor: torch.Tensor, num: int):
    """
    Repeats each element in input_tensor 'num' times.
    """
    return input_tensor.repeat_interleave(num)


class PeftModel(Problem):
    def __init__(self, number_of_trainable_params, random_seed, solution_length):
        super().__init__(
            objective_sense="max",
            solution_length=solution_length,
            initial_bounds=(-0.15, 0.15),
            num_actors=number_of_actors,
            num_gpus_per_actor=(1 / number_of_actors),
            dtype=dtype,
            device=device,
            store_solution_stats=True,
        )

        # Since the number_of_trainable_params could be very large, we limit the solution length
        # to solution_length, which is much smaller than number_of_trainable_params. We then repeat
        # each element in the solution such that we have a tensor of size number_of_trainable_params.
        # How many times we repeat each element is ceiling of (number_of_trainable_params/solution_length)
        self.random_seed = random_seed
        self.number_of_trainable_params = number_of_trainable_params
        self.number_of_repeats = math.ceil(number_of_trainable_params / solution_length)
        self.train_dataloader, self.test_dataloader, self.eval_dataloader = get_dataloader(self.random_seed)

    def _evaluate(self, solution: Solution):
        param_vector = solution.values.to(device)
        param_vector_expanded = repeat_elements(param_vector, self.number_of_repeats)[:self.number_of_trainable_params]
        update_model(param_vector_expanded)
        accuracy, f1, loss = get_accuracy_f1_and_loss(self.train_dataloader)
        solution.set_evals(-loss)


def evolve_peft_model(output_dir, random_seed, solution_length):

    number_of_trainable_params = sum(p.numel() for p in model_with_adapter.parameters() if p.requires_grad)
    print(f"number_of_trainable_params: {number_of_trainable_params}")

    problem = PeftModel(number_of_trainable_params, random_seed, solution_length)

    # Create the SNES searcher
    searcher = SNES(
        problem,
        popsize=population_size,        # population size
        # radius_init=0.15,
        stdev_init=0.005,             # initial exploration scale
        center_learning_rate=0.2,   # mean update step
        stdev_learning_rate=0.0014,    # std update step. 0.2 * (3 + log(n)) / sqrt(n) where n is the length of a solution
        optimizer="clipup",
        optimizer_config={"max_speed": 0.0002, "momentum": 0.9},
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
    model_with_adapter.load_state_dict(torch.load(os.path.join(output_dir, "model_weights.pth")))
    accuracy, f1, loss = get_accuracy_f1_and_loss(problem.train_dataloader)
    print(f"Best Model -> Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, loss : {loss:.4f}")

    print("Save Pandas logger dataframe")
    pandas_logger.to_dataframe().to_csv(os.path.join(output_dir, "pandas_logger.csv"), index=False)


if __name__ == "__main__":
    argumentParser = argparse.ArgumentParser()
    argumentParser.add_argument(
        "--output_dir", "-o", type=str, required=True, help="Directory to save the best model, logger file, etc."
    )
    argumentParser.add_argument("-s", "--random_seed", type=int, help="Random seed for bootstrapping the training dataset", required=True)
    argumentParser.add_argument("-l", "--solution_length", type=int, help="Length of a solution. Potentially less than the number of parameters being learned", required=True)

    args = argumentParser.parse_args()

    # Set the random seed for bootstrapping the training dataset so the process is repeatable
    # Also, makes the shuffling of the datasets repeatable
    torch.manual_seed(args.random_seed)

    evolve_peft_model(args.output_dir, args.random_seed, args.solution_length)
