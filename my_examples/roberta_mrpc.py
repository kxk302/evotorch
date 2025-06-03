import os

import evaluate
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evotorch.algorithms import SNES
from evotorch.core import Problem, Solution
from evotorch.logging import PandasLogger, StdOutLogger

os.environ["CUDA_VISIBLE_DEVICES"] = '5'

minibatch_size = 32
population_size = 10
number_of_generations = 2
device = "cpu"
dtype = torch.float16
model_dir = "/opt/dlami/nvme/kaivan/evotorch/my_examples/roberta_mrpc_model"


# After the evolution completes, save the best solution to file
def save_best_solution(searcher, model_with_adapter):
    best_solution: Solution = searcher.status["best"].clone()

    # Flattened parameters from EvoTorch (1D tensor)
    param_vector = best_solution.values

    model = update_model(model_with_adapter, param_vector)

    torch.save(model.state_dict(), os.path.join(model_dir, "model_weights.pth"))


# Calculate and return the accuracy and F1
# measure of the model for the dataset
def get_accuracy_and_f1(model, dataloader, device):
    all_preds = []
    all_labels = []

    model.to(device)
    model.eval()
    for step, batch in enumerate(dataloader):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")  # or 'macro', 'micro', 'binary'

    return accuracy, f1


# Convert param_vector to model state_dict. Then call
# load_state_dict() on model and return the updated model
# This method updates only the model's the PEFT adapter
# weights based on the solution's parameter vector
def update_model(model, param_vector):
    new_state_dict = model.state_dict().copy()
    pointer = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            numel = param.numel()
            # print(f"name: {name}, numel : {numel}")

            # Extract and reshape
            new_param = param_vector[pointer : pointer + numel].view_as(param).to(param.dtype)
            new_state_dict[name] = new_param
            pointer += numel

    model.load_state_dict(new_state_dict)
    return model


def get_dataloader(split, model_name_or_path):
    print("Loading mrpc dataset")

    allowed = {"train", "test", "validatation"}
    if split not in allowed:
        raise ValueError(f"split must be one of {allowed}, got '{split}'")

    if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
        padding_side = "left"
    else:
        padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    datasets = load_dataset("glue", "mrpc")

    def tokenize_function(examples):
        # max_length=None => use the model max length (it's actually the default)
        outputs = tokenizer(
            examples["sentence1"], examples["sentence2"], truncation=True, max_length=128, padding="max_length"
        )
        return outputs

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )

    # We also rename the 'label' column to 'labels' which is the expected name for labels
    # by the models of the transformers library
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    def collate_fn(examples):
        return tokenizer.pad(examples, padding="longest", return_tensors="pt")

    print(f"Number of rows in training set: {len(tokenized_datasets['train'])}")
    print(f"Number of rows in training set: {len(tokenized_datasets['test'])}")
    print(f"Number of rows in validation set: {len(tokenized_datasets['validation'])}")

    num_minibatches = len(tokenized_datasets[split]) // minibatch_size

    print(f"Number of minibatches: {num_minibatches}")
    print(f"Minibatch size: {minibatch_size}")

    train_dataloader = DataLoader(
        tokenized_datasets[split], shuffle=True, collate_fn=collate_fn, batch_size=minibatch_size
    )

    return train_dataloader


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


class PeftModel(Problem):
    def __init__(self, solution_length, model_name_or_path, model_with_adapter, dtype, device):
        super().__init__(
            objective_sense="max",
            solution_length=solution_length,
            initial_bounds=(-1, 1),
            num_actors=10,
            num_gpus_per_actor=0.1,
            dtype=dtype,
            device=device,
            store_solution_stats=True,
        )

        self.model_name_or_path = model_name_or_path
        self.model_with_adapter = model_with_adapter
        self.train_dataloader = get_dataloader(split="train", model_name_or_path=model_name_or_path)

    def _evaluate(self, solution: Solution):
        param_vector = solution.values
        updated_model = update_model(self.model_with_adapter, param_vector)
        metric = evaluate.load("glue", "mrpc")
        accuracy, f1 = get_accuracy_and_f1(updated_model, self.train_dataloader, self.aux_device)
        solution.set_evals(accuracy)


def evolve_peft_model():
    model_name_or_path = "roberta-large"

    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
    peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
    model_with_adapter = get_peft_model(model, peft_config)
    number_of_trainable_params = sum(p.numel() for p in model_with_adapter.parameters() if p.requires_grad)

    problem = PeftModel(number_of_trainable_params, model_name_or_path, model_with_adapter, dtype=dtype, device=device)
    searcher = SNES(problem, popsize=population_size, stdev_init=5)
    stdout_logger = StdOutLogger(searcher, interval=1)
    pandas_logger = PandasLogger(searcher, interval=1)

    searcher.run(number_of_generations)

    # Save the best solution
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
    model_with_adapter = get_peft_model(model, peft_config)
    save_best_solution(searcher, model_with_adapter)

    # Reconstruct the model architecture
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
    model_with_adapter = get_peft_model(model, peft_config)
    model_with_adapter.load_state_dict(torch.load(os.path.join(model_dir, "model_weights.pth")))
    accuracy, f1 = get_accuracy_and_f1(model_with_adapter, problem.train_dataloader, device)
    print(f"Best Model -> Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    print("Visualizing the progress")
    pandas_logger.to_dataframe().mean_eval.plot()


if __name__ == "__main__":
    evolve_peft_model()
