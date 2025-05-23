import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, PeftType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from evotorch.algorithms import SNES
from evotorch.core import Solution
from evotorch.logging import PandasLogger, StdOutLogger
from evotorch.neuroevolution import SupervisedTransformerNE
from evotorch.neuroevolution.net import count_parameters

number_of_generations = 200
population_size = 12
minibatch_size = 32
radius_init = 0.1
# subbatch_size = population_size // 2
searcher = None
mrpc_training_dataset = None
train_dataloader = None
device = "cpu"
model_name_or_path = "roberta-large"
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True)
peft_config = LoraConfig(task_type="SEQ_CLS", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
model_with_adapter = get_peft_model(model, peft_config)


# Print the accuracy and F1 measure for
# the best solution in each generation
def evaluate_best_solution_in_generation():
    pop_best_solution: Solution = searcher.status["pop_best"].clone()

    # Flattened parameters from EvoTorch (1D tensor)
    param_vector = pop_best_solution.values

    model = vector_to_model(model_with_adapter, param_vector)

    accuracy, f1 = get_accuracy_and_f1(model)

    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")


# Convert param_vector to model state_dict. Then call
# load_state_dict() on model and return the model
def vector_to_model(model, param_vector):
    state_dict = model.state_dict()
    new_state_dict = {}
    pointer = 0

    for name, param in state_dict.items():
        numel = param.numel()
        # Extract and reshape
        new_param = param_vector[pointer : pointer + numel].view_as(param).to(param.dtype)
        new_state_dict[name] = new_param
        pointer += numel

    model.load_state_dict(new_state_dict)
    return model


# Calculate and return the accuracy and F1
# measure of the model for the dataset
def get_accuracy_and_f1(model):
    all_preds = []
    all_labels = []

    model.to(device)
    model.eval()
    for step, batch in enumerate(train_dataloader):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")  # or 'macro', 'micro', 'binary'

    return accuracy, f1


# 1. Load mrpc dataset
print("Loading mrpc dataset")

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
print(f"Number of rows in validation set: {len(tokenized_datasets['validation'])}")

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=minibatch_size
)

# Create TensorDataset with masked data
mrpc_training_dataset = TensorDataset(
    tokenized_datasets["train"]["input_ids"],
    tokenized_datasets["train"]["attention_mask"],
    tokenized_datasets["train"]["labels"],
)

mrpc_validation_dataset = TensorDataset(
    tokenized_datasets["validation"]["input_ids"],
    tokenized_datasets["validation"]["attention_mask"],
    tokenized_datasets["validation"]["labels"],
)

mrpc_test_dataset = TensorDataset(
    tokenized_datasets["test"]["input_ids"],
    tokenized_datasets["test"]["attention_mask"],
    tokenized_datasets["test"]["labels"],
)

# 2. Load a pretrained RobertaLarge model
print("Loading RobertaLarge model")
model_with_adapter.print_trainable_parameters()

# 3. Define loss and optimizer
criterion = nn.CrossEntropyLoss()

# 5. Define the problem
mrpc_problem = SupervisedTransformerNE(
    dataset=mrpc_training_dataset,  # Using the dataset specified earlier
    network=model,  # Training the RobertaLarge module loaded earlier
    loss_func=criterion,  # Minimizing CrossEntropyLoss
    minibatch_size=minibatch_size,  # With a minibatch size of 1024
    # common_minibatch = True,  # Always using the same minibatch across all solutions on an actor
    num_actors="max",  # The total number of CPUs used
    # num_gpus_per_actor = 'max',  # Dividing all available GPUs between the actors
    # subbatch_size = subbatch_size,  # Evaluating solutions in sub-batches of size 50 ensures we won't run out of GPU memory for individual workers
)

# 6. Define the searcher
searcher = SNES(
    mrpc_problem,
    popsize=population_size,
    radius_init=radius_init,  # Initial radius of the search distribution
)

searcher.after_step_hook.append(evaluate_best_solution_in_generation)

# 7. Define the loggers
stdout_logger = StdOutLogger(searcher, interval=1)
pandas_logger = PandasLogger(searcher, interval=1)

# 8. Run the evolution
print("Starting the evolution")
for idx in range(number_of_generations):
    searcher.step()

# 10. Visualize the progress
print("Visualizing the progress")
pandas_logger.to_dataframe().mean_eval.plot()
