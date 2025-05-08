# %%
# Implementation of DeepSeek Finetuning using Unsloth and Ollama
# Inspired from https://unsloth.ai/blog/deepseek-r1

# %%
!pip install unsloth

!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

# %%
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# %%
model = FastLanguageModel.get_peft_model(
    model,
    r = 4,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 42,
    use_rslora = False,
    loftq_config = None,
)

# %% [markdown]
# <a name="Data"></a>
# ### Data Prep

# %%
from datasets import load_dataset
dataset = load_dataset("vicgalle/alpaca-gpt4", split = "train")
print(dataset.column_names)

# %%
dataset[0]

# %%
from unsloth import to_sharegpt

dataset = to_sharegpt(
    dataset,
    merged_prompt = "{instruction}[[\nYour input is:\n{input}]]",
    output_column_name = "output",
    conversation_extension = 3, # Select more to handle longer conversations
)

# %%
from unsloth import standardize_sharegpt
dataset = standardize_sharegpt(dataset)

# %%
dataset[0]['conversations']

# %% [markdown]
# ### Customizable Chat Templates

# %%
chat_template = """Below are some instructions that describe some tasks. Write responses that appropriately complete each request.

### Instruction:
{INPUT}

### Response:
{OUTPUT}"""

from unsloth import apply_chat_template
dataset = apply_chat_template(
    dataset,
    tokenizer = tokenizer,
    chat_template = chat_template,
    # default_system_message = "You are a helpful assistant", << [OPTIONAL]
)

# %% [markdown]
# <a name="Train"></a>
# ### Train the model

# %%
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

# %%
trainer_stats = trainer.train()

# %% [markdown]
# <a name="Ollama"></a>
# ### Ollama

# %%
!curl -fsSL https://ollama.com/install.sh | sh

# %%
model.save_pretrained_gguf("model", tokenizer)

# %%
import subprocess
subprocess.Popen(["ollama", "serve"])
import time
time.sleep(3)

# %%
print(tokenizer._ollama_modelfile)

# %% [markdown]
# We now will create an `Ollama` model called `unsloth_model` using the `Modelfile` which we auto generated!

# %%
!ollama create deepseek_finetuned_model -f ./model/Modelfile

# %%
!pip install ollama

# %%
import ollama

response = ollama.chat(model="deepseek_finetuned_model",
            messages=[{ "role": "user", "content": "Continue the Fibonacci sequence: 1, 1, 2, 3, 5, 8,"
            },
                      ])

print(response.message.content)

# %%
from IPython.display import Markdown
import ollama

response = ollama.chat(model="deepseek_finetuned_model",
                       messages=[{"role": "user",
                                  "content": "How to add chart to a document?"},
                      ])

Markdown(response.message.content)


