from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from flask import Flask, request
from transformers import AutoTokenizer
import transformers
import torch
import json

app = Flask(__name__)

# model = "codellama/CodeLlama-34b-hf"
# model = "WizardLM/WizardCoder-15B-V1.0"
# model = "codellama/CodeLlama-7b-hf"
model = "nomic-ai/gpt4all-j"

# tokenizer = tokenizer.to("cpu")

# device_map=device_map
# Ideally this could be loaded via accelerate https://huggingface.co/blog/accelerate-large-models
# But apple silicon has issues
# tokenizer = AutoTokenizer.from_pretrained(
#     model,
#     offload_folder="offload",
#     offload_state_dict=True,
#     device_map="auto",
# )

# torch.backends.mps.set_default_dtype(torch.float32)


tokenizer = AutoTokenizer.from_pretrained(model)
# tokenizer = AutoTokenizer.from_pretrained('../model/WizardCoder-15B-V1.0', local_files_only=True)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    # May need to change this depending on whether you have GPU support or not
    # torch_dtype=torch.float16,
    torch_dtype=torch.float32,
    # device_map="auto",
    device="cpu",
)
# pipeline = pipeline.to("cpu")

sequences = pipeline(
    "test",
    do_sample=True,
    top_k=10,
    temperature=0.1,
    top_p=0.95,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)


@app.route("/")
def hello_world():
    q = request.args.get("query")
    sequences = pipeline(
        q,
        do_sample=True,
        top_k=10,
        temperature=0.1,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )

    return f"<code>{json.dumps(sequences, indent=4)}</code>"
