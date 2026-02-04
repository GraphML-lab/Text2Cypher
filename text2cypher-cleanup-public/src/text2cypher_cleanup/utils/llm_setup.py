import outlines
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import getpass
import os

if not os.environ.get("PATH_TO_LOCAL_LLM"):
    os.environ["PATH_TO_LOCAL_LLM"] = getpass.getpass(
        "Enter your absolute path to local LLM (otherwise this project won't work!): "
    )

model_path = os.getenv("PATH_TO_LOCAL_LLM")
tok = AutoTokenizer.from_pretrained(model_path)
mdl = AutoModelForCausalLM.from_pretrained(
    model_path, dtype=torch.float32, device_map="auto"
)

llm = outlines.from_transformers(mdl, tok)
