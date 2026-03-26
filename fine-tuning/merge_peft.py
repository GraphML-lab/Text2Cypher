from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

llm_name = "granite-20b-code-instruct-8k"
base_model = AutoModelForCausalLM.from_pretrained(
    f"/data/home/yilin/projects/llm/{llm_name}", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(f"/data/home/yilin/projects/llm/{llm_name}")
output_dir_lora = f"/data/home/yilin/projects/lora-models/{llm_name}-lora"
peft_model = PeftModel.from_pretrained(base_model, output_dir_lora)
merged_model_dir = (
    f"/data/home/yilin/projects/lora-fine-tuned-llm/{llm_name}-lora-tuned-second"
)
merged_model = peft_model.merge_and_unload(safe_merge=True)
merged_model.save_pretrained(merged_model_dir)
tokenizer.save_pretrained(merged_model_dir)
