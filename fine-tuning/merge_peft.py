from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

llm_name = "your_model_name_goes_here"
base_model = AutoModelForCausalLM.from_pretrained(
    f"root_dir_you_like/projects/llm/{llm_name}", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(f"root_dir_you_like/projects/llm/{llm_name}")
output_dir_lora = f"root_dir_you_like/projects/lora-models/{llm_name}-lora"
peft_model = PeftModel.from_pretrained(base_model, output_dir_lora)
merged_model_dir = (
    f"root_dir_you_like/projects/lora-fine-tuned-llm/{llm_name}-lora-tuned-second"
)
merged_model = peft_model.merge_and_unload(safe_merge=True)
merged_model.save_pretrained(merged_model_dir)
tokenizer.save_pretrained(merged_model_dir)
