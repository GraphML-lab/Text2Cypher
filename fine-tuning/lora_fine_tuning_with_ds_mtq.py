from CyVer import SyntaxValidator
import neo4j
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, EvalPrediction
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

import numpy as np
import torch
import re
from neo4j import GraphDatabase

# Simple post-processing, like HF examples
def _pp(xs: list[str]):
    return [x.strip().lower() for x in xs]

def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

table = {"bloom50": 7682, "er":7680, "wwc":7678, "healthcare":7679, "covid":7681}

def compute_metrics(eval_pred: EvalPrediction):
    global length_of_eval_dataset, table
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    inputs = eval_pred.inputs

    # HF sometimes passes a tuple; if so, first element is usually token ids or logits
    if isinstance(preds, tuple):
        preds = preds[0]
    if isinstance(inputs, tuple):
        inputs = inputs[0]
    
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    inputs = np.asarray(inputs)
    # Replace -100 in labels so we can decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
    # Decode
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

    decoded_preds = _pp(decoded_preds)
    decoded_labels = _pp(decoded_labels)
    decoded_inputs = _pp(decoded_inputs)

    flag = True
    score=0

    for decoded_pred, decoded_label, decoded_input in zip(decoded_preds, decoded_labels, decoded_inputs):
        decoded_pred = decoded_pred.replace("<think>", "").strip()
        decoded_pred = decoded_pred.replace("</think>", "").strip()
        decoded_label = decoded_label.replace("<think>", "").strip()
        decoded_label = decoded_label.replace("</think>", "").strip()
        
        database_targeted = re.search(r"##(.*?)##", decoded_input).group(1)
        if not database_targeted:
            database_targeted = re.findall(r"##(.*?)##", decoded_input)[-1]
        port = table[database_targeted]
        database_driver = GraphDatabase.driver(
            f"neo4j://localhost:{port}",
            auth=("neo4j", "password"),
            database="neo4j",
            connection_timeout=10.0,
            liveness_check_timeout=10.0,
        )
        syntax_validator = SyntaxValidator(database_driver)
        is_valid, syntax_metadata = syntax_validator.validate(decoded_pred, "neo4j")
        if is_valid:
            with database_driver.session(
                default_access_mode=neo4j.READ_ACCESS
            ) as session:
                results = [
                    r.values()
                    for r in session.run(
                        neo4j.Query(decoded_pred, timeout=20.0)
                    )
                ]
                if results == [r.values() for r in session.run(decoded_label)]:
                    score += 1

    cypher_quality = score/length_of_eval_dataset

    return {"eval_cypher_quality": round(cypher_quality, 6)}

def format_prompt(example):
    return {
        "prompt": [
            {
                "role": "system",
                "content": "You are a senior and professional Cypher query generator that only outputs semantically correct and syntactically correct Cypher query based on given database schema and intent of user provided question.",
            },
            {"role": "user", "content": example["question"] + "\n" + "##"+example["source_dataset"]+"##"+"\n"+example["schema"]},
        ],
        "completion": [{"role": "assistant", "content": example["cypher"]}],
    }


def start_train(llm_name, eval_dataset, train_dataset):
    global tokenizer
    path_model = f"root_dir_you_like/projects/llm/{llm_name}"
    tokenizer = AutoTokenizer.from_pretrained(
        path_model, trust_remote_code=True, use_fast=True, padding_side="left"
    )

    output_dir_lora = f"root_dir_you_like/projects/lora-models-mtq/{llm_name}-lora"
    merged_model_dir = (
        f"root_dir_you_like/projects/lora-fine-tuned-llm-mtq/{llm_name}-lora-tuned"
    )

    torch.cuda.empty_cache()
    ds_config_dict = {
        "fp16": {"enabled": True, "loss_scale": 0},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 1e6,
            "overlap_comm": False,
            "reduce_scatter": True,
            "reduce_bucket_size": 1e6,
            "contiguous_gradients": True,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "ignore_unused_parameters": True,
        },
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 8,
        "gradient_clipping": 1.0,
    }

    training_arguments = SFTConfig(
        completion_only_loss=True,
        seed=39,
        output_dir=output_dir_lora,
        do_eval=True,
        do_train=True,
        eval_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=2,
        gradient_checkpointing=True,
        save_strategy="steps",
        logging_strategy="no",
        eval_steps=80,
        greater_is_better=True,
        optim="adamw_torch",
        eval_on_start=True,
        load_best_model_at_end=True,
        dataloader_drop_last=True,
        save_steps=80,
        metric_for_best_model="eval_cypher_quality",
        eval_accumulation_steps=5,
        torch_empty_cache_steps=4,
        fp16_full_eval=True,
        deepspeed=ds_config_dict,
        fp16=True,
        include_for_metrics=["inputs"]
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        path_model, trust_remote_code=True
    )

    # A pretty general LoRAConfig
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        bias="none",
        target_modules="all-linear",
        lora_dropout=0.1,
        init_lora_weights=True,
        use_rslora=True,
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()
    trainer = SFTTrainer(
        model=lora_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        args=training_arguments,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()
    trainer.model.save_pretrained(output_dir_lora)

    peft_model = PeftModel.from_pretrained(base_model, output_dir_lora)

    merged_model = peft_model.merge_and_unload(safe_merge=True)
    merged_model.save_pretrained(merged_model_dir)
    tokenizer.save_pretrained(merged_model_dir)


if "__main__" == __name__:  # entry of program
    train_dataset = load_dataset(
        "json",
        data_files="root_dir_you_like/projects/fine-tuning/train_mtq.json",
        split = "train"
    )
    # format it using format_prompt (if the model does have chat_template, it would raise errors)
    train_dataset = train_dataset.map(
        format_prompt,
        remove_columns=train_dataset.column_names,
    )

    eval_dataset = load_dataset(
        "json",
        data_files="root_dir_you_like/projects/fine-tuning/eval_mtq.json",
        split="train"
    )

    eval_dataset = eval_dataset.map(
        format_prompt,
        remove_columns=eval_dataset.column_names,
    )
    length_of_eval_dataset = len(eval_dataset)
    
    start_train("your_model_goes_here", eval_dataset, train_dataset)
