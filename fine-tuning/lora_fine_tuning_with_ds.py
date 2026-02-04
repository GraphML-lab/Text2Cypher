from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
)
import statistics
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import evaluate
import numpy as np
import torch

exact_match = evaluate.load("exact_match")
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")


# Simple post-processing, like HF examples
def _pp(xs: list[str]):
    return [x.strip().lower() for x in xs]


# jaccard similarity calculation
def jac(i: str, j: str):
    set_i = set(i.lower().split())
    set_j = set(j.lower().split())
    return len(set_i.intersection(set_j)) / len(set_i.union(set_j))


def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def jaro_similarity(s1: str, s2: str) -> float:
    """
    Compute the Jaro similarity between two strings (value in [0, 1]).

    Reference formula:
      If m = number of matching characters, t = number of transpositions (half the number of
      matched characters that are in different order), then
      Jaro = 1/3 * (m/len(s1) + m/len(s2) + (m - t)/m)  if m > 0 else 0

    Matching window (match_distance) = max(len(s1), len(s2)) // 2 - 1, minimum 0.
    """
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    match_distance = max(len1, len2) // 2 - 1
    if match_distance < 0:
        match_distance = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    # Count matching characters.
    matches = 0
    for i in range(len1):
        start = max(0, i - match_distance)
        end = min(len2, i + match_distance + 1)
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    # Count transpositions.
    k = 0
    transpositions = 0
    for i in range(len1):
        if s1_matches[i]:
            # find next matched char in s2
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

    transpositions = transpositions / 2.0

    jaro = (
        (matches / len1) + (matches / len2) + ((matches - transpositions) / matches)
    ) / 3.0

    return jaro


def jaro_winkler(s1: str, s2: str, scaling: float = 0.1, prefix_max: int = 4) -> float:
    """
    Compute the Jaro–Winkler similarity between s1 and s2.

    scaling: the Winkler scaling factor (usually 0.1). Typical range [0, 0.25].
    prefix_max: max prefix length to use (usually 4).

    Returns value in [0, 1]. Higher means more similar.
    """
    jaro = jaro_similarity(s1, s2)
    if jaro == 0.0:
        return 0.0

    # length of common prefix up to prefix_max
    prefix_len = 0
    for a, b in zip(s1, s2):
        if a == b:
            prefix_len += 1
            if prefix_len >= prefix_max:
                break
        else:
            break

    jw = jaro + (prefix_len * scaling * (1.0 - jaro))
    # ensure rounding / numeric stability
    if jw > 1.0:
        jw = 1.0
    if jw < 0.0:
        jw = 0.0
    return jw


def compute_metrics(eval_pred):
    preds, labels = eval_pred

    # HF sometimes passes a tuple; if so, first element is usually token ids or logits
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 in labels so we can decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    # Decode
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = _pp(decoded_preds)
    decoded_labels = _pp(decoded_labels)

    # Metrics
    em_score = exact_match.compute(
        predictions=decoded_preds, references=decoded_labels, ignore_case=True
    )["exact_match"]

    rougeL_score = rouge.compute(
        predictions=decoded_preds,
        references=[[lab] for lab in decoded_labels],
        rouge_types=["rougeL"],
    )["rougeL"]

    bleu_score = bleu.compute(
        predictions=decoded_preds, references=[[lab] for lab in decoded_labels]
    )["bleu"]

    jaccard_score = statistics.mean(
        [jac(i, j) for i, j in zip(decoded_preds, decoded_labels)]
    )
    jaro_wink_score = statistics.mean(
        [jaro_winkler(i, j) for i, j in zip(decoded_preds, decoded_labels)]
    )

    cypher_quality = (
        0.2 * em_score
        + 0.2 * rougeL_score
        + 0.2 * bleu_score
        + 0.2 * jaccard_score
        + 0.2 * jaro_wink_score
    )

    return {"eval_cypher_quality": round(cypher_quality, 6)}


def format_prompt(example):
    return {
        "prompt": [
            {
                "role": "system",
                "content": "You are a senior and professional Cypher query generator that only outputs semantically correct and syntactically correct Cypher query based on given database schema and intent of user provided question.",
            },
            {"role": "user", "content": example["question"] + "\n" + example["schema"]},
        ],
        "completion": [{"role": "assistant", "content": example["cypher"]}],
    }


def start_train(llm_name, eval_dataset, train_dataset):
    global tokenizer
    path_model = f"path_to_a_folder/llm/{llm_name}"
    tokenizer = AutoTokenizer.from_pretrained(
        path_model, trust_remote_code=True, use_fast=True, padding_side="left"
    )

    output_dir_lora = f"path_to_a_folder/lora-models/{llm_name}-lora"
    merged_model_dir = (
        f"path_to_a_folder/lora-fine-tuned-llm/{llm_name}-lora-tuned"
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
        overwrite_output_dir=True,
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
    # load train.parquet (it's train split)
    train_dataset = load_dataset(
        "parquet",
        data_files="path_to_a_folder/fine-tuning/train.parquet",
        split="train",
    )
    # format it using format_prompt (if the model does have chat_template, it would raise errors)
    train_dataset = train_dataset.map(
        format_prompt,
        remove_columns=train_dataset.column_names,
    )

    eval_dataset = load_dataset(
        "parquet",
        data_files="path_to_a_folder/fine-tuning/eval.parquet",
        split="train",
    )

    eval_dataset = eval_dataset.map(
        format_prompt,
        remove_columns=eval_dataset.column_names,
    )
    start_train("your_model_name_goes_here", eval_dataset, train_dataset)
