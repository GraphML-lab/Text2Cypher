export CUDA_VISIBLE_DEVICES=1
vllm serve root_dir_you_like/projects/llm/deepseek-coder-33b-instruct --dtype auto --api-key abc --max-model-len 2048 --gpu-memory-utilization 0.96 --trust-remote-code