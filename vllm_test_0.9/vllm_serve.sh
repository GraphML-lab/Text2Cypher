export CUDA_VISIBLE_DEVICES=1
vllm serve /data/home/yilin/projects/llm/deepseek-coder-33b-instruct --dtype auto --api-key abc --max-model-len 2048 --gpu-memory-utilization 0.96 --trust-remote-code