export LD_LIBRARY_PATH=/data/home/yilin/projects/fine-tuning/.venv/lib64/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
deepspeed lora_fine_tuning_with_ds.py