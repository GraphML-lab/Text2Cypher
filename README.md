Github Repo to Thesis named "Think Beyond LLM: Can Agentic Workflow outperform Fine Tuned LLM in terms of Text2Cypher?" in GraphML Lab.

I was told that it's pretty likely that someone would try to reproduce my experiments. I think using AI to write the readme.md won't make my life that easy so i would write some hints.

First, read my thesis, if you could and would like to.

Second, to fine tune LLMs, use the scripts in "fine-tuning" folder and dependencies could be installed by running uv sync. Please be advised that you might have to choose a specific torch that works on your machine. Change the paths of datasets and then you are good to go. To fine tune LLMs using Mind the query dataset using lora_fine_tuning_with_ds_mtq.py, deploy Neo4j docker containers accordingly first, which could be found in Neo4j github repos. A little bit tedious.
If you run into some errors such as "current loss scale already in minimum" during fine tuning, simply rerun it because fine tuning is stochastic.
Fine tuning would sometimes raise error in the middle. Rerun it.
At the end of fine tuning, my code snippet would try to merge lora model (or lora adapter, if you find it easier to understand) and base model and save the combination as a standalone LLM. Sometimes this attempt would fail. Use merge_peft to merge them explicitly if it fails.

Third, enjoy troubleshooting vllm. The most difficult thing to vLLM is to download it. Most of the github issues of vLLM are related to installation issues. If you download my code from folder vllm_test_0.9 (confusing name 1) and uv sync, then notice that it doesn't work. Congrats! Enjoy troubleshooting vLLM!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Forth, to benchmark LLMs or agentic workflows, simply use the scripts in folder called "test" (confusing name 2). To install dependencies, this time, use uv add -r requirements.txt and use --frozen to oppress uv complaints.
(uv would complain about dependency inconsistency but it doesn't matter. The root cause is because of old langchain, which is needed. Ignore pyproject.toml and uv.lock for now.)
(Before benchmarking, deploy Neo4j docker containers then import data to these Neo4j databases using Neo4j data importers i made, which could be found through my thesis. Tedious.)
(vLLM sometimes halts in the middle. Take a break then rerun it maybe it won't halt in the middle.)

Fifth, benchmark dataset is in the folder "benchmark_dataset".

(Why not rename your folders if they are so confusing? Ans: The name of a folder is stored in the configuration file such as uv.lock and pyproject.toml, which are used to reproduce. Renaming folders would break reproducibility.)
