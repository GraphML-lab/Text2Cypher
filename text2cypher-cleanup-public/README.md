# Text2Cypher-2024v1 Dataset Cleanup

This Python project was developed for cleaning a Neo4j-created [Text2Cypher-2024v1](https://huggingface.co/datasets/neo4j/text2cypher-2024v1) dataset, which is hosted in HuggingFace.  
This project automates the process of cleaning dataset splits using a local decoder-only Large Language Model (LLM). (Please prepare your own GPU resources and download LLM if you want to run this project on your own.)

Paper that introduces Text2Cypher-2024v1 dataset can be found [here](https://arxiv.org/abs/2412.10064).

Cleaned splits, which are further adjusted for training and evaluation, are available to download from the folder /data of this repository.

## Features

- **Singleton**: For entires with no database alias specified (in other words, no corresponding neo4j demo database), a pre-selected neo4j driver would handle these with limited functionality. A simple singleton pattern was implemented.
- **Structured LLM Output**: A project, [Outlines](https://github.com/dottxt-ai/outlines), provides the best approach of structured LLM output. Can't thank them enough.
- **Structure of Pipeline**: A giant for loop plus a function that looks into every single entry.

(I just realized I should have used CyVer when i was writing these few lines. But i don't think it could improve a lot after i read their docs carefully. A good open source project btw.)

## Prerequisites

- Python 3.11+
- Access to HuggingFace datasets & language models
- uv

## Installation & Usage

### 1. Clone and Set Up Environment

```bash
# Clone the repository
git clone https://github.com/Yacklin/text2cypher-cleanup-public.git
cd text2cypher-cleanup-public

# Install dependencies using uv
uv sync
```

### 2. Run the Cleanup Tool

Navigate to the main module directory and run the script:

```bash
cd src/text2cypher_cleanup
uv run main.py
```

You will be prompted to:
1. Enter the **dataset split** you want to clean (e.g., `train`, or `test`)
2. Provide the **path to your local LLM** (Using API is probably a bad idea. Even the cheapest API like DeepSeek could make you run out of money)

**After** processing, a parquet file will be saved in output folder (`output/`) (Github ignores empty folder so i can't upload empty 'output' folder to repo).

## Contributing

Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## Dataset Information

The cleaned neo4j/text2cypher-2024v1 dataset contains natural language questions, corresponding Cypher queries and relevant schema.

One of the authors has run this project to clean neo4j/text2cypher-2024v1 dataset using **Qwen/Qwen3-4B-Instruct-2507** and H100s.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Neo4j](https://neo4j.com/) and its scientists (notably, Makbule Ozsoy) for producing the original Text2Cypher-2024v1 dataset.
- Jiju Poovvancheri for providing me access to server of GPUs. Without his support, it would not be possible for me to do all these things.
