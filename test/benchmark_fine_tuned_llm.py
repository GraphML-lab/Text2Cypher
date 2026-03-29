import json
from CyVer import PropertiesValidator, SchemaValidator, SyntaxValidator
from neo4j import GraphDatabase
import neo4j
from openai import OpenAI
import time

with open(r"root_dir_you_like/projects/test/benchmark_dataset.json", "r") as f:
    test_dataset = json.load(f)

mapping_of_db_name_to_port = {
    "food_ingredients_allergens": "7687",
    "drugs_proteins_diseases": "7686",
    "movie_director_show_actor": "7685",
    "police_investigation_crime": "7683",
    "network_management": "7684",
}
client = OpenAI(
    base_url="http://localhost:8000/v1", api_key="abc", timeout=70.0, max_retries=2
)
is_seed = False
model_name: str = "your_model_name_goes_here"

if (
    model_name
    == "Seed-OSS-36B-Instruct-lora-tuned"
):
    is_seed = True
cyver_pass_counter = 0
correct_cypher_query_counter = 0
counter_for_debug = 0
for item in test_dataset:  # item is a dict
    print("-" * 10 + str(counter_for_debug) + "-" * 10)
    time.sleep(1)
    port = mapping_of_db_name_to_port[item["database"]]
    database_driver = GraphDatabase.driver(
        f"neo4j://localhost:{port}",
        auth=("neo4j", "password"),
        database="neo4j",
        connection_timeout=20.0,
        liveness_check_timeout=20.0,
    )

    if is_seed:
        completion = client.chat.completions.create(
            model=f"root_dir_you_like/projects/lora-fine-tuned-llm-mtq/{model_name}",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior and professional Cypher query generator that only outputs semantically correct and syntactically correct Cypher query based on given database schema and intent of user provided question.",
                },
                {
                    "role": "user",
                    "content": f"{item['question']}\n{item['schema']}",
                },
            ],
            extra_body={
                "chat_template_kwargs": {"thinking_budget": 0},
            },
        )
    else:
        completion = client.chat.completions.create(
            model=f"root_dir_you_like/projects/lora-fine-tuned-llm-mtq/{model_name}",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior and professional Cypher query generator that only outputs semantically correct and syntactically correct Cypher query based on given database schema and intent of user provided question.",
                },
                {
                    "role": "user",
                    "content": f"{item['question']}\n{item['schema']}",
                },
            ],
        )
    candidate_Cypher_query = completion.choices[0].message.content

    syntax_validator = SyntaxValidator(database_driver)
    schema_validator = SchemaValidator(database_driver)
    props_validator = PropertiesValidator(database_driver)

    schema_score, schema_metadata = schema_validator.validate(
        candidate_Cypher_query, "neo4j"
    )
    is_valid, syntax_metadata = syntax_validator.validate(
        candidate_Cypher_query, "neo4j"
    )
    props_score, properties_metadata = props_validator.validate(
        candidate_Cypher_query, "neo4j", strict=False
    )
    if props_score is None:
        props_score = 1

    if is_valid and int(props_score) == 1 and int(schema_score) == 1:
        cyver_pass_counter += 1
        with database_driver.session(default_access_mode=neo4j.READ_ACCESS) as session:
            try:
                results = [r.values() for r in session.run(neo4j.Query(candidate_Cypher_query, timeout=20.0))]
            except:
                counter_for_debug += 1
                continue
            else:
                if results == [r.values() for r in session.run(item["cypher"])]:
                    correct_cypher_query_counter += 1
    counter_for_debug += 1

record = f"Score on Benchmark for fine-tuned LLM ({model_name}): {correct_cypher_query_counter / len(test_dataset)} and the CyVer pass rate is:{cyver_pass_counter / len(test_dataset)}\n"
print(record)
with open(
    "root_dir_you_like/projects/test/results_of_experiments.txt", "a", encoding="utf-8"
) as f:
    f.write(record)

# it works
