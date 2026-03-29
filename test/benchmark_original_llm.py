import json
from CyVer import PropertiesValidator, SchemaValidator, SyntaxValidator
from neo4j import GraphDatabase
import neo4j
from openai import OpenAI

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
    base_url="http://localhost:8000/v1", api_key="abc", timeout=20.0, max_retries=1
)
is_seed = False
is_gpt = False
model_name: str = "granite-20b-code-instruct-8k"


if model_name == "Seed-OSS-36B-Instruct":
    is_seed = True
cyver_pass_counter = 0
correct_cypher_query_counter = 0
counter_for_debug = 0
for item in test_dataset:  # item is a dict
    print("-" * 10 + str(counter_for_debug) + "-" * 10)
    port = mapping_of_db_name_to_port[item["database"]]
    database_driver = GraphDatabase.driver(
        f"neo4j://localhost:{port}", auth=("neo4j", "password"), database="neo4j", connection_timeout = 10.0, liveness_check_timeout=10.0
    )

    if is_seed:
        completion = client.chat.completions.create(
            model="root_dir_you_like/projects/llm/Seed-OSS-36B-Instruct",
            messages=[
                {
                    "role": "system",
                    "content": "You are an experienced, details-oriented and professional Cypher query generator.\nMUST ONLY output correct Cypher query. All you output would be validated by Cypher Validation module.\nNO EXPLANATION. No NUMBERING. No EMOJI and No textual decoration! Just correct Cypher query.",
                },
                {
                    "role": "user",
                    "content": f"User-provided question:\n{item['question']}\nSchema:\n{item['schema']}",
                },
            ],
            extra_body={
                "chat_template_kwargs": {"thinking_budget": 0},
            }
        )
    else:
        completion = client.chat.completions.create(
            model=f"root_dir_you_like/projects/llm/{model_name}",
            messages=[
                {
                    "role": "system",
                    "content": "You are an experienced, details-oriented and professional Cypher query generator.\nMUST ONLY output correct Cypher query. All you output would be validated by Cypher Validation module.\nNO EXPLANATION. No NUMBERING. No EMOJI and No textual decoration! Just correct Cypher query.",
                },
                {
                    "role": "user",
                    "content": f"User-provided question:\n{item['question']}\nSchema:\n{item['schema']}",
                },
            ]
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
        cyver_pass_counter +=1
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

record = f"Score on Benchmark for original LLM ({model_name}): {correct_cypher_query_counter / len(test_dataset)} and the CyVer pass rate is:{cyver_pass_counter / len(test_dataset)}\n"
print(record)
with open(
    "root_dir_you_like/projects/test/results_of_experiments.txt", "a", encoding="utf-8"
) as f:
    f.write(record)

# it works
