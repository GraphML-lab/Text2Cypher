import json
from typing import List
import neo4j
from openai import OpenAI
import Retrieve as r
from CyVer import SyntaxValidator, SchemaValidator, PropertiesValidator
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from neo4j import GraphDatabase


def structured_response(llm: str, messages, response_format):
    global openai_client

    extra_body = None
    if llm == "Seed-OSS-36B-Instruct":
        extra_body = {"chat_template_kwargs": {"thinking_budget": 0}}

    completion = openai_client.beta.chat.completions.parse(
        model=f"root_dir_you_like/projects/llm/{llm}",
        messages=messages,
        response_format=response_format,
        extra_body=extra_body,
    )
    return completion


def agentic_workflow(vllm_model: str, router_disabled: bool = True):
    global \
        timeout_bar, \
        openai_client, \
        syntax_validator, \
        schema_db, \
        schema_validator, \
        props_validator

    class llm_router_response_format(BaseModel):
        is_convertible: bool = Field(
            description="It would be True if user query could be converted to Cypher query. False otherwise."
        )
        explanation_if_non_convertible: str = Field(
            None, description="If user query is non-convertible, explain why briefly."
        )

    class State(TypedDict):
        design: List[str]
        user_input: str
        router_decision: bool
        output: str
        docs: str
        errors: str
        validation_decision: bool

    class PseudoCode(BaseModel):
        lines_of_pseudo_code: List[str] = Field(
            description="pseudo code line by line in natural language that details how Cypher query should be designed to achieve intent of user provided question"
        )

    class CypherQuery(BaseModel):
        cypher_clauses: List[str] = Field(
            description="Cypher query generated clause by clause. One string in the list corresponds to one clause of Cypher query. These clauses form the final Cypher query to return"
        )

    def pass_or_not(state: State) -> str:
        if state["validation_decision"]:
            return END
        else:
            return "cypher_generator"

    def route_decision(state: State) -> str:
        if state["router_decision"]:
            return "pseudo_code_generator"
        else:
            return END

    # Node
    def llm_router(state: State):
        if router_disabled:
            return {"router_decision": True}
        else:
            completion = structured_response(
                llm=vllm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You would decide whether user provided query could be theorectically converted to Cypher query based on its relevancy to given database schema contents ",
                    },
                    {
                        "role": "user",
                        "content": f"Given Schema: {schema_db}\nUser provided question: {state['user_input']}",
                    },
                ],
                response_format=llm_router_response_format,
            )
            result = completion.choices[0].message.parsed
            if result.is_convertible:
                return {"router_decision": True}
            else:
                return {
                    "output": result.explanation_if_non_convertible,
                    "router_decision": False,
                }

    # Node
    def pseudo_code_generator(state: State) -> dict[str, List[str]]:
        completion = structured_response(
            llm=vllm_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional Cypher query pseudo code generator.\nyou don't directly output Cypher query but the most feasible and step-by-step lines of pseudo code detailing how the best Cypher query should be designed to achieve intent of user provided question based on provided database schema.",
                },
                {
                    "role": "user",
                    "content": f"{state['user_input']}\nDatabase Schema:\n{schema_db}",
                },
            ],
            response_format=PseudoCode,
        )

        return {"design": completion.choices[0].message.parsed.lines_of_pseudo_code}

    def cyver(state: State):
        global timeout_bar

        if timeout_bar >= 4:
            return {"validation_decision": True, "output": "timeout"}
        candidate_Cypher = state["output"]
        if candidate_Cypher == "timeout":
            return {"validation_decision": True, "output": "timeout"}
        print("validation called")
        schema_score, schema_metadata = schema_validator.validate(
            candidate_Cypher, "neo4j"
        )
        is_valid, syntax_metadata = syntax_validator.validate(candidate_Cypher, "neo4j")
        props_score, properties_metadata = props_validator.validate(
            candidate_Cypher, "neo4j", strict=False
        )

        if props_score is None:
            props_score = 1

        if is_valid and int(props_score) == 1 and int(schema_score) == 1:
            return {"validation_decision": True, "errors": ""}
        else:
            timeout_bar += 1
            errors_list_dicts = syntax_metadata + schema_metadata + properties_metadata
            error_messages = "\n".join(
                [f"{i['code']}:{i['description']}" for i in errors_list_dicts]
            )
            return {
                "validation_decision": False,
                "errors": f"Previously generated Cypher query: {candidate_Cypher}\nError Messages: {error_messages}",
            }

    def retriever(state: State) -> dict[str, str]:
        lines_of_pseudo_code: List[str] = state["design"]
        retrieved_docs_combined = []
        for line in lines_of_pseudo_code:
            retrieved_docs_combined.extend(r.retrieve(line))
        deduplicated_retrieved_docs = "\n".join(list(set(retrieved_docs_combined)))

        return {"docs": deduplicated_retrieved_docs}

    # Node
    def cypher_generator(state: State) -> dict[str, str]:
        print("generation")
        try:
            completion = structured_response(
                llm=vllm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an experienced, details-oriented and professional Cypher query generator.\nMUST ONLY output correct Cypher query clause by clause. All you output would be validated by Cypher Validation module.\nNO EXPLANATION. No NUMBERING. No EMOJI and No textual decoration! Just correct Cypher query.\nYou are provided with relevant documents of Cypher query, database schema, lines of Cypher pseudo code that tells you how to write Cypher query, and errors messages provided by CyVer about your previous generated Cypher query (if any).",
                    },
                    {
                        "role": "user",
                        "content": f"{state['errors']}\nDocuments:{state['docs']}\nDatabase Schema:{schema_db}\nPseudo Code:{state['design']}",
                    },
                ],
                response_format=CypherQuery,
            )
        except:
            return {"output": "timeout"}
        else:
            return {
                "output": " ".join(completion.choices[0].message.parsed.cypher_clauses)
            }

    graph = StateGraph(State)
    graph.add_node("llm_router", llm_router)
    graph.add_node("pseudo_code_generator", pseudo_code_generator)
    graph.add_node("retriever", retriever)
    graph.add_node("cypher_generator", cypher_generator)
    graph.add_node("cyver", cyver)
    graph.add_edge(START, "llm_router")
    graph.add_conditional_edges("llm_router", route_decision)
    graph.add_edge("pseudo_code_generator", "retriever")
    graph.add_edge("retriever", "cypher_generator")
    graph.add_edge("cypher_generator", "cyver")
    graph.add_conditional_edges("cyver", pass_or_not)

    compiled_workflow = graph.compile()
    # png_bytes = agentic_workflow.get_graph().draw_mermaid_png()
    # img = Image.open(io.BytesIO(png_bytes))
    # plt.imshow(img)
    # plt.axis('off')
    # plt.show()

    return compiled_workflow


if __name__ == "__main__":
    correct_cypher_query_counter = 0
    timeout_bar: int = 0
    aw = None
    counter_for_debug = 0
    cyver_pass_counter = 0

    model_name = "your_model_name_goes_here"

    openai_client = OpenAI(
        base_url="http://localhost:8000/v1", api_key="abc", timeout=30.0, max_retries=1
    )

    with open(r"root_dir_you_like/projects/test/benchmark_dataset.json", "r") as f:
        test_dataset = json.load(f)

    mapping_of_db_name_to_port = {
        "food_ingredients_allergens": "7687",
        "drugs_proteins_diseases": "7686",
        "movie_director_show_actor": "7685",
        "police_investigation_crime": "7683",
        "network_management": "7684",
    }
    aw = agentic_workflow(model_name)
    for item in test_dataset:  # item is a dict
        timeout_bar = 0  # timeout_bar as a global var should be initialized to 0 before each iteration

        print("-" * 10 + str(counter_for_debug) + "-" * 10)  # for debugging

        port = mapping_of_db_name_to_port[item["database"]]
        database_driver = GraphDatabase.driver(
            f"neo4j://localhost:{port}",
            auth=("neo4j", "password"),
            database="neo4j",
            connection_timeout=10.0,
            liveness_check_timeout=10.0,
        )

        # schema_db = generate_schema(database_driver, full_schema_Cypher_query)
        schema_db = item["schema"]
        syntax_validator = SyntaxValidator(database_driver)
        schema_validator = SchemaValidator(database_driver)
        props_validator = PropertiesValidator(database_driver)
        try:
            state = aw.invoke({"user_input": item["question"], "errors": ""})
        except:
            counter_for_debug += 1
            continue
        else:
            candidate_Cypher_query = state["output"]
        if candidate_Cypher_query != "timeout":
            cyver_pass_counter += 1
            with database_driver.session(
                default_access_mode=neo4j.READ_ACCESS
            ) as session:
                try:
                    results = [
                        r.values()
                        for r in session.run(
                            neo4j.Query(candidate_Cypher_query, timeout=20.0)
                        )
                    ]
                except:
                    counter_for_debug += 1
                    continue
                else:
                    if results == [r.values() for r in session.run(item["cypher"])]:
                        correct_cypher_query_counter += 1
        counter_for_debug += 1

    record = f"Score on Benchmark for original LLM ({model_name}): {correct_cypher_query_counter / len(test_dataset)} wrapped in agentic workflow and the CyVer pass rate is:{cyver_pass_counter / len(test_dataset)}\n"
    print(record)
    with open(
        "root_dir_you_like/projects/test/results_of_experiments.txt",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(record)
