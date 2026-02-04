from enum import Enum
import re
from typing import Literal
import pandas as pd
from database.neo4j_demo_db import Neo4jConnector
from utils.constants import (
    CYPHER,
    DATABASE_REFERENCE_ALIAS,
    INSTANCE_ID,
    ISSUES_COLUMN_NAME,
    QUERY_RUN_EXCEPTION,
    QUESTION,
    SCHEMA,
)
from database.neo4j_demo_db import Neo4JDemoDatabases
from utils.llm_setup import llm
from utils.logger import logger_factory

logger = logger_factory(__name__)


# All types of issues that can be detected
class IssueType(str, Enum):
    DEPRECATION = "deprecation_contained_in_Cypher_query"
    EMPTY_RESULT = "empty_result"
    SYNTAX_ERROR = "syntax_error_of_Cypher_query"
    NON_ENGLISH = "non_english(or commonly used latin)_characters_contained_in_question"
    AMBIGUOUS_QUESTION = "ambiguous_question"

    # The following is experimental. Its quality was partially determined by quality of schema
    INACCURATE_QUERY = "the_query_is_not_what_the question_is_looking for"  # if the question is not ambiguous, see if the query reflects what the user question is looking for (semantics)


# Currently all the questions should be written in English with some combinations of latin characters that are commonly seen in english phrases
def only_contains_latin_characters_helper(row: pd.Series):
    pattern = re.compile(
        r"^[A-Za-z0-9"
        r"À-ÖØ-öø-ÿ"  # Latin-1 Supplement
        r"ÆæŒœ"  # extra FR ligatures
        r"ÑñÇç"  # ES/French chars (also covered by ranges, kept explicit)
        r"ÁÉÍÓÚáéíóú"  # ES accented vowels
        r"ÀÈÌÒÙàèìòù"  # IT/French grave accents
        r"ÂÊÎÔÛâêîôû"  # French circumflex
        r"ÜüŸÿ"  # umlauts
        r"¿¡"  # Spanish inverted punct.
        r'"\'“”‘’«»–—…'  # quotes, guillemets, en/em dash, ellipsis
        r"!#$%&()*+,\-./:;<=>?@\[\]\\\^_`{|}~"  # full ASCII punctuation set
        r"\s"  # whitespace (space/tab/newline etc.)
        r"]+$"
    )
    if not bool(pattern.match(row[QUESTION])):
        row[ISSUES_COLUMN_NAME].append(IssueType.NON_ENGLISH.value)


# Rows with database alias could be examined by complete functionality
def execution_for_row_with_alias_helper(
    row: pd.Series, neo4j_connector: Neo4jConnector
):
    global logger
    instance_id = row[INSTANCE_ID]
    cypher_query = row[CYPHER]
    issues_column = row[ISSUES_COLUMN_NAME]
    assert isinstance(issues_column, list)

    output, notifications = neo4j_connector.execute_query_with_gql_objects(cypher_query)

    if len(output) == 1 and QUERY_RUN_EXCEPTION in output[0]:
        logger.info(f"Syntax error detected for instance {instance_id}")
        issues_column.append(IssueType.SYNTAX_ERROR.value)
        return
    else:
        is_deprecated = False
        for notf in notifications:
            if (
                "warn: feature deprecated with replacement." in notf
                and not is_deprecated
            ):
                logger.info(f"Deprecation detected for {instance_id}")
                issues_column.append(IssueType.DEPRECATION.value)
                is_deprecated = True
            elif "note: no data" == notf:
                logger.info(f"Empty Result detected for {instance_id}")
                issues_column.append(IssueType.EMPTY_RESULT.value)


# It would do two things: if the question is ambiguous, tag it with ambiguous_question. If the question is not ambiguous, see if the Cypher query correctly represents the user question by using advanced LLM. This helper is experimental.
def semantics_issues_helper(row: pd.Series):
    global logger

    question = row[QUESTION]
    schema = row[SCHEMA]
    cypher_query = row[CYPHER]
    issues_column = row[ISSUES_COLUMN_NAME]
    assert isinstance(issues_column, list)
    decision_for_question = llm(
        f"determine if the given user question is vague or not: {question}",
        Literal["vague", "clear"],
    )

    if decision_for_question == "vague":
        issues_column.append(IssueType.AMBIGUOUS_QUESTION.value)
        return
    else:
        decision_for_correctness_of_cypher_query = llm(
            f"determine whether given Cypher query semantically reflects the intent of user question or not (schema would be provided but could be useless. you make your choice):\nuser question:\n{question}\nCypher query:\n{cypher_query}\nschema:{schema}",
            Literal["yes it reflects", "no it doesn't reflect"],
        )

        if decision_for_correctness_of_cypher_query == "no it doesn't reflect":
            issues_column.append(IssueType.INACCURATE_QUERY.value)
            return


# Rows with no database alias would be examined by limited functionality. That being said, only syntax could be checked.
def execution_for_row_with_no_alias_helper(
    row: pd.Series, neo4j_connector: Neo4jConnector
):
    cypher_query = row[CYPHER]
    issues_column = row[ISSUES_COLUMN_NAME]
    output, notifications = neo4j_connector.execute_query_with_gql_objects(
        cypher_query=f"EXPLAIN {cypher_query}"
    )
    if len(output) == 1 and QUERY_RUN_EXCEPTION in output[0]:
        issues_column.append(IssueType.SYNTAX_ERROR.value)
        return
    else:
        for notf in notifications:
            if "warn: feature deprecated with replacement" in notf:
                issues_column.append(IssueType.DEPRECATION.value)
                return


class Detector:
    def __init__(self):
        self.logger = logger
        super().__init__()

    def detect_issues(
        self,
        row: pd.Series,
        neo4j_connector: Neo4jConnector,
        index,
        dataframe: pd.DataFrame,
    ) -> None:
        instance_id = row[INSTANCE_ID]
        db_alias = row[DATABASE_REFERENCE_ALIAS]
        cypher_query = row[CYPHER]
        issues_column = row[ISSUES_COLUMN_NAME]
        assert isinstance(issues_column, list)

        """If there is no database alias for this row, this row can only be checked in terms of whether or not the question is vague, whether or not the question contains non-latin characters, syntax (execute the Cypher query against a database with 'EXPLAIN' prepended using a random neo4jconnector)
        
        If the database_reference_alias is null in parquet file, it would be none in dataframe.
        """
        if db_alias is None:
            self.logger.debug(f"db_alias is None for instance {instance_id}")

            only_contains_latin_characters_helper(row=row)
            execution_for_row_with_no_alias_helper(row, neo4j_connector=neo4j_connector)
            semantics_issues_helper(row=row)
            return
        else:
            # Updating schema within issues detector might sound a little bit confusing. But if non-standardized schema is treated as an issue, it somehow makes sense.
            Neo4JDemoDatabases.schema_update(db_alias, index, dataframe=dataframe)
            only_contains_latin_characters_helper(row)
            execution_for_row_with_alias_helper(row, neo4j_connector)
            semantics_issues_helper(row)
            return
