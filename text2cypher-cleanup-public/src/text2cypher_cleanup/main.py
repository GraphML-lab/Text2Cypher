#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from datasets import Dataset
from typing import cast
from analysis.dataset_issues_analyzer import DatasetIssueAnalyzer
from database.neo4j_demo_db import Neo4JDemoDatabases
from utils.constants import DATABASE_REFERENCE_ALIAS, ISSUES_COLUMN_NAME
from datasets import load_dataset

if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    OUT_DIR = REPO_ROOT / "output"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load HuggingFace dataset and convert to pandas DataFrame
    split = input("Which split you want to clean? test or train?: ")
    text2cypher2024_dataset_test = load_dataset("neo4j/text2cypher-2024v1", split=split)
    assert isinstance(
        text2cypher2024_dataset_test, Dataset
    ), "Expected Dataset object when using split parameter"
    text2cypher2024_dataframe = cast(
        pd.DataFrame, text2cypher2024_dataset_test.to_pandas()
    )
    text2cypher2024_dataframe[ISSUES_COLUMN_NAME] = [
        [] for _ in range(len(text2cypher2024_dataframe))
    ]
    db_aliases = (
        text2cypher2024_dataframe[DATABASE_REFERENCE_ALIAS].dropna().unique().tolist()
    )
    Neo4JDemoDatabases.populate_db_alias_enum_2_neo4j_connector(
        db_aliases=db_aliases,
        neo4j_timeout_in_seconds=30,
    )

    print("Starting issue detection")
    print(f"Dataset has {len(text2cypher2024_dataframe)} instances to process")
    output_df = DatasetIssueAnalyzer.add_issue(
        dataset_df=text2cypher2024_dataframe,
    )
    print("Processing complete!")
    print(f"Final dataset shape: {output_df.shape}")
    # Get summary of detected issues
    issue_summary = DatasetIssueAnalyzer.get_issue_summary(output_df)
    print(f"Issue summary: {issue_summary}")

    output_dataframe_with_no_issue = output_df[
        output_df[ISSUES_COLUMN_NAME].apply(lambda x: len(x) == 0)
    ]
    output_dataframe_with_no_issue.drop("issues", axis=1, inplace=True)
    output_dataframe_with_no_issue.to_parquet(
        path=OUT_DIR / f"{split}_split_cleaned.parquet", index=False
    )
