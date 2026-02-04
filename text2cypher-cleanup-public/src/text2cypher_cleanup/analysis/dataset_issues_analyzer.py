import pandas as pd
from tqdm import tqdm
from analysis.issues_detector import Detector
from database.neo4j_demo_db import (
    Neo4JDemoDatabases,
    Neo4jConnectorSingleton,
)
from utils.constants import DATABASE_REFERENCE_ALIAS, ISSUES_COLUMN_NAME
from utils.logger import logger_factory


class DatasetIssueAnalyzer:
    """Analyzes entire datasets for issues."""

    LOGGER = logger_factory(__name__)

    @staticmethod
    def add_issue(
        dataset_df: pd.DataFrame,
    ):
        """Add issues to the issues column.
        Args:
            dataset_df: The DataFrame to analyze
            db_alias_2_neo4j_connector: Dict of database aliases to Neo4j connectors
        Returns:
            DataFrame with issues column populated
        """
        detector = Detector()
        # Process the dataset
        for index, row in tqdm(
            dataset_df.iterrows(),
            total=len(dataset_df),
            desc=f"Dectector is processing",
        ):
            db_alias = row[DATABASE_REFERENCE_ALIAS]
            if db_alias != None:
                neo4j_connector = Neo4JDemoDatabases.convert_db_alias_to_neo4jconnector(
                    db_alias
                )
            else:
                neo4j_connector = Neo4jConnectorSingleton.instance()
            # Detect issues using the detector
            detector.detect_issues(row, neo4j_connector, index, dataset_df)
        return dataset_df

    @staticmethod
    def get_issue_summary(dataset_df: pd.DataFrame) -> dict[str, int]:
        """Get summary statistics of detected issues."""
        logger = DatasetIssueAnalyzer.LOGGER
        issue_counts = {}

        for issues in dataset_df[ISSUES_COLUMN_NAME]:
            for issue in issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        logger.info(f"Issue counts: {issue_counts}")
        return issue_counts
