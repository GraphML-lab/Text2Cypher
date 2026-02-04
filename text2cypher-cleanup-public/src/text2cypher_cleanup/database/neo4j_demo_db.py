from enum import Enum
from typing import LiteralString, Optional, Union, cast
import neo4j
import pandas as pd
from utils.constants import (
    FULL_SCHEMA_CYPHER_QUERY,
    NEO4JLABS_DEMO_URI,
    QUERY_RUN_EXCEPTION,
    SCHEMA,
)
from utils.logger import logger_factory


class DatabaseAliasEnum(str, Enum):
    # Database aliases used by the Neo4j Text2Cypher-2024v1 dataset
    NEO4JLABS_DEMO_DB_MOVIES = "neo4jlabs_demo_db_movies"
    NEO4JLABS_DEMO_DB_COMPANIES = "neo4jlabs_demo_db_companies"
    NEO4JLABS_DEMO_DB_NETWORK = "neo4jlabs_demo_db_network"
    NEO4JLABS_DEMO_DB_RECOMMENDATIONS = "neo4jlabs_demo_db_recommendations"
    NEO4JLABS_DEMO_DB_BLUESKY = "neo4jlabs_demo_db_bluesky"
    NEO4JLABS_DEMO_DB_BUZZOVERFLOW = "neo4jlabs_demo_db_buzzoverflow"
    NEO4JLABS_DEMO_DB_FINCEN = "neo4jlabs_demo_db_fincen"
    NEO4JLABS_DEMO_DB_GAMEOFTHRONES = "neo4jlabs_demo_db_gameofthrones"
    NEO4JLABS_DEMO_DB_GRANDSTACK = "neo4jlabs_demo_db_grandstack"
    NEO4JLABS_DEMO_DB_NEOFLIX = (
        "neo4jlabs_demo_db_eoflix"  # Note the typo in the dataset
    )
    NEO4JLABS_DEMO_DB_NORTHWIND = "neo4jlabs_demo_db_northwind"
    NEO4JLABS_DEMO_DB_OFFSHORELEAKS = "neo4jlabs_demo_db_offshoreleaks"
    NEO4JLABS_DEMO_DB_OPENSTREETMAP = "neo4jlabs_demo_db_openstreetmap"
    NEO4JLABS_DEMO_DB_STACKOVERFLOW2 = "neo4jlabs_demo_db_stackoverflow2"
    NEO4JLABS_DEMO_DB_TWITCH = "neo4jlabs_demo_db_twitch"
    NEO4JLABS_DEMO_DB_TWITTER = "neo4jlabs_demo_db_twitter"
    NEO4JLABS_DEMO_DB_STACKOVERFLOW = "neo4jlabs_demo_db_stackoverflow"


class Neo4jConnector:
    """Neo4j database connector for executing Cypher queries
    and managing connections."""

    def __init__(
        self,
        db_uri: str,
        db_username: str,
        db_password: str,
        db_name: str = "neo4j",
        neo4j_timeout_in_seconds: Optional[int] = None,
    ) -> None:
        # Initialize logger first
        self.logger = logger_factory(self.__class__.__name__)
        self.logger.debug(f"Initializing Neo4j connector for database: {db_name}")
        self.db_uri = db_uri
        self.db_username = db_username
        self.db_password = db_password
        self.db_name = db_name
        self.neo4j_timeout_in_seconds = neo4j_timeout_in_seconds
        self.driver = self.init_driver()
        self.logger.debug(
            f"Neo4j connector initialized successfully for database: {db_name}"
        )

    def init_driver(self) -> neo4j.Driver:
        self.logger.debug(f"Initializing Neo4j driver for database: {self.db_name}")
        driver = neo4j.GraphDatabase.driver(
            self.db_uri,
            auth=(self.db_username, self.db_password),
            database=self.db_name,
        )
        self.logger.debug(
            f"Neo4j driver initialized successfully for database: {self.db_name}"
        )
        return driver

    def execute_query_with_gql_objects(
        self, cypher_query: str, params: Optional[dict] = None, for_schema: bool = False
    ):
        if not for_schema:
            self.logger.debug(f"Executing single query on database: {self.db_name}")
            try:
                with self.driver.session(database=self.db_name) as session:
                    query = neo4j.Query(
                        cast(LiteralString, cypher_query),
                        timeout=self.neo4j_timeout_in_seconds,
                    )
                    intermediate_result = session.run(query=query, parameters=params)
                    output = intermediate_result.data()
                    notifications = [
                        obj.status_description
                        for obj in intermediate_result.consume().gql_status_objects
                    ]

            except Exception as e:
                self.logger.warning(f"Exception in execute_query: {e}")
                output = [{QUERY_RUN_EXCEPTION: type(e).__name__}]
                notifications = []
            else:
                self.logger.debug("Cypher query is executed successfully")
            finally:
                return output, notifications
        else:
            with self.driver.session(database=self.db_name) as session:
                query = neo4j.Query(
                    cast(LiteralString, cypher_query),
                    timeout=self.neo4j_timeout_in_seconds,
                )
                return session.run(query=query, parameters=params).data()[0][
                    "FullSchema"
                ]


class Neo4jConnectorSingleton:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = Neo4jConnector(
                db_uri="neo4j+s://demo.neo4jlabs.com",
                db_username="northwind",
                db_password="northwind",
                db_name="northwind",
            )
        return cls._instance


class Neo4JDemoDatabases:
    LOGGER = logger_factory(__name__)

    db_alias_2_schema = {}
    db_alias_enum_2_neo4jconnector = {}

    # Database alias to database name mapping
    DB_ALIAS_ENUM_TO_NAME = {
        DatabaseAliasEnum.NEO4JLABS_DEMO_DB_MOVIES: "movies",
        DatabaseAliasEnum.NEO4JLABS_DEMO_DB_COMPANIES: "companies",
        DatabaseAliasEnum.NEO4JLABS_DEMO_DB_NETWORK: "network",
        DatabaseAliasEnum.NEO4JLABS_DEMO_DB_RECOMMENDATIONS: "recommendations",
        DatabaseAliasEnum.NEO4JLABS_DEMO_DB_BLUESKY: "bluesky",
        DatabaseAliasEnum.NEO4JLABS_DEMO_DB_BUZZOVERFLOW: "buzzoverflow",
        DatabaseAliasEnum.NEO4JLABS_DEMO_DB_FINCEN: "fincen",
        DatabaseAliasEnum.NEO4JLABS_DEMO_DB_GAMEOFTHRONES: "gameofthrones",
        DatabaseAliasEnum.NEO4JLABS_DEMO_DB_GRANDSTACK: "grandstack",
        DatabaseAliasEnum.NEO4JLABS_DEMO_DB_NEOFLIX: "neoflix",
        DatabaseAliasEnum.NEO4JLABS_DEMO_DB_NORTHWIND: "northwind",
        DatabaseAliasEnum.NEO4JLABS_DEMO_DB_OFFSHORELEAKS: "offshoreleaks",
        DatabaseAliasEnum.NEO4JLABS_DEMO_DB_OPENSTREETMAP: "openstreetmap",
        DatabaseAliasEnum.NEO4JLABS_DEMO_DB_STACKOVERFLOW2: "stackoverflow2",
        DatabaseAliasEnum.NEO4JLABS_DEMO_DB_TWITCH: "twitch",
        DatabaseAliasEnum.NEO4JLABS_DEMO_DB_TWITTER: "twitter",
        DatabaseAliasEnum.NEO4JLABS_DEMO_DB_STACKOVERFLOW: "stackoverflow",
    }

    @staticmethod
    def _create_neo4j_connector(
        db_alias_enum: DatabaseAliasEnum,
        neo4j_timeout_in_seconds: Optional[int] = None,
    ) -> Neo4jConnector:
        neo4j_uri = NEO4JLABS_DEMO_URI
        db_name = Neo4JDemoDatabases.DB_ALIAS_ENUM_TO_NAME[db_alias_enum]
        # Demo databases use the same username and password as the database name
        neo4j_username = db_name
        neo4j_password = db_name

        # Create a logger for this static method context
        Neo4JDemoDatabases.LOGGER.info(
            f"Creating Neo4j connector for database: {db_name}"
        )

        neo4j_connector = Neo4jConnector(
            db_uri=neo4j_uri,
            db_username=neo4j_username,
            db_password=neo4j_password,
            db_name=db_name,
            neo4j_timeout_in_seconds=neo4j_timeout_in_seconds,
        )
        return neo4j_connector

    @staticmethod
    def populate_db_alias_enum_2_neo4j_connector(
        db_aliases: list[Union[DatabaseAliasEnum, str]],
        neo4j_timeout_in_seconds: Optional[int] = None,
    ):
        logger = Neo4JDemoDatabases.LOGGER

        # Convert string elements to DemoDatabaseAlias enums if needed
        db_alias_enums = [DatabaseAliasEnum(db_alias) for db_alias in db_aliases]

        logger.debug(
            f"Populating Neo4j connectors for {len(db_alias_enums)} database aliases"
        )
        for db_alias_enum in db_alias_enums:
            if db_alias_enum not in Neo4JDemoDatabases.db_alias_enum_2_neo4jconnector:
                Neo4JDemoDatabases.LOGGER.debug(
                    f"Creating connector for database alias: {db_alias_enum}"
                )
                neo4j_connector = Neo4JDemoDatabases._create_neo4j_connector(
                    db_alias_enum=db_alias_enum,
                    neo4j_timeout_in_seconds=neo4j_timeout_in_seconds,
                )
                Neo4JDemoDatabases.db_alias_enum_2_neo4jconnector[db_alias_enum] = (
                    neo4j_connector
                )

        logger.debug(
            f"Successfully created {len(Neo4JDemoDatabases.db_alias_enum_2_neo4jconnector)} Neo4j connectors"
        )

    @staticmethod
    def schema_update(db_alias: str, index, dataframe: pd.DataFrame):
        if db_alias not in Neo4JDemoDatabases.db_alias_2_schema:
            neo4j_connector = Neo4JDemoDatabases.convert_db_alias_to_neo4jconnector(
                db_alias=db_alias
            )
            updated_schema = f"{neo4j_connector.execute_query_with_gql_objects(FULL_SCHEMA_CYPHER_QUERY, for_schema=True)}"
            Neo4JDemoDatabases.db_alias_2_schema[db_alias] = updated_schema
        dataframe.loc[index, SCHEMA] = Neo4JDemoDatabases.db_alias_2_schema[db_alias]

    @staticmethod
    def convert_db_alias_to_neo4jconnector(db_alias: str) -> Neo4jConnector:
        return Neo4JDemoDatabases.db_alias_enum_2_neo4jconnector[
            DatabaseAliasEnum(db_alias)
        ]
