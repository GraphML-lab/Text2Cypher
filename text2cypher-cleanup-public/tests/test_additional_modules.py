import unittest
from unittest.mock import patch, MagicMock
import logging
from src.text2cypher_cleanup.database.neo4j_demo_db import (
    DatabaseAliasEnum,
    Neo4jConnector,
    Neo4jConnectorSingleton,
    Neo4JDemoDatabases,
)
from src.text2cypher_cleanup.utils import constants, logger


class TestNeo4jDemoDB(unittest.TestCase):
    @patch("src.text2cypher_cleanup.database.neo4j_demo_db.neo4j.GraphDatabase.driver")
    def test_connector_initialization(self, mock_driver):
        connector = Neo4jConnector(
            "neo4j+s://demo.neo4jlabs.com", "northwind", "northwind", "northwind"
        )
        self.assertIsNotNone(connector.driver)
        mock_driver.assert_called_once()

    @patch("src.text2cypher_cleanup.database.neo4j_demo_db.neo4j.GraphDatabase.driver")
    def test_execute_query_with_gql_objects(self, mock_driver):
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.data.return_value = [{"result": 1}]
        mock_result.consume.return_value.gql_status_objects = []
        mock_session.run.return_value = mock_result
        mock_driver.return_value.session.return_value.__enter__.return_value = (
            mock_session
        )

        connector = Neo4jConnector(
            "neo4j+s://demo.neo4jlabs.com", "northwind", "northwind", "northwind"
        )
        output, notifications = connector.execute_query_with_gql_objects(
            "MATCH (n) RETURN n"
        )
        self.assertIsInstance(output, list)
        self.assertIsInstance(notifications, list)

    def test_database_alias_enum_values(self):
        self.assertIn("neo4jlabs_demo_db_movies", [e.value for e in DatabaseAliasEnum])

    def test_singleton_instance(self):
        instance1 = Neo4jConnectorSingleton.instance()
        instance2 = Neo4jConnectorSingleton.instance()
        self.assertIs(instance1, instance2)

    @patch.object(
        Neo4JDemoDatabases, "_create_neo4j_connector", return_value=MagicMock()
    )
    def test_populate_db_alias_enum_2_neo4j_connector(self, mock_create):
        Neo4JDemoDatabases.db_alias_enum_2_neo4jconnector.clear()
        Neo4JDemoDatabases.populate_db_alias_enum_2_neo4j_connector(
            ["neo4jlabs_demo_db_movies"], neo4j_timeout_in_seconds=5
        )
        self.assertTrue(Neo4JDemoDatabases.db_alias_enum_2_neo4jconnector)


class TestConstants(unittest.TestCase):
    def test_constants_exist(self):
        self.assertEqual(constants.ISSUES_COLUMN_NAME, "issues")
        self.assertIn("CALL db.schema", constants.FULL_SCHEMA_CYPHER_QUERY)


class TestLogger(unittest.TestCase):
    def test_logger_factory(self):
        log = logger.logger_factory("TestClass")
        self.assertIsInstance(log, logging.Logger)
        self.assertEqual(log.name, "TestClass")

    def test_setup_logging(self):
        logger.setup_logging(log_level=logging.DEBUG)
        log = logging.getLogger("test")
        self.assertTrue(log)


if __name__ == "__main__":
    unittest.main()
