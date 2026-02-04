# Column names
ISSUES_COLUMN_NAME = "issues"
DATABASE_REFERENCE_ALIAS = "database_reference_alias"
CYPHER = "cypher"
INSTANCE_ID = "instance_id"
QUESTION = "question"
SCHEMA = "schema"

# Exceptions
QUERY_RUN_EXCEPTION = "query_run_exception"

# URI for neo4j demo databases
NEO4JLABS_DEMO_URI = "neo4j+s://demo.neo4jlabs.com"

FULL_SCHEMA_CYPHER_QUERY = r"""
CALL {
    CALL db.schema.nodeTypeProperties() YIELD nodeLabels, propertyName, propertyTypes
    WITH nodeLabels[0] AS label, propertyName + ": " + propertyTypes[0] AS prop
    WITH label, collect(prop) AS props
    RETURN "Nodes' properties and types of properties:\n" + apoc.text.join(collect(label + " {" + apoc.text.join(props, ", ") + "}"), "\n") AS nodeSchema
}
CALL {
    CALL db.schema.relTypeProperties() YIELD relType, propertyName, propertyTypes
    WITH relType, propertyName, propertyTypes 
    WHERE propertyName IS NOT NULL
    WITH relType, propertyName + ": " + propertyTypes[0] AS prop
    WITH relType, collect(prop) AS props
    // Clean up the relationship type string (removes ':' and backticks)
    WITH replace(replace(relType, ":", ""), "`", "") AS typeName, props
    WITH typeName + " {" + apoc.text.join(props, ", ") + "}" AS relDefinition
    RETURN "Relationships' properties and types of properties:\n" + apoc.text.join(collect(relDefinition), "\n") AS relPropSchema
}
CALL {
    MATCH (n)-[r]->(m)
    WITH DISTINCT labels(n)[0] AS s, type(r) AS t, labels(m)[0] AS e
    RETURN "The relationships:\n" + apoc.text.join(collect("(:" + s + ")-[:" + t + "]->(:" + e + ")"), "\n") AS patternSchema
}
WITH nodeSchema, patternSchema,relPropSchema + "\n\n" AS finalRelProps
RETURN nodeSchema + "\n\n" + finalRelProps + patternSchema AS FullSchema
"""
