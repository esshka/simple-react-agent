from dotenv import load_dotenv
from typing import Dict, Any
from bson import ObjectId
import pymongo
import os
from simple_or_agent.simple_agent import ToolSpec

# --- Init ---

load_dotenv()
mongo_client: pymongo.MongoClient = pymongo.MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))

def _serialize_doc(doc: Any) -> Any:
    """Recursively convert ObjectId to string for JSON serialization."""
    if isinstance(doc, ObjectId):
        return str(doc)
    if isinstance(doc, list):
        return [_serialize_doc(item) for item in doc]
    if isinstance(doc, dict):
        return {key: _serialize_doc(value) for key, value in doc.items()}
    return doc

# --- CRUD Tools ---

def make_mongo_insert_one_tool() -> ToolSpec:
    """Inserts a single document into a collection."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            db = mongo_client[args["db_name"]]
            collection = db[args["collection_name"]]
            result = collection.insert_one(args["document"])
            return {"inserted_id": str(result.inserted_id)}
        except Exception as e:
            return {"error": f"mongo_insert_one failed: {e}"}

    return ToolSpec(
        name="mongo_insert_one",
        description="Inserts a single document into a specified MongoDB collection.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "db_name": {"type": "string", "description": "The name of the database."},
                "collection_name": {"type": "string", "description": "The name of the collection."},
                "document": {"type": "object", "description": "The JSON document to insert."},
            },
            "required": ["db_name", "collection_name", "document"],
        },
    )

def make_mongo_find_one_tool() -> ToolSpec:
    """Finds a single document in a collection."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            db = mongo_client[args["db_name"]]
            collection = db[args["collection_name"]]
            doc = collection.find_one(args.get("filter", {}))
            return _serialize_doc(doc) if doc else None
        except Exception as e:
            return {"error": f"mongo_find_one failed: {e}"}

    return ToolSpec(
        name="mongo_find_one",
        description="Finds a single document in a collection that matches the filter.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "db_name": {"type": "string", "description": "The name of the database."},
                "collection_name": {"type": "string", "description": "The name of the collection."},
                "filter": {"type": "object", "description": "A JSON object to filter the query."},
            },
            "required": ["db_name", "collection_name"],
        },
    )

def make_mongo_update_one_tool() -> ToolSpec:
    """Updates a single document in a collection."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            db = mongo_client[args["db_name"]]
            collection = db[args["collection_name"]]
            result = collection.update_one(args["filter"], args["update"])
            return {
                "matched_count": result.matched_count,
                "modified_count": result.modified_count,
            }
        except Exception as e:
            return {"error": f"mongo_update_one failed: {e}"}

    return ToolSpec(
        name="mongo_update_one",
        description="Updates a single document matching the filter in a collection.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "db_name": {"type": "string", "description": "The name of the database."},
                "collection_name": {"type": "string", "description": "The name of the collection."},
                "filter": {"type": "object", "description": "A JSON object to select the document to update."},
                "update": {"type": "object", "description": "A JSON object specifying the update operations (e.g., using $set)."},
            },
            "required": ["db_name", "collection_name", "filter", "update"],
        },
    )
    
def make_mongo_delete_one_tool() -> ToolSpec:
    """Deletes a single document from a collection."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            db = mongo_client[args["db_name"]]
            collection = db[args["collection_name"]]
            result = collection.delete_one(args["filter"])
            return {"deleted_count": result.deleted_count}
        except Exception as e:
            return {"error": f"mongo_delete_one failed: {e}"}

    return ToolSpec(
        name="mongo_delete_one",
        description="Deletes a single document matching the filter from a collection.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "db_name": {"type": "string", "description": "The name of the database."},
                "collection_name": {"type": "string", "description": "The name of the collection."},
                "filter": {"type": "object", "description": "A JSON object to select the document to delete."},
            },
            "required": ["db_name", "collection_name", "filter"],
        },
    )

# --- Advanced Data Tools ---

def make_mongo_find_tool() -> ToolSpec:
    """Finds multiple documents in a collection."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            db = mongo_client[args["db_name"]]
            collection = db[args["collection_name"]]
            limit = args.get("limit", 25) # Default limit to prevent huge outputs
            cursor = collection.find(args.get("filter", {})).limit(limit)
            return [_serialize_doc(doc) for doc in cursor]
        except Exception as e:
            return {"error": f"mongo_find failed: {e}"}

    return ToolSpec(
        name="mongo_find",
        description="Finds multiple documents in a collection that match the filter. Returns up to 25 documents by default.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "db_name": {"type": "string", "description": "The name of the database."},
                "collection_name": {"type": "string", "description": "The name of the collection."},
                "filter": {"type": "object", "description": "A JSON object to filter the query."},
                "limit": {"type": "integer", "description": "The maximum number of documents to return."},
            },
            "required": ["db_name", "collection_name"],
        },
    )

def make_mongo_aggregate_tool() -> ToolSpec:
    """Runs a MongoDB aggregation pipeline."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            db = mongo_client[args["db_name"]]
            collection = db[args["collection_name"]]
            pipeline = args["pipeline"]
            cursor = collection.aggregate(pipeline)
            return [_serialize_doc(doc) for doc in cursor]
        except Exception as e:
            return {"error": f"mongo_aggregate failed: {e}"}

    return ToolSpec(
        name="mongo_aggregate",
        description="Performs complex data aggregation using a pipeline of stages.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "db_name": {"type": "string", "description": "The name of the database."},
                "collection_name": {"type": "string", "description": "The name of the collection."},
                "pipeline": {"type": "array", "description": "A list of JSON objects representing the aggregation stages."},
            },
            "required": ["db_name", "collection_name", "pipeline"],
        },
    )

def make_mongo_count_documents_tool() -> ToolSpec:
    """Counts documents matching a filter."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            db = mongo_client[args["db_name"]]
            collection = db[args["collection_name"]]
            count = collection.count_documents(args.get("filter", {}))
            return {"count": count}
        except Exception as e:
            return {"error": f"mongo_count_documents failed: {e}"}

    return ToolSpec(
        name="mongo_count_documents",
        description="Counts the number of documents in a collection that match the given filter.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "db_name": {"type": "string", "description": "The name of the database."},
                "collection_name": {"type": "string", "description": "The name of the collection."},
                "filter": {"type": "object", "description": "A JSON object to filter the documents to be counted."},
            },
            "required": ["db_name", "collection_name"],
        },
    )

# --- Database and Collection Management Tools ---

def make_mongo_list_databases_tool() -> ToolSpec:
    """Lists all database names on the server."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            return {"databases": mongo_client.list_database_names()}
        except Exception as e:
            return {"error": f"mongo_list_databases failed: {e}"}

    return ToolSpec(
        name="mongo_list_databases",
        description="Lists the names of all databases on the MongoDB server.",
        handler=handler,
        parameters={"type": "object", "properties": {}},
    )

def make_mongo_list_collections_tool() -> ToolSpec:
    """Lists all collection names in a database."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            db = mongo_client[args["db_name"]]
            return {"collections": db.list_collection_names()}
        except Exception as e:
            return {"error": f"mongo_list_collections failed: {e}"}

    return ToolSpec(
        name="mongo_list_collections",
        description="Lists the names of all collections within a specified database.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "db_name": {"type": "string", "description": "The name of the database to inspect."},
            },
            "required": ["db_name"],
        },
    )

def make_mongo_drop_collection_tool() -> ToolSpec:
    """Drops (deletes) an entire collection."""

    def handler(args: Dict[str, Any]) -> Any:
        try:
            db = mongo_client[args["db_name"]]
            db.drop_collection(args["collection_name"])
            return {"message": f"Collection '{args['collection_name']}' dropped successfully from database '{args['db_name']}'."}
        except Exception as e:
            return {"error": f"mongo_drop_collection failed: {e}"}

    return ToolSpec(
        name="mongo_drop_collection",
        description="Deletes an entire collection from a database. This action is irreversible.",
        handler=handler,
        parameters={
            "type": "object",
            "properties": {
                "db_name": {"type": "string", "description": "The name of the database."},
                "collection_name": {"type": "string", "description": "The name of the collection to drop."},
            },
            "required": ["db_name", "collection_name"],
        },
    )

__all__ = [
    "make_mongo_insert_one_tool",
    "make_mongo_find_one_tool",
    "make_mongo_update_one_tool",
    "make_mongo_delete_one_tool",
    "make_mongo_find_tool",
    "make_mongo_aggregate_tool",
    "make_mongo_count_documents_tool",
    "make_mongo_list_databases_tool",
    "make_mongo_list_collections_tool",
    "make_mongo_drop_collection_tool",
]