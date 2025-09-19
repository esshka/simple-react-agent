from bson import ObjectId
import unittest
import sys
import os

# Add the src directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from simple_or_agent.tools.mongo import (
    mongo_client,
    _serialize_doc,
    make_mongo_insert_one_tool,
    make_mongo_find_one_tool,
    make_mongo_update_one_tool,
    make_mongo_delete_one_tool,
    make_mongo_find_tool,
    make_mongo_aggregate_tool,
    make_mongo_count_documents_tool,
    make_mongo_list_databases_tool,
    make_mongo_list_collections_tool,
    make_mongo_drop_collection_tool
)


class TestMongoTools(unittest.TestCase):
    """Test suite for MongoDB tools."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_db_name = "test_mongo_tools_db"
        self.test_collection_name = "test_collection"
        self.test_doc = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com",
            "preferences": {"theme": "dark", "notifications": True}
        }
        self.test_filter = {"name": "John Doe"}
        self.test_update = {"$set": {"age": 31, "email": "john.updated@example.com"}}

    def tearDown(self):
        """Clean up after tests."""
        # Clean up test database if it exists
        try:
            if self.test_db_name in mongo_client.list_database_names():
                mongo_client.drop_database(self.test_db_name)
        except:
            pass

    def test_serialize_doc(self):
        """Test document serialization with ObjectId conversion."""
        # Test ObjectId conversion
        obj_id = ObjectId()
        doc = {"_id": obj_id, "name": "test"}
        serialized = _serialize_doc(doc)
        self.assertEqual(serialized["_id"], str(obj_id))
        self.assertEqual(serialized["name"], "test")

        # Test nested ObjectId
        nested_doc = {"user": {"_id": obj_id}, "items": [{"_id": obj_id}]}
        serialized_nested = _serialize_doc(nested_doc)
        self.assertEqual(serialized_nested["user"]["_id"], str(obj_id))
        self.assertEqual(serialized_nested["items"][0]["_id"], str(obj_id))

        # Test non-ObjectId values remain unchanged
        simple_doc = {"name": "test", "age": 25, "active": True}
        serialized_simple = _serialize_doc(simple_doc)
        self.assertEqual(serialized_simple, simple_doc)

    def test_mongo_insert_one_tool_success(self):
        """Test successful document insertion."""
        tool = make_mongo_insert_one_tool()
        args = {
            "db_name": self.test_db_name,
            "collection_name": self.test_collection_name,
            "document": self.test_doc
        }

        result = tool.handler(args)

        # Should return inserted_id as string
        self.assertIn("inserted_id", result)
        self.assertIsInstance(result["inserted_id"], str)

        # Verify document was actually inserted
        db = mongo_client[self.test_db_name]
        collection = db[self.test_collection_name]
        inserted_doc = collection.find_one({"name": "John Doe"})
        self.assertIsNotNone(inserted_doc)
        self.assertEqual(inserted_doc["age"], 30)

    def test_mongo_insert_one_tool_error(self):
        """Test document insertion with invalid parameters."""
        tool = make_mongo_insert_one_tool()
        args = {
            "db_name": "",  # Invalid database name
            "collection_name": self.test_collection_name,
            "document": self.test_doc
        }

        result = tool.handler(args)
        self.assertIn("error", result)
        self.assertIn("mongo_insert_one failed", result["error"])

    def test_mongo_find_one_tool_success(self):
        """Test successful single document finding."""
        # First insert a document
        db = mongo_client[self.test_db_name]
        collection = db[self.test_collection_name]
        inserted_id = collection.insert_one(self.test_doc).inserted_id

        # Test finding the document
        tool = make_mongo_find_one_tool()
        args = {
            "db_name": self.test_db_name,
            "collection_name": self.test_collection_name,
            "filter": self.test_filter
        }

        result = tool.handler(args)

        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "John Doe")
        self.assertEqual(result["age"], 30)
        self.assertEqual(result["_id"], str(inserted_id))

    def test_mongo_find_one_tool_not_found(self):
        """Test finding non-existent document."""
        tool = make_mongo_find_one_tool()
        args = {
            "db_name": self.test_db_name,
            "collection_name": self.test_collection_name,
            "filter": {"name": "NonExistent"}
        }

        result = tool.handler(args)
        self.assertIsNone(result)

    def test_mongo_update_one_tool_success(self):
        """Test successful document update."""
        # First insert a document
        db = mongo_client[self.test_db_name]
        collection = db[self.test_collection_name]
        collection.insert_one(self.test_doc)

        # Test updating the document
        tool = make_mongo_update_one_tool()
        args = {
            "db_name": self.test_db_name,
            "collection_name": self.test_collection_name,
            "filter": self.test_filter,
            "update": self.test_update
        }

        result = tool.handler(args)

        self.assertIn("matched_count", result)
        self.assertIn("modified_count", result)
        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(result["modified_count"], 1)

        # Verify the update
        updated_doc = collection.find_one({"name": "John Doe"})
        self.assertEqual(updated_doc["age"], 31)
        self.assertEqual(updated_doc["email"], "john.updated@example.com")

    def test_mongo_delete_one_tool_success(self):
        """Test successful document deletion."""
        # First insert a document
        db = mongo_client[self.test_db_name]
        collection = db[self.test_collection_name]
        collection.insert_one(self.test_doc)

        # Test deleting the document
        tool = make_mongo_delete_one_tool()
        args = {
            "db_name": self.test_db_name,
            "collection_name": self.test_collection_name,
            "filter": self.test_filter
        }

        result = tool.handler(args)

        self.assertIn("deleted_count", result)
        self.assertEqual(result["deleted_count"], 1)

        # Verify the deletion
        deleted_doc = collection.find_one({"name": "John Doe"})
        self.assertIsNone(deleted_doc)

    def test_mongo_find_tool_success(self):
        """Test successful multiple document finding."""
        # Insert multiple documents
        db = mongo_client[self.test_db_name]
        collection = db[self.test_collection_name]
        docs = [
            {"name": "John Doe", "age": 30},
            {"name": "Jane Smith", "age": 25},
            {"name": "Bob Johnson", "age": 35}
        ]
        collection.insert_many(docs)

        # Test finding documents
        tool = make_mongo_find_tool()
        args = {
            "db_name": self.test_db_name,
            "collection_name": self.test_collection_name,
            "filter": {},
            "limit": 10
        }

        result = tool.handler(args)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

        # Test with filter
        args_filter = {
            "db_name": self.test_db_name,
            "collection_name": self.test_collection_name,
            "filter": {"age": {"$gte": 30}},
            "limit": 10
        }

        result_filtered = tool.handler(args_filter)
        self.assertEqual(len(result_filtered), 2)  # John and Bob

    def test_mongo_aggregate_tool_success(self):
        """Test successful aggregation pipeline."""
        # Insert test data
        db = mongo_client[self.test_db_name]
        collection = db[self.test_collection_name]
        docs = [
            {"name": "John", "age": 30, "department": "Engineering"},
            {"name": "Jane", "age": 25, "department": "Marketing"},
            {"name": "Bob", "age": 35, "department": "Engineering"}
        ]
        collection.insert_many(docs)

        # Test aggregation
        tool = make_mongo_aggregate_tool()
        args = {
            "db_name": self.test_db_name,
            "collection_name": self.test_collection_name,
            "pipeline": [
                {"$group": {"_id": "$department", "count": {"$sum": 1}}}
            ]
        }

        result = tool.handler(args)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)  # Engineering and Marketing

        # Should have grouped by department
        departments = {doc["_id"] for doc in result}
        self.assertIn("Engineering", departments)
        self.assertIn("Marketing", departments)

    def test_mongo_count_documents_tool_success(self):
        """Test successful document counting."""
        # Insert test data
        db = mongo_client[self.test_db_name]
        collection = db[self.test_collection_name]
        docs = [
            {"name": "John", "active": True},
            {"name": "Jane", "active": True},
            {"name": "Bob", "active": False}
        ]
        collection.insert_many(docs)

        # Test counting all documents
        tool = make_mongo_count_documents_tool()
        args = {
            "db_name": self.test_db_name,
            "collection_name": self.test_collection_name,
            "filter": {}
        }

        result = tool.handler(args)

        self.assertIn("count", result)
        self.assertEqual(result["count"], 3)

        # Test counting with filter
        args_filtered = {
            "db_name": self.test_db_name,
            "collection_name": self.test_collection_name,
            "filter": {"active": True}
        }

        result_filtered = tool.handler(args_filtered)
        self.assertEqual(result_filtered["count"], 2)

    def test_mongo_list_databases_tool_success(self):
        """Test successful database listing."""
        # First create the test database by inserting a document
        db = mongo_client[self.test_db_name]
        db[self.test_collection_name].insert_one({"test": "data"})

        tool = make_mongo_list_databases_tool()
        args = {}

        result = tool.handler(args)

        self.assertIn("databases", result)
        self.assertIsInstance(result["databases"], list)

        # Should include our test database
        self.assertIn(self.test_db_name, result["databases"])

    def test_mongo_list_collections_tool_success(self):
        """Test successful collection listing."""
        # Create a test collection
        db = mongo_client[self.test_db_name]
        db[self.test_collection_name].insert_one({"test": "data"})

        tool = make_mongo_list_collections_tool()
        args = {
            "db_name": self.test_db_name
        }

        result = tool.handler(args)

        self.assertIn("collections", result)
        self.assertIsInstance(result["collections"], list)
        self.assertIn(self.test_collection_name, result["collections"])

    def test_mongo_drop_collection_tool_success(self):
        """Test successful collection dropping."""
        # Create a test collection
        db = mongo_client[self.test_db_name]
        db[self.test_collection_name].insert_one({"test": "data"})

        # Verify collection exists
        collections_before = db.list_collection_names()
        self.assertIn(self.test_collection_name, collections_before)

        # Test dropping collection
        tool = make_mongo_drop_collection_tool()
        args = {
            "db_name": self.test_db_name,
            "collection_name": self.test_collection_name
        }

        result = tool.handler(args)

        self.assertIn("message", result)
        self.assertIn("dropped successfully", result["message"])

        # Verify collection is dropped
        collections_after = db.list_collection_names()
        self.assertNotIn(self.test_collection_name, collections_after)

    def test_tool_error_handling(self):
        """Test error handling across all tools."""
        # Test with invalid database name
        invalid_args = {"db_name": "invalid/db/name", "collection_name": "test"}

        tools_to_test = [
            (make_mongo_insert_one_tool, {**invalid_args, "document": {"test": "data"}}),
            (make_mongo_find_one_tool, {**invalid_args, "filter": {}}),
            (make_mongo_update_one_tool, {**invalid_args, "filter": {}, "update": {"$set": {"test": "value"}}}),
            (make_mongo_delete_one_tool, {**invalid_args, "filter": {}}),
            (make_mongo_find_tool, {**invalid_args, "filter": {}}),
            (make_mongo_aggregate_tool, {**invalid_args, "pipeline": []}),
            (make_mongo_count_documents_tool, {**invalid_args, "filter": {}}),
            (make_mongo_list_collections_tool, invalid_args),
        ]

        for tool_func, args in tools_to_test:
            with self.subTest(tool=tool_func.__name__):
                tool = tool_func()
                result = tool.handler(args)
                self.assertIn("error", result)
                self.assertIn("failed", result["error"])

    def test_missing_required_parameters(self):
        """Test behavior with missing required parameters."""
        tool = make_mongo_insert_one_tool()
        incomplete_args = {
            "db_name": self.test_db_name,
            # Missing collection_name and document
        }

        result = tool.handler(incomplete_args)
        self.assertIn("error", result)


if __name__ == "__main__":
    # Run the tests
    unittest.main()