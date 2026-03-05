""" 
MongoDB Atlas Vector Search Handler
Production-grade vector store with versioning and enhanced metadata
"""
from typing import List, Dict, Optional
import logging
from datetime import datetime
import hashlib

from pymongo import MongoClient
from pymongo.errors import PyMongoError
import numpy as np

from ingestion.embeddings import EmbeddingGenerator
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDBVectorStore:
    """
    Vector store using MongoDB Atlas Vector Search.
    
    Features:
    - Automatic embedding generation
    - Vector similarity search
    - Hybrid search (vector + metadata filtering)
    - Dynamic document ingestion (any number of PDFs)
    - Section-aware retrieval
    """
    
    def __init__(self):
        try:
            self.client = MongoClient(settings.mongodb_uri)
            self.db = self.client[settings.mongodb_db_name]
            self.collection = self.db[settings.mongodb_collection_name]
            
            # Initialize embedding generator
            self.embedding_generator = EmbeddingGenerator()
            
            # Ensure indexes
            self._ensure_indexes()
            
            logger.info(f"Connected to MongoDB: {settings.mongodb_db_name}")
        except PyMongoError as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
    
    def _ensure_indexes(self):
        """
        Ensure necessary indexes exist.
        
        Note: Vector search index must be created via Atlas UI or CLI:
        
        Index definition for Atlas Vector Search:
        {
          "fields": [
            {
              "type": "vector",
              "path": "embedding",
              "numDimensions": 768,
              "similarity": "cosine"
            },
            {
              "type": "filter",
              "path": "metadata.patient_id"
            },
            {
              "type": "filter",
              "path": "metadata.section_type"
            },
            {
              "type": "filter",
              "path": "metadata.filename"
            }
          ]
        }

        CRITICAL: metadata.patient_id MUST be a filter field.
        Without it, patient_search() filters are ignored and patients see each other's records.
        """
        # Create text index for fallback search
        try:
            self.collection.create_index([("text", "text")])
            logger.info("Text index ensured")
        except Exception as e:
            logger.debug(f"Text index creation: {e}")
        
        # Create compound index for metadata queries
        try:
            self.collection.create_index([
                ("metadata.filename", 1),
                ("metadata.section_type", 1)
            ])
            logger.info("Metadata index ensured")
        except Exception as e:
            logger.debug(f"Metadata index creation: {e}")
    
    def add_documents(self, chunks: List[Dict[str, any]]) -> List[str]:
        """
        Add document chunks to vector store with automatic embedding generation.
        
        Fully dynamic - handles any number of chunks from any number of documents.
        
        Args:
            chunks: List of chunk dictionaries containing 'text' and metadata
            
        Returns:
            List of inserted document IDs
        """
        if not chunks:
            logger.warning("No chunks provided to add_documents")
            return []
        
        logger.info(f"Adding {len(chunks)} chunks to vector store...")
        
        # Extract texts for batch embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings in batch (more efficient)
        try:
            embeddings = self.embedding_generator.generate_embeddings_batch(texts)
        except Exception as e:
            logger.error(f"Batch embedding failed, falling back to individual: {e}")
            embeddings = [
                self.embedding_generator.generate_embedding(text) 
                for text in texts
            ]
        
        # Prepare documents for insertion (PRODUCTION SCHEMA)
        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            # Generate unique doc_id (hash of source + chunk_id)
            doc_id = self._generate_doc_id(
                chunk.get("filename", "unknown"),
                chunk.get("chunk_id", 0)
            )
            
            # Production document structure
            doc = {
                "doc_id": doc_id,
                "text": chunk["text"],  # Raw text (separate from embedding)
                "embedding": embedding,  # Vector embedding
                "metadata": {
                    # Core identifiers
                    "chunk_id": chunk.get("chunk_id", 0),
                    "chunk_type": chunk.get("chunk_type", "text"),  # text or table
                    "page": chunk.get("page", None),
                    "position_in_doc": chunk.get("position_in_doc", 0),
                    
                    # Source tracking
                    "source": chunk.get("source", ""),
                    "filename": chunk.get("filename", ""),
                    
                    # Semantic classification
                    "section_type": chunk.get("section_type", "general"),
                    
                    # Table-specific metadata
                    "table_number": chunk.get("table_number", None),
                    
                    # Processing metadata
                    "extraction_method": chunk.get("extraction_method", "unknown"),
                    
                    # Timestamps
                    "created_at": datetime.utcnow().isoformat(),
                    "ingestion_date": chunk.get("date", datetime.utcnow().isoformat()),
                    
                    # Rich metadata (Production requirement)
                    "doctor_name": chunk.get("doctor_name"),
                    "hospital_name": chunk.get("hospital_name"),
                    "report_date": chunk.get("report_date"),
                    "report_type": chunk.get("report_type"),
                    "patient_id": chunk.get("patient_id"),
                    
                    # Versioning (production requirement)
                    "version": settings.document_version if settings.enable_document_versioning else "1.0",
                    "schema_version": "2.1",  # Updated for rich metadata
                }
            }
            documents.append(doc)
        
        # Batch insert
        try:
            result = self.collection.insert_many(documents, ordered=False)
            logger.info(f"Successfully inserted {len(result.inserted_ids)} documents")
            return [str(_id) for _id in result.inserted_ids]
        except PyMongoError as e:
            logger.error(f"Document insertion failed: {e}")
            raise
    
    def add_patient_document(
        self,
        chunks: List[Dict[str, any]],
        patient_id: str,
        document_type: str = "other"
    ) -> tuple[str, List[str]]:
        """
        Add document chunks for a MediVault patient with shared doc_id.
        All chunks from the same file share one doc_id for easy deletion.
        
        Args:
            chunks: List of chunk dictionaries
            patient_id: Patient's MongoDB ObjectId from MediVault
            document_type: Type of medical document
            
        Returns:
            Tuple of (doc_id, list of inserted MongoDB IDs)
        """
        if not chunks:
            logger.warning("No chunks provided to add_patient_document")
            return ("", [])
        
        # Generate ONE shared doc_id for all chunks (timestamp-based for uniqueness)
        import uuid
        doc_id = uuid.uuid4().hex[:16]
        
        logger.info(f"Adding patient document: patient_id={patient_id}, doc_id={doc_id}, chunks={len(chunks)}")
        
        # Extract texts for batch embedding
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings in batch
        try:
            embeddings = self.embedding_generator.generate_embeddings_batch(texts)
        except Exception as e:
            logger.error(f"Batch embedding failed, falling back to individual: {e}")
            embeddings = [
                self.embedding_generator.generate_embedding(text) 
                for text in texts
            ]
        
        # Prepare documents with SHARED doc_id
        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            doc = {
                "doc_id": doc_id,  # SHARED across all chunks from this file
                "text": chunk["text"],
                "embedding": embedding,
                "metadata": {
                    # MediVault patient isolation (MANDATORY)
                    "patient_id": patient_id,
                    
                    # Core identifiers
                    "chunk_id": chunk.get("chunk_id", 0),
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "page": chunk.get("page", None),
                    "position_in_doc": chunk.get("position_in_doc", 0),
                    
                    # Source tracking
                    "source": chunk.get("source", ""),
                    "filename": chunk.get("filename", ""),
                    
                    # Semantic classification
                    "section_type": chunk.get("section_type", "general"),
                    
                    # Table-specific
                    "table_number": chunk.get("table_number", None),
                    
                    # Processing metadata
                    "extraction_method": chunk.get("extraction_method", "unknown"),
                    
                    # Medical metadata
                    "doctor_name": chunk.get("doctor_name"),
                    "hospital_name": chunk.get("hospital_name"),
                    "report_date": chunk.get("report_date"),
                    "report_type": document_type,
                    
                    # Timestamps
                    "created_at": datetime.utcnow().isoformat(),
                    "ingestion_date": chunk.get("date", datetime.utcnow().isoformat()),
                    
                    # Versioning
                    "version": settings.document_version if settings.enable_document_versioning else "1.0",
                    "schema_version": "2.2",  # MediVault patient-isolation
                }
            }
            documents.append(doc)
        
        # Batch insert
        try:
            result = self.collection.insert_many(documents, ordered=False)
            logger.info(f"Successfully inserted {len(result.inserted_ids)} chunks for doc_id={doc_id}")
            return (doc_id, [str(_id) for _id in result.inserted_ids])
        except PyMongoError as e:
            logger.error(f"Patient document insertion failed: {e}")
            raise
    
    def _generate_doc_id(self, filename: str, chunk_id: int) -> str:
        """
        Generate unique document ID from filename and chunk.
        
        Production requirement: Consistent IDs for versioning and deduplication.
        
        Args:
            filename: Source filename
            chunk_id: Chunk identifier
            
        Returns:
            Unique document ID (hash)
        """
        content = f"{filename}_{chunk_id}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict[str, any]]:
        """
        Perform vector similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            metadata_filter: Optional filter on metadata fields
            
        Returns:
            List of matching documents with scores
        """
        # Generate query embedding
        try:
            query_embedding = self.embedding_generator.generate_embedding(query)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return self._fallback_search(query, k)
        
        # Build aggregation pipeline for vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": settings.vector_index_name,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": k * settings.vector_search_candidates_multiplier,
                    "limit": k
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        # Add metadata filter if provided
        if metadata_filter:
            # Insert filter stage after vector search
            filter_stage = {"$match": {}}
            for key, value in metadata_filter.items():
                filter_stage["$match"][f"metadata.{key}"] = value
            pipeline.insert(1, filter_stage)
        
        try:
            results = list(self.collection.aggregate(pipeline))
            logger.info(f"Vector search found {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            logger.warning("Falling back to text search")
            return self._fallback_search(query, k)
    
    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        section_type: Optional[str] = None,
        filename: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Hybrid search combining vector similarity and metadata filtering.
        
        Perfect for dynamic RAG:
        - Searches across all documents
        - Filters by section when relevant
        - Adapts to any number of documents
        
        Args:
            query: Search query
            k: Number of results
            section_type: Optional section filter (e.g., 'medications')
            filename: Optional filename filter
            
        Returns:
            Filtered and ranked results
        """
        # Build metadata filter
        metadata_filter = {}
        if section_type:
            metadata_filter['section_type'] = section_type
        if filename:
            metadata_filter['filename'] = filename
        
        # Perform vector search with filters
        results = self.similarity_search(
            query,
            k=k * 2,  # Get more candidates for better filtering
            metadata_filter=metadata_filter if metadata_filter else None
        )
        
        # If section filter was applied but we got few results, try without it
        if section_type and len(results) < k // 2:
            logger.info("Insufficient results with section filter, searching without")
            results = self.similarity_search(query, k=k)
        
        return results[:k]
    
    def filter_by_metadata(
        self,
        metadata_filter: Dict[str, any],
        limit: int = 50
    ) -> List[Dict[str, any]]:
        """
        Retrieve all chunks matching metadata filter.
        
        Useful for queries like "list ALL medications across all documents"
        
        Args:
            metadata_filter: Filter dictionary (e.g., {'section_type': 'medications'})
            limit: Maximum documents to return
            
        Returns:
            All matching chunks
        """
        query = {}
        for key, value in metadata_filter.items():
            query[f"metadata.{key}"] = value
        
        try:
            results = list(self.collection.find(query).limit(limit))
            logger.info(f"Metadata filter found {len(results)} documents")
            
            # Add a placeholder score
            for result in results:
                result['score'] = 1.0
            
            return results
        except PyMongoError as e:
            logger.error(f"Metadata filter query failed: {e}")
            return []
    
    def get_all_filenames(self) -> List[str]:
        """
        Get list of all unique filenames in the database.
        Enables dynamic multi-document queries.
        """
        try:
            filenames = self.collection.distinct("metadata.filename")
            logger.info(f"Found {len(filenames)} unique documents")
            return filenames
        except PyMongoError as e:
            logger.error(f"Failed to get filenames: {e}")
            return []
    
    def _fallback_search(self, query: str, k: int) -> List[Dict]:
        """Fallback to text search if vector search fails."""
        logger.warning("Using fallback text search")
        try:
            results = list(
                self.collection.find(
                    {"$text": {"$search": query}},
                    {"score": {"$meta": "textScore"}}
                ).sort([("score", {"$meta": "textScore"})]).limit(k)
            )
            return results
        except Exception as e:
            logger.error(f"Fallback search also failed: {e}")
            return []
    
    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source file."""
        try:
            result = self.collection.delete_many({"metadata.source": source})
            logger.info(f"Deleted {result.deleted_count} documents from {source}")
            return result.deleted_count
        except PyMongoError as e:
            logger.error(f"Delete operation failed: {e}")
            return 0
    
    def delete_by_filename(self, filename: str) -> int:
        """Delete all chunks from a specific filename."""
        try:
            result = self.collection.delete_many({"metadata.filename": filename})
            logger.info(f"Deleted {result.deleted_count} documents for {filename}")
            return result.deleted_count
        except PyMongoError as e:
            logger.error(f"Delete operation failed: {e}")
            return 0
    
    def clear_collection(self) -> int:
        """Clear all documents (use with caution!)."""
        try:
            result = self.collection.delete_many({})
            logger.warning(f"Cleared {result.deleted_count} documents from collection")
            return result.deleted_count
        except PyMongoError as e:
            logger.error(f"Clear operation failed: {e}")
            return 0
    
    def get_stats(self) -> Dict:
        """
        Get collection statistics.
        Useful for understanding the current state of the vector database.
        """
        try:
            count = self.collection.count_documents({})
            
            # Get section type distribution
            pipeline = [
                {"$group": {
                    "_id": "$metadata.section_type",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}}
            ]
            section_stats = list(self.collection.aggregate(pipeline))
            
            # Get document distribution
            pipeline = [
                {"$group": {
                    "_id": "$metadata.filename",
                    "chunks": {"$sum": 1}
                }},
                {"$sort": {"chunks": -1}}
            ]
            doc_stats = list(self.collection.aggregate(pipeline))
            
            return {
                "total_chunks": count,
                "total_documents": len(doc_stats),
                "section_distribution": {
                    item["_id"]: item["count"] 
                    for item in section_stats
                },
                "documents": [
                    {"filename": item["_id"], "chunks": item["chunks"]}
                    for item in doc_stats
                ]
            }
        except PyMongoError as e:
            logger.error(f"Stats query failed: {e}")
            return {"error": str(e)}
    
    # ========== MEDIVAULT PATIENT-SPECIFIC METHODS ==========
    
    def patient_search(
        self,
        patient_id: str,
        query: str,
        k: int = 5,
        section_type: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Vector search filtered by patient_id (MediVault multi-tenant isolation).
        
        Args:
            patient_id: Patient's MongoDB ObjectId from MediVault backend
            query: Search query
            k: Number of results
            section_type: Optional section filter
            
        Returns:
            Results belonging only to this patient
        """
        # Generate query embedding
        try:
            query_embedding = self.embedding_generator.generate_embedding(query)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return []
        
        # Build metadata filter with MANDATORY patient_id
        metadata_filter = {"metadata.patient_id": patient_id}
        if section_type:
            metadata_filter["metadata.section_type"] = section_type
        
        # Build vector search pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": settings.vector_index_name,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": k * settings.vector_search_candidates_multiplier,
                    "limit": k,
                    "filter": metadata_filter
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "text": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        try:
            results = list(self.collection.aggregate(pipeline))
            logger.info(f"Patient search (patient_id={patient_id}): {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Patient search failed: {e}")
            return []
    
    def get_patient_documents(self, patient_id: str) -> List[Dict]:
        """
        Get deduplicated list of documents for a patient.
        Used for MediVault "My Documents" screen.
        
        Args:
            patient_id: Patient's MongoDB ObjectId
            
        Returns:
            List of unique documents with metadata
        """
        try:
            pipeline = [
                # Filter by patient
                {"$match": {"metadata.patient_id": patient_id}},
                # Group by doc_id (all chunks from same file share doc_id)
                {"$group": {
                    "_id": "$doc_id",
                    "filename": {"$first": "$metadata.filename"},
                    "document_type": {"$first": "$metadata.report_type"},
                    "document_date": {"$first": "$metadata.report_date"},
                    "created_at": {"$first": "$metadata.created_at"},
                    "chunks": {"$sum": 1}
                }},
                {"$sort": {"created_at": -1}}
            ]
            
            docs = list(self.collection.aggregate(pipeline))
            
            # Format for MediVault response
            formatted_docs = []
            for doc in docs:
                formatted_docs.append({
                    "doc_id": doc["_id"],
                    "filename": doc.get("filename", "unknown.pdf"),
                    "document_type": doc.get("document_type") or "other",
                    "document_date": doc.get("document_date"),
                    "chunks": doc["chunks"],
                    "ingested_at": doc.get("created_at")
                })
            
            return formatted_docs
        except PyMongoError as e:
            logger.error(f"Get patient documents failed: {e}")
            return []
    
    def get_patient_stats(self, patient_id: str) -> Dict:
        """Get statistics for a specific patient's documents."""
        try:
            # Total chunks
            total_chunks = self.collection.count_documents({"metadata.patient_id": patient_id})
            
            # Unique documents
            pipeline = [
                {"$match": {"metadata.patient_id": patient_id}},
                {"$group": {"_id": "$doc_id"}},
                {"$count": "total"}
            ]
            doc_count_result = list(self.collection.aggregate(pipeline))
            total_docs = doc_count_result[0]["total"] if doc_count_result else 0
            
            return {
                "patient_id": patient_id,
                "total_documents": total_docs,
                "total_chunks": total_chunks
            }
        except PyMongoError as e:
            logger.error(f"Get patient stats failed: {e}")
            return {"error": str(e)}
    
    def delete_by_doc_id(self, patient_id: str, doc_id: str) -> int:
        """
        Delete all chunks for a specific document, with patient_id verification.
        
        Args:
            patient_id: Patient's MongoDB ObjectId (for security)
            doc_id: Document ID to delete
            
        Returns:
            Number of chunks deleted
        """
        try:
            result = self.collection.delete_many({
                "doc_id": doc_id,
                "metadata.patient_id": patient_id
            })
            logger.info(f"Deleted {result.deleted_count} chunks for doc_id={doc_id}, patient_id={patient_id}")
            return result.deleted_count
        except PyMongoError as e:
            logger.error(f"Delete by doc_id failed: {e}")
            return 0
    
    def check_patient_has_documents(self, patient_id: str) -> bool:
        """Check if a patient has any documents ingested."""
        try:
            count = self.collection.count_documents({"metadata.patient_id": patient_id}, limit=1)
            return count > 0
        except PyMongoError as e:
            logger.error(f"Check patient documents failed: {e}")
            return False


# Example usage
if __name__ == "__main__":
    print("MongoDB Vector Store")
    print("="*60)
    print(f"Database: {settings.mongodb_db_name}")
    print(f"Collection: {settings.mongodb_collection_name}")
    print(f"Vector Index: {settings.vector_index_name}")
    print()
    print("This vector store supports dynamic RAG:")
    print("   Any number of documents")
    print("   Any number of pages per document")
    print("   Section-aware retrieval")
    print("   Metadata filtering")
    print("   Hybrid search")
    print()
    print("To set up Atlas Vector Search:")
    print("  1. Create a MongoDB Atlas cluster")
    print("  2. Create a database and collection")
    print("  3. Create a Vector Search index via Atlas UI")
    print("  4. Configure index with 1536 dimensions, cosine similarity")
