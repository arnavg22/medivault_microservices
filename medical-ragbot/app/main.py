"""
Medical RAG Bot - Main API Application
FastAPI application for dynamic RAG system
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
from pathlib import Path
import shutil
from datetime import datetime

from config import settings
from ingestion.pdf_loader import PDFProcessor
from ingestion.text_splitter import MedicalTextSplitter
from vectorstore.mongodb_handler import MongoDBVectorStore
from rag.llm_handler import MedicalLLMHandler
from rag.qa_chain import MedicalQAChain
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical RAG Bot API",
    description="Dynamic RAG system for medical document analysis using LLAMA 3",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global component references (initialized on startup)
pdf_processor = None
text_splitter = None
vector_store = None
llm_handler = None
qa_chain = None
startup_complete = False  # Track if initialization succeeded


@app.on_event("startup")
async def startup_event():
    """Initialize components after server starts (allows port binding first)"""
    global pdf_processor, text_splitter, vector_store, llm_handler, qa_chain, startup_complete
    
    # Production: Log environment information
    port = os.getenv("PORT", "10000")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    logger.info("=" * 80)
    logger.info("🚀 Medical RAG Bot API Starting...")
    logger.info("=" * 80)
    logger.info(f"📍 Environment: {'Render.com' if os.getenv('RENDER') else 'Local'}")
    logger.info(f"🐍 Python Version: {python_version}")
    logger.info(f"🔌 Binding to Port: {port} on host 0.0.0.0")
    logger.info(f"🤖 LLM Provider: {settings.llm_provider}")
    logger.info(f"🧠 LLM Model: {settings.llm_model}")
    logger.info(f"💾 Vector Store: MongoDB Atlas")
    logger.info(f"📄 Chunk Size: {settings.chunk_size} chars")
    logger.info(f"🔗 Chunk Overlap: {settings.chunk_overlap_percent * 100}%")
    logger.info("=" * 80)
    
    # Ensure data directories exist
    Path(settings.raw_pdfs_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.processed_text_dir).mkdir(parents=True, exist_ok=True)
    logger.info("✅ Data directories ready")
    
    # Initialize components with detailed logging
    try:
        logger.info("⏳ Initializing PDF processor...")
        pdf_processor = PDFProcessor()
        
        logger.info("⏳ Initializing text splitter...")
        text_splitter = MedicalTextSplitter()
        
        logger.info("⏳ Connecting to MongoDB Atlas...")
        vector_store = MongoDBVectorStore()
        logger.info("✅ MongoDB connected successfully")
        
        logger.info("⏳ Initializing LLM handler...")
        llm_handler = MedicalLLMHandler()
        
        logger.info("⏳ Building QA chain...")
        qa_chain = MedicalQAChain(vector_store=vector_store)
        qa_chain.set_llm_handler(llm_handler)
        
        startup_complete = True
        logger.info("=" * 80)
        logger.info("✅ ALL COMPONENTS INITIALIZED SUCCESSFULLY")
        logger.info("🎉 Medical RAG Bot is ready to serve requests!")
        logger.info("=" * 80)
    except Exception as e:
        startup_complete = False
        logger.error("=" * 80)
        logger.error(f"❌ COMPONENT INITIALIZATION FAILED: {e}")
        logger.error(f"📋 Error Type: {type(e).__name__}")
        logger.warning("⚠️  App running in DEGRADED mode - health check will report status")
        logger.warning("⚠️  RAG features will not work until components are initialized")
        logger.error("=" * 80)
        # Don't crash - let health endpoint report the problem


@app.on_event("shutdown")
async def shutdown_event():
    """Graceful shutdown with proper cleanup"""
    global vector_store
    
    logger.info("=" * 80)
    logger.info("🛑 Medical RAG Bot API shutting down...")
    
    try:
        # Close MongoDB connection gracefully
        if vector_store is not None and hasattr(vector_store, 'client'):
            logger.info("📦 Closing MongoDB connection...")
            vector_store.client.close()
            logger.info("✅ MongoDB connection closed")
    except Exception as e:
        logger.error(f"⚠️  Error during shutdown cleanup: {e}")
    
    logger.info("👋 Shutdown complete")
    logger.info("=" * 80)


# ========== Request/Response Models ==========

class QueryRequest(BaseModel):
    question: str
    k: int = 5
    conversation_history: Optional[List[Dict]] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    metadata: Dict


class IngestionResponse(BaseModel):
    status: str
    filename: str
    chunks_created: int
    extraction_method: str


class StatsResponse(BaseModel):
    total_chunks: int
    total_documents: int
    section_distribution: Dict[str, int]
    documents: List[Dict]


# ========== API Endpoints ==========

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Medical RAG Bot API",
        "version": "2.0.0",
        "llm": settings.llm_model,
        "provider": settings.llm_provider,
        "status": "running",
        "features": [
            "Dynamic document ingestion",
            "Section-aware chunking",
            "LLAMA 3 generation",
            "Multi-document queries",
            "Semantic search"
        ]
    }


@app.get("/health")
async def health_check():
    """
    Production-ready health check endpoint.
    
    Returns:
    - 200: Service healthy and ready
    - 503: Service degraded (MongoDB or initialization failed)
    
    Render uses this for:
    - Initial deployment health check (must return 200 within timeout)
    - Ongoing uptime monitoring
    - Load balancer routing decisions
    """
    health_data = {
        "service": "medivault-ragbot",
        "version": "2.0.0",
        "startup_complete": startup_complete,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }
    
    # Check startup status
    if not startup_complete:
        health_data["status"] = "initializing"
        health_data["ready"] = False
        health_data["message"] = "Components still initializing or initialization failed"
        raise HTTPException(status_code=503, detail=health_data)
    
    # Check MongoDB connection (if initialized)
    try:
        if vector_store is not None:
            vector_store.client.admin.command('ping', maxTimeMS=2000)  # 2 second timeout
            mongodb_connected = True
        else:
            mongodb_connected = False
    except Exception as e:
        logger.warning(f"MongoDB health check failed: {e}")
        mongodb_connected = False
    
    # Check if Groq is configured
    groq_configured = bool(settings.groq_api_key and settings.groq_api_key != "")
    
    # Build detailed health response
    health_data.update({
        "status": "healthy" if mongodb_connected else "degraded",
        "ready": mongodb_connected,
        "components": {
            "pdf_processor": pdf_processor is not None,
            "text_splitter": text_splitter is not None,
            "vector_store": vector_store is not None,
            "mongodb_connected": mongodb_connected,
            "llm_handler": llm_handler is not None,
            "groq_configured": groq_configured,
            "qa_chain": qa_chain is not None
        }
    })
    
    # Return 503 if MongoDB is down (critical dependency)
    if not mongodb_connected:
        health_data["message"] = "MongoDB connection unavailable - RAG features disabled"
        raise HTTPException(status_code=503, detail=health_data)
    
    # Return 200 - service healthy
    return health_data


@app.post("/ingest/upload", response_model=IngestionResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and ingest a single PDF file.
    
    Fully dynamic - handles any PDF regardless of size or content.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file to raw_pdfs directory
        upload_dir = Path(settings.raw_pdfs_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved uploaded file: {file.filename}")
        
        # Process the PDF
        document = pdf_processor.extract_text_from_pdf(str(file_path))
        
        # Split into chunks
        chunks = text_splitter.split_document(document)
        
        # Add to vector store
        vector_store.add_documents(chunks)
        
        return IngestionResponse(
            status="success",
            filename=file.filename,
            chunks_created=len(chunks),
            extraction_method=document.get("extraction_method", "unknown")
        )
    
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/ingest/directory")
async def ingest_directory(directory_path: Optional[str] = None):
    """
    Ingest all PDFs from a directory.
    
    Perfect for batch processing - any number of documents.
    """
    dir_to_process = directory_path or settings.raw_pdfs_dir
    
    try:
        # Extract from directory
        documents = pdf_processor.extract_from_directory(dir_to_process)
        
        if not documents:
            raise HTTPException(status_code=404, detail="No PDF files found in directory")
        
        # Split all documents
        all_chunks = text_splitter.batch_split(documents)
        
        # Add to vector store
        vector_store.add_documents(all_chunks)
        
        return {
            "status": "success",
            "documents_processed": len(documents),
            "total_chunks_created": len(all_chunks),
            "directory": dir_to_process
        }
    
    except Exception as e:
        logger.error(f"Directory ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question about the medical documents.
    
    Dynamically searches across all ingested documents.
    """
    try:
        result = qa_chain.answer_question(
            question=request.question,
            k=request.k,
            conversation_history=request.conversation_history,
            use_multi_stage=True
        )
        
        return QueryResponse(**result)
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/query/section/{section_type}")
async def query_section(section_type: str, question: Optional[str] = None):
    """
    Query a specific section type across ALL documents.
    
    Examples:
    - /query/section/medications - Lists ALL medications
    - /query/section/diagnosis - Lists ALL diagnoses
    - /query/section/lab_results - Shows ALL lab results
    """
    try:
        result = qa_chain.answer_with_specific_section(
            section_type=section_type,
            question=question
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Section query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/documents")
async def list_documents():
    """
    List all ingested documents.
    
    Shows the dynamic corpus of available documents.
    """
    try:
        filenames = vector_store.get_all_filenames()
        
        return {
            "total_documents": len(filenames),
            "documents": filenames
        }
    
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get vector store statistics.
    
    Shows the current state of the RAG system.
    """
    try:
        stats = vector_store.get_stats()
        return StatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """
    Delete a specific document from the vector store.
    
    Removes all chunks associated with the document.
    """
    try:
        deleted_count = vector_store.delete_by_filename(filename)
        
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "status": "success",
            "filename": filename,
            "chunks_deleted": deleted_count
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.delete("/clear")
async def clear_all():
    """
    Clear all documents from the vector store.
    
     WARNING: This cannot be undone!
    """
    try:
        deleted_count = vector_store.clear_collection()
        
        return {
            "status": "success",
            "chunks_deleted": deleted_count,
            "warning": "All documents cleared"
        }
    
    except Exception as e:
        logger.error(f"Clear failed: {e}")
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")


# ========== MEDIVAULT-SPECIFIC ROUTES (Patient Isolation) ==========

class MediVaultIngestRequest(BaseModel):
    patient_id: str
    document_type: str = "other"  # prescription | lab_report | discharge_summary | consultation_note | other
    document_date: Optional[str] = None  # YYYY-MM-DD
    source_encounter_id: Optional[str] = None


class MediVaultQueryRequest(BaseModel):
    patient_id: str
    question: str
    conversation_history: Optional[List[Dict]] = None
    section_filter: Optional[str] = None  # medications | diagnosis | lab_results | vitals | allergies | procedures | follow_up | discharge


class MediVaultSummarizeRequest(BaseModel):
    patient_id: str
    summary_type: str = "full"  # full | medications | conditions | recent


@app.post("/ingest/pdf")
async def ingest_patient_pdf(
    file: UploadFile = File(...),
    patient_id: str = File(...),
    document_type: str = File(default="other"),
    document_date: Optional[str] = File(default=None),
    source_encounter_id: Optional[str] = File(default=None)
):
    """
    MediVault: Upload and ingest a patient's PDF document.
    All chunks are tagged with patient_id for multi-tenant isolation.
    """
    temp_path = None
    
    try:
        # Validation
        if not file.filename.endswith('.pdf'):
            return {
                "success": False,
                "message": "Only PDF files are supported",
                "error": "INVALID_FILE_TYPE"
            }
        
        # Save temp file
        upload_dir = Path(settings.raw_pdfs_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Unique filename to avoid collisions
        import uuid
        unique_filename = f"{patient_id}_{uuid.uuid4().hex[:8]}_{file.filename}"
        temp_path = upload_dir / unique_filename
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"[MediVault] Processing PDF for patient_id={patient_id}: {file.filename}")
        
        # Extract text from PDF
        try:
            document = pdf_processor.extract_text_from_pdf(str(temp_path))
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return {
                "success": False,
                "message": "Failed to extract text from PDF. The file may be corrupted or encrypted.",
                "error": "TEXT_EXTRACTION_FAILED"
            }
        
        # Add patient metadata to document
        document["patient_id"] = patient_id
        document["document_type"] = document_type
        document["report_type"] = document_type
        if document_date:
            document["report_date"] = document_date
        if source_encounter_id:
            document["source_encounter_id"] = source_encounter_id
        
        # Split into chunks
        chunks = text_splitter.split_document(document)
        
        if not chunks:
            return {
                "success": False,
                "message": "No content could be extracted from the PDF",
                "error": "NO_CONTENT_EXTRACTED"
            }
        
        # Add to vector store with shared doc_id
        try:
            doc_id, _ = vector_store.add_patient_document(
                chunks,
                patient_id=patient_id,
                document_type=document_type
            )
        except Exception as e:
            logger.error(f"Vector store insertion failed: {e}")
            return {
                "success": False,
                "message": "Failed to store document in database",
                "error": "DATABASE_ERROR"
            }
        
        return {
            "success": True,
            "message": "Document ingested successfully",
            "data": {
                "doc_id": doc_id,
                "filename": file.filename,
                "chunks_created": len(chunks),
                "document_type": document_type,
                "patient_id": patient_id
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[MediVault] Ingestion failed: {e}")
        return {
            "success": False,
            "message": f"Unexpected error during ingestion: {str(e)}",
            "error": "INTERNAL_ERROR"
        }
    
    finally:
        # Always clean up temp file
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
                logger.info(f"Cleaned up temp file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")


@app.get("/ingest/status/{patient_id}")
async def get_patient_ingestion_status(patient_id: str):
    """
    MediVault: Get list of all documents ingested for a patient.
    Used for "My Documents" screen in mobile app.
    """
    try:
        # Get patient's documents
        documents = vector_store.get_patient_documents(patient_id)
        stats = vector_store.get_patient_stats(patient_id)
        
        return {
            "success": True,
            "message": "Patient documents retrieved successfully",
            "data": {
                "patient_id": patient_id,
                "total_documents": stats.get("total_documents", 0),
                "total_chunks": stats.get("total_chunks", 0),
                "documents": documents
            }
        }
    
    except Exception as e:
        logger.error(f"[MediVault] Get patient status failed: {e}")
        return {
            "success": False,
            "message": "Failed to retrieve patient documents",
            "error": "DATABASE_ERROR"
        }


@app.delete("/ingest/document/{patient_id}/{doc_id}")
async def delete_patient_document(patient_id: str, doc_id: str):
    """
    MediVault: Delete a specific document for a patient.
    Verifies patient_id before deletion for security.
    """
    try:
        deleted_count = vector_store.delete_by_doc_id(patient_id, doc_id)
        
        if deleted_count == 0:
            return {
                "success": False,
                "message": "Document not found or does not belong to this patient",
                "error": "DOCUMENT_NOT_FOUND"
            }
        
        return {
            "success": True,
            "message": "Document removed successfully",
            "data": {
                "chunks_deleted": deleted_count
            }
        }
    
    except Exception as e:
        logger.error(f"[MediVault] Delete document failed: {e}")
        return {
            "success": False,
            "message": "Failed to delete document",
            "error": "DATABASE_ERROR"
        }


@app.post("/chat/query")
async def medivault_query(request: MediVaultQueryRequest):
    """
    MediVault: Core RAG chat endpoint.
    Answers patient questions using ONLY their medical records.
    """
    try:
        # Check if patient has any documents
        has_docs = vector_store.check_patient_has_documents(request.patient_id)
        
        if not has_docs:
            return {
                "success": False,
                "message": "No medical records found. Please upload your documents first.",
                "error": "NO_DOCUMENTS"
            }
        
        # Retrieve relevant context with patient_id filtering
        try:
            retrieved_chunks = vector_store.patient_search(
                patient_id=request.patient_id,
                query=request.question,
                k=5,
                section_type=request.section_filter
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return {
                "success": False,
                "message": "Failed to search medical records",
                "error": "SEARCH_ERROR"
            }
        
        if not retrieved_chunks:
            return {
                "success": True,
                "message": "Query processed successfully",
                "data": {
                    "answer": "I couldn't find relevant information in your medical records to answer this question.",
                    "sources": [],
                    "question": request.question
                }
            }
        
        # Build context from retrieved chunks
        context_parts = []
        for chunk in retrieved_chunks:
            filename = chunk.get('metadata', {}).get('filename', 'Unknown')
            section = chunk.get('metadata', {}).get('section_type', 'general')
            text = chunk.get('text', '')
            context_parts.append(f"[From {filename} - {section}]\n{text}\n")
        
        context = "\n".join(context_parts)
        
        # Build prompt with context — MedicalLLMHandler applies safety system prompt internally
        user_prompt = f"""Based on the following medical records, please answer the question.

MEDICAL RECORDS:
{context}

USER QUESTION:
{request.question}

ANSWER:"""

        # Generate response with Groq
        try:
            response = llm_handler.generate_response(
                user_prompt,
                conversation_history=request.conversation_history
            )
            answer = response.get("answer", "")
        except Exception as e:
            logger.error(f"[MediVault] Groq API failed: {e}")
            
            # Check for specific error types
            if "rate_limit" in str(e).lower() or "429" in str(e):
                return {
                    "success": False,
                    "message": "Rate limit exceeded. Please try again in a moment.",
                    "error": "RATE_LIMIT"
                }
            elif "503" in str(e) or "service" in str(e).lower():
                return {
                    "success": False,
                    "message": "AI service temporarily unavailable. Please try again later.",
                    "error": "SERVICE_UNAVAILABLE"
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to generate response",
                    "error": "LLM_ERROR"
                }
        
        # Format sources
        sources = []
        seen_docs = set()
        for chunk in retrieved_chunks[:5]:  # Top 5 sources
            metadata = chunk.get('metadata', {})
            filename = metadata.get('filename', 'Unknown')
            
            if filename not in seen_docs:
                sources.append({
                    "filename": filename,
                    "document_type": metadata.get('report_type', 'other'),
                    "document_date": metadata.get('report_date'),
                    "section_type": metadata.get('section_type', 'general'),
                    "relevance_score": round(chunk.get('score', 0), 2)
                })
                seen_docs.add(filename)
        
        return {
            "success": True,
            "message": "Query processed successfully",
            "data": {
                "answer": answer,
                "sources": sources,
                "question": request.question
            }
        }
    
    except Exception as e:
        logger.error(f"[MediVault] Query endpoint failed: {e}")
        return {
            "success": False,
            "message": "An unexpected error occurred",
            "error": "INTERNAL_ERROR"
        }


@app.post("/chat/summarize")
async def medivault_summarize(request: MediVaultSummarizeRequest):
    """
    MediVault: Generate structured health summary from all patient documents.
    Used for "My Health Summary" screen.
    """
    try:
        # Check if patient has documents
        has_docs = vector_store.check_patient_has_documents(request.patient_id)
        
        if not has_docs:
            return {
                "success": False,
                "message": "No medical records found. Please upload your documents first.",
                "error": "NO_DOCUMENTS"
            }
        
        # Get documents based on summary type
        if request.summary_type == "medications":
            section_filter = "medications"
            prompt_template = "List ALL medications with dosages and frequencies."
        elif request.summary_type == "conditions":
            section_filter = "diagnosis"
            prompt_template = "List ALL diagnosed conditions and their status."
        elif request.summary_type == "recent":
            section_filter = None
            prompt_template = "Summarize the most recent medical findings and recommendations."
        else:  # full
            section_filter = None
            prompt_template = """Provide a comprehensive health summary including:
1. Current medications
2. Diagnoses and conditions
3. Recent lab results
4. Allergies (if any)
5. Follow-up recommendations"""
        
        # Retrieve relevant information
        try:
            if section_filter:
                retrieved_chunks = vector_store.patient_search(
                    patient_id=request.patient_id,
                    query=prompt_template,
                    k=20,  # Get more for comprehensive summary
                    section_type=section_filter
                )
            else:
                retrieved_chunks = vector_store.patient_search(
                    patient_id=request.patient_id,
                    query=prompt_template,
                    k=20
                )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return {
                "success": False,
                "message": "Failed to retrieve medical records",
                "error": "SEARCH_ERROR"
            }
        
        if not retrieved_chunks:
            return {
                "success": True,
                "message": "No relevant information found",
                "data": {
                    "summary": "No medical information available for summary.",
                    "summary_type": request.summary_type,
                    "documents_analyzed": 0,
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
        
        # Build context
        context_parts = []
        docs_analyzed = set()
        for chunk in retrieved_chunks:
            metadata = chunk.get('metadata', {})
            filename = metadata.get('filename', 'Unknown')
            docs_analyzed.add(filename)
            text = chunk.get('text', '')
            context_parts.append(text)
        
        context = "\n\n".join(context_parts)
        
        # Build summary prompt
        user_prompt = f"""Based on the patient's medical records, {prompt_template}

MEDICAL RECORDS:
{context}

Provide a clear, organized summary. Use markdown formatting for readability.

SUMMARY:"""
        
        # Generate summary with Groq
        try:
            response = llm_handler.generate_response(user_prompt, None)
            summary = response.get("answer", "")
        except Exception as e:
            logger.error(f"[MediVault] Groq API failed during summarization: {e}")
            return {
                "success": False,
                "message": "Failed to generate summary. Please try again later.",
                "error": "LLM_ERROR"
            }
        
        return {
            "success": True,
            "message": "Summary generated successfully",
            "data": {
                "summary": summary,
                "summary_type": request.summary_type,
                "documents_analyzed": len(docs_analyzed),
                "generated_at": datetime.utcnow().isoformat()
            }
        }
    
    except Exception as e:
        logger.error(f"[MediVault] Summarize endpoint failed: {e}")
        return {
            "success": False,
            "message": "An unexpected error occurred",
            "error": "INTERNAL_ERROR"
        }


# ========== Run Server ==========

if __name__ == "__main__":
    import os
    import uvicorn

    # reload=True is development-only — must be False in Docker/production.
    # Set APP_RELOAD=true in local .env to enable hot-reload during development.
    _reload = os.getenv("APP_RELOAD", "false").lower() == "true"

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=_reload,
        log_level="info"
    )
