"""
MediVault RAG Bot - Local Route Testing Script
Tests all routes sequentially with realistic scenarios
"""
import httpx
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_PATIENT_ID = "test_patient_001"
doc_id = None  # will be set after upload test


def print_result(test_name, passed, response=None, error=None):
    """Print test result with details"""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{status} — {test_name}")
    if not passed:
        if response:
            print(f"  Status code: {response.status_code}")
            try:
                print(f"  Response: {json.dumps(response.json(), indent=2)}")
            except:
                print(f"  Response text: {response.text}")
        if error:
            print(f"  Error: {error}")


# TEST 1: Health Check
def test_health():
    """Test health endpoint - must pass before other tests"""
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=30)
        data = r.json()
        passed = (
            r.status_code == 200 and
            data.get("success") == True and
            data.get("data", {}).get("mongodb_connected") == True and
            data.get("data", {}).get("groq_configured") == True
        )
        print_result("GET /health", passed, r if not passed else None)
        if not passed:
            print("  ⚠️  Fix health check before proceeding")
            print("  → Check MongoDB connection in .env")
            print("  → Verify GROQ_API_KEY is set")
            exit(1)
        return True
    except httpx.ConnectError as e:
        print_result("GET /health", False, error=str(e))
        print("  ⚠️  Cannot connect to server. Is it running?")
        print("  → Run: uvicorn app.main:app --reload --port 8000")
        exit(1)
    except Exception as e:
        print_result("GET /health", False, error=str(e))
        exit(1)


# TEST 2: Upload PDF
def test_upload_pdf():
    """Test PDF upload with patient_id"""
    global doc_id
    
    # Check if real test PDF exists
    test_pdf_path = Path("data/raw_pdfs/test_prescription.pdf")
    
    if test_pdf_path.exists():
        # Use real PDF
        with open(test_pdf_path, "rb") as f:
            test_content = f.read()
        print("  Using real test PDF")
    else:
        # Fallback to minimal PDF
        test_content = b"""%PDF-1.4
Patient: Test Patient
Date: 2025-03-20
Doctor: Dr. Arjun Mehta

Diagnosis: Hypertension (I10)

Medications:
- Amlodipine 5mg - once daily in the morning
- Metoprolol 25mg - twice daily with meals

Lab Results:
- Blood Pressure: 145/92 mmHg (HIGH)
- Cholesterol: 210 mg/dL (BORDERLINE HIGH)

Follow-up: 2 weeks
"""
        print("  Using minimal test PDF (run create_test_pdf.py for better testing)")
    
    try:
        r = httpx.post(
            f"{BASE_URL}/ingest/pdf",
            data={
                "patient_id": TEST_PATIENT_ID,
                "document_type": "prescription",
                "document_date": "2025-03-20",
            },
            files={"file": ("test_prescription.pdf", test_content, "application/pdf")},
            timeout=120  # embedding model is slow on first load
        )
        data = r.json()
        passed = r.status_code == 200 and data.get("success") == True
        if passed:
            doc_id = data.get("data", {}).get("doc_id")
            print(f"  Document ID: {doc_id}")
            print(f"  Chunks created: {data.get('data', {}).get('chunks_created', 0)}")
        print_result("POST /ingest/pdf", passed, r if not passed else None)
        return passed
    except Exception as e:
        print_result("POST /ingest/pdf", False, error=str(e))
        return False


# TEST 3: Check Ingest Status
def test_ingest_status():
    """Test getting patient's document list"""
    try:
        r = httpx.get(f"{BASE_URL}/ingest/status/{TEST_PATIENT_ID}", timeout=30)
        data = r.json()
        passed = (
            r.status_code == 200 and
            data.get("success") == True and
            data.get("data", {}).get("total_documents", 0) >= 1
        )
        if passed:
            total_docs = data.get("data", {}).get("total_documents", 0)
            total_chunks = data.get("data", {}).get("total_chunks", 0)
            print(f"  Total documents: {total_docs}")
            print(f"  Total chunks: {total_chunks}")
        print_result("GET /ingest/status/{patient_id}", passed, r if not passed else None)
        return passed
    except Exception as e:
        print_result("GET /ingest/status/{patient_id}", False, error=str(e))
        return False


# TEST 4: Chat Query
def test_chat_query():
    """Test RAG query endpoint"""
    try:
        r = httpx.post(
            f"{BASE_URL}/chat/query",
            json={
                "patient_id": TEST_PATIENT_ID,
                "question": "What medications are mentioned in my records?",
                "conversation_history": []
            },
            timeout=60
        )
        data = r.json()
        passed = (
            r.status_code == 200 and
            data.get("success") == True and
            "answer" in data.get("data", {})
        )
        if passed:
            answer = data['data']['answer']
            print(f"  Answer preview: {answer[:150]}...")
            sources = data['data'].get('sources', [])
            print(f"  Sources: {len(sources)}")
        print_result("POST /chat/query", passed, r if not passed else None)
        return passed
    except Exception as e:
        print_result("POST /chat/query", False, error=str(e))
        return False


# TEST 5: Chat Query with section filter
def test_chat_query_with_filter():
    """Test query with section filter"""
    try:
        r = httpx.post(
            f"{BASE_URL}/chat/query",
            json={
                "patient_id": TEST_PATIENT_ID,
                "question": "What is my diagnosis?",
                "conversation_history": [],
                "section_filter": "medications"
            },
            timeout=60
        )
        data = r.json()
        passed = r.status_code == 200 and data.get("success") == True
        if passed:
            print(f"  Query with section_filter processed successfully")
        print_result("POST /chat/query (with section_filter)", passed, r if not passed else None)
        return passed
    except Exception as e:
        print_result("POST /chat/query (section_filter)", False, error=str(e))
        return False


# TEST 6: Chat Summarize
def test_chat_summarize():
    """Test health summary generation"""
    try:
        r = httpx.post(
            f"{BASE_URL}/chat/summarize",
            json={
                "patient_id": TEST_PATIENT_ID,
                "summary_type": "full"
            },
            timeout=60
        )
        data = r.json()
        passed = (
            r.status_code == 200 and
            data.get("success") == True and
            "summary" in data.get("data", {})
        )
        if passed:
            summary = data['data']['summary']
            print(f"  Summary preview: {summary[:150]}...")
            docs_analyzed = data['data'].get('documents_analyzed', 0)
            print(f"  Documents analyzed: {docs_analyzed}")
        print_result("POST /chat/summarize", passed, r if not passed else None)
        return passed
    except Exception as e:
        print_result("POST /chat/summarize", False, error=str(e))
        return False


# TEST 7: Query non-existent patient (should return NO_DOCUMENTS error)
def test_no_documents_error():
    """Test error handling for patient with no documents"""
    try:
        r = httpx.post(
            f"{BASE_URL}/chat/query",
            json={
                "patient_id": "nonexistent_patient_xyz",
                "question": "What medications am I taking?",
                "conversation_history": []
            },
            timeout=30
        )
        data = r.json()
        # Should return 200 with success=False and error=NO_DOCUMENTS
        passed = (
            r.status_code == 200 and
            data.get("success") == False and
            data.get("error") == "NO_DOCUMENTS"
        )
        if passed:
            print(f"  Error handling working correctly")
        print_result("POST /chat/query (no docs — expect NO_DOCUMENTS)", passed, r if not passed else None)
        return passed
    except Exception as e:
        print_result("POST /chat/query (no docs)", False, error=str(e))
        return False


# TEST 8: Cleanup — delete test document
def test_delete_document():
    """Test document deletion"""
    if not doc_id:
        print("\n⚠️  SKIP — DELETE /ingest/document (no doc_id from upload test)")
        return True
    try:
        r = httpx.delete(
            f"{BASE_URL}/ingest/document/{TEST_PATIENT_ID}/{doc_id}",
            timeout=30
        )
        data = r.json()
        passed = r.status_code == 200 and data.get("success") == True
        if passed:
            chunks_deleted = data.get("data", {}).get("chunks_deleted", 0)
            print(f"  Chunks deleted: {chunks_deleted}")
        print_result("DELETE /ingest/document/{patient_id}/{doc_id}", passed, r if not passed else None)
        return passed
    except Exception as e:
        print_result("DELETE /ingest/document", False, error=str(e))
        return False


# TEST 9: Verify deletion worked
def test_verify_deletion():
    """Verify document was successfully deleted"""
    try:
        r = httpx.get(f"{BASE_URL}/ingest/status/{TEST_PATIENT_ID}", timeout=30)
        data = r.json()
        total = data.get("data", {}).get("total_documents", -1)
        passed = r.status_code == 200 and total == 0
        if passed:
            print(f"  Confirmed: no documents remain for test patient")
        print_result("GET /ingest/status (verify deletion)", passed, r if not passed else None)
        return passed
    except Exception as e:
        print_result("GET /ingest/status (verify deletion)", False, error=str(e))
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("MediVault RAG Bot — Local Route Tests")
    print("=" * 60)
    print(f"Testing against: {BASE_URL}")
    print(f"Test patient ID: {TEST_PATIENT_ID}")
    print("\nNOTE: First run will be slower (embedding model downloads)")
    print("=" * 60)
    
    results = []
    start_time = time.time()
    
    # Run tests in sequence
    results.append(("Health Check", test_health()))
    results.append(("Upload PDF", test_upload_pdf()))
    results.append(("Ingest Status", test_ingest_status()))
    results.append(("Chat Query", test_chat_query()))
    results.append(("Chat Query + Filter", test_chat_query_with_filter()))
    results.append(("Chat Summarize", test_chat_summarize()))
    results.append(("No Docs Error", test_no_documents_error()))
    results.append(("Delete Document", test_delete_document()))
    results.append(("Verify Deletion", test_verify_deletion()))
    
    elapsed = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, result in results:
        print(f"  {'✅' if result else '❌'} {name}")
    
    print(f"\n📊 {passed}/{total} tests passed")
    print(f"⏱️  Total time: {elapsed:.1f}s")
    
    if passed == total:
        print("\n🎉 All tests passed — ready to deploy to Render!")
    else:
        print("\n⚠️  Fix failing tests before deploying.")
        exit(1)
