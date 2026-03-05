"""
MediVault RAG Bot - Startup Verification Script
Checks all prerequisites before starting the server
Run this BEFORE uvicorn to catch configuration issues early
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def print_check(name, passed, details=""):
    """Print check result"""
    status = "✅" if passed else "❌"
    print(f"{status} {name}", end="")
    if details:
        print(f": {details}")
    else:
        print()
    return passed


def check_env_variables():
    """Check all required environment variables"""
    print("\n📋 Checking environment variables...")
    
    required_vars = [
        ("GROQ_API_KEY", "Groq API key for LLM"),
        ("MONGODB_URI", "MongoDB connection string"),
        ("MONGODB_DB_NAME", "MongoDB database name"),
        ("MONGODB_COLLECTION_NAME", "MongoDB collection name"),
        ("VECTOR_INDEX_NAME", "MongoDB Atlas vector search index name"),
    ]
    
    all_present = True
    for var_name, description in required_vars:
        value = os.getenv(var_name)
        if value and value.strip():
            print_check(f"{var_name}", True, "present")
        else:
            print_check(f"{var_name}", False, f"MISSING - {description}")
            all_present = False
    
    if all_present:
        print_check(".env variables", True, "all present")
        return True
    else:
        print("\n❌ Missing environment variables!")
        print("   → Check your .env file")
        print("   → Compare with .env.template")
        return False


def check_mongodb_connection():
    """Check MongoDB connection"""
    print("\n🗄️  Checking MongoDB connection...")
    
    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
        
        mongodb_uri = os.getenv("MONGODB_URI")
        if not mongodb_uri:
            print_check("MongoDB", False, "MONGODB_URI not set")
            return False
        
        # Try to connect with short timeout
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        
        # Ping to verify connection
        client.admin.command('ping')
        
        # Get database and collection info
        db_name = os.getenv("MONGODB_DB_NAME", "medical_ragbot")
        collection_name = os.getenv("MONGODB_COLLECTION_NAME", "medical_vectors")
        
        db = client[db_name]
        collection = db[collection_name]
        
        # Count documents
        doc_count = collection.count_documents({})
        
        print_check("MongoDB", True, f"connected to {db_name}.{collection_name}")
        print(f"   → {doc_count} documents in collection")
        
        client.close()
        return True
        
    except ConnectionFailure:
        print_check("MongoDB", False, "connection failed")
        print("   → Check MONGODB_URI in .env")
        print("   → Verify network connectivity")
        print("   → Check MongoDB Atlas IP whitelist (add 0.0.0.0/0 for testing)")
        return False
    except ServerSelectionTimeoutError:
        print_check("MongoDB", False, "connection timeout")
        print("   → Check if MongoDB Atlas cluster is running")
        print("   → Verify MONGODB_URI is correct")
        return False
    except Exception as e:
        print_check("MongoDB", False, f"error: {e}")
        return False


def check_vector_index():
    """Check if MongoDB vector search index exists and is ready"""
    print("\n🔍 Checking vector search index...")
    
    try:
        from pymongo import MongoClient
        
        mongodb_uri = os.getenv("MONGODB_URI")
        db_name = os.getenv("MONGODB_DB_NAME", "medical_ragbot")
        collection_name = os.getenv("MONGODB_COLLECTION_NAME", "medical_vectors")
        index_name = os.getenv("VECTOR_INDEX_NAME", "vector_index")
        
        client = MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000)
        db = client[db_name]
        
        # List search indexes (Atlas-specific command)
        try:
            # This command works on MongoDB Atlas
            result = db.command({
                "aggregate": collection_name,
                "pipeline": [{"$listSearchIndexes": {}}],
                "cursor": {}
            })
            
            indexes = result.get("cursor", {}).get("firstBatch", [])
            
            # Find our vector index
            vector_index = None
            for idx in indexes:
                if idx.get("name") == index_name:
                    vector_index = idx
                    break
            
            if vector_index:
                status = vector_index.get("status", "unknown")
                if status == "READY":
                    print_check("Vector index", True, f"{index_name} is READY")
                    client.close()
                    return True
                else:
                    print_check("Vector index", False, f"{index_name} status: {status}")
                    print(f"   → Wait for index to finish building in MongoDB Atlas")
                    client.close()
                    return False
            else:
                print_check("Vector index", False, f"{index_name} not found")
                print(f"   → Create vector search index in MongoDB Atlas")
                print(f"   → Database: {db_name}, Collection: {collection_name}")
                print(f"   → Index name: {index_name}")
                print(f"   → Path: embedding, Dimensions: 768, Similarity: cosine")
                client.close()
                return False
                
        except Exception as e:
            # If listSearchIndexes fails, might not be Atlas or index doesn't exist
            print_check("Vector index", False, "cannot verify (might not be Atlas)")
            print(f"   ⚠️  Warning: {e}")
            print(f"   → Ensure vector index '{index_name}' exists in Atlas")
            client.close()
            return False
            
    except Exception as e:
        print_check("Vector index", False, f"error: {e}")
        return False


def check_groq_api():
    """Check if Groq API key is valid"""
    print("\n🤖 Checking Groq API...")
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print_check("Groq API", False, "GROQ_API_KEY not set")
        return False
    
    try:
        import requests
        
        # Make a minimal test call to Groq
        headers = {
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            print_check("Groq API", True, "key valid")
            return True
        elif response.status_code == 401:
            print_check("Groq API", False, "invalid API key")
            print("   → Check GROQ_API_KEY in .env")
            print("   → Get a new key from https://console.groq.com")
            return False
        elif response.status_code == 429:
            print_check("Groq API", False, "rate limit exceeded")
            print("   → Wait a moment and try again")
            print("   → API key is valid but hitting rate limits")
            return True  # Key is valid, just rate limited
        else:
            print_check("Groq API", False, f"unexpected response: {response.status_code}")
            print(f"   → Response: {response.text[:100]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_check("Groq API", False, "connection failed")
        print("   → Check internet connectivity")
        return False
    except Exception as e:
        print_check("Groq API", False, f"error: {e}")
        return False


def check_embedding_model():
    """Check if embedding model can be loaded"""
    print("\n🧠 Checking embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        print(f"   Loading {model_name}...")
        
        # This will download the model on first run (~400MB)
        model = SentenceTransformer(model_name)
        dimension = model.get_sentence_embedding_dimension()
        
        print_check("Embeddings model", True, f"{model_name} loaded ({dimension}D)")
        return True
        
    except Exception as e:
        print_check("Embeddings model", False, f"error: {e}")
        print("   → Run: pip install sentence-transformers")
        return False


def check_directories():
    """Check if required directories exist"""
    print("\n📁 Checking directories...")
    
    directories = [
        "data",
        "data/raw_pdfs",
        "data/processed_text"
    ]
    
    all_ok = True
    for dir_path in directories:
        path = Path(dir_path)
        if path.exists():
            print_check(dir_path, True, "exists")
        else:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print_check(dir_path, True, "created")
            except Exception as e:
                print_check(dir_path, False, f"cannot create: {e}")
                all_ok = False
    
    # Check if data/raw_pdfs is writable
    try:
        test_file = Path("data/raw_pdfs/.test_write")
        test_file.write_text("test")
        test_file.unlink()
        print_check("data/raw_pdfs writable", True)
    except Exception as e:
        print_check("data/raw_pdfs writable", False, f"error: {e}")
        all_ok = False
    
    return all_ok


def main():
    """Run all checks"""
    print("=" * 60)
    print("MediVault RAG Bot - Startup Verification")
    print("=" * 60)
    
    checks = []
    
    # Run all checks
    checks.append(check_env_variables())
    checks.append(check_mongodb_connection())
    checks.append(check_vector_index())
    checks.append(check_groq_api())
    checks.append(check_embedding_model())
    checks.append(check_directories())
    
    # Summary
    passed = sum(checks)
    total = len(checks)
    
    print("\n" + "=" * 60)
    print("SETUP VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"✅ {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! Ready to start the server.")
        print("\nNext steps:")
        print("  1. uvicorn app.main:app --reload --port 8000")
        print("  2. python test_routes.py (in new terminal)")
        return 0
    else:
        print("\n⚠️  Fix the issues above before starting the server.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
