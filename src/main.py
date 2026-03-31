import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from minio import Minio
from minio.error import S3Error
from .utils import process_pdf_to_vectors
from .utils import generate_answer_with_context

app = FastAPI(title="AI Doc-Assistant Backend")

# 1. MinIO İstemci Yapılandırması (Docker'daki bilgilerimizle)
minio_client = Minio(
    "localhost:9000",
    access_key="admin",
    secret_key="password123",
    secure=False # Lokal çalıştığımız için SSL kapalı
)

# 2. "documents" adında bir depolama alanı (bucket) oluştur
BUCKET_NAME = "pdf-documents"

def initialize_minio():
    try:
        if not minio_client.bucket_exists(BUCKET_NAME):
            minio_client.make_bucket(BUCKET_NAME)
            print(f"'{BUCKET_NAME}' bucket oluşturuldu.")
    except S3Error as e:
        print(f"MinIO Hatası: {e}")

initialize_minio()

@app.get("/")
def read_root():
    return {"status": "AI Doc-Assistant API is Running"}

@app.post("/upload-doc/")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Sadece PDF dosyaları yüklenebilir.")

    try:
        # 1. Dosyayı BİR KEZ oku
        file_content = await file.read()
        file_size = len(file_content)
        
        # 2. MinIO'ya yükle
        minio_client.put_object(
            BUCKET_NAME,
            file.filename,
            io.BytesIO(file_content), # Okuduğumuz değişkeni kullanıyoruz
            length=file_size,
            content_type=file.content_type
        )

        # 3. Vektörleştirme işlemini başlat (Return'den ÖNCE!)
        num_chunks = process_pdf_to_vectors(file_content, file.filename)
        
        # 4. ŞİMDİ yanıt dön
        return {
            "filename": file.filename,
            "size": file_size,
            "chunks_processed": num_chunks,
            "message": "Dosya hem MinIO'ya kaydedildi hem de Qdrant'a indekslendi!"
        }

    except Exception as e:
        print(f"HATA: {e}") # Debug için terminale yazdır
        raise HTTPException(status_code=500, detail=f"İşlem hatası: {str(e)}")

@app.get("/search/")
async def search_documents(query: str, limit: int = 3):
    from . import utils # utils.py'yi modül olarak içe aktar
    
    try:
        # Sorguyu vektöre çevir
        query_vector = utils.model.encode(query).tolist()
        
        # DÜZELTME: search yerine query_points kullanıyoruz
        # Yeni SDK formatında 'query' parametresi vektörü doğrudan kabul eder
        search_result = utils.q_client.query_points(
            collection_name=utils.COLLECTION_NAME,
            query=query_vector,
            limit=limit
        ).points # Sonuçlar .points içinde döner
        
        results = [
            {
                "text": res.payload["text"], 
                "score": res.score, 
                "source": res.payload["source"]
            }
            for res in search_result
        ]
        return {"query": query, "results": results}
    except Exception as e:
        print(f"Arama Hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask/")
async def ask_assistant(query: str):
    try:
        # 1. Semantik arama yap
        # Not: Kendi endpoint'ini içeriden çağırırken await kullanmalısın
        search_results = await search_documents(query, limit=3)
        
        # search_documents'ın dönüş yapısı: {"query": ..., "results": [...]}
        chunks = search_results.get("results", [])
        
        if not chunks:
            return {"answer": "Üzgünüm, bu konuda sistemde kayıtlı bir doküman bulamadım."}

        # 2. Bulunan parçaları LLM'e gönder ve cevap üret
        final_prompt = generate_answer_with_context(query, chunks)
        
        return {
            "query": query,
            "context_sources": list(set([c["source"] for c in chunks])), # Tekil kaynaklar
            "generated_prompt_ready_for_llm": final_prompt
        }
    except Exception as e:
        print(f"HATA (ask_assistant): {str(e)}")
        raise HTTPException(status_code=500, detail=f"Asistan hatası: {str(e)}")