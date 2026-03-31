import io
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import PyPDF2

# 1. HuggingFace Modelini Yükle (Vektör Boyutu: 384)
# İlanda belirtilen "HuggingFace aşinalığı" için bu model standarttır.
print("DEBUG: Model yükleniyor (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Qdrant Bağlantısı
q_client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "document_chunks"

# Koleksiyon Kontrolü ve Oluşturma
def ensure_collection():
    try:
        # q_client kullanarak kontrol et
        collections = q_client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if not exists:
            print(f"DEBUG: '{COLLECTION_NAME}' koleksiyonu oluşturuluyor...")
            q_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            print("DEBUG: Koleksiyon başarıyla oluşturuldu.")
    except Exception as e:
        print(f"HATA: Qdrant bağlantısı kurulamadı: {e}")

ensure_collection()

def process_pdf_to_vectors(file_content, filename):
    """
    PDF içeriğini okur, parçalar, vektöre çevirir ve Qdrant'a yükler.
    """
    try:
        # PDF'ten metni çıkar
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        print(f"DEBUG: Okunan metin uzunluğu: {len(text)} karakter.")

        if len(text.strip()) == 0:
            print("HATA: PDF içeriği boş veya okunamaz (Resim tabanlı olabilir).")
            return 0

        # Metni parçalara böl (Chunking)
        # 500 karakterlik parçalar, anlam bütünlüğü için idealdir.
        chunk_size = 500
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        print(f"DEBUG: {len(chunks)} adet parça oluşturuldu.")

        points = []
        for idx, chunk in enumerate(chunks):
            # Metni vektöre (embedding) çevir
            vector = model.encode(chunk).tolist()
            
            # Benzersiz bir ID oluştur (Qdrant UUID veya Integer bekler)
            point_id = str(uuid.uuid4())
            
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk, 
                    "source": filename,
                    "chunk_index": idx
                }
            ))

        # Qdrant'a toplu yükleme (Upsert)
        operation_info = q_client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points
        )
        
        print(f"DEBUG: Qdrant Yükleme Durumu: {operation_info.status}")
        return len(chunks)

    except Exception as e:
        print(f"HATA: process_pdf_to_vectors sırasında hata: {str(e)}")
        return 0

def generate_answer_with_context(query, context_chunks):
    """
    Bulunan parçaları birleştirip bir LLM Prompt'u hazırlar.
    """
    context_text = "\n\n".join([f"Kaynak ({c['source']}): {c['text']}" for c in context_chunks])
    
    prompt = f"""
    Sen teknik bir döküman asistanısın. Aşağıdaki bağlamı kullanarak soruyu cevapla.
    Eğer bilgi bağlamda yoksa, bilmediğini belirt.
    
    BAĞLAM:
    {context_text}
    
    SORU: {query}
    
    CEVAP:
    """
    return prompt