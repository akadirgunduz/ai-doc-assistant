📝 AI-Powered Semantic Search & Doc-Assistant (RAG Pipeline)
Bu proje, modern yapay zeka tekniklerini ve büyük veri araçlarını kullanarak dokümanlar üzerinde anlam odaklı (semantic) arama ve soru-cevap işlemi gerçekleştiren uçtan uca bir RAG (Retrieval-Augmented Generation) sistemidir.

🚀 Öne Çıkan Özellikler
Vector DB (Qdrant): En güncel Query API (v1.17+) kullanılarak milisaniyeler içinde anlamsal arama ve veri indeksleme.

Object Storage (MinIO): Dokümanların S3 uyumlu, ölçeklenebilir ve güvenli bir ortamda fiziksel olarak saklanması.

HuggingFace Integration: all-MiniLM-L6-v2 embedding modeli ile metinlerin yüksek başarımlı vektörlere dönüştürülmesi.

Asenkron FastAPI: Yüksek performanslı, async/await yapılı ve otomatik Swagger dökümantasyonlu Backend mimarisi.

Containerization: Tüm altyapının (Qdrant, MinIO) Docker Compose ile izole ve taşınabilir şekilde kurgulanması.

🏗️ Sistem Mimarisi ve Akış
Proje, karmaşık verileri anlamlı bilgiye dönüştüren 4 ana fazdan oluşur:

Ingestion (Veri Girişi): Kullanıcı tarafından yüklenen PDF dosyaları doğrulanır ve MinIO bucket'larına fiziksel olarak kaydedilir.

Processing (İşleme): PyPDF2 ile metinler çıkarılır ve anlamsal bütünlüğü korumak adına 500 karakterlik parçalara (chunks) ayrılır.

Embedding & Indexing: Her parça vektör uzayına aktarılır ve Qdrant üzerinde PointStruct yapısıyla indekslenir.

Retrieval (Geri Getirme): Kullanıcı sorguları vektörize edilerek Qdrant üzerinde query_points ile en alakalı bağlamlar (context) bulunur.

🛠️ Teknik Zorluklar ve Çözümler (İşe Alımcı İçin Önemli Not)
SDK Metot Uyumluluğu: Qdrant v1.17+ sürümüyle gelen SDK değişiklikleri analiz edilerek, eski search metodu yerine daha kapsayıcı olan query_points yapısına başarılı bir geçiş sağlanmıştır.

Bellek Yönetimi: Dosya okuma işlemlerinde io.BytesIO ve stream yapıları kullanılarak bellek verimliliği optimize edilmiştir.

💻 Kurulum ve Çalıştırma
1. Altyapıyı Başlat (Docker)
Bash
docker-compose up -d
2. Bağımlılıkları Yükle
Bash
pip install -r requirements.txt
3. Uygulamayı Başlat
Bash
uvicorn src.main:app --reload
📬 İletişim & Geliştirici
Ad Soyad: Abdulkadir Gündüz

Eğitim: Kocaeli Üniversitesi - Bilgisayar Mühendisliği

İlgi Alanları: Python, Agentic AI, Büyük Veri Araçları
