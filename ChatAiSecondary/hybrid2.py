import os
import sys
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Any
import argparse
import platform
import PyPDF2
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import re
import warnings
import google.generativeai as genai
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()


# ==================== Belge Okuyucu ====================
class DocumentProcessor:
    """Farklı formatlardaki belgeleri işleyen sınıf"""

    def __init__(self):
        self.supported_formats = ['.txt', '.pdf', '.docx']

    def read_txt(self, file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            encodings = ['latin-1', 'cp1254', 'iso-8859-9']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            raise Exception(f"Dosya okunamadı: {file_path}")

    def read_pdf(self, file_path: str) -> str:
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n\f"
        except Exception as e:
            raise Exception(f"PDF okunamadı: {e}")
        return text

    def read_docx(self, file_path: str) -> str:
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"DOCX okunamadı: {e}")

    def read_document(self, file_path: str) -> str:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

        extension = file_path.suffix.lower()

        if extension == '.txt':
            return self.read_txt(str(file_path))
        elif extension == '.pdf':
            return self.read_pdf(str(file_path))
        elif extension == '.docx':
            return self.read_docx(str(file_path))
        else:
            raise ValueError(f"Desteklenmeyen format: {extension}")


# ==================== Chunker ====================
class TextChunker:
    """Metni parçalara bölen sınıf"""

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            if len(paragraph) > self.chunk_size:
                sentences = re.split(r'[.!?]+', paragraph)
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > self.chunk_size:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence
            else:
                if len(current_chunk) + len(paragraph) > self.chunk_size:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph
                else:
                    current_chunk += "\n\n" + paragraph

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if len(chunk.strip()) > 50]


# ==================== Vektör Store ====================
class VectorStore:
    """Belge parçalarını vektörleştiren ve saklayan sınıf"""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        print(f"Embedding modeli yükleniyor: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = None

    def add_documents(self, chunks: List[str]):
        print(f"{len(chunks)} parça vektörleştiriliyor...")
        self.chunks = chunks
        self.embeddings = self.model.encode(chunks, show_progress_bar=True)
        print("Vektörleştirme tamamlandı!")

    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        if self.embeddings is None:
            return []
        # Çok dilli model için özel prefix eklenir
        query_embedding = self.model.encode([f"query: {query}"])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(self.chunks[i], similarities[i]) for i in top_indices]
        return results


# ==================== Page Index Store ====================
class PageIndexStore:
    """Sayfa bazlı indeksleme yapan sınıf"""

    def __init__(self):
        self.pages = []  # (sayfa_no, sayfa_metni)

    def add_document_pages(self, text: str):
        pages = text.split("\f")
        if len(pages) == 1:
            pages = text.split("\n\n\n")
        if len(pages) == 1:
            pages = [text]

        self.pages.extend([(i + 1, p.strip()) for i, p in enumerate(pages) if p.strip()])

    def search(self, query: str, top_k: int = 3) -> List[tuple]:
        results = []
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        for page_no, content in self.pages:
            content_words = set(re.findall(r'\b\w+\b', content.lower()))
            score = len(query_words.intersection(content_words))
            if score > 0:
                results.append((page_no, content, score))
        results = sorted(results, key=lambda x: x[2], reverse=True)
        return results[:top_k]


# ==================== Gemini Client ====================
class GeminiClient:
    """Gemini API ile iletişim kuran sınıf"""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        if not api_key:
            raise ValueError("Gemini API anahtarı sağlanmalıdır.")
        self.api_key = api_key
        self.model_name = model_name
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            print(f"✅ Gemini sistemi hazır! (Model: {self.model_name})")
        except Exception as e:
            raise Exception(f"Gemini API yapılandırma hatası: {e}")

    def generate(self, prompt: str, context: str = "") -> str:
        full_prompt = f"""
        Aşağıdaki bağlam bilgilerini kullanarak soruya detaylı ve net bir şekilde cevap verin. 
        Eğer bağlam bilgileri yeterli değilse, bunu belirtin. Yanıtı Türkçe olarak oluşturun.

        Bağlam bilgileri:
        {context}

        Soru: {prompt}
        """
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"❌ Gemini API hatası: {e}"


# ==================== QA Sistemi ====================
class DocumentQASystem:
    def __init__(self, api_key: str, embedding_model: str, gemini_model: str):
        self.doc_processor = DocumentProcessor()
        self.chunker = TextChunker(chunk_size=800, overlap=100)
        self.vector_store = VectorStore(embedding_model)
        self.page_index = PageIndexStore()
        self.gemini_client = GeminiClient(api_key, gemini_model)
        self.loaded_documents = []

    def load_document(self, file_path: str):
        print(f"📄 Belge yükleniyor: {file_path}")
        text = self.doc_processor.read_document(file_path)
        self.page_index.add_document_pages(text)
        chunks = self.chunker.chunk_text(text)
        self.vector_store.add_documents(chunks)
        self.loaded_documents.append(file_path)
        print(f"✅ {Path(file_path).name} yüklendi ({len(chunks)} chunk, {len(text)} karakter)")

    def load_multiple_documents(self, file_paths: List[str]):
        all_chunks = []
        for file_path in file_paths:
            print(f"📄 Belge yükleniyor: {file_path}")
            text = self.doc_processor.read_document(file_path)
            self.page_index.add_document_pages(text)
            chunks = self.chunker.chunk_text(text)
            all_chunks.extend(chunks)
            self.loaded_documents.append(file_path)
            print(f"✅ {Path(file_path).name} için chunk'lar hazır.")
        self.vector_store.add_documents(all_chunks)
        print("✅ Tüm belgeler başarıyla yüklendi!")

    def answer_question(self, question: str) -> str:
        if not self.loaded_documents:
            return "⚠️ Henüz hiç belge yüklenmedi."

        # 1️⃣ Page Index
        page_hits = self.page_index.search(question, top_k=2)
        context_parts = []
        if page_hits:
            print("\n📑 Page Index ile bulunan sayfalar:")
            for page_no, content, score in page_hits:
                print(f"  - Sayfa {page_no} (Skor: {score})")
                context_parts.append(f"[Sayfa {page_no}]\n{content[:800]}...")

        # 2️⃣ Embedding
        relevant_chunks = self.vector_store.search(question, top_k=3)
        if relevant_chunks:
            print("\n🔍 Embedding ile bulunan chunklar:")
            for i, (chunk, score) in enumerate(relevant_chunks, 1):
                print(f"  - Chunk {i} (Benzerlik: {score:.3f})")
                context_parts.append(f"[Chunk {i}]\n{chunk}")

        if not context_parts:
            return "🔍 İlgili bilgi bulunamadı."

        context = "\n\n".join(context_parts[:4])
        response = self.gemini_client.generate(question, context)
        return response

    def interactive_mode(self):
        print("\n" + "=" * 50)
        print("🤖 Belge Soru-Cevap Sistemi (Çok Dilli Destek)")
        print("=" * 50)
        print("📋 Komutlar:")
        print("   • 'çıkış' veya 'quit': Programdan çık")
        print("   • 'belgeler': Yüklü belgeleri listele")
        print("   • 'yükle [dosya_yolu]': Yeni belge yükle")
        print("   • Soru sorun!\n")

        while True:
            try:
                question = input("🤔 Sorunuz: ").strip()

                if question.lower() in ['çıkış', 'quit', 'exit']:
                    print("👋 Görüşmek üzere!")
                    break

                if question.lower() == 'belgeler':
                    if self.loaded_documents:
                        print("📚 Yüklü belgeler:")
                        for i, doc in enumerate(self.loaded_documents, 1):
                            print(f"   {i}. {Path(doc).name}")
                    else:
                        print("⚠️ Henüz belge yüklenmedi.")
                    continue

                if question.lower().startswith('yükle '):
                    file_path = question[6:].strip()
                    try:
                        self.load_document(file_path)
                    except Exception as e:
                        print(f"❌ Belge yükleme hatası: {e}")
                    continue

                if not question:
                    continue

                print("\n⏳ Yanıt hazırlanıyor...")
                answer = self.answer_question(question)
                print(f"\n🤖 Cevap:\n{answer}\n")

            except KeyboardInterrupt:
                print("\n\n⏹️ Program sonlandırıldı.")
                break
            except Exception as e:
                print(f"❌ Hata: {e}")


# ==================== Main ====================
def main():
    parser = argparse.ArgumentParser(description="Belge Soru-Cevap Sistemi")
    parser.add_argument("--belgeler", "-b", nargs="+", help="Yüklenecek belge dosyaları")
    parser.add_argument("--embedding_model", "-e", default="intfloat/multilingual-e5-large",
                        help="Kullanılacak embedding modeli (örn. 'intfloat/multilingual-e5-large')")
    parser.add_argument("--gemini_model", "-g", default="gemini-2.0-flash",
                        help="Kullanılacak Gemini modeli (örn. 'gemini-2.0-flash' veya 'gemini-2.0-pro')")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Hata: 'GEMINI_API_KEY' ortam değişkeni ayarlanmamış.")
        return

    try:
        print("🚀 Sistem başlatılıyor...")
        qa_system = DocumentQASystem(
            api_key=api_key,
            embedding_model=args.embedding_model,
            gemini_model=args.gemini_model
        )

        if args.belgeler:
            qa_system.load_multiple_documents(args.belgeler)
        else:
            print("ℹ️ Belge belirtilmedi. Etkileşimli modda manuel yükleme yapabilirsiniz.")

        qa_system.interactive_mode()

    except KeyboardInterrupt:
        print("\n⏹️ Program kullanıcı tarafından sonlandırıldı.")
    except Exception as e:
        print(f"❌ Kritik hata: {e}")


if __name__ == "__main__":
    main()