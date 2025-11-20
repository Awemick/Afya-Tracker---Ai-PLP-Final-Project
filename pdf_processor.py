import os
import fitz  # PyMuPDF for PDF text extraction
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import json
from typing import List, Dict, Tuple
import re

class PDFKnowledgeBase:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.model = None

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF and split into chunks"""
        doc = fitz.open(pdf_path)
        chunks = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()

            # Clean and split text into sentences
            sentences = self.split_into_sentences(text)

            for sentence in sentences:
                if len(sentence.strip()) > 20:  # Filter out very short chunks
                    chunks.append({
                        'text': sentence.strip(),
                        'source': os.path.basename(pdf_path),
                        'page': page_num + 1,
                        'chunk_id': len(chunks)
                    })

        doc.close()
        return chunks

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, handling both English and Swahili"""
        # Handle common sentence endings
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter very short fragments
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def process_all_pdfs(self, pdf_directory: str = "Ai datasets"):
        """Process all PDFs in the directory"""
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]

        all_chunks = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            print(f"Processing {pdf_file}...")
            chunks = self.extract_text_from_pdf(pdf_path)
            all_chunks.extend(chunks)
            print(f"Extracted {len(chunks)} text chunks from {pdf_file}")

        self.documents = all_chunks
        print(f"Total chunks extracted: {len(self.documents)}")

        return self.documents

    def create_embeddings(self, model_name: str = "sentence-transformers/LaBSE"):
        """Create embeddings for all text chunks using multilingual model"""
        print(f"Loading multilingual model: {model_name}")
        self.model = SentenceTransformer(model_name)

        texts = [doc['text'] for doc in self.documents]
        print(f"Creating embeddings for {len(texts)} text chunks...")

        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Embeddings created with shape: {self.embeddings.shape}")

        return self.embeddings

    def save_knowledge_base(self, output_dir: str = "knowledge_base"):
        """Save the knowledge base for later use"""
        os.makedirs(output_dir, exist_ok=True)

        # Save documents
        with open(os.path.join(output_dir, 'documents.json'), 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

        # Save embeddings
        np.save(os.path.join(output_dir, 'embeddings.npy'), self.embeddings)

        # Save model info
        model_info = {
            'model_name': 'sentence-transformers/LaBSE',
            'embedding_dim': self.embeddings.shape[1],
            'num_documents': len(self.documents)
        }

        with open(os.path.join(output_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)

        print(f"Knowledge base saved to {output_dir}")

    def load_knowledge_base(self, input_dir: str = "knowledge_base"):
        """Load a previously saved knowledge base"""
        # Load documents
        with open(os.path.join(input_dir, 'documents.json'), 'r', encoding='utf-8') as f:
            self.documents = json.load(f)

        # Load embeddings
        self.embeddings = np.load(os.path.join(input_dir, 'embeddings.npy'))

        # Load model
        self.model = SentenceTransformer('sentence-transformers/LaBSE')

        print(f"Knowledge base loaded: {len(self.documents)} documents")

    def search_similar(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for most similar documents to the query"""
        if self.model is None or self.embeddings is None:
            raise ValueError("Knowledge base not loaded. Call load_knowledge_base() first.")

        # Encode query
        query_embedding = self.model.encode([query])[0]

        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))

        return results

    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Generate an answer based on retrieved documents with medical disclaimers"""
        relevant_texts = [doc['text'] for doc in context_docs]

        # Find the most relevant sentence containing keywords from the query
        query_words = set(query.lower().split())
        best_match = ""
        best_score = 0

        for text in relevant_texts:
            text_words = set(text.lower().split())
            overlap = len(query_words.intersection(text_words))
            score = overlap / len(query_words) if query_words else 0

            if score > best_score:
                best_score = score
                best_match = text

        # Medical disclaimer - CRITICAL for healthcare apps
        disclaimer = "\n\n⚠️ IMPORTANT: This is general information only and not a substitute for professional medical advice. Please consult your healthcare provider for personalized guidance."

        if best_match:
            return f"Based on general medical guidelines: {best_match}{disclaimer}"
        else:
            # Fallback: return the most relevant document
            fallback_text = relevant_texts[0] if relevant_texts else 'I recommend consulting a healthcare provider for personalized advice.'
            return f"According to general medical resources: {fallback_text}{disclaimer}"

def main():
    # Initialize knowledge base
    kb = PDFKnowledgeBase()

    # Process PDFs
    print("Starting PDF processing...")
    kb.process_all_pdfs()

    # Create multilingual embeddings
    print("Creating embeddings...")
    kb.create_embeddings()

    # Save knowledge base
    kb.save_knowledge_base()

    # Test search
    test_queries = [
        "How often should I feel my baby move?",
        "What foods should I eat during pregnancy?",
        "When should I call the doctor?",
        "Ninawezaje kupima mtoto anavyotembea?",  # Swahili: How to check if baby is moving?
        "Ni chakula gani kizuri wakati wa ujauzito?"  # Swahili: What foods are good during pregnancy?
    ]

    print("\nTesting search functionality:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = kb.search_similar(query, top_k=2)
        for doc, score in results:
            print(".3f")

if __name__ == "__main__":
    main()