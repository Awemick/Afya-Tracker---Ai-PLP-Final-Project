#!/usr/bin/env python3
"""
Cloud-based chatbot service for mobile deployment
Uses Firebase Functions or similar serverless architecture
"""

import os
import json
import requests
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle

class CloudChatbotService:
    """Cloud-based chatbot that can be deployed as serverless function"""

    def __init__(self):
        self.model = None
        self.index = None
        self.documents = []
        self.embeddings = None

    def initialize(self):
        """Initialize the service (called once during cold start)"""
        print("Initializing cloud chatbot service...")

        # Load pre-computed optimized embeddings
        try:
            self.embeddings = np.load('knowledge_base/embeddings_optimized.npy')

            # Reconstruct full embeddings from PCA + quantization
            with open('knowledge_base/pca_model.pkl', 'rb') as f:
                pca = pickle.load(f)

            # Dequantize
            embeddings_norm = self.embeddings.astype(np.float32) / 127.0
            self.embeddings = pca.inverse_transform(embeddings_norm)

            # Load documents
            with open('knowledge_base/documents.json', 'r', encoding='utf-8') as f:
                self.documents = json.load(f)

            # Create FAISS index for fast search
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            faiss.normalize_L2(self.embeddings)  # Normalize for cosine similarity
            self.index.add(self.embeddings)

            print(f"Loaded {len(self.documents)} documents")
            print(f"Embeddings shape: {self.embeddings.shape}")

        except Exception as e:
            print(f"Error initializing service: {e}")
            # Fallback to basic functionality
            self.documents = []
            self.embeddings = None

    def search_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents using FAISS"""

        if self.index is None or self.embeddings is None:
            return []

        try:
            # Load embedding model (could be cached)
            if self.model is None:
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

            # Encode query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)

            # Search
            scores, indices = self.index.search(query_embedding, top_k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = float(score)
                    results.append(doc)

            return results

        except Exception as e:
            print(f"Search error: {e}")
            return []

    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Generate answer using retrieved context"""

        if not context_docs:
            return "I'm sorry, I don't have enough information to answer that question."

        # Simple extractive answer generation
        context_text = "\n".join([doc.get('content', '') for doc in context_docs[:2]])

        # For production, you might want to use a smaller LLM or rule-based system
        # For now, return the most relevant text chunk

        best_doc = max(context_docs, key=lambda x: x.get('similarity_score', 0))

        answer = best_doc.get('content', '').strip()

        # Truncate if too long
        if len(answer) > 500:
            answer = answer[:500] + "..."

        return answer

    def process_query(self, query: str) -> Dict[str, Any]:
        """Main endpoint for processing user queries"""

        # Search for relevant documents
        similar_docs = self.search_similar(query, top_k=3)

        # Generate answer
        answer = self.generate_answer(query, similar_docs)

        return {
            'query': query,
            'answer': answer,
            'sources': [
                {
                    'content': doc.get('content', '')[:200] + '...',
                    'source': doc.get('source', ''),
                    'page': doc.get('page', 0),
                    'similarity': doc.get('similarity_score', 0)
                }
                for doc in similar_docs
            ],
            'confidence': max([doc.get('similarity_score', 0) for doc in similar_docs]) if similar_docs else 0
        }

# Global service instance (for serverless)
service = CloudChatbotService()

def initialize_service():
    """Initialize service on cold start"""
    global service
    if not service.documents:  # Only initialize once
        service.initialize()

def handle_chatbot_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle incoming chatbot requests"""

    initialize_service()

    query = request_data.get('query', '').strip()

    if not query:
        return {'error': 'No query provided'}

    try:
        result = service.process_query(query)
        return result

    except Exception as e:
        return {'error': f'Processing failed: {str(e)}'}

# Example usage for testing
if __name__ == "__main__":
    # Test the service
    initialize_service()

    test_queries = [
        "How often should I feel my baby move?",
        "What foods should I eat during pregnancy?",
        "Ni dalili gani za kuzaa?"  # Swahili
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = handle_chatbot_request({'query': query})
        print(f"Answer: {result.get('answer', 'No answer')}")
        print(f"Sources: {len(result.get('sources', []))}")