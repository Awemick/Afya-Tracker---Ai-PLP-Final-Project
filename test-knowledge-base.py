#!/usr/bin/env python3
"""
Standalone test script for the knowledge base model
Tests the PDF knowledge base loading and querying functionality
"""

import sys
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import json

class KnowledgeBaseTester:
    def __init__(self):
        self.model = None
        self.documents = []
        self.embeddings = None
        self.knowledge_base_path = 'knowledge_base'

    def load_knowledge_base(self):
        """Load the knowledge base components"""
        try:
            print("Loading knowledge base...")

            # Load model info
            with open(os.path.join(self.knowledge_base_path, 'model_info.json'), 'r') as f:
                model_info = json.load(f)
            print(f"Model: {model_info['model_name']}")
            print(f"Embedding dimension: {model_info['embedding_dim']}")
            print(f"Number of documents: {model_info['num_documents']}")

            # Load documents
            with open(os.path.join(self.knowledge_base_path, 'documents.json'), 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            print(f"Loaded {len(self.documents)} documents")

            # Load embeddings
            self.embeddings = np.load(os.path.join(self.knowledge_base_path, 'embeddings.npy'))
            print(f"Loaded embeddings with shape: {self.embeddings.shape}")

            # Load the sentence transformer model
            self.model = SentenceTransformer(model_info['model_name'])
            print("Model loaded successfully")

        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return False
        return True

    def search_similar(self, query, top_k=3):
        """Search for similar documents"""
        if self.model is None or self.embeddings is None:
            return []

        # Encode query
        query_embedding = self.model.encode([query])[0]

        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append((self.documents[idx], similarities[idx]))

        return results

    def test_queries(self):
        """Test the knowledge base with sample queries"""
        test_queries = [
            "How often should I feel my baby move?",
            "What foods should I eat during pregnancy?",
            "When should I call the doctor?",
            "What are signs of labor?",
            "Ninawezaje kupima mtoto anavyotembea?",  # Swahili
            "Ni chakula gani kizuri wakati wa ujauzito?",  # Swahili
        ]

        print("\n" + "="*60)
        print("TESTING KNOWLEDGE BASE QUERIES")
        print("="*60)

        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")

            results = self.search_similar(query, top_k=2)

            print("   Top matching documents:")
            for j, (doc, score) in enumerate(results, 1):
                print(".3f")
                print(f"      Source: {doc.get('source', 'Unknown')} (Page {doc.get('page', 'N/A')})")
                print(f"      Preview: {doc.get('content', '')[:100]}...")

def main():
    print("Knowledge Base Model Tester")
    print("=" * 40)

    tester = KnowledgeBaseTester()

    if not tester.load_knowledge_base():
        print("Failed to load knowledge base")
        sys.exit(1)

    tester.test_queries()

    print("\n" + "="*60)
    print("Knowledge base testing completed!")
    print("="*60)

if __name__ == "__main__":
    main()