#!/usr/bin/env python3
"""
Test script for the offline PDF-based chatbot
"""

from pdf_processor import PDFKnowledgeBase

def test_offline_chatbot():
    print("Testing Offline PDF-based Chatbot")
    print("=" * 50)

    # Load the knowledge base
    kb = PDFKnowledgeBase()
    kb.load_knowledge_base()

    # Test queries in English and Swahili
    test_queries = [
        "How often should I feel my baby move?",
        "What foods should I eat during pregnancy?",
        "When should I call the doctor?",
        "What are signs of labor?",
        "Ninawezaje kupima mtoto anavyotembea?",  # Swahili: How to check if baby is moving?
        "Ni chakula gani kizuri wakati wa ujauzito?",  # Swahili: What foods are good during pregnancy?
        "Ni dalili gani za kuzaa?",  # Swahili: What are signs of labor?
    ]

    print(f"Knowledge Base loaded: {len(kb.documents)} text chunks")
    print(f"Testing {len(test_queries)} queries:\n")

    for i, query in enumerate(test_queries, 1):
        print(f"{i}. Query: '{query}'")

        # Search for relevant documents
        results = kb.search_similar(query, top_k=3)

        print("   Top matching documents:")
        for j, (doc, score) in enumerate(results, 1):
            print(".3f")
            print(f"      Source: {doc['source']} (Page {doc['page']})")

        # Generate answer
        answer = kb.generate_answer(query, [doc for doc, _ in results])
        # Remove emoji for console output to avoid encoding issues
        clean_answer = answer.replace('⚠️', 'WARNING:')
        print(f"   Answer: {clean_answer[:200]}{'...' if len(clean_answer) > 200 else ''}")
        print()

if __name__ == "__main__":
    test_offline_chatbot()