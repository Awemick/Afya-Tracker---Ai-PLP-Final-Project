/**
 * Firebase Cloud Functions for Afya Tracker Chatbot
 * Deploy with: firebase deploy --only functions
 */

const functions = require('firebase-functions');
const admin = require('firebase-admin');

// Initialize Firebase Admin
admin.initializeApp();

// Import required libraries (these will be available in Firebase Functions environment)
const { SentenceTransformer } = require('@xenova/transformers'); // Alternative to sentence-transformers
const faiss = require('faiss-node'); // FAISS for Node.js

// Global variables for caching (cold start optimization)
let model = null;
let index = null;
let documents = [];
let embeddings = null;

// Initialize the chatbot service
async function initializeChatbot() {
    if (model && index) return; // Already initialized

    try {
        console.log('Initializing chatbot service...');

        // Load optimized embeddings and documents
        // Note: In production, these would be stored in Cloud Storage
        const bucket = admin.storage().bucket();
        const [embeddingsFile] = await bucket.file('knowledge_base/embeddings_optimized.npy').download();
        const [documentsFile] = await bucket.file('knowledge_base/documents.json').download();

        // Parse the data (simplified - you'd need proper numpy parsing)
        embeddings = JSON.parse(embeddingsFile.toString()); // This is simplified
        documents = JSON.parse(documentsFile.toString());

        // Initialize sentence transformer model
        model = new SentenceTransformer('Xenova/paraphrase-multilingual-MiniLM-L12-v2');

        // Create FAISS index
        const dimension = embeddings[0].length;
        index = new faiss.IndexFlatIP(dimension);

        // Convert to Float32Array for FAISS
        const floatEmbeddings = new Float32Array(embeddings.flat());
        index.add(floatEmbeddings);

        console.log(`Loaded ${documents.length} documents`);
    } catch (error) {
        console.error('Error initializing chatbot:', error);
        // Fallback to basic responses
    }
}

// Cloud Function for chatbot queries
exports.chatbotQuery = functions
    .runWith({
        timeoutSeconds: 60,
        memory: '1GB', // Adjust based on your needs
        maxInstances: 10,
    })
    .https.onCall(async (data, context) => {
        try {
            // Initialize if needed
            await initializeChatbot();

            const query = data.query?.trim();
            if (!query) {
                return { error: 'No query provided' };
            }

            // Encode query
            const queryEmbedding = await model.encode([query], { pooling: 'mean', normalize: true });

            // Search similar documents
            const k = 3; // Number of results
            const { distances, indices } = index.search(queryEmbedding, k);

            // Format results
            const results = [];
            for (let i = 0; i < Math.min(k, indices.length); i++) {
                const docIndex = indices[i];
                if (docIndex < documents.length) {
                    results.push({
                        content: documents[docIndex].content?.substring(0, 500) + '...',
                        source: documents[docIndex].source,
                        page: documents[docIndex].page,
                        similarity: distances[i]
                    });
                }
            }

            // Generate answer (simple extractive for now)
            const bestMatch = results[0];
            const answer = bestMatch ? bestMatch.content : 'I apologize, but I don\'t have information on that topic.';

            return {
                query: query,
                answer: answer,
                sources: results,
                confidence: bestMatch ? bestMatch.similarity : 0,
                timestamp: admin.firestore.FieldValue.serverTimestamp()
            };

        } catch (error) {
            console.error('Chatbot query error:', error);
            return {
                error: 'Service temporarily unavailable',
                fallback: 'Please try again later or contact your healthcare provider.'
            };
        }
    });

// Health check function
exports.healthCheck = functions.https.onRequest((req, res) => {
    res.status(200).json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        service: 'afya-chatbot'
    });
});

// Alternative implementation using simpler approach (no ML models)
exports.simpleChatbot = functions
    .runWith({
        timeoutSeconds: 30,
        memory: '256MB',
    })
    .https.onCall(async (data, context) => {
        const query = data.query?.toLowerCase()?.trim();

        if (!query) {
            return { error: 'No query provided' };
        }

        // Simple rule-based responses for common queries
        const responses = {
            'fetal movement': {
                answer: 'Fetal movements are important signs of your baby\'s health. You should feel at least 10 movements in 2 hours. If you notice decreased movement, contact your healthcare provider immediately.',
                sources: ['WHO Guidelines', 'UNICEF Recommendations']
            },
            'pregnancy nutrition': {
                answer: 'A balanced diet during pregnancy should include proteins, carbohydrates, healthy fats, vitamins, and minerals. Focus on folate-rich foods, calcium, iron, and stay hydrated.',
                sources: ['WHO Nutrition Guidelines', 'CDC Recommendations']
            },
            'labor signs': {
                answer: 'Signs of labor include regular contractions, water breaking, back pain, and bloody show. Contact your healthcare provider if you experience these symptoms.',
                sources: ['American College of Obstetricians and Gynecologists']
            },
            'baby kick': {
                answer: 'Most women feel their baby\'s first movements between 18-25 weeks. Keep track of movements daily. Report any changes to your doctor.',
                sources: ['Mayo Clinic', 'Johns Hopkins Medicine']
            }
        };

        // Find matching response
        for (const [key, response] of Object.entries(responses)) {
            if (query.includes(key)) {
                return {
                    query: data.query,
                    answer: response.answer,
                    sources: response.sources.map(source => ({ source, content: response.answer })),
                    confidence: 0.8,
                    type: 'rule-based'
                };
            }
        }

        // Default response
        return {
            query: data.query,
            answer: 'For personalized medical advice, please consult with your healthcare provider. I can provide general information about pregnancy and fetal health.',
            sources: [{ source: 'General Medical Guidance', content: 'Always consult healthcare professionals for specific concerns.' }],
            confidence: 0.3,
            type: 'general'
        };
    });