#!/bin/bash

# Deployment script for Afya Tracker Cloud Chatbot
# This script sets up and deploys the chatbot to Firebase Functions

echo "🚀 Deploying Afya Tracker Cloud Chatbot"
echo "========================================"

# Check if Firebase CLI is installed
if ! command -v firebase &> /dev/null; then
    echo "❌ Firebase CLI not found. Installing..."
    npm install -g firebase-tools
fi

# Check if user is logged in
if ! firebase projects:list &> /dev/null; then
    echo "🔐 Please login to Firebase:"
    firebase login
fi

# Initialize Firebase project if not already done
if [ ! -f ".firebaserc" ]; then
    echo "🔧 Initializing Firebase project..."
    echo "Please select your Firebase project or create a new one:"
    firebase init --only functions,storage

    # Copy our functions file
    cp firebase-functions-chatbot.js functions/index.js
    cp firebase-functions-package.json functions/package.json
fi

# Install dependencies
echo "📦 Installing dependencies..."
cd functions
npm install
cd ..

# Upload optimized knowledge base to Firebase Storage
echo "☁️  Uploading knowledge base to Firebase Storage..."
firebase storage:upload knowledge_base/embeddings_optimized.npy gs://[YOUR-PROJECT-ID].appspot.com/knowledge_base/
firebase storage:upload knowledge_base/documents.json gs://[YOUR-PROJECT-ID].appspot.com/knowledge_base/

# Deploy functions
echo "🚀 Deploying Firebase Functions..."
firebase deploy --only functions

echo "✅ Deployment complete!"
echo ""
echo "🌐 Your chatbot API endpoints:"
echo "   - Health Check: https://[REGION]-[PROJECT-ID].cloudfunctions.net/healthCheck"
echo "   - Simple Chatbot: https://[REGION]-[PROJECT-ID].cloudfunctions.net/simpleChatbot"
echo "   - Full Chatbot: https://[REGION]-[PROJECT-ID].cloudfunctions.net/chatbotQuery"
echo ""
echo "📱 Update your mobile app to use these endpoints!"
echo "🔧 Don't forget to update Firebase security rules for storage access."