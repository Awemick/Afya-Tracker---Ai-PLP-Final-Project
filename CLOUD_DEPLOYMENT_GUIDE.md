# Cloud Deployment Guide for Afya Tracker Chatbot

## Free Cloud Services Comparison

| Service | Free Tier | Setup Complexity | Scalability | Best For |
|---------|-----------|------------------|-------------|----------|
| **Firebase Functions** | 2M invocations/month, 400k GB-seconds | Easy (if using Firebase) | Excellent | Firebase projects |
| **Vercel Functions** | 100GB-hours/month, 100k invocations | Easy | Good | Web-focused |
| **Netlify Functions** | 125k invocations/month | Easy | Good | Static sites |
| **Railway** | $5/month credit, then pay-as-you-go | Medium | Excellent | Full-stack apps |
| **Render** | 750 hours/month free | Easy | Good | Web services |
| **Fly.io** | 3 shared CPUs, 256MB RAM free | Medium | Excellent | Global deployment |
| **AWS Lambda** | 1M requests/month, 400k GB-seconds | Complex | Excellent | Enterprise |

## Recommended: Firebase Functions (Since you're already using Firebase)

### Prerequisites
1. Firebase CLI: `npm install -g firebase-tools`
2. Google Cloud Project with billing enabled (required for Functions)

### Step 1: Initialize Firebase
```bash
firebase login
firebase init functions
```

### Step 2: Deploy Functions
```bash
# Copy the provided files to functions directory
cp firebase-functions-chatbot.js functions/index.js
cp firebase-functions-package.json functions/package.json

# Install dependencies
cd functions && npm install && cd ..

# Deploy
firebase deploy --only functions
```

### Step 3: Upload Knowledge Base
```bash
# Upload optimized embeddings to Firebase Storage
firebase storage:upload knowledge_base/embeddings_optimized.npy gs://your-project.appspot.com/knowledge_base/
firebase storage:upload knowledge_base/documents.json gs://your-project.appspot.com/knowledge_base/
```

## Alternative: Vercel Functions (Easiest Setup)

### Create `vercel.json`
```json
{
  "functions": {
    "api/chatbot.js": {
      "runtime": "nodejs18.x"
    }
  }
}
```

### Create `api/chatbot.js`
```javascript
const { SentenceTransformer } = require('@xenova/transformers');

let model, index, documents, embeddings;

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { query } = req.body;
  if (!query) {
    return res.status(400).json({ error: 'No query provided' });
  }

  try {
    // Initialize on first call (cold start)
    if (!model) {
      // Load your optimized knowledge base
      // (Implementation similar to Firebase Functions)
    }

    // Process query and return response
    const result = await processQuery(query);
    res.status(200).json(result);

  } catch (error) {
    res.status(500).json({ error: 'Service error' });
  }
}
```

### Deploy to Vercel
```bash
npm i -g vercel
vercel --prod
```

## Alternative: Netlify Functions

### Create `netlify/functions/chatbot.js`
```javascript
const { SentenceTransformer } = require('@xenova/transformers');

exports.handler = async (event) => {
  if (event.httpMethod !== 'POST') {
    return { statusCode: 405, body: 'Method not allowed' };
  }

  const { query } = JSON.parse(event.body);

  try {
    // Your chatbot logic here
    const result = await processQuery(query);

    return {
      statusCode: 200,
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'POST, OPTIONS'
      },
      body: JSON.stringify(result)
    };
  } catch (error) {
    return {
      statusCode: 500,
      body: JSON.stringify({ error: 'Service error' })
    };
  }
};
```

### Deploy
```bash
npm install -g netlify-cli
netlify deploy --prod
```

## Alternative: Railway (Full Application)

### Create `Dockerfile`
```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
EXPOSE 8080

CMD ["node", "server.js"]
```

### Create Express Server
```javascript
const express = require('express');
const { SentenceTransformer } = require('@xenova/transformers');

const app = express();
app.use(express.json());

// Your chatbot logic here

app.post('/chatbot', async (req, res) => {
  const result = await processQuery(req.body.query);
  res.json(result);
});

app.listen(8080, () => console.log('Server running on port 8080'));
```

### Deploy to Railway
1. Connect GitHub repo
2. Railway auto-deploys with Docker

## Mobile App Integration

### Update Flutter Dependencies
```yaml
dependencies:
  cloud_functions: ^4.0.0
  http: ^0.13.5
```

### Use the Cloud Service
```dart
import 'package:cloud_functions/cloud_functions.dart';

class ChatbotService {
  final FirebaseFunctions _functions = FirebaseFunctions.instance;

  Future<Map<String, dynamic>> query(String question) async {
    final callable = _functions.httpsCallable('simpleChatbot');
    final result = await callable.call({'query': question});
    return result.data;
  }
}
```

## Cost Optimization Tips

1. **Caching**: Implement response caching in your mobile app
2. **Rate Limiting**: Add delays between requests
3. **Batch Queries**: Send multiple queries at once when possible
4. **Compression**: Use gzip for responses
5. **Monitoring**: Track usage to optimize free tier limits

## Security Considerations

1. **API Keys**: Never expose API keys in client code
2. **Rate Limiting**: Implement on server-side
3. **Input Validation**: Sanitize all inputs
4. **CORS**: Configure properly for web access
5. **Authentication**: Consider Firebase Auth for user management

## Testing Your Deployment

```bash
# Test Firebase Functions locally
firebase emulators:start --only functions

# Test with curl
curl -X POST https://your-region-your-project.cloudfunctions.net/simpleChatbot \
  -H "Content-Type: application/json" \
  -d '{"query": "How often should I feel my baby move?"}'
```

## Monitoring & Maintenance

- **Firebase**: Use Firebase Console for logs and monitoring
- **Vercel**: Vercel Analytics dashboard
- **Netlify**: Netlify Analytics
- **Railway**: Railway dashboard

## Scaling Up (When Free Tier is Not Enough)

When your app grows beyond free tiers:

1. **Firebase Blaze Plan**: Pay-as-you-go pricing
2. **Vercel Pro**: $20/month for higher limits
3. **Railway**: Usage-based pricing
4. **AWS/GCP**: Enterprise solutions

## Backup Deployment Strategy

Always have a fallback plan:

1. **Rule-based responses** for common queries (no ML required)
2. **Offline mode** with cached responses
3. **Progressive enhancement** - basic features work without cloud

This hybrid approach ensures your healthcare app remains functional even during service outages.