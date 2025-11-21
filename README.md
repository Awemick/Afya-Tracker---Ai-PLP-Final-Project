# Afya Tracker Elsa AI

[![Live Demo](https://afyatrackerwebapp.vercel.app/)

## Overview

Afya Tracker Elsa AI is an AI-powered health tracking system designed to assist healthcare providers and patients with fetal health monitoring. The system includes a machine learning model for predicting fetal health status based on cardiotocography (CTG) data.

## Features

- Fetal health prediction model (Normal, Suspect, Pathological)
- PDF document processing for medical knowledge base
- Gemini AI integration for chat support
- Offline chatbot functionality
- Cross-platform support (Web and Flutter)

## Applications

### Web Application
The web app is built with React.js and provides a comprehensive healthcare management platform. It includes:
- User authentication and role-based access (Admin, Provider, Patient)
- Fetal health assessment with AI-powered predictions
- Medical document management and search
- Real-time chat support with AI assistant
- Dashboard analytics and reporting

**Live Demo:** [View Web App](https://afya-tracker-elsa.vercel.app)

### Mobile Application
The mobile app is developed using Flutter for cross-platform compatibility (iOS/Android). Features include:
- Offline AI chatbot for medical queries
- Fetal health monitoring and tracking
- Emergency contact integration
- Multilingual support (English/Swahili)
- Push notifications for health reminders

## Development

This project was built using **Vibe Coding AI for Software Engineering**, an AI-assisted development approach that accelerates the software development lifecycle through:
- Intelligent code generation and completion
- Automated testing and debugging assistance
- Architecture design recommendations
- Cross-platform compatibility optimization
- Performance monitoring and optimization suggestions

The AI-assisted development methodology enabled rapid prototyping, iterative improvements, and high-quality code generation across web, mobile, and AI components.

## Prerequisites

Before getting started, ensure you have the following installed:

- **Node.js** (version 14 or higher) - for running JavaScript tests and web app
- **npm** - Node.js package manager
- **Python** (version 3.7 or higher) - for AI model training and processing
- **pip** - Python package manager

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd afya_tracker_elsa_ai
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Node.js dependencies** (for model testing):
   ```bash
   npm install --save-dev @tensorflow/tfjs-node
   ```

## Testing the Fetal Health Model

You can test the trained TensorFlow.js model using either Node.js or a web browser:

### Option 1: Browser-based Testing (Recommended for first-time users)

1. Open `test-model.html` in your web browser (double-click the file or drag it into your browser)

This will automatically load the model from the local `models/fetal_health_model/` directory and run tests with sample data. The results will display in the browser window with improved formatting.

### Option 2: Node.js Testing

1. Install the required package:
   ```bash
   npm install --save-dev @tensorflow/tfjs-node
   ```
   **Note:** This may require Node.js version 18-20 and Visual Studio Build Tools on Windows. If installation fails, use the browser option above.

2. Ensure the model files are present in `../afya_tracker_web/public/models/fetal_health_model/`

3. Run the test script:
   ```bash
   node test-model.js
   ```

Both methods will load the model and test it with sample data representing:
- Normal fetal health
- Suspect fetal health
- Pathological fetal health

The output will show predicted classes and confidence levels for each test case.

## Testing the Knowledge Base

The knowledge base is a semantic search system using sentence embeddings for retrieving relevant medical information.

To test the knowledge base:

```bash
python test-knowledge-base.py
```

This will:
- Load the LaBSE sentence transformer model
- Load pre-processed document embeddings
- Test queries in English and Swahili
- Show similarity scores and matching document previews

The knowledge base contains ~17,000 text chunks from medical PDFs with 768-dimensional embeddings.

## Model Information

- **Input Features**: 21 cardiotocography measurements
- **Output Classes**:
  - 0: Normal
  - 1: Suspect
  - 2: Pathological
- **Model Format**: TensorFlow.js (converted from Keras/TensorFlow)

## Project Structure

### AI/ML Components
- `train_fetal_model.py` - Script to train the fetal health model
- `test_fetal_model.py` - Python testing for the model
- `convert_to_tfjs.py` - Convert trained model to TensorFlow.js format
- `pdf_processor.py` - Process medical PDF documents
- `test-model.js` - Node.js standalone model testing
- `test-knowledge-base.py` - Test the knowledge base retrieval system
- `knowledge_base/` - Processed medical documents and embeddings
- `Ai datasets/` - Training data and medical resources

### Web Application (`afya_tracker_web/`)
- Built with React.js, TypeScript, and Material-UI
- Firebase integration for authentication and database
- TensorFlow.js for client-side AI inference
- Responsive design for desktop and mobile browsers

### Mobile Application (`afya_tracker/`)
- Built with Flutter for cross-platform development
- Offline AI capabilities using pre-trained models
- SQLite for local data storage
- Multilingual support and accessibility features

## Getting Help

If you encounter issues:

1. Check that all prerequisites are installed
2. Ensure model files exist in the correct location
3. Verify Python/Node.js versions meet requirements
4. Check console output for specific error messages

## Getting Started with Applications

### Web App Setup
```bash
cd afya_tracker_web
npm install
npm start
```
Access at `http://localhost:3000`

### Mobile App Setup
```bash
cd afya_tracker
flutter pub get
flutter run
```
Requires Android Studio or Xcode for device simulation.

## Contributing

This is a healthcare AI project. Please ensure all contributions follow medical data privacy and ethical guidelines.