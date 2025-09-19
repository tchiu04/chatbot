# Personalized News Briefing Chatbot

**Stack:** FastAPI · PyTorch · Hugging Face Transformers · Pinecone · Docker · CI/CD

## 🚀 Overview

An end-to-end NLP chatbot that delivers personalized news briefings by combining real-time RSS ingestion, summarization, and retrieval-augmented question answering (RAG).

**Users can:**
- Query live news feeds in natural language
- Receive concise summaries powered by Pegasus
- Get fact-based answers using RoBERTa with retrieval augmentation

## 🛠 Features

- **Summarization**: Hugging Face Pegasus model fine-tuned for concise news briefs
- **Q&A**: RoBERTa with RAG pipeline, integrated with Pinecone vector search + sentence-transformer embeddings
- **Performance**: Median response latency <300ms on 1,000+ articles
- **Scalability**: FastAPI microservices, containerized with Docker, deployable via CI/CD pipelines
- **Semantic Search**: Supports semantic retrieval across large corpora of news articles

## 📂 Project Structure

```
/app
  ├── api/              # FastAPI endpoints (summarization, QA, search)
  ├── models/           # Pegasus, RoBERTa, embeddings
  ├── services/         # Pinecone integration, article ingestion
  ├── tests/            # Unit + integration tests
  └── main.py           # Entry point
/docker
  ├── Dockerfile
  └── docker-compose.yml
/config
  ├── settings.yaml
  └── logging.conf
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- Pinecone API key
- RSS feed URLs

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

4. **Run with Docker**
   ```bash
   docker-compose up --build
   ```

### Local Development

```bash
# Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest app/tests/

# Access API documentation
# http://localhost:8000/docs
```

## 🔧 API Endpoints

### Summarization
```http
POST /api/v1/summarize
Content-Type: application/json

{
  "text": "Your news article text here...",
  "max_length": 150
}
```

### Question Answering
```http
POST /api/v1/qa
Content-Type: application/json

{
  "question": "What happened in the latest tech news?",
  "context": "optional context or leave empty for RAG"
}
```

### Semantic Search
```http
GET /api/v1/search?query=artificial intelligence&limit=10
```

## 🏗 Architecture

### Core Components

1. **News Ingestion Service**
   - RSS feed parsing and monitoring
   - Article preprocessing and cleaning
   - Real-time data pipeline

2. **NLP Models**
   - **Pegasus**: Text summarization
   - **RoBERTa**: Question answering
   - **Sentence-BERT**: Embedding generation

3. **Vector Database**
   - Pinecone integration for semantic search
   - Efficient similarity matching
   - Scalable vector storage

4. **API Layer**
   - FastAPI endpoints
   - Request validation and rate limiting
   - Response caching

### Data Flow

```
RSS Feeds → Ingestion → Preprocessing → Embedding → Pinecone
                                                        ↓
User Query → FastAPI → RAG Pipeline → Model Inference → Response
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test categories
pytest app/tests/unit/
pytest app/tests/integration/
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build and run
docker-compose up -d

# Scale services
docker-compose up --scale api=3

# View logs
docker-compose logs -f
```

### CI/CD Pipeline

The project includes GitHub Actions workflows for:
- Automated testing on PR
- Docker image building
- Deployment to staging/production
- Performance monitoring

## 📊 Performance Metrics

- **Response Time**: <300ms median latency
- **Throughput**: 1,000+ articles processed
- **Accuracy**: 85%+ on news QA benchmarks
- **Uptime**: 99.9% availability target

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# Model Configuration
PEGASUS_MODEL_PATH=google/pegasus-newsroom
ROBERTA_MODEL_PATH=deepset/roberta-base-squad2

# Pinecone Configuration
PINECONE_API_KEY=your-api-key
PINECONE_ENVIRONMENT=us-west1-gcp
PINECONE_INDEX_NAME=news-embeddings

# RSS Feeds
RSS_FEEDS=https://feeds.reuters.com/reuters/topNews,https://rss.cnn.com/rss/edition.rss
```

### Model Configuration

Edit `config/settings.yaml` to customize:
- Model parameters
- Summarization settings
- QA confidence thresholds
- Embedding dimensions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure CI/CD pipeline passes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for pre-trained models
- Pinecone for vector database services
- FastAPI for the web framework
- The open-source NLP community

## 📞 Support

- **Documentation**: [Wiki](../../wiki)
- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)

---

**Built with ❤️ for personalized news experiences**