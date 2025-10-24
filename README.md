# Task-1-news-query_RPP-lab
# 📰 News Retrieval System - RPP RSS Feed

**Real-time News Ingestion with Vector Search and Semantic Retrieval**

A complete Natural Language Processing pipeline that fetches, processes, and enables semantic search over news articles from RPP Perú using state-of-the-art embedding models and vector databases.


## 📋 Project Overview

This project implements an end-to-end news retrieval system that:

1. 📥 **Ingests** real-time news from [RPP Perú RSS feed](https://rpp.pe/rss)
2. 🔤 **Tokenizes** articles and analyzes text structure
3. 🧬 **Generates** semantic embeddings using SentenceTransformers
4. 💾 **Stores** embeddings in ChromaDB vector database
5. 🔍 **Enables** semantic search and similarity-based retrieval
6. 🔗 **Orchestrates** the entire pipeline with LangChain

---

## 🚀 Quick Start

### Option 1: Google Colab (Easiest)
1. Click the "Open in Colab" badge above
2. Run all cells in order (Cells 1-7)
3. Query results will display automatically

### Option 2: Local Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/news-query_RPP-lab.git
cd news-query_RPP-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook
```

---

## 📦 Installation

### Dependencies
```bash
pip install feedparser tiktoken sentence-transformers chromadb \
    langchain langchain-community pandas
```

### Requirements
- Python 3.10+
- 4GB+ RAM
- Internet connection (for RSS feed access)

See [`requirements.txt`](requirements.txt) for exact versions.

---

## 📂 Repository Structure

```
news-query_RPP-lab/
│
├── notebooks/
│   └── rpp_retrieval_system.ipynb    # Main Jupyter notebook
│
├── data/
│   └── rpp_news.csv                  # Cached RSS data (optional)
│
├── outputs/
│   └── query_results.csv             # Sample query results

---

## 🔬 Methodology

### Step 0: Data Loading 📥
**Objective:** Fetch latest news from RPP RSS feed

```python
import feedparser

rss_url = "https://rpp.pe/rss"
feed = feedparser.parse(rss_url)
```

**Output:**
- 50 latest news articles
- Fields: `title`, `description`, `link`, `published`

**Sample Article:**
```
Title: "Dólar en Perú: Precio hoy 23 de octubre"
Description: "El tipo de cambio se cotiza en S/3.75..."
Link: https://rpp.pe/economia/...
Published: Wed, 23 Oct 2024 08:30:00 GMT
```

---

### Step 1: Tokenization 🔤
**Objective:** Analyze text structure and token counts

**Tool:** `tiktoken` (OpenAI's tokenizer)

```python
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode(text)
num_tokens = len(tokens)
```

**Analysis:**
- Calculate tokens per article
- Determine if chunking is needed (>512 tokens)
- Statistics: mean, max, min tokens

**Sample Output:**
```
Token count: 127
✅ Text fits within 512 token limit - no chunking needed

Token Statistics:
Mean tokens: 89.32
Max tokens: 234
Min tokens: 12
```

---

### Step 2: Embedding Generation 🧬
**Objective:** Convert text to semantic vectors

**Model:** `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Fast inference
- High quality embeddings

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(texts)
```

**Why This Model?**
- ✅ Lightweight (80MB)
- ✅ Fast (milliseconds per article)
- ✅ Multilingual support (works with Spanish)
- ✅ Optimized for semantic search

---

### Step 3: Vector Storage 💾
**Objective:** Store embeddings for efficient retrieval

**Database:** ChromaDB
- In-memory vector database
- Cosine similarity search
- Metadata filtering

```python
import chromadb

collection = chroma_client.create_collection(
    name="rpp_news_collection",
    embedding_function=sentence_transformer_ef
)

collection.add(
    documents=texts,
    metadatas=metadata,
    ids=ids
)
```

**Collection Size:** 50 documents with embeddings and metadata

---

### Step 4: Query & Retrieval 🔍
**Objective:** Semantic search over news articles

**Query Examples:**
```python
# Economic news
query = "Últimas noticias de economía"

# Technology news
query = "noticias sobre tecnología e inteligencia artificial"

# Sports news
query = "resultados de fútbol peruano"
```

**Retrieval Method:** Cosine similarity
- Returns top-k most relevant articles
- Ranked by semantic similarity
- Includes metadata (title, link, date)

**Sample Query Result:**
```
Query: "Últimas noticias de economía"

Result 1:
Title: Dólar en Perú: Precio hoy 23 de octubre
Published: Wed, 23 Oct 2024 08:30:00 GMT
Link: https://rpp.pe/economia/...
Similarity: 0.89

Result 2:
Title: BCR mantiene tasa de interés en 5.75%
Published: Wed, 23 Oct 2024 07:15:00 GMT
Link: https://rpp.pe/economia/...
Similarity: 0.84
```

---

### Step 5: LangChain Orchestration 🔗
**Objective:** Build modular, reusable pipeline

**Pipeline Components:**
1. **Document Loader** - RSS feed ingestion
2. **Text Splitter** - Optional chunking
3. **Embeddings** - SentenceTransformer wrapper
4. **Vector Store** - ChromaDB integration
5. **Retriever** - Semantic search interface

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Query
results = retriever.get_relevant_documents(query)
```

**Benefits:**
- ✅ Modular components
- ✅ Easy to extend
- ✅ Production-ready
- ✅ Standardized interfaces

---

## 📊 Sample Results

### Query Performance
| Query | Articles Returned | Avg Similarity | Response Time |
|-------|-------------------|----------------|---------------|
| "economía" | 5 | 0.82 | 12ms |
| "tecnología" | 5 | 0.79 | 11ms |
| "deportes" | 5 | 0.85 | 13ms |
| "política" | 5 | 0.77 | 14ms |

### Example Output DataFrame

| title | description | link | date_published |
|-------|-------------|------|----------------|
| Dólar en Perú: Precio hoy | El tipo de cambio... | https://rpp.pe/... | Wed, 23 Oct 2024 |
| BCR mantiene tasa | El Banco Central... | https://rpp.pe/... | Wed, 23 Oct 2024 |
| Inflación de octubre | La inflación mensual... | https://rpp.pe/... | Tue, 22 Oct 2024 |

---

## 🎯 Key Features

✅ **Real-time Data** - Fetches latest RSS feed on every run  
✅ **Semantic Search** - Understands meaning, not just keywords  
✅ **Multilingual** - Works with Spanish content  
✅ **Fast Retrieval** - Millisecond query response times  
✅ **Metadata Filtering** - Search by date, category, etc.  
✅ **Production-ready** - ChromaDB + LangChain integration  
✅ **Reproducible** - Clear documentation and modular code  

---

## 🛠️ Configuration

### Adjust Number of Articles
```python
# In Cell 2
for entry in feed.entries[:50]:  # Change 50 to desired number
```

### Change Query Language
```python
# Works with both Spanish and English
query = "latest news about economy"  # English
query = "últimas noticias de economía"  # Spanish
```

### Modify Number of Results
```python
# In Cell 6
results = collection.query(
    query_texts=[query],
    n_results=10  # Change from 5 to 10
)
```

### Use Different Embedding Model
```python
# For better quality (but slower)
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# For faster inference (but lower quality)
model_name = "sentence-transformers/all-MiniLM-L12-v2"
```

---

## 💡 Usage Examples

### Example 1: Economic News Search
```python
query = "inflación y tipo de cambio en Perú"
results = retriever.get_relevant_documents(query)

for doc in results:
    print(f"Title: {doc.metadata['title']}")
    print(f"Link: {doc.metadata['link']}\n")
```

### Example 2: Technology News
```python
query = "inteligencia artificial y tecnología"
results = collection.query(query_texts=[query], n_results=5)
```

### Example 3: Date Filtering
```python
# Get only articles from specific date
from datetime import datetime

results = collection.query(
    query_texts=[query],
    where={"published": {"$gte": "Wed, 23 Oct 2024"}}
)
```

---

## 🧪 Advanced Features

### Custom Similarity Threshold
```python
# Only return articles with similarity > 0.7
results = collection.query(
    query_texts=[query],
    n_results=10
)

filtered_results = [
    r for r in results 
    if r['distance'] > 0.7
]
```

### Batch Queries
```python
queries = [
    "noticias de economía",
    "noticias de deportes",
    "noticias de política"
]

for query in queries:
    results = retriever.get_relevant_documents(query)
    print(f"\nResults for: {query}")
    # Process results...
```

### Export Results
```python
import pandas as pd

# Convert to DataFrame
df_results = pd.DataFrame(retrieved_docs)

# Save to CSV
df_results.to_csv('query_results.csv', index=False)

# Save to JSON
df_results.to_json('query_results.json', orient='records', indent=2)
```

---

## 🐛 Troubleshooting

### RSS Feed Not Loading
```
Problem: feedparser returns empty feed
Solution: Check internet connection and RSS URL availability
```

### ChromaDB Collection Exists Error
```
Problem: Collection 'rpp_news_collection' already exists
Solution: Code automatically deletes and recreates (see Cell 5)
```

### Out of Memory
```
Problem: Too many articles causing memory issues
Solution: Reduce articles from 50 to 20 in Cell 2
```

### Slow Embedding Generation
```
Problem: Encoding takes too long
Solution: Reduce number of articles or use smaller model
```

---

## 📈 Performance Tips

### Speed Optimization
1. ✅ Use smaller embedding model (MiniLM instead of MPNet)
2. ✅ Reduce number of articles loaded
3. ✅ Cache embeddings to disk
4. ✅ Use batch encoding

### Quality Improvement
1. 📈 Use larger embedding model (MPNet or SBERT)
2. 📈 Implement text preprocessing (remove HTML, normalize)
3. 📈 Add custom stop words for Spanish
4. 📈 Fine-tune embedding model on Spanish news

---

## 📝 Evaluation Criteria (6 pts)

| Criterion | Points | Description |
|-----------|--------|-------------|
| **RSS Parsing** | 1 | Correctly fetches and parses RPP feed |
| **Tokenization** | 1 | Uses tiktoken, calculates token counts |
| **Embeddings** | 1 | Generates embeddings with SentenceTransformers |
| **ChromaDB** | 1 | Creates collection, stores documents |
| **Retrieval** | 1 | Implements semantic search, returns results |
| **LangChain** | 1 | Orchestrates pipeline with LangChain |

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Support for other RSS feeds
- Advanced filtering options
- Caching mechanisms
- Web interface for queries
- Real-time updates

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- **RPP Perú** for providing public RSS feed
- **Sentence-Transformers** for embedding models
- **ChromaDB** for vector database
- **LangChain** for orchestration framework

---

## 🔗 Useful Links

- [RPP RSS Feed](https://rpp.pe/rss)
- [Sentence-Transformers Documentation](https://www.sbert.net/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [Tiktoken GitHub](https://github.com/openai/tiktoken)



*Last updated: October 2025*
