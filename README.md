# ⚖️ Lawgorithm — Legal Query Classification

> AI-powered legal query classifier for the Indian Legal System using BERT

## What it does
Lawgorithm takes a natural language legal query and classifies it across two dimensions:

| Dimension | Classes |
|-----------|---------|
| **Domain** | Criminal, Civil, Property, Family, Tax |
| **Intent** | Needs Lawyer, Self Solvable, Urgent, General Info |

## Architecture

```
User Query
    ↓
BERT (bert-base-uncased)
    ↓
[CLS] Token Pooling
    ↓
    ├── Domain Head → Linear(768→256) → ReLU → Linear(256→5)
    └── Intent Head → Linear(768→256) → ReLU → Linear(256→4)
         ↓
FastAPI Backend (/predict)
         ↓
Chat UI (index.html)
```

## Why BERT?
- BERT's bidirectional attention captures full context of legal queries
- Pre-trained on large text corpus — understands legal terminology
- Fine-tuned on domain-specific Indian legal queries
- Multi-task learning: two heads trained simultaneously on shared BERT backbone

## Project Structure
```
lawgorithm/
├── data/
│   ├── generate_dataset.py     # Dataset generator
│   └── legal_queries.csv       # Training data (189 samples)
├── model/
│   ├── train.py                # BERT fine-tuning script
│   └── model_utils.py          # Inference utilities
├── api/
│   └── main.py                 # FastAPI server
├── ui/
│   └── index.html              # Chat interface
├── saved_model/                # Generated after training
│   ├── best_model.pt
│   ├── domain_encoder.pkl
│   ├── intent_encoder.pkl
│   └── tokenizer files
└── requirements.txt
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset
```bash
python data/generate_dataset.py
```

### 3. Train the model
```bash
cd model
python train.py
```
Training takes ~10-15 mins on CPU, ~3 mins on GPU.

### 4. Start the API
```bash
cd api
uvicorn main:app --reload --port 8000
```

### 5. Open the UI
Open `ui/index.html` in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/` | API info |
| GET  | `/health` | Health check |
| POST | `/predict` | Classify a query |
| GET  | `/docs` | Swagger UI |

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"query": "My landlord is refusing to return my security deposit after 2 years"}'
```

### Example Response
```json
{
  "query": "My landlord is refusing to return my security deposit after 2 years",
  "domain": "civil",
  "domain_label": "Civil Law",
  "domain_scores": {"civil": 78.3, "property": 15.2, "criminal": 3.1, "family": 2.1, "tax": 1.3},
  "intent": "needs_lawyer",
  "intent_scores": {"needs_lawyer": 82.1, "urgent": 9.3, "self_solvable": 5.4, "general_info": 3.2},
  "message": "This situation requires professional legal representation.",
  "action": "We strongly recommend consulting a qualified lawyer immediately.",
  "urgency_color": "red",
  "confidence": 78.3
}
```

## Technical Details
- **Model**: bert-base-uncased fine-tuned with two classification heads
- **Loss**: Combined CrossEntropyLoss for domain + intent (equal weights)
- **Optimizer**: AdamW with weight decay 0.01
- **Scheduler**: Linear LR decay
- **Max sequence length**: 128 tokens
- **Batch size**: 16
- **Epochs**: 5

## Model
Due to size limitations, trained model is not included.

You can:
- Train using: `python3 train.py`

## Author

**Arnav Singh**  
ML Engineering | Production ML Systems | MLOps  
[LinkedIn](https://linkedin.com/in/your-profile) · [Kaggle](https://kaggle.com/singharnav18)

