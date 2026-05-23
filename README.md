# 📊 NLP Transformers Playbook

End-to-end NLP tasks with HuggingFace Transformers — text classification, NER, summarization, Q&A, and translation — all with fine-tuning notebooks and evaluation.

## 🚀 Tasks Covered
| Task | Model | Dataset |
|------|-------|---------|
| Text Classification | BERT, RoBERTa | SST-2, AG News |
| Named Entity Recognition | BERT-NER | CoNLL-2003 |
| Summarization | BART, T5 | CNN/DailyMail |
| Question Answering | DistilBERT | SQuAD 2.0 |
| Translation | Helsinki-NLP/opus | WMT |

## 📁 Structure
```
nlp-transformers-playbook/
├── tasks/
│   ├── classification.py
│   ├── ner.py
│   ├── summarization.py
│   ├── qa.py
│   └── translation.py
├── notebooks/
│   └── 01_full_nlp_pipeline.ipynb
├── requirements.txt
└── README.md
```

## ⚡ Quick Start
```python
from tasks.classification import TextClassifier
clf = TextClassifier(model_name="distilbert-base-uncased-finetuned-sst-2-english")
print(clf.predict("This movie was absolutely fantastic!"))
# [{"label": "POSITIVE", "score": 0.9998}]
```
