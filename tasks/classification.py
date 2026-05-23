"""Text Classification with HuggingFace Transformers"""
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import evaluate
import numpy as np


class TextClassifier:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.pipe = pipeline("text-classification", model=model_name)

    def predict(self, texts: str | list[str]) -> list[dict]:
        if isinstance(texts, str):
            texts = [texts]
        return self.pipe(texts)

    @staticmethod
    def fine_tune(base_model: str = "distilbert-base-uncased",
                  dataset_name: str = "sst2", output_dir: str = "./clf-finetuned"):
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        dataset   = load_dataset("glue", dataset_name)
        tokenized = dataset.map(lambda x: tokenizer(x["sentence"], truncation=True, padding="max_length"), batched=True)
        model     = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
        metric    = evaluate.load("accuracy")

        def compute_metrics(p):
            preds = np.argmax(p.predictions, axis=1)
            return metric.compute(predictions=preds, references=p.label_ids)

        Trainer(
            model=model,
            args=TrainingArguments(output_dir=output_dir, num_train_epochs=3,
                                   per_device_train_batch_size=16, evaluation_strategy="epoch"),
            train_dataset=tokenized["train"].select(range(2000)),
            eval_dataset=tokenized["validation"],
            compute_metrics=compute_metrics,
        ).train()
        print(f"Fine-tuned model saved to {output_dir}")


if __name__ == "__main__":
    clf = TextClassifier()
    results = clf.predict(["I loved this film!", "Worst movie ever."])
    for r in results:
        print(r)
