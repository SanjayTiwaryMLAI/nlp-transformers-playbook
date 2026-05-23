"""Abstractive Summarization with BART / T5"""
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import evaluate


class Summarizer:
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.pipe = pipeline("summarization", model=model_name)

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> str:
        return self.pipe(text, max_length=max_length, min_length=min_length, do_sample=False)[0]["summary_text"]

    def batch_summarize(self, texts: list[str], **kwargs) -> list[str]:
        return [self.summarize(t, **kwargs) for t in texts]

    @staticmethod
    def evaluate_rouge(predictions: list[str], references: list[str]) -> dict:
        rouge = evaluate.load("rouge")
        return rouge.compute(predictions=predictions, references=references)


if __name__ == "__main__":
    s = Summarizer()
    article = (
        "The Transformer model, introduced in the paper Attention is All You Need by Vaswani et al. (2017), "
        "revolutionized natural language processing. Unlike RNNs and LSTMs, Transformers rely entirely on "
        "attention mechanisms to draw global dependencies between input and output, enabling parallelization "
        "and achieving state-of-the-art results on machine translation benchmarks."
    )
    print(s.summarize(article))
