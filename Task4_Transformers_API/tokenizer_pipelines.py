"""Task 4 requirement: show DistilBERT tokenizer outputs and demo sentiment/NER pipelines."""
from transformers import AutoTokenizer, pipeline

MODEL = "distilbert/distilbert-base-uncased"

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    sentences = [
        "The delivery was fast and excellent.",
        "I am unhappy because my payment failed."
    ]
    encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    for i, sentence in enumerate(sentences):
        print(f"\nSentence {i+1}: {sentence}")
        print("Tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"][i]))
        print("Input IDs:", encoded["input_ids"][i].tolist())
        print("Attention mask:", encoded["attention_mask"][i].tolist())

    sentiment = pipeline("sentiment-analysis")
    ner = pipeline("ner", aggregation_strategy="simple")
    sentiment_examples = [
        "I love this product; it works perfectly.",
        "The service was disappointing and slow.",
        "The package arrived yesterday."
    ]
    ner_examples = [
        "Sundar Pichai works at Google.",
        "Dhanush lives in Chennai.",
        "Microsoft opened an office in London."
    ]
    print("\nSentiment-analysis pipeline:")
    for sentence in sentiment_examples:
        print(sentence, "->", sentiment(sentence))
    print("\nNER pipeline:")
    for sentence in ner_examples:
        print(sentence, "->", ner(sentence))
