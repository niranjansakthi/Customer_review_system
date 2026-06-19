from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

model_name = "facebook/bart-large-cnn"

sum_tokenizer = AutoTokenizer.from_pretrained(model_name)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def analyze_review(df):
    reviews = df["review"].astype(str).tolist()
    sentiment_results = sentiment_model(reviews)
    df["sentiment"] = [result["label"] for result in sentiment_results]

    # Emotion Classification
    emotion_results = emotion_model(reviews)
    df["emotion"] = [result["label"] for result in emotion_results]

    # Direct Tokenizer + Model approach for Summarization
    summaries = []
    for review in reviews:

        if len(review.split()) > 30:
            inputs = sum_tokenizer(review, max_length=1024, truncation=True, return_tensors="pt")
            summary_ids = sum_model.generate(inputs["input_ids"], max_length=25, min_length=5, do_sample=False)
            summary_text = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary_text)
        else:
            summaries.append("Short review - no summary needed")

    df["summary"] = summaries
    return df
