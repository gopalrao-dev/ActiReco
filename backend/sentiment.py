from transformers import pipeline
import logging

logger = logging.getLogger("ActiReco")

class SentimentModel:
    """Robust HuggingFace sentiment model wrapper."""

    def __init__(self):
        try:
            self.classifier = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            logger.info("✅ Sentiment model loaded successfully")
        except Exception as e:
            logger.error(f"⚠️ Failed to load sentiment model: {e}")
            self.classifier = None

    def analyze(self, text: str) -> str:
        if not text or not text.strip():
            return "neutral"

        if not self.classifier:
            logger.warning("Sentiment fallback: model not available")
            return "neutral"

        try:
            result = self.classifier(text)[0]
            label = result["label"].lower()

            if label in ["positive", "negative", "neutral"]:
                return label
            elif label in ["label_0", "1 star", "2 stars"]:
                return "negative"
            elif label in ["label_1", "3 stars"]:
                return "neutral"
            elif label in ["label_2", "4 stars", "5 stars"]:
                return "positive"
            else:
                return "neutral"
        except Exception as e:
            logger.error(f"⚠️ Sentiment analysis failed: {e}")
            return "neutral"