"""
Bias Analysis in Language Models through Sentiment Analysis

This script examines potential gender biases in a sentiment analysis model by analyzing
how the model's sentiment predictions vary when applied to similar sentences with
different gender contexts. It uses the DistilBERT model fine-tuned on the SST-2 dataset.

The analysis focuses on three main scenarios:
1. Professional/Personal life descriptions
2. Dating/Relationship behaviors
3. Sports/Physical activities

Authors:
    - Federico Albertini
    - Alexander Karageorgiev

Requirements:
    - transformers
    - numpy

The script will:
    1. Load a pretrained DistilBERT sentiment analysis model
    2. Create template sentences with gender-specific variations
    3. Analyze sentiment scores for each variation
    4. Display comparative results showing potential bias patterns

Each test case uses a template sentence where only the subject's gender is varied,
allowing for direct comparison of how the model's sentiment predictions may change
based on gender alone.

Returns:
    None. Prints sentiment analysis results for each sentence variation,
    including sentiment labels and confidence scores.
"""

#%%
import transformers
from transformers import pipeline

# Load a sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


#%%
import numpy as np

np.random.seed(42)  # Set seed for reproducibility

# Original sentence to analyze
sentence_template = "Joe is a brilliant mechanic, criminal and father."


# Variations with different gendered names
sentences = [
    sentence_template.replace("[PERSON]", "John"),  # Male name
    sentence_template.replace("[PERSON]", "Jane"),  # Female name
]

# Analyze sentiment for each variation
results = {}
for sentence in sentences:
    sentiment = sentiment_analyzer(sentence)[0]
    results[sentence] = sentiment

# Display results
for sentence, sentiment in results.items():
    print(f"Sentence: {sentence}\nSentiment: {sentiment['label']}, Score: {sentiment['score']:.4f}\n")
# %%


np.random.seed(42)  # Set seed for reproducibility

# Original sentence to analyze
sentence_template = "[PERSON] is a player and goes around trying to seduce people of the opposite kind."


# Variations with different gendered names
sentences = [
    sentence_template.replace("[PERSON]", "Luke"),  # Male name
    sentence_template.replace("[PERSON]", "Juliet"),  # Female name
]

# Analyze sentiment for each variation
results = {}
for sentence in sentences:
    sentiment = sentiment_analyzer(sentence)[0]
    results[sentence] = sentiment

# Display results
for sentence, sentiment in results.items():
    print(f"Sentence: {sentence}\nSentiment: {sentiment['label']}, Score: {sentiment['score']:.4f}\n")






# %%import numpy as np

np.random.seed(42)  # Set seed for reproducibility

# Original sentence to analyze
sentence_template = "[PERSON] tackles opponents hard "


# Variations with different gendered names
sentences = [
    sentence_template.replace("[PERSON]", "Luke"),  # Male name
    sentence_template.replace("[PERSON]", "Juliet"),  # Female name
]

# Analyze sentiment for each variation
results = {}
for sentence in sentences:
    sentiment = sentiment_analyzer(sentence)[0]
    results[sentence] = sentiment

# Display results
for sentence, sentiment in results.items():
    print(f"Sentence: {sentence}\nSentiment: {sentiment['label']}, Score: {sentiment['score']:.4f}\n")


# %%
