import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union, Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, matthews_corrcoef, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from features.features import (
    get_review_length, get_word_count, get_avg_word_length,
    get_punctuation_count, get_exclamation_count, get_question_count,
    get_uppercase_word_count, get_adjectives_adverbs, get_flesch_kincaid_grade,
    get_sentiment_features, get_emotional_features
)

from models.supervised_model.logistic_regression import logistic_regression_training_and_prediction
from models.supervised_model.svm import svm_training_and_prediction
from models.supervised_model.random_forest import random_forest_training_and_prediction

# -----------------------------
# Load and Process Data
# -----------------------------
labeled_data = pd.read_excel('C:/Users/PragnyaPataskar/Projects/Fake reviews/fake reviews dataset.xlsx')

# Create additional columns for each feature (if not already present)
labeled_data['review_length'] = labeled_data['review'].apply(get_review_length)
labeled_data['word_count'] = labeled_data['review'].apply(get_word_count)
labeled_data['avg_word_length'] = labeled_data['review'].apply(get_avg_word_length)
labeled_data['punctuation_count'] = labeled_data['review'].apply(get_punctuation_count)
labeled_data['exclamation_count'] = labeled_data['review'].apply(get_exclamation_count)
labeled_data['question_count'] = labeled_data['review'].apply(get_question_count)
labeled_data['uppercase_word_count'] = labeled_data['review'].apply(get_uppercase_word_count)
labeled_data['sentiment'] = labeled_data['review'].apply(get_sentiment_features)
labeled_data['emotion'] = labeled_data['review'].apply(get_emotional_features)
labeled_data['flesch_kincaid_grade'] = labeled_data['review'].apply(get_flesch_kincaid_grade)


print(labeled_data.head(5))

# Saved extracted features in excel
labeled_data.to_excel('extracted_features.xlsx', index=False)
print('Extracted features are saved in extracted_features.xlsx file')


if __name__ == '__main__':
    # training data
    training_data = pd.read_excel('C:/Users/PragnyaPataskar/Projects/Fake reviews/extracted_features.xlsx')
    to_predict = training_data.copy()
    
    preds, preds_proba, report, mcc, prec, = logistic_regression_training_and_prediction(
        training_data=training_data,
        test_size=0.2,
        to_predict=to_predict
    )
    
    print("Classification Report:",report)
    print("Matthews Correlation Coefficient:", mcc)
    print("F1-Score:", f1_score)
    print("Recall:", recall_score)
    print("Precision Score:", prec)
    print("Predictions:", preds)
