#imports
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

# Load and Process Data
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

# method to train the models
def train_model(model: str, training_data: pd.DataFrame, to_predict: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
    """
    Train a specified model using TF-IDF embedding and save predictions to an Excel file.
    
    Args:
        model (str): The model type. Supported: 'logistic_regression', 'random_forest', 'svm'.
        training_data (pd.DataFrame): Training data containing at least 'review' and 'label' columns.
        to_predict (pd.DataFrame): Data for prediction. Must include a 'review' column.
        test_size (float): Proportion of the training data to use for validation.
    
    Returns:
        pd.DataFrame: The 'to_predict' DataFrame updated with the model's predictions and probabilities.
    """
    if model == 'logistic_regression':
        preds, preds_proba, report, mcc, prec = logistic_regression_training_and_prediction(
            training_data=training_data,
            test_size=test_size,
            to_predict=to_predict
        )
    elif model == 'random_forest':
        preds, preds_proba, report, mcc, prec = random_forest_training_and_prediction(
            training_data=training_data,
            test_size=test_size,
            to_predict=to_predict
        )
    elif model == 'svm':
        preds, preds_proba, report, mcc, prec = svm_training_and_prediction(
            training_data=training_data,
            test_size=test_size,
            to_predict=to_predict
        )
    else:
        raise ValueError("Model is not present in this method")
    
    print(f"{model.capitalize()} Classification Report:")
    print(report)
    print(f"{model.capitalize()} Matthews Correlation Coefficient:", mcc)
    print(f"{model.capitalize()} Precision Score:", prec)
    
    # Add predictions and probabilities to the prediction DataFrame
    to_predict[f'{model}_predictions'] = preds
    # Save probabilities as lists.
    to_predict[f'{model}_probabilities'] = preds_proba.tolist() if isinstance(preds_proba, np.ndarray) else preds_proba
    
    # Save the updated DataFrame to an Excel file
    output_filename = f'{model}_predictions_testsize_{str(test_size).replace(".", "_")}.xlsx'
    to_predict.to_excel(output_filename, index=False)
    print(f"Predictions for {model} saved to {output_filename}")
    
    return to_predict


if __name__ == '__main__':
    # training data
    training_data = pd.read_excel('C:/Users/PragnyaPataskar/Projects/Fake reviews/extracted_features.xlsx')
    df_to_predict = training_data
    
    # training and predicting the data for each models
    df_lr = train_model('logistic_regression', training_data=labeled_data, to_predict=df_to_predict, test_size=0.2)
    df_rf = train_model('random_forest', training_data=labeled_data, to_predict=df_to_predict, test_size=0.2)
    df_svm = train_model('svm', training_data=labeled_data, to_predict=df_to_predict, test_size=0.2)