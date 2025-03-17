from typing import Tuple, Union
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, matthews_corrcoef

def svm_training_and_prediction(
    training_data: pd.DataFrame, 
    test_size: float, 
    to_predict: pd.DataFrame, 
    text_column: str = 'review', 
    output_column: str = 'label_encoded'
) -> Tuple[np.ndarray, np.ndarray, Union[str, dict], float, Union[float, np.ndarray]]:
    """
    Train and predict with an SVM model using TF-IDF for text features.
    
    Args:
        training_data (pd.DataFrame): DataFrame with training data.
        test_size (float): Proportion for test split.
        to_predict (pd.DataFrame): DataFrame for new predictions.
        text_column (str): Column name with text (default 'review').
        output_column (str): Column name with labels (default 'label_encoded').
        
    Returns:
        Tuple containing:
          - Final predicted labels on new data.
          - Predicted probabilities on new data.
          - Classification report on test split.
          - Matthews correlation coefficient on test split.
          - Weighted precision score on test split.
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', SVC(probability=True, random_state=42))
    ])
    
    X = training_data[text_column]
    y = training_data[output_column].astype(str)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=21)
    
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    class_report = classification_report(y_test, y_pred)
    precision_score_report = precision_score(y_test, y_pred, average='weighted')
    mcc_value = matthews_corrcoef(y_test, y_pred)
    
    final_prediction = pipeline.predict(to_predict[text_column])
    final_prediction_proba = pipeline.predict_proba(to_predict[text_column])
    
    return final_prediction, final_prediction_proba, class_report, mcc_value, precision_score_report
