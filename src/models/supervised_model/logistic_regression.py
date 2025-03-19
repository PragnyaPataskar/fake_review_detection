from typing import Tuple, Union
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, matthews_corrcoef, precision_score

def logistic_regression_training_and_prediction(
    training_data: pd.DataFrame,
    test_size: float,
    to_predict: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, Union[str, dict], float, Union[float, np.ndarray]]:
    """
    method to train and predict data with Logistic Regression model using TF-IDF embedding.
    
    Args:
        training_data (pd.DataFrame): Training data containing at least 'review' and 'label' columns.
        test_size (float): Proportion of training data to set aside for validation.
        to_predict (pd.DataFrame): Data that needs to be predicted. Must include a 'review' column.
        
    Returns:
        Tuple containing:
            - final_prediction_logistic_regression (np.ndarray): Predicted labels for to_predict.
            - final_prediction_proba_logistic_regression (np.ndarray): Predicted probabilities.
            - class_report (Union[str, dict]): Classification report from the validation split.
            - mcc_value (float): Matthews correlation coefficient from the validation split.
            - precision_score_report (Union[float, np.ndarray]): Precision score (weighted) from the validation split.
    """
    # Replace missing reviews with empty strings for both training and prediction data.
    training_data['review'] = training_data['review'].fillna("")
    to_predict['review'] = to_predict['review'].fillna("")
    
    # Vectorize the training text using TF-IDF
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(training_data['review'])
    
    tfidf_transformer = TfidfTransformer()
    vectorized_data = tfidf_transformer.fit_transform(X_counts)
    matrix = vectorized_data.toarray()
    
    # Split data into training and validation sets
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        matrix,
        training_data['label'].astype(str),
        test_size=test_size,
        random_state=21
    )
    
    # Train Logistic Regression classifier
    lr_clf = LogisticRegression(random_state=42, max_iter=1000)
    lr_clf.fit(x_train, y_train)
    
    # Evaluate on the validation set
    lr_preds = lr_clf.predict(x_test)
    class_report = classification_report(y_test, lr_preds)
    precision_score_report = precision_score(y_test, lr_preds, average='weighted')
    mcc_value = matthews_corrcoef(y_test, lr_preds)
    
    # Prepare the prediction data using the same TF-IDF transformation
    matrix_for_prediction = count_vect.transform(to_predict['review'])
    matrix_for_prediction = tfidf_transformer.transform(matrix_for_prediction)
    
    # Make final predictions and compute prediction probabilities
    final_prediction_logistic_regression = lr_clf.predict(matrix_for_prediction)
    final_prediction_proba_logistic_regression = lr_clf.predict_proba(matrix_for_prediction)
    
    return (final_prediction_logistic_regression,
            final_prediction_proba_logistic_regression,
            class_report,
            mcc_value,
            precision_score_report)