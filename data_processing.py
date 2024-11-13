import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import pandas as pd


def preprocess_data(df):
    # Create a copy of the dataframe to avoid modifying the original
    df_processed = df.copy()
    df_processed.drop(columns='Hand',inplace=True)

    # Iterate over columns to handle categorical and numerical columns differently
    for column in df_processed.columns:
        if df_processed[column].dtype == 'object':  # Categorical data
            # Replace null values with the mode (most frequent value)
            mode_value = df_processed[column].mode()[0]
            df_processed[column] = df_processed[column].fillna(mode_value)
            
            # Apply label encoding for categorical values
            label_encoder = LabelEncoder()
            df_processed[column] = label_encoder.fit_transform(df_processed[column])
        
        else:  # Numerical data
            # Replace null values with the mean
            mean_value = df_processed[column].mean()
            df_processed[column] = df_processed[column].fillna(mean_value)
    
    return df_processed

# data=pd.read_csv("dementia_dataset.csv")
# preprocess_data(data.iloc[:,3:])


def select_chi_square_features(X, y, k='all'):
    # Apply SelectKBest with Chi-Square as the score function
    chi_selector = SelectKBest(score_func=chi2, k=k)
    X_kbest = chi_selector.fit_transform(X, y)

    # Get selected feature names and their scores
    selected_features = X.columns[chi_selector.get_support()]
    chi_scores = chi_selector.scores_[chi_selector.get_support()]

    # Create a DataFrame for selected features and their Chi-Square scores
    selected_features_df = pd.DataFrame({
        'Feature': selected_features,
        'Chi-Square Score': chi_scores
    }).sort_values(by='Chi-Square Score', ascending=False)

    print("Selected Features Based on Chi-Square Test:\n", selected_features_df)

    # Return DataFrame with selected features
    return pd.DataFrame(X_kbest, columns=selected_features)


from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

def k_fold_cross_validation(model, X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = []
    precision_list, recall_list, f1_list, support_list = [], [], [], []
    for train_index, test_index in kf.split(X):
        # Split the data using iloc for integer-based indexing
        X_train = X.iloc[train_index] if isinstance(X, pd.DataFrame) else X[train_index]
        X_test = X.iloc[test_index] if isinstance(X, pd.DataFrame) else X[test_index]

        y_train = y.iloc[train_index] if isinstance(y, pd.DataFrame) else y[train_index]
        y_test = y.iloc[test_index] if isinstance(y, pd.DataFrame) else y[test_index]

        # Fit the model and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate accuracy and store it
        accuracy = accuracy_score(y_test, y_pred)
        scores.append(accuracy)

        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=np.unique(y), zero_division=0)
        
        # Append metrics to lists for aggregation
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        support_list.append(support)

    # Calculate average metrics across all folds
    avg_precision = np.mean(precision_list, axis=0)
    avg_recall = np.mean(recall_list, axis=0)
    avg_f1 = np.mean(f1_list, axis=0)
    total_support = np.sum(support_list, axis=0)

    # Display fold-wise accuracy
    avg_score = np.mean(scores)
    print(f"Accuracy scores for each fold: {scores}")
    print(f"Average accuracy across {k} folds: {avg_score:.4f}")

    # Create a DataFrame for the aggregated classification report
    report_df = pd.DataFrame({
        'Precision': avg_precision,
        'Recall': avg_recall,
        'F1-Score': avg_f1,
        'Support': total_support
    }, index=np.unique(y))

    print("\nAggregated Classification Report across all folds:")
    print(report_df)
    
    return avg_score


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Remove or Combine Highly Correlated Features
def remove_high_corr_features(df, threshold=0.9):
    """
    Remove features that are highly correlated (above the threshold).
    If correlation is above threshold, drop one of the two.
    """
    corr_matrix = df.corr()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df = df.drop(columns=to_drop)
    return df

# 2. Create Interaction Features
def create_interaction_features(df, features):
    """
    Create interaction terms by multiplying given pairs of features.
    """
    for f1, f2 in features:
        df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
    return df

# 3. Create Polynomial Features
def create_polynomial_features(df, features, degree=2):
    """
    Create polynomial features (up to given degree) for selected features.
    """
    for feature in features:
        for d in range(2, degree + 1):
            df[f'{feature}^{d}'] = df[feature] ** d
    return df

# 4. Scale Features
def scale_features(df, features):
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

# 5. Bin Continuous Features
def bin_continuous_features(df, feature_bins):
    for feature, bins in feature_bins.items():
        if isinstance(bins, int):  # If bins is an integer, use pd.cut to bin into equal-sized bins
            df[f'{feature}_binned'] = pd.cut(df[feature], bins=bins, labels=False)
        else:  # If bins is a list, use pd.cut to bin based on given bin edges
            df[f'{feature}_binned'] = pd.cut(df[feature], bins=bins, labels=False)
    return df

# Example function to apply all steps
def apply_feature_engineering(df):
    # Step 1: Remove highly correlated features
    df = remove_high_corr_features(df)

    # Step 2: Create interaction features (example: 'Age' x 'MMSE' and 'Age' x 'CDR')
    interaction_features = [('Age', 'MMSE'), ('Age', 'CDR')]
    df = create_interaction_features(df, interaction_features)

    # Step 3: Create polynomial features (example: create squares of 'Age' and 'MMSE')
    polynomial_features = ['Age', 'MMSE']
    df = create_polynomial_features(df, polynomial_features)

    # Step 4: Scale numerical features (example: scaling 'Age', 'MMSE', and 'CDR')
    scale_features_list = ['Age', 'MMSE', 'CDR']
    df = scale_features(df, scale_features_list)

    # Step 5: Bin continuous features (example: bin 'Age' and 'MMSE' into ranges)
    feature_bins = {
        'Age': [50, 60, 70, 80, 90],  # Specify the bins manually for Age
        'MMSE': [0, 17, 23, 30]       # MMSE: 0-17 (Severe), 18-23 (Mild), 24-30 (Normal)
    }
    df = bin_continuous_features(df, feature_bins)

    return df

