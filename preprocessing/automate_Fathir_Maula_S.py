import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from joblib import dump
from sklearn.model_selection import train_test_split

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    
    df.drop(columns=['id'], inplace=True)
    df['hypertension'] = df["hypertension"].astype('object')
    df['heart_disease'] = df["heart_disease"].astype('object')
    df['stroke'] = df["stroke"].astype('object')

    numeric = []
    categorical = []
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col].fillna(df[col].mean(), inplace=True)
        elif df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
   

    for feature in numeric:
        mean = df[feature].mean()
        std = df[feature].std()
        df[feature] = (df[feature] - mean) / std
    

    for feature in numeric:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    

    le = LabelEncoder()
    for feature in categorical:
        df[feature] = le.fit_transform(df[feature])
    
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    df.to_csv("healthcare-dataset-stroke_preprocessing-automated.csv", index=False)
    return df
