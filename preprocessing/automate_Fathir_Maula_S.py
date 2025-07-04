from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
 
def preprocess_data(data, target_column, save_path, label_path, data_path):
    # Menentukan fitur numerik dan kategoris
    column_names = data.columns
    # Mendapatkan nama kolom tanpa kolom target
    column_names = data.columns.drop(target_column)
    data.drop(columns=['id'], inplace=True, axis=1)
    data['hypertension'] = data["hypertension"].astype('object')
    data['heart_disease'] = data["heart_disease"].astype('object')
    data['stroke'] = data["stroke"].astype('object')
    
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    
 
    # Membuat DataFrame kosong dengan nama kolom
    df_header = pd.DataFrame(columns=column_names)
 
    # Menyimpan nama kolom sebagai header tanpa data
    df_header.to_csv(label_path, index=False)
    print(f"Nama kolom berhasil disimpan ke: {label_path}")
 
   
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)
 
    # Pipeline untuk fitur numerik
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
 
    # Pipeline untuk fitur kategoris
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
 
    # Column Transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    data_olah = preprocessor.fit_transform(data.drop(columns=[target_column]))    
    data_olah = np.concatenate((data_olah, data[[target_column]].values), axis=1)
    # Mengubah hasil preprocessing menjadi DataFrame
    data_olah = pd.DataFrame(data_olah, columns=np.append(preprocessor.get_feature_names_out(), target_column))
    data_olah.to_csv(data_path, index=False)
    print(f"Data berhasil disimpan ke: {data_path}")

    # Memisahkan target
    X = data.drop(columns=[target_column])
    y = data[target_column]
 
    # Membagi data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
    # Fitting dan transformasi data pada training set
    X_train = preprocessor.fit_transform(X_train)
    # Transformasi data pada testing set
    X_test = preprocessor.transform(X_test)
    # Simpan pipeline
    dump(preprocessor, save_path)
 
    return X_train, X_test, y_train, y_test