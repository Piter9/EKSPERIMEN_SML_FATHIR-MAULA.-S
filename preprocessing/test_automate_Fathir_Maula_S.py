import pandas as pd
from automate_Fathir_Maula_S import preprocess_data

# 1. Load data mentah
raw_path = "dataset_raw/healthcare-dataset-stroke_raw.csv"
df = pd.read_csv(raw_path)

# 2. Tentukan parameter untuk menyimpan hasil
target_col = "stroke"
output_path = "preprocessing/test_preprocessor_pipeline.joblib"
label_path = "preprocessing/test_label.csv"
meta_path = "preprocessing/test_healthcare-dataset-stroke_preprocessing_automated.csv"

# 3. Panggil fungsi preprocessing
preprocess_data(df, target_col, output_path, label_path, meta_path)
