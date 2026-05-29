# -----------------------------
# One-Hot Encoding
# -----------------------------
input_data_encoded = pd.get_dummies(
    input_data_raw,
    columns=['성별'],
    drop_first=True,
    dtype=float
)

# -----------------------------
# 컬럼 맞추기
# -----------------------------
final_input_data = pd.DataFrame(
    columns=X_columns,
    index=[0]
).fillna(0)

for col in input_data_encoded.columns:

    if col in final_input_data.columns:

        final_input_data[col] = input_data_encoded[col]

# -----------------------------
# 숫자형 강제 변환
# -----------------------------
final_input_data = final_input_data.apply(
    pd.to_numeric,
    errors='coerce'
)

final_input_data = final_input_data.fillna(0)

final_input_data = final_input_data.astype(float)

# -----------------------------
# 스케일링
# -----------------------------
input_scaled = scaler.transform(final_input_data)
