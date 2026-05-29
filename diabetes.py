from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# 기본 설정
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent


def load_artifact(filename):

    artifact_path = BASE_DIR / filename

    if not artifact_path.exists():

        st.error(f"""
❌ 파일을 찾을 수 없습니다.

파일명: {filename}

예상 위치:
{artifact_path}

👉 프로젝트 폴더에 파일을 넣어주세요.
""")
        st.stop()

    return joblib.load(artifact_path)


# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="AI 당뇨 예측 시스템",
    page_icon="🩺",
    layout="wide"
)


# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>

.main {
    background-color: #f5f9ff;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1565c0;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #1976d2, #42a5f5);
    color: white;
    font-size: 20px;
    font-weight: bold;
    border-radius: 12px;
    height: 55px;
    border: none;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #1565c0, #1e88e5);
}

.result-box {
    background-color: #e3f2fd;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    margin-top: 25px;
}

.result-text {
    font-size: 38px;
    font-weight: bold;
    color: #0d47a1;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# 모델 로드 (여기만 변경됨)
# -----------------------------
model = load_artifact("rf_model.joblib")
scaler = load_artifact("scaler.pkl")
X_columns = load_artifact("X_columns.pkl")
X_mean = load_artifact("X_mean.pkl")


# -----------------------------
# 제목
# -----------------------------
st.markdown('<div class="title">🩺 AI 당뇨 예측 시스템</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">건강 데이터를 기반으로 당뇨 위험도를 예측합니다.</div>', unsafe_allow_html=True)


# -----------------------------
# 입력
# -----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    total_chol = st.number_input("총콜레스테롤", 200.0)
    stab_glu = st.number_input("공복혈당", 90.0)
    hdl = st.number_input("HDL콜레스테롤", 40.0)
    ratio = st.number_input("콜레스테롤비율", 5.0)
    glyhb = st.number_input("당화혈색소", 5.5)

with col2:
    age = st.number_input("나이", 45)
    gender = st.selectbox("성별", ["female", "male"])
    height = st.number_input("키(cm)", 165.0)
    weight = st.number_input("몸무게(kg)", 70.0)
    waist = st.number_input("허리둘레", 35.0)

with col3:
    hip = st.number_input("엉덩이둘레", 40.0)
    bp_1s = st.number_input("1차 수축기혈압", 130.0)
    bp_1d = st.number_input("1차 이완기혈압", 80.0)
    bp_2s = st.number_input("2차 수축기혈압", 140.0)
    bp_2d = st.number_input("2차 이완기혈압", 90.0)
    time_ppn = st.number_input("식후경과시간(분)", 180.0)

st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# 예측
# -----------------------------
if st.button("🧠 당뇨 예측하기"):

    input_data = pd.DataFrame([[
        total_chol, stab_glu, hdl, ratio, glyhb,
        age, gender, height, weight,
        bp_1s, bp_1d, bp_2s, bp_2d,
        waist, hip, time_ppn
    ]], columns=[
        '총콜레스테롤','공복혈당','HDL콜레스테롤','콜레스테롤비율','당화혈색소',
        '나이','성별','키','몸무게',
        '1차_수축기혈압','1차_이완기혈압','2차_수축기혈압','2차_이완기혈압',
        '허리둘레','엉덩이둘레','식후경과시간'
    ])

    # Feature Engineering
    input_data["복부비만도"] = input_data["허리둘레"] / input_data["엉덩이둘레"]
    input_data["BMI"] = input_data["몸무게"] / ((input_data["키"]/100)**2)
    input_data["나쁜콜레스테롤_비율"] = (
        (input_data["총콜레스테롤"] - input_data["HDL콜레스테롤"]) /
        input_data["HDL콜레스테롤"]
    )

    # One-hot
    input_encoded = pd.get_dummies(input_data, columns=["성별"], drop_first=True, dtype=float)

    # 컬럼 맞추기
    final_input = pd.DataFrame(columns=X_columns, index=[0]).fillna(0)

    for c in input_encoded.columns:
        if c in final_input.columns:
            final_input[c] = input_encoded[c]

    final_input = final_input.astype(float)

    # scaling
    scaled = scaler.transform(final_input)

    # predict
    pred = model.predict(scaled)
    prob = model.predict_proba(scaled)[0][1] * 100

    st.markdown(f"""
    <div class="result-box">
        <div style="font-size:22px;">결과</div>
        <div class="result-text">
            {"⚠ 당뇨 위험" if pred[0]==1 else "✅ 정상"}
        </div>
        <div style="font-size:26px;">
            확률: {prob:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)
