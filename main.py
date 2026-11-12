import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 한글 폰트 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(
    page_title="건물 에너지 예측 분석",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ 머신러닝 기반 건물 에너지 소비량 예측 시스템")
st.write("전처리된 데이터를 바탕으로 다중 선형 회귀 모델을 학습시키고, 환경 조건에 따른 에너지 사용량을 예측합니다.")

# 1. 데이터 로드 (캐시 적용)
@st.cache_data
def load_data():
    return pd.read_csv('Energy_consumption_cleaned (1).csv')

try:
    df = load_data()

    # 데이터 요약 섹션 (에러 원인이었던 info 대신 dtypes와 describe 사용)
    with st.expander("📁 데이터셋 구조 및 샘플 확인하기"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**상위 5개 데이터 샘플**")
            st.dataframe(df.head(), use_container_width=True)
        with col2:
            st.write("**변수별 데이터 타입 및 결측치 현황**")
            info_df = pd.DataFrame({
                '컬럼명': df.columns,
                '데이터 타입': df.dtypes.astype(str),
                '결측치 수': df.isnull().sum()
            })
            st.dataframe(info_df, use_container_width=True, hide_index=True)

    # 상관관계 시각화
    st.subheader("📊 변수 간 상관관계 분석 (Heatmap)")
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.markdown("---")

    # 2. 모델 학습 및 평가 (Train / Test Split)
    st.subheader("🤖 다중 선형 회귀 모델 학습 및 평가")
    
    # 주요 수치형 독립 변수 설정
    features = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy']
    target = 'EnergyConsumption'

    X = df[features]
    y = df[target]

    # 데이터 분할 (8:2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 회귀 모델 학습
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # 예측 및 성능 평가
    predictions = regressor.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # 지표 출력
    m1, m2, m3 = st.columns(3)
    m1.metric("결정 계수 (R² Score)", f"{r2:.4f}")
    m2.metric("평균 제곱근 오차 (RMSE)", f"{rmse:.4f}")
    m3.metric("학습 데이터 수", f"{len(X_train)}건")

    # 회귀 계수 확인
    with st.expander("🔍 모델 회귀 계수(가중치) 상세 보기"):
        coef_table = pd.DataFrame({'변수': features, '가중치(Coefficient)': regressor.coef_})
        st.dataframe(coef_table, use_container_width=True)
        st.write(f"**절편 (Intercept):** `{regressor.intercept_:.4f}`")

    st.markdown("---")

    # 3. 실시간 시뮬레이션 예측
    st.subheader("🎛️ 실시간 에너지 소비량 시뮬레이션")
    st.write("슬라이더를 조작하여 원하는 건물 환경 조건을 입력하고 예측된 에너지 소비량을 확인해보세요.")

    c1, c2 = st.columns(2)
    with c1:
        t_val = st.slider("온도 (Temperature)", float(X['Temperature'].min()), float(X['Temperature'].max()), float(X['Temperature'].mean()))
        h_val = st.slider("습도 (Humidity)", float(X['Humidity'].min()), float(X['Humidity'].max()), float(X['Humidity'].mean()))
        s_val = st.slider("건물 면적 (SquareFootage)", float(X['SquareFootage'].min()), float(X['SquareFootage'].max()), float(X['SquareFootage'].mean()))

    with c2:
        o_val = st.slider("재실 인원 (Occupancy)", int(X['Occupancy'].min()), int(X['Occupancy'].max()), int(X['Occupancy'].mean()))
        r_val = st.slider("신재생에너지 (RenewableEnergy)", float(X['RenewableEnergy'].min()), float(X['RenewableEnergy'].max()), float(X['RenewableEnergy'].mean()))

    # 사용자 입력 예측 실행
    user_data = pd.DataFrame([[t_val, h_val, s_val, o_val, r_val]], columns=features)
    pred_result = regressor.predict(user_data)[0]

    st.success(f"⚡ **예측된 에너지 소비량:** **{pred_result:.2f} kWh**")

except FileNotFoundError:
    st.error("데이터 파일을 찾을 수 없습니다. 파일명을 확인해 주세요.")
