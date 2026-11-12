import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. 페이지 및 스타일 설정 ---
st.set_page_config(
    page_title="스마트 빌딩 에너지 소비량 예측 시스템",
    page_icon="🏢",
    layout="wide"
)

# 한글 폰트 깨짐 방지
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 타이틀 및 개요 ---
st.title("🏢 머신러닝 기반 스마트 빌딩 에너지 소비량 예측 및 분석")
st.markdown("""
본 웹 애플리케이션은 **다중 선형 회귀(Multiple Linear Regression)** 모델을 활용하여 건물 내외부 환경 변수와 에너지 소비량 간의 상관관계를 분석하고, 실시간으로 에너지 사용량을 예측하기 위해 제작되었습니다.
""")
st.markdown("---")

# --- 3. 데이터 로드 및 전처리 ---
@st.cache_data
def load_data():
    return pd.read_csv('Energy_consumption_cleaned (1).csv')

try:
    df = load_data()

    # 상단 요약 탭 구성
    tab1, tab2 = st.tabs(["📋 데이터셋 개요 및 통계", "📊 변수 간 상관관계 분석"])

    with tab1:
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("데이터 샘플 (Top 5)")
            st.dataframe(df.head(), use_container_width=True)
        with col_t2:
            st.subheader("데이터 통계 요약 (describe)")
            st.dataframe(df.describe(), use_container_width=True)

    with tab2:
        st.subheader("변수 간 상관관계 히트맵 (Correlation Heatmap)")
        st.write("수치형 변수들 사이의 선형적 상관관계를 시각적으로 파악합니다.")
        
        numeric_df = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='YlGnBu', fmt=".2f", linewidths=.5, ax=ax)
        st.pyplot(fig)

    st.markdown("---")

    # --- 4. 머신러닝 모델 학습 (Train/Test Split & Linear Regression) ---
    st.header("🤖 다중 선형 회귀 모델(ML) 학습 및 성능 평가")
    
    # 독립변수(Feature)와 종속변수(Target) 설정
    features = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy', 'RenewableEnergy']
    target = 'EnergyConsumption'

    X = df[features]
    y = df[target]

    # 데이터셋 분할 (학습:테스트 = 8:2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 객체 생성 및 학습
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 예측 수행
    y_pred = model.predict(X_test)

    # 성능 지표 계산
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 성과 지표 대시보드 출력
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("결정 계수 (R² Score)", f"{r2:.4f}", help="모델의 설명력을 나타냅니다 (1에 가까울수록 우수)")
    m_col2.metric("평균 제곱근 오차 (RMSE)", f"{rmse:.4f}", help="실제값과 예측값 간의 평균 오차 크기")
    m_col3.metric("학습 데이터 크기", f"{len(X_train)} 건")
    m_col4.metric("테스트 데이터 크기", f"{len(X_test)} 건")

    # 회귀 계수(가중치) 시각화
    with st.expander("🔍 모델 세부 파라미터 (회귀 계수 및 절편) 확인"):
        coef_df = pd.DataFrame({
            '독립 변수 (Feature)': features,
            '회귀 계수 (Coefficient)': model.coef_
        })
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
        st.info(f"📌 **모델 절편 (Intercept):** `{model.intercept_:.4f}`")

    st.markdown("---")

    # --- 5. 실시간 인터랙티브 예측 시뮬레이션 ---
    st.header("🎛️ 실시간 에너지 소비량 예측 시뮬레이션")
    st.write("아래의 슬라이더를 조작하여 환경 변수 값을 변경하고, 머신러닝 모델이 예측하는 실시간 에너지 소비량을 확인하세요.")

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        in_temp = st.slider("🌡️ 온도 (Temperature)", float(X['Temperature'].min()), float(X['Temperature'].max()), float(X['Temperature'].mean()))
        in_hum = st.slider("💧 습도 (Humidity)", float(X['Humidity'].min()), float(X['Humidity'].max()), float(X['Humidity'].mean()))
        in_sqft = st.slider("🏢 건물 면적 (SquareFootage)", float(X['SquareFootage'].min()), float(X['SquareFootage'].max()), float(X['SquareFootage'].mean()))

    with sim_col2:
        in_occ = st.slider("👥 재실 인원 (Occupancy)", int(X['Occupancy'].min()), int(X['Occupancy'].max()), int(X['Occupancy'].mean()))
        in_reno = st.slider("🌱 신재생에너지 생산량 (RenewableEnergy)", float(X['RenewableEnergy'].min()), float(X['RenewableEnergy'].max()), float(X['RenewableEnergy'].mean()))

    # 사용자 입력 기반 예측 실행
    user_input_df = pd.DataFrame([[in_temp, in_hum, in_sqft, in_occ, in_reno]], columns=features)
    predicted_energy = model.predict(user_input_df)[0]

    st.markdown("---")
    
    # 결과 강조 박스
    st.success(f"### ⚡ 예측된 예상 에너지 소비량: **{predicted_energy:,.2f} kWh**")
    
    st.markdown("---")
    st.caption("© 2026 Electronic Engineering & Semiconductor Track - Energy Prediction Project")

except FileNotFoundError:
    st.error("⚠️ 데이터 파일(`Energy_consumption_cleaned (1).csv`)을 찾을 수 없습니다. 경로와 파일명을 다시 확인해주세요.")
