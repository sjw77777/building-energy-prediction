import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --- 1. 페이지 설정 및 디자인 ---
st.set_page_config(
    page_title="스마트 빌딩 에너지 소비량 다중 회귀 분석 리포트",
    page_icon="⚡",
    layout="wide"
)

# 한글 폰트 및 마이너스 기호 깨짐 방지
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 메인 타이틀 및 소개 ---
st.title("🏢 스마트 빌딩 에너지 소비량 예측 및 다중 선형 회귀 분석 리포트")
st.markdown("""
본 대시보드는 건물 내외부 환경 요인 및 설비 가동 상태 데이터를 활용하여 **다중 선형 회귀(Multiple Linear Regression)** 
모델을 구축하고, 에너지 소비량에 미치는 영향을 다각도로 분석 및 예측하기 위해 제작되었습니다.
""")
st.markdown("---")

# --- 3. 데이터 로딩 및 전처리 (캐시 적용) ---
@st.cache_data
def load_and_preprocess_data():
    # 데이터 로드
    df = pd.read_csv('Energy_consumption_cleaned (1).csv')
    
    # Timestamp(날짜) 컬럼 존재 시 모델링용에서 제외
    if 'Timestamp' in df.columns:
        df_model_base = df.drop(columns=['Timestamp'])
    else:
        df_model_base = df.copy()
        
    # 범주형 변수 원-핫 인코딩 (HVACUsage, LightingUsage, DayOfWeek, Holiday 등)
    df_encoded = pd.get_dummies(df_model_base, drop_first=True)
    
    return df, df_encoded

try:
    df_raw, df_encoded = load_and_preprocess_data()

    # --- 4. 섹션 1: 데이터 탐색적 분석 (EDA) ---
    st.header("📌 1. 데이터 탐색적 분석 (Exploratory Data Analysis)")
    
    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["📋 데이터 샘플 및 구조", "📊 통계적 요약", "🔥 상관관계 히트맵"])

    with eda_tab1:
        st.subheader("전처리된 데이터셋 상위 샘플 (Top 5)")
        st.dataframe(df_raw.head(), use_container_width=True)
        st.info(f"💡 전체 데이터 크기: **{df_raw.shape[0]}행 × {df_raw.shape[1]}열** (Timestamp 제외 총 10개 분석 변수 활용)")

    with eda_tab2:
        st.subheader("수치형 변수 기술통계량")
        st.dataframe(df_raw.describe(), use_container_width=True)

    with eda_tab3:
        st.subheader("전체 수치형 변수 간 상관관계 분석")
        st.write("변수들 간의 선형 연관성을 파악하기 위해 피어슨 상관계수를 시각화했습니다.")
        
        numeric_df = df_raw.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)

    st.markdown("---")

    # --- 5. 섹션 2: 머신러닝 모델 학습 (Train/Test Split & Linear Regression) ---
    st.header("🤖 2. 다중 선형 회귀 모델 학습 및 성능 평가")
    
    # 타겟 및 독립변수 설정 (10개 변수 전면 활용 - 범주형 원핫인코딩 포함)
    target_col = 'EnergyConsumption'
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    # 데이터 분할 (Train:Test = 8:2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 다중 선형 회귀 모델 학습
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 예측 수행
    y_pred = model.predict(X_test)

    # 성능 지표 계산
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # 성과 지표 대시보드 출력
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("결정 계수 (R² Score)", f"{r2:.4f}", help="모델의 설명력 (1에 가까울수록 완벽)")
    m_col2.metric("RMSE (평균제곱근오차)", f"{rmse:.4f}", help="예측 오차의 표준적인 크기")
    m_col3.metric("MAE (평균절대오차)", f"{mae:.4f}", help="실제값과 예측값 절대 오차 평균")
    m_col4.metric("학습/테스트 데이터 비율", f"{len(X_train)} / {len(X_test)} 건")

    st.markdown("<br>", unsafe_allow_html=True)

    # 회귀 계수 및 잔차 시각화 탭
    model_tab1, model_tab2 = st.tabs(["📋 모델 회귀 계수 (Weights) 분석", "📈 실제값 vs 예측값 & 잔차 시각화"])

    with model_tab1:
        st.subheader("변수별 회귀 계수(가중치) 확인")
        st.write("각 독립 변수가 종속 변수(EnergyConsumption)에 미치는 영향력의 크기와 방향을 나타냅니다.")
        
        coef_df = pd.DataFrame({
            '변수명 (Feature)': X.columns,
            '회귀 계수 (Coefficient)': model.coef_
        }).sort_values(by='회귀 계수 (Coefficient)', ascending=False)
        
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
        st.success(f"📌 **모델 절편 (Intercept):** `{model.intercept_:.4f}`")

    with model_tab2:
        st.subheader("모델 예측력 검증 그래프")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. 실제값 vs 예측값 산점도
        ax1.scatter(y_test, y_pred, alpha=0.5, color='royalblue', edgecolors='k')
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
        ax1.set_xlabel("Actual Energy Consumption")
        ax1.set_ylabel("Predicted Energy Consumption")
        ax1.set_title("Actual vs Predicted Energy Consumption")
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 2. 잔차(Residual) 분포 히스토그램
        residuals = y_test - y_pred
        sns.histplot(residuals, kde=True, ax=ax2, color='purple', bins=30)
        ax2.set_xlabel("Residual (Actual - Predicted)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Residual Distribution (Error Analysis)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        st.pyplot(fig)

    st.markdown("---")

    # --- 6. 섹션 3: 실시간 인터랙티브 예측 시뮬레이션 ---
    st.header("🎛️ 3. 실시간 빌딩 에너지 소비량 시뮬레이션")
    st.markdown("환경 및 설비 조건을 실시간으로 조작하여, 머신러닝 모델이 산출하는 예상 에너지 소비량을 즉각적으로 확인해보세요.")

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        st.subheader("🌡️ 수치형 환경 변수 조절")
        in_temp = st.slider("Temperature (온도)", float(df_raw['Temperature'].min()), float(df_raw['Temperature'].max()), float(df_raw['Temperature'].mean()))
        in_hum = st.slider("Humidity (습도)", float(df_raw['Humidity'].min()), float(df_raw['Humidity'].max()), float(df_raw['Humidity'].mean()))
        in_sqft = st.slider("SquareFootage (건물 면적)", float(df_raw['SquareFootage'].min()), float(df_raw['SquareFootage'].max()), float(df_raw['SquareFootage'].mean()))
        in_occ = st.slider("Occupancy (재실 인원)", int(df_raw['Occupancy'].min()), int(df_raw['Occupancy'].max()), int(df_raw['Occupancy'].mean()))
        in_reno = st.slider("RenewableEnergy (신재생에너지)", float(df_raw['RenewableEnergy'].min()), float(df_raw['RenewableEnergy'].max()), float(df_raw['RenewableEnergy'].mean()))

    with sim_col2:
        st.subheader("⚙️ 범주형 설비 및 일정 상태 조절")
        in_hvac = st.selectbox("HVACUsage (냉난방기 사용 여부)", df_raw['HVACUsage'].unique())
        in_light = st.selectbox("LightingUsage (조명 사용 여부)", df_raw['LightingUsage'].unique())
        in_day = st.selectbox("DayOfWeek (요일)", df_raw['DayOfWeek'].unique())
        in_holiday = st.selectbox("Holiday (휴일 여부)", df_raw['Holiday'].unique())
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 범주형 입력값은 원-핫 인코딩 과정을 거쳐 모델의 전체 피처 공간에 자동 동기화됩니다.")

    # 사용자 입력 데이터 프레임 구성 및 인코딩 맞춤화
    user_input_raw = pd.DataFrame({
        'Temperature': [in_temp],
        'Humidity': [in_hum],
        'SquareFootage': [in_sqft],
        'Occupancy': [in_occ],
        'RenewableEnergy': [in_reno],
        'HVACUsage': [in_hvac],
        'LightingUsage': [in_light],
        'DayOfWeek': [in_day],
        'Holiday': [in_holiday]
    })
    
    # 모델 학습 때와 동일하게 원-핫 인코딩 적용 후 컬럼 구조 맞추기
    user_input_encoded = pd.get_dummies(user_input_raw)
    user_input_encoded = user_input_encoded.reindex(columns=X.columns, fill_value=0)

    # 예측 수행
    predicted_energy = model.predict(user_input_encoded)[0]

    st.markdown("---")
    
    # 최종 결과 출력 박스
    st.success(f"### ⚡ 시뮬레이션 결과 - 예상 에너지 소비량: **{predicted_energy:,.2f} kWh**")
    
    st.markdown("---")
    st.caption("© 2026 Electronic Engineering & Semiconductor Track - Smart Building Energy Analytics System")

except FileNotFoundError:
    st.error("⚠️ 데이터 파일(`Energy_consumption_cleaned (1).csv`)을 찾을 수 없습니다. 프로젝트 루트 폴더에 파일이 있는지 확인해주세요.")
