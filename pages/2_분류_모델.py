import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

st.title("분류 모델")

# 데이터 로드 함수
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jeongwon164/data/refs/heads/main/stroke.csv"
    df = pd.read_csv(url)
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 데 실패했습니다: {e}")
    st.stop()

# 속성 이름 매핑 (영어 -> 우리말)
feature_names_kr = {
    'age': '나이',
    'avg_glucose_level': '평균 혈당',
    'bmi': '체질량지수',
    'hypertension': '고혈압',
    'heart_disease': '심장병'
}

# 기본 선택: 체질량지수(bmi)를 뺀 넷
default_features = ['age', 'avg_glucose_level', 'hypertension', 'heart_disease']
default_features_kr = [feature_names_kr[f] for f in default_features]

st.sidebar.header("모델 설정")
selected_kr = st.sidebar.multiselect(
    "입력으로 사용할 속성 선택",
    options=list(feature_names_kr.values()),
    default=default_features_kr
)

# 두 개보다 적게 고르면 안내문 표시 후 멈춤
if len(selected_kr) < 2:
    st.warning("분류 모델을 학습하기 위해 속성을 2개 이상 선택해 주세요.")
    st.stop()

kr_to_en = {v: k for k, v in feature_names_kr.items()}
selected_features = [kr_to_en[name] for name in selected_kr]

# 번호 순으로 정렬한 뒤 10명씩 묶어 앞 3명 테스트용, 나머지 7명 훈련용
df = df.sort_index().reset_index(drop=True)

test_indices = []
train_indices = []

for i in range(len(df)):
    if (i % 10) < 3:
        test_indices.append(i)
    else:
        train_indices.append(i)

train_df = df.iloc[train_indices].copy()
test_df = df.iloc[test_indices].copy()

# 체질량지수(bmi) 결측치 처리 (고른 경우에만 훈련용의 중앙값으로 채움)
if 'bmi' in df.columns:
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    train_df['bmi'] = pd.to_numeric(train_df['bmi'], errors='coerce')
    test_df['bmi'] = pd.to_numeric(test_df['bmi'], errors='coerce')

if 'bmi' in selected_features:
    train_bmi_median = train_df['bmi'].median()
    train_df['bmi'] = train_df['bmi'].fillna(train_bmi_median)
    test_df['bmi'] = test_df['bmi'].fillna(train_bmi_median)

# X, y 분리 (stroke: 1=뇌졸중, 0=아님)
X_train = train_df[selected_features]
y_train = train_df['stroke']
X_test = test_df[selected_features]
y_test = test_df['stroke']

# 모델 학습
# 1. 로지스틱 회귀
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)

lr_train_acc = accuracy_score(y_train, lr_model.predict(X_train_scaled))
lr_test_acc = accuracy_score(y_test, lr_model.predict(X_test_scaled))

# 2. 의사결정트리 (질문 최대 3번, 마지막 마디 5명 미만 분할 금지, 난수 고정)
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)

dt_train_acc = accuracy_score(y_train, dt_model.predict(X_train))
dt_test_acc = accuracy_score(y_test, dt_model.predict(X_test))

# 3. 기준 모델 (입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)
dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_model.fit(X_train, y_train)

dummy_train_acc = accuracy_score(y_train, dummy_model.predict(X_train))
dummy_test_acc = accuracy_score(y_test, dummy_model.predict(X_test))

# 정확도 카드 표시
st.subheader("모델 정확도 비교")

col1, col2, col3 = st.columns(3)

def render_metric_card(title, test_acc, train_acc):
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 10px; border: 1px solid #e0e0e0;">
        <h4 style="margin: 0; color: #31333F; font-size: 16px;">{title}</h4>
        <h1 style="margin: 10px 0; color: #1f77b4; font-size: 32px;">{test_acc:.4f}</h1>
        <p style="margin: 0; font-size: 13px; color: #555;">훈련 정확도: {train_acc:.4f} &nbsp;|&nbsp; 테스트 정확도: {test_acc:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

with col1:
    render_metric_card("로지스틱 회귀<br>(확률로 답하는 모델)", lr_test_acc, lr_train_acc)

with col2:
    render_metric_card("의사결정트리<br>(질문으로 답하는 모델)", dt_test_acc, dt_train_acc)

with col3:
    render_metric_card("기준 모델<br>(많은 쪽으로만 답하는 모델)", dummy_test_acc, dummy_train_acc)

st.markdown("---")

# 산점도 및 결정 경계 시각화
st.subheader("데이터 산점도 및 결정 경계")

col_a, col_b = st.columns(2)
with col_a:
    x_axis_kr = st.selectbox("가로축 속성", options=selected_kr, index=0)
with col_b:
    default_y_idx = 1 if len(selected_kr) > 1 else 0
    y_axis_kr = st.selectbox("세로축 속성", options=selected_kr, index=default_y_idx)

x_axis = kr_to_en[x_axis_kr]
y_axis = kr_to_en[y_axis_kr]

# 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워둠
other_features = [f for f in selected_features if f != x_axis and f != y_axis]
median_values = {f: test_df[f].median() for f in other_features}

if other_features:
    median_desc = ", ".join([f"**{feature_names_kr[f]}**: {val:.2f}" for f, val in median_values.items()])
    st.write(f"ℹ️ 두 축이 아닌 나머지 속성은 테스트 데이터의 중앙값에 고정했습니다: {median_desc}")

# Plotly 산점도 생성
test_df['stroke_label'] = test_df['stroke'].map({1: '뇌졸중 (양성)', 0: '아님'})

fig = px.scatter(
    test_df,
    x=x_axis,
    y=y_axis,
    color='stroke_label',
    color_discrete_map={'뇌졸중 (양성)': '#ff4b4b', '아님': '#1f77b4'},
    labels={x_axis: x_axis_kr, y_axis: y_axis_kr, 'stroke_label': '실제 뇌졸중 여부'},
    title=f"{x_axis_kr} (가로) vs {y_axis_kr} (세로)"
)

# 배경 영역 예측용 그리드 생성
x_min, x_max = test_df[x_axis].min(), test_df[x_axis].max()
y_min, y_max = test_df[y_axis].min(), test_df[y_axis].max()

x_range = x_max - x_min
y_range = y_max - y_min
