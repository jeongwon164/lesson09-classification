import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

st.title("채점표 (혼동행렬 및 성능 지표)")

# 데이터 로드 함수 ('분류 모델' 페이지와 동일한 방식)
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

# 기본 선택: 체질량지수(bmi)를 뺀 넷 ('분류 모델'과 동일)
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
    st.warning("모델을 평가하기 위해 속성을 2개 이상 선택해 주세요.")
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
lr_pred = lr_model.predict(X_test_scaled)

# 2. 의사결정트리
dt_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# 3. 기준 모델 (훈련용에서 많은 쪽으로만 답하는 모델)
dummy_model = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_model.fit(X_train, y_train)
dummy_pred = dummy_model.predict(X_test)

# 테스트용 인원수 및 뇌졸중 환자 수
total_test_count = len(y_test)
actual_stroke_count = int(y_test.sum())

st.markdown(f"**테스트용 사람 수:** {total_test_count}명 &nbsp;|&nbsp; **그중 실제 뇌졸중인 사람 수:** {actual_stroke_count}명")
st.markdown("---")

# 혼동행렬 (2x2 표) 그리기 함수
def display_confusion_matrix(model_title, y_true, y_pred):
    st.subheader(model_title)
    # sklearn confusion_matrix 순서: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    # 표 데이터 구성 (행: 실제, 열: 예측)
    cm_df = pd.DataFrame(
        [
            [tn, fp, tn + fp],
            [fn, tp, fn + tp],
            [tn + fn, fp + tp, total_test_count]
        ],
        index=['실제 아님 (0)', '실제 뇌졸중 (1)', '합계'],
        columns=['예측 아님 (0)', '예측 뇌졸중 (1)', '합계']
    )
    
    # 텍스트로 보기 쉽게 칸별 지표 표기 설명 추가
    st.markdown(f"""
    - **TN (진음성):** {tn}명 (실제 아님 $\rightarrow$ 아님 예측)
    - **FP (위양성):** {fp}명 (실제 아님 $\rightarrow$ 뇌졸중 예측)
    - **FN (위음성):** {fn}명 (실제 뇌졸중 $\rightarrow$ 아님 예측)
    - **TP (진양성):** {tp}명 (실제 뇌졸중 $\rightarrow$ 뇌졸중 예측)
    """)
    st.dataframe(cm_df, use_container_width=True)

col_cm1, col_cm2 = st.columns(2)
with col_cm1:
    display_confusion_matrix("로지스틱 회귀(확률로 답하는 모델)", y_test, lr_pred)
with col_cm2:
    display_confusion_matrix("의사결정트리(질문으로 답하는 모델)", y_test, dt_pred)

st.markdown("---")

# 성능 지표 한 표로 비교
st.subheader("성능 지표 비교표")

def get_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    # 재현율 분모: 실제 뇌졸중인 사람 수
    recall = recall_score(y_true, y_pred, zero_division=0)
    
    # 정밀도 분모: 뇌졸중이라 예측한 사람 수
    pred_positive = y_pred.sum()
    if pred_positive == 0:
        precision = None
        f1 = None
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return acc, recall, precision, f1

lr_acc, lr_rec, lr_prec, lr_f1 = get_metrics(y_test, lr_pred)
dt_acc, dt_rec, dt_prec, dt_f1 = get_metrics(y_test, dt_pred)
dummy_acc, dummy_rec, dummy_prec, dummy_f1 = get_metrics(y_test, dummy_pred)

def format_val(val):
    return f"{val:.4f}" if val is not None else "계산할 수 없음"

metrics_data = {
    "평가 지표": ["정확도 (Accuracy)", "재현율 (Recall)", "정밀도 (Precision)", "F1 Score"],
    "로지스틱 회귀(확률로 답하는 모델)": [
        format_val(lr_acc), format_val(lr_rec), format_val(lr_prec), format_val(lr_f1)
    ],
    "의사결정트리(질문으로 답하는 모델)": [
        format_val(dt_acc), format_val(dt_rec), format_val(dt_prec), format_val(dt_f1)
    ],
    "기준 모델(훈련용에서 많은 쪽으로만 답하는 모델)": [
        format_val(dummy_acc), format_val(dummy_rec), format_val(dummy_prec), format_val(dummy_f1)
    ]
}

metrics_df = pd.DataFrame(metrics_data)
st.table(metrics_df)

st.markdown("---")

# 환자 오류 목록 분석 (놓친 환자 vs 헛짚은 사람)
st.subheader("오류 환자 분석 목록")

model_choice = st.selectbox(
    "목록을 확인할 모델 선택",
    options=["로지스틱 회귀(확률로 답하는 모델)", "의사결정트리(질문으로 답하는 모델)"]
)

chosen_pred = lr_pred if "로지스틱 회귀" in model_choice else dt_pred

# 테스트 데이터 프레임에 예측값 및 인덱스 추가
eval_df = test_df.copy()
eval_df['pred'] = chosen_pred
eval_df['original_idx'] = test_indices

# 1. 실제 뇌졸중인데 아니라고 예측한 사람 (놓친 환자: FN)
fn_df = eval_df[(eval_df['stroke'] == 1) & (eval_df['pred'] == 0)].copy()
# 2. 아닌데 뇌졸중이라고 예측한 사람 (헛짚은 사람: FP)
fp_df = eval_df[(eval_df['stroke'] == 0) & (eval_df['pred'] == 1)].copy()

# 나이가 많은 순서로 정렬
fn_df = fn_df.sort_values(by='age', ascending=False).reset_index(drop=True)
fp_df = fp_df.sort_values(by='age', ascending=False).reset_index(drop=True)

fn_total_count = len(fn_df)
fp_total_count = len(fp_df)

# 보여줄 컬럼 정리 (번호, 나이, 평균 혈당, 고혈압 여부, 심장병 여부)
# 번호는 원래 데이터프레임의 인덱스 또는 순번 사용
def format_list_df(df_target):
    if df_target.empty:
        return pd.DataFrame(columns=['번호', '나이', '평균 혈당', '고혈압', '심장병'])
    
    formatted = pd.DataFrame({
        '번호': df_target['original_idx'] + 1,
        '나이': df_target['age'],
        '평균 혈당': df_target['avg_glucose_level'],
        '고혈압': df_target['hypertension'].map({1: '있음', 0: '없음'}),
        '심장병': df_target['heart_disease'].map({1: '있음', 0: '없음'})
    })
    return formatted

fn_display = format_list_df(fn_df.head(20))
fp_display = format_list_df(fp_df.head(20))

col_l, col_r = st.columns(2)

with col_l:
    st.markdown(f"#### 🛑 실제 뇌졸중인데 아니라고 예측한 사람 (놓친 환자) - 전체 {fn_total_count}명")
    if fn_total_count > 20:
        st.caption("인원이 20명을 초과하여 앞 20페이지만 표시합니다.")
    st.dataframe(fn_display, use_container_width=True)

with col_r:
    st.markdown(f"#### 🔍 아닌데 뇌졸중이라고 예측한 사람 (헛짚은 사람) - 전체 {fp_total_count}명")
    if fp_total_count > 20:
        st.caption("인원이 20명을 초과하여 앞 20페이지만 표시합니다.")
    st.dataframe(fp_display, use_container_width=True)

# 혼동행렬의 FN, FP 수와 일치하는지 확인하는 문구
st.markdown(f"ℹ️ **확인:** 위 목록의 인원 수 (놓친 환자 **{fn_total_count}명**, 헛짚은 사람 **{fp_total_count}명**)는 선택한 모델의 혼동행렬 내 **FN(위음성) 칸** 및 **FP(위양성) 칸**의 사람 수와 정확히 일치합니다.")
