import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

st.title("성능 높이기 (클래스 가중치 조절)")

# 데이터 로드 함수 ('채점표' 페이지와 동일한 방식)
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

# 기본 선택: age, avg_glucose_level, hypertension, heart_disease (bmi 제외 넷)
default_features = ['age', 'avg_glucose_level', 'hypertension', 'heart_disease']
default_features_kr = [feature_names_kr[f] for f in default_features]

# 화면 맨 위에서 입력으로 사용할 속성 선택
st.subheader("입력 속성 선택")
selected_kr = st.multiselect(
    "학습에 사용할 속성을 선택해 주세요 (2개 이상)",
    options=list(feature_names_kr.values()),
    default=default_features_kr
)

# 2개 미만 고르면 경고 메시지 후 중단
if len(selected_kr) < 2:
    st.warning("모델을 학습하고 평가하기 위해 속성을 2개 이상 선택해 주세요.")
    st.stop()

kr_to_en = {v: k for k, v in feature_names_kr.items()}
selected_features = [kr_to_en[name] for name in selected_kr]

# 번호 순 정렬 후 10명씩 묶어 앞 3명 테스트용(1,533명), 나머지 7명 훈련용
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

# 체질량지수(bmi) 결측치 처리 (고른 경우에만 훈련용 중앙값으로 채움)
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

# 스케일링 (로지스틱 회귀용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 모델 학습 (가중치 조절 전 vs 조절 후)
# ---------------------------------------------------------

# 1. 로지스틱 회귀
lr_before = LogisticRegression(random_state=42)
lr_before.fit(X_train_scaled, y_train)
pred_lr_before = lr_before.predict(X_test_scaled)

lr_after = LogisticRegression(class_weight='balanced', random_state=42)
lr_after.fit(X_train_scaled, y_train)
pred_lr_after = lr_after.predict(X_test_scaled)

# 2. 의사결정트리
dt_before = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
dt_before.fit(X_train, y_train)
pred_dt_before = dt_before.predict(X_test)

dt_after = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, class_weight='balanced', random_state=42)
dt_after.fit(X_train, y_train)
pred_dt_after = dt_after.predict(X_test)

# 지표 계산 함수
def eval_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    
    pred_pos = y_pred.sum()
    if pred_pos == 0:
        prec_str = "계산할 수 없음"
        f1_str = "계산할 수 없음"
    else:
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec_str = f"{prec:.4f}"
        f1_str = f"{f1:.4f}"
        
    return {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'acc': f"{acc:.4f}", 'rec': f"{rec:.4f}",
        'prec': prec_str, 'f1': f1_str
    }

m_lr_bef = eval_metrics(y_test, pred_lr_before)
m_lr_aft = eval_metrics(y_test, pred_lr_after)
m_dt_bef = eval_metrics(y_test, pred_dt_before)
m_dt_aft = eval_metrics(y_test, pred_dt_after)

# ---------------------------------------------------------
# 모델별 전·후 평가 비교 표
# ---------------------------------------------------------
st.markdown("---")
st.subheader("모델 평가 비교 (가중치 맞추기 전 vs 맞춘 뒤)")

def build_comparison_table(bef, aft):
    table_data = {
        "항목": [
            "진음성 (TN - 아닌데 맞춤)",
            "위양성 (FP - 헛짚은 사람)",
            "위음성 (FN - 놓친 환자)",
            "진양성 (TP - 찾아낸 환자)",
            "정확도 (Accuracy)",
            "재현율 (Recall)",
            "정밀도 (Precision)",
            "F1 Score"
        ],
        "가중치 맞추기 전": [
            f"{bef['tn']}명", f"{bef['fp']}명", f"{bef['fn']}명", f"{bef['tp']}명",
            bef['acc'], bef['rec'], bef['prec'], bef['f1']
        ],
        "가중치 맞춘 뒤": [
            f"{aft['tn']}명", f"{aft['fp']}명", f"{aft['fn']}명", f"{aft['tp']}명",
            aft['acc'], aft['rec'], aft['prec'], aft['f1']
        ]
    }
    return pd.DataFrame(table_data)

col_tbl1, col_tbl2 = st.columns(2)

with col_tbl1:
    st.markdown("#### 로지스틱 회귀(확률로 답하는 모델)")
    st.table(build_comparison_table(m_lr_bef, m_lr_aft))
    st.caption(
        f"💡 **변화 요약:** 찾아낸 환자 수 **{m_lr_bef['tp']}명 → {m_lr_aft['tp']}명**, "
        f"놓친 환자 수 **{m_lr_bef['fn']}명 → {m_lr_aft['fn']}명**, "
        f"헛짚은 사람 수 **{m_lr_bef['fp']}명 → {m_lr_aft['fp']}명**으로 바뀌었습니다."
    )

with col_tbl2:
    st.markdown("#### 의사결정트리(질문으로 답하는 모델)")
    st.table(build_comparison_table(m_dt_bef, m_dt_aft))
    st.caption(
        f"💡 **변화 요약:** 찾아낸 환자 수 **{m_dt_bef['tp']}명 → {m_dt_aft['tp']}명**, "
        f"놓친 환자 수 **{m_dt_bef['fn']}명 → {m_dt_aft['fn']}명**, "
        f"헛짚은 사람 수 **{m_dt_bef['fp']}명 → {m_dt_aft['fp']}명**으로 바뀌었습니다."
    )

# ---------------------------------------------------------
# 2차원 산점도 및 0.5 경계선 시각화 (Plotly)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("2차원 산점도 및 결정 경계 (0.5 선 비교)")

col_ax1, col_ax2 = st.columns(2)
with col_ax1:
    x_axis_kr = st.selectbox("가로축 속성 선택", options=selected_kr, index=0)
with col_ax2:
    default_y_idx = 1 if len(selected_kr) > 1 else 0
    y_axis_kr = st.selectbox("세로축 속성 선택", options=selected_kr, index=default_y_idx)

x_axis = kr_to_en[x_axis_kr]
y_axis = kr_to_en[y_axis_kr]

# 선택 안 된 속성은 테스트 데이터의 중앙값 고정
other_features = [f for f in selected_features if f != x_axis and f != y_axis]
median_values = {f: test_df[f].median() for f in other_features}

if other_features:
    median_desc = ", ".join([f"**{feature_names_kr[f]}**: {val:.2f}" for f, val in median_values.items()])
    st.write(f"ℹ️ 고정된 축 외 나머지 속성(중앙값): {median_desc}")

# 2D Plotly 객체 생성
test_df['stroke_label'] = test_df['stroke'].map({1: '뇌졸중 (양성)', 0: '아님'})

fig_2d = px.scatter(
    test_df,
    x=x_axis,
    y=y_axis,
    color='stroke_label',
    color_discrete_map={'뇌졸중 (양성)': '#ff4b4b', '아님': '#1f77b4'},
    labels={x_axis: x_axis_kr, y_axis: y_axis_kr, 'stroke_label': '실제 뇌졸중 여부'},
    title=f"2D 결정 경계선 비교 ({x_axis_kr} vs {y_axis_kr})"
)

# 격자 생성
x_min, x_max = test_df[x_axis].min(), test_df[x_axis].max()
y_min, y_max = test_df[y_axis].min(), test_df[y_axis].max()
x_range, y_range = x_max - x_min, y_max - y_min
x_min_p, x_max_p = x_min - x_range * 0.1, x_max + x_range * 0.1
y_min_p, y_max_p = y_min - y_range * 0.1, y_max + y_range * 0.1

xx, yy = np.meshgrid(np.linspace(x_min_p, x_max_p, 100), np.linspace(y_min_p, y_max_p, 100))

grid_df = pd.DataFrame(index=np.arange(xx.ravel().shape[0]))
for f in selected_features:
    if f == x_axis:
        grid_df[f] = xx.ravel()
    elif f == y_axis:
        grid_df[f] = yy.ravel()
    else:
        grid_df[f] = median_values[f]

grid_scaled = scaler.transform(grid_df[selected_features])

# 가중치 맞추기 전 확률 (로지스틱 회귀)
prob_bef = lr_before.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)
# 가중치 맞춘 뒤 확률 (로지스틱 회귀)
prob_aft = lr_after.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

# 경계선 추가 (전: 파란색 선 / 후: 빨간색 선)
fig_2d.add_trace(
    go.Contour(
        z=prob_bef, x=np.linspace(x_min_p, x_max_p, 100), y=np.linspace(y_min_p, y_max_p, 100),
        showscale=False, contours=dict(start=0.5, end=0.5, size=0, coloring='lines'),
        line=dict(color='blue', width=3, dash='dash'), name='가중치 맞추기 전 (0.5 선)'
    )
)

fig_2d.add_trace(
    go.Contour(
        z=prob_aft, x=np.linspace(x_min_p, x_max_p, 100), y=np.linspace(y_min_p, y_max_p, 100),
        showscale=False, contours=dict(start=0.5, end=0.5, size=0, coloring='lines'),
        line=dict(color='red', width=3), name='가중치 맞춘 뒤 (0.5 선)'
    )
)

st.plotly_chart(fig_2d, use_container_width=True)

# 경계선 범위 이탈 여부 확인 문구
bef_in = (prob_bef.min() <= 0.5 <= prob_bef.max())
aft_in = (prob_aft.min() <= 0.5 <= prob_aft.max())

out_msgs = []
if not bef_in:
    out_msgs.append("가중치를 맞추기 전의 0.5 경계선이 그림 밖에 있습니다.")
if not aft_in:
    out_msgs.append("가중치를 맞춘 뒤의 0.5 경계선이 그림 밖에 있습니다.")

if out_msgs:
    for msg in out_msgs:
        st.info(f"⚠️ {msg}")

# ---------------------------------------------------------
# 3차원 산점도 및 0.5 평면 시각화 (속성 3개 이상 선택 시)
# ---------------------------------------------------------
if len(selected_features) >= 3:
    st.markdown("---")
    st.subheader("3차원 산점도 및 결정 평면 (0.5 평면 비교)")
    
    col_3d1, col_3d2, col_3d3 = st.columns(3)
    with col_3d1:
        x_3d_kr = st.selectbox("3D X축 속성", options=selected_kr, index=0, key='3d_x')
    with col_3d2:
        y_3d_kr = st.selectbox("3D Y축 속성", options=selected_kr, index=1, key='3d_y')
    with col_3d3:
        z_3d_kr = st.selectbox("3D Z축 속성", options=selected_kr, index=2, key='3d_z')

    x_3d, y_3d, z_3d = kr_to_en[x_3d_kr], kr_to_en[y_3d_kr], kr_to_en[z_3d_kr]

    other_3d = [f for f in selected_features if f not in [x_3d, y_3d, z_3d]]
    med_3d = {f: test_df[f].median() for f in other_3d}

    fig_3d = px.scatter_3d(
        test_df,
        x=x_3d, y=y_3d, z=z_3d,
        color='stroke_label',
        color_discrete_map={'뇌졸중 (양성)': '#ff4b4b', '아님': '#1f77b4'},
        labels={x_3d: x_3d_kr, y_3d: y_3d_kr, z_3d: z_3d_kr, 'stroke_label': '실제 뇌졸중 여부'},
        title=f"3D 결정 평면 (마우스로 회전해 볼 수 있습니다)"
    )

    # 3D 평면을 그리기 위한 메시지 그리드 생성 (20x20x20)
    x_min3, x_max3 = test_df[x_3d].min(), test_df[x_3d].max()
    y_min3, y_max3 = test_df[y_3d].min(), test_df[y_3d].max()

    xx3, yy3 = np.meshgrid(np.linspace(x_min3, x_max3, 20), np.linspace(y_min3, y_max3, 20))

    grid3_df = pd.DataFrame(index=np.arange(xx3.ravel().shape[0]))
    for f in selected_features:
        if f == x_3d:
            grid3_df[f] = xx3.ravel()
        elif f == y_3d:
            grid3_df[f] = yy3.ravel()
        elif f == z_3d:
            grid3_df[f] = 0 # 임시 초기화
        else:
            grid3_df[f] = med_3d[f]

    # 평면 방점 계산: w0*x0 + w1*x1 + ... + b = 0 (z_3d에 대해 풀기)
    # 로지스틱 회귀 Decision Surface: w_scaled . ((x - mean)/scale) + b = 0
    def compute_z_plane(model):
        coefs = model.coef_[0]
        intercept = model.intercept_[0]
        means = scaler.mean_
        scales = scaler.scale_

        idx_x = selected_features.index(x_3d)
        idx_y = selected_features.index(y_3d)
        idx_z = selected_features.index(z_3d)

        w_x = coefs[idx_x] / scales[idx_x]
        w_y = coefs[idx_y] / scales[idx_y]
        w_z = coefs[idx_z] / scales[idx_z]

        const_term = intercept
        for idx, f in enumerate(selected_features):
            if f == z_3d:
                const_term -= (coefs[idx] * means[idx] / scales[idx])
            elif f == x_3d:
                const_term -= (coefs[idx] * means[idx] / scales[idx])
            elif f == y_3d:
                const_term -= (coefs[idx] * means[idx] / scales[idx])
            else:
                const_term += coefs[idx] * (med_3d[f] - means[idx]) / scales[idx]

        if w_z != 0:
            zz = -(w_x * xx3 + w_y * yy3 + const_term) / w_z
            return zz
        return None

    zz_bef = compute_z_plane(lr_before)
    zz_aft = compute_z_plane(lr_after)

    if zz_bef is not None:
        fig_3d.add_trace(
            go.Surface(
                x=xx3, y=yy3, z=zz_bef,
                colorscale=[[0, 'rgba(31, 119, 180, 0.5)'], [1, 'rgba(31, 119, 180, 0.5)']],
                showscale=False, name='가중치 전 0.5 평면'
            )
        )

    if zz_aft is not None:
        fig_3d.add_trace(
            go.Surface(
                x=xx3, y=yy3, z=zz_aft,
                colorscale=[[0, 'rgba(255, 75, 75, 0.5)'], [1, 'rgba(255, 75, 75, 0.5)']],
                showscale=False, name='가중치 후 0.5 평면'
            )
        )

    st.plotly_chart(fig_3d, use_container_width=True)
