import pandas as pd
import plotly.express as px
import streamlit as st

APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/jeongwon164/data/refs/heads/main/stroke.csv"

st.set_page_config(
    page_title=f"탐색 | {APP_TITLE}",
    page_icon="🔎",
    layout="wide",
)


@st.cache_data
def load_data():
    """첫 화면과 같은 주소의 CSV를 UTF-8 인코딩으로 불러옵니다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


def label_stroke(value):
    if pd.isna(value):
        return None
    try:
        return {0: "뇌졸중 없음", 1: "뇌졸중 있음"}.get(int(value), None)
    except (TypeError, ValueError):
        return None


def label_binary(value):
    if pd.isna(value):
        return None
    try:
        return {0: "없음", 1: "있음"}.get(int(value), None)
    except (TypeError, ValueError):
        return None


st.title("🔎 데이터 탐색")
st.caption("분포와 그룹별 차이를 그래프로 살펴봅니다. 관찰된 차이만으로 원인이나 개인의 위험을 단정할 수는 없습니다.")

try:
    raw_df = load_data()
except Exception as exc:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 CSV 주소를 확인해 주세요.")
    st.exception(exc)
    st.stop()

required_columns = [
    "age", "avg_glucose_level", "stroke", "hypertension",
    "heart_disease", "bmi", "smoking_status"
]
missing_columns = [col for col in required_columns if col not in raw_df.columns]
if missing_columns:
    st.error("CSV에 필요한 열이 없습니다: " + ", ".join(missing_columns))
    st.stop()

df = raw_df.copy()
for col in ["age", "avg_glucose_level", "stroke", "hypertension", "heart_disease", "bmi"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["뇌졸중 그룹"] = df["stroke"].map({0: "뇌졸중 없음", 1: "뇌졸중 있음"})
stroke_order = ["뇌졸중 없음", "뇌졸중 있음"]

# ------------------------------------------------------------
# 1. 나이와 평균 혈당 분포: 히스토그램 두 개를 나란히 배치
# ------------------------------------------------------------
st.subheader("1. 나이와 평균 혈당의 분포")
col_age, col_glucose = st.columns(2)

with col_age:
    age_data = df.dropna(subset=["age"])
    fig_age = px.histogram(
        age_data, x="age", nbins=30,
        title="나이 분포",
        labels={"age": "나이", "count": "사람 수"},
    )
    fig_age.update_layout(yaxis_title="사람 수")
    st.plotly_chart(fig_age, use_container_width=True)
    st.caption("이 그래프로 알 수 있는 것: 사람들의 나이가 어느 구간에 많이 모여 있는지 확인할 수 있습니다.")

with col_glucose:
    glucose_data = df.dropna(subset=["avg_glucose_level"])
    fig_glucose = px.histogram(
        glucose_data, x="avg_glucose_level", nbins=30,
        title="평균 혈당 분포",
        labels={"avg_glucose_level": "평균 혈당", "count": "사람 수"},
    )
    fig_glucose.update_layout(yaxis_title="사람 수")
    st.plotly_chart(fig_glucose, use_container_width=True)
    st.caption("이 그래프로 알 수 있는 것: 평균 혈당이 어느 구간에 집중되어 있는지 살펴볼 수 있습니다.")

st.divider()

# ------------------------------------------------------------
# 2. 뇌졸중 유무별 상자그림과 평균값 표
# ------------------------------------------------------------
st.subheader("2. 뇌졸중 유무에 따른 나이와 평균 혈당 비교")
box_age_col, box_glucose_col = st.columns(2)
grouped = df.dropna(subset=["뇌졸중 그룹"])

with box_age_col:
    age_box = grouped.dropna(subset=["age"])
    fig_age_box = px.box(
        age_box, x="뇌졸중 그룹", y="age",
        category_orders={"뇌졸중 그룹": stroke_order},
        points="outliers",
        title="뇌졸중 유무별 나이",
        labels={"뇌졸중 그룹": "그룹", "age": "나이"},
    )
    st.plotly_chart(fig_age_box, use_container_width=True)

with box_glucose_col:
    glucose_box = grouped.dropna(subset=["avg_glucose_level"])
    fig_glucose_box = px.box(
        glucose_box, x="뇌졸중 그룹", y="avg_glucose_level",
        category_orders={"뇌졸중 그룹": stroke_order},
        points="outliers",
        title="뇌졸중 유무별 평균 혈당",
        labels={"뇌졸중 그룹": "그룹", "avg_glucose_level": "평균 혈당"},
    )
    st.plotly_chart(fig_glucose_box, use_container_width=True)

mean_table = (
    grouped.groupby("뇌졸중 그룹", observed=False)
    .agg(
        사람_수=("stroke", "size"),
        평균_나이=("age", "mean"),
        평균_혈당=("avg_glucose_level", "mean"),
    )
    .reindex(stroke_order)
    .reset_index()
    .rename(columns={
        "뇌졸중 그룹": "그룹",
        "사람_수": "사람 수",
        "평균_나이": "평균 나이",
        "평균_혈당": "평균 혈당",
    })
)
mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균 혈당"].round(2)
st.markdown("**그룹별 평균값**")
st.dataframe(mean_table, use_container_width=True, hide_index=True)
st.caption("상자그림으로 중앙값과 사분위 범위를 비교하고, 평균값은 표에서 확인할 수 있습니다.")

st.divider()

# ------------------------------------------------------------
# 3. 고혈압/심장병 유무별 뇌졸중 비율
# ------------------------------------------------------------
st.subheader("3. 고혈압과 심장병 유무별 뇌졸중 비율")


def make_rate_summary(source, column):
    temp = source.dropna(subset=[column, "stroke"]).copy()
    temp["상태"] = temp[column].map({0: "없음", 1: "있음"})
    temp = temp.dropna(subset=["상태"])
    summary = (
        temp.groupby("상태", observed=False)
        .agg(
            전체_사람수=("stroke", "size"),
            뇌졸중_사람수=("stroke", lambda values: int((values == 1).sum())),
        )
        .reset_index()
        .rename(columns={
            "전체_사람수": "전체 사람 수",
            "뇌졸중_사람수": "뇌졸중 사람 수",
        })
    )
    summary["뇌졸중 비율(%)"] = (
        summary["뇌졸중 사람 수"] / summary["전체 사람 수"] * 100
    ).round(2)
    summary["상태"] = pd.Categorical(summary["상태"], categories=["없음", "있음"], ordered=True)
    return summary.sort_values("상태")


rate_hypertension_col, rate_heart_col = st.columns(2)

with rate_hypertension_col:
    hypertension_summary = make_rate_summary(df, "hypertension")
    fig_hypertension = px.bar(
        hypertension_summary,
        x="상태", y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title="고혈압 유무별 뇌졸중 비율",
        labels={"상태": "고혈압", "뇌졸중 비율(%)": "뇌졸중 비율(%)"},
    )
    fig_hypertension.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_hypertension.update_layout(yaxis_rangemode="tozero")
    st.plotly_chart(fig_hypertension, use_container_width=True)
    st.dataframe(hypertension_summary, use_container_width=True, hide_index=True)

with rate_heart_col:
    heart_summary = make_rate_summary(df, "heart_disease")
    fig_heart = px.bar(
        heart_summary,
        x="상태", y="뇌졸중 비율(%)",
        text="뇌졸중 비율(%)",
        title="심장병 유무별 뇌졸중 비율",
        labels={"상태": "심장병", "뇌졸중 비율(%)": "뇌졸중 비율(%)"},
    )
    fig_heart.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_heart.update_layout(yaxis_rangemode="tozero")
    st.plotly_chart(fig_heart, use_container_width=True)
    st.dataframe(heart_summary, use_container_width=True, hide_index=True)

st.caption("비율은 해당 상태 그룹에서 stroke 값이 1인 사람의 비율입니다. 두 비율의 차이만으로 원인이라고 결론 내릴 수는 없습니다.")

st.divider()

# ------------------------------------------------------------
# 4. bmi 결측 그룹과 전체 뇌졸중 비율 비교
# ------------------------------------------------------------
st.subheader("4. bmi가 비어 있는 사람들과 전체의 뇌졸중 비율")
missing_bmi = df[df["bmi"].isna()]
missing_stroke = missing_bmi["stroke"].dropna()
all_stroke = df["stroke"].dropna()

missing_bmi_rate = (
    (missing_stroke == 1).sum() / len(missing_stroke) * 100
    if len(missing_stroke) else float("nan")
)
overall_rate = (
    (all_stroke == 1).sum() / len(all_stroke) * 100
    if len(all_stroke) else float("nan")
)

bmi_comparison = pd.DataFrame([
    {
        "대상": "bmi가 비어 있는 사람",
        "사람 수": len(missing_bmi),
        "stroke 값이 있는 사람 수": len(missing_stroke),
        "뇌졸중 사람 수": int((missing_stroke == 1).sum()),
        "뇌졸중 비율(%)": round(missing_bmi_rate, 2) if pd.notna(missing_bmi_rate) else None,
    },
    {
        "대상": "전체 사람",
        "사람 수": len(df),
        "stroke 값이 있는 사람 수": len(all_stroke),
        "뇌졸중 사람 수": int((all_stroke == 1).sum()),
        "뇌졸중 비율(%)": round(overall_rate, 2) if pd.notna(overall_rate) else None,
    },
])
st.dataframe(bmi_comparison, use_container_width=True, hide_index=True)
st.caption("뇌졸중 비율은 stroke 값이 확인되는 사람을 기준으로 계산합니다. bmi의 빈 값은 임의로 채우지 않습니다.")

st.divider()

# ------------------------------------------------------------
# 5. 흡연 상태별 사람 수
# ------------------------------------------------------------
st.subheader("5. 흡연 상태별 사람 수")
smoking_counts = (
    df["smoking_status"]
    .fillna("(빈 값)")
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking_counts, use_container_width=True, hide_index=True)

fig_smoking = px.bar(
    smoking_counts,
    x="흡연 상태", y="사람 수",
    text="사람 수",
    title="흡연 상태별 사람 수",
    labels={"흡연 상태": "흡연 상태", "사람 수": "사람 수"},
)
fig_smoking.update_traces(textposition="outside")
fig_smoking.update_layout(yaxis_rangemode="tozero")
st.plotly_chart(fig_smoking, use_container_width=True)
st.caption("흡연 상태 범주별 인원수를 비교할 수 있습니다. '(빈 값)'은 기록되지 않은 상태입니다.")
