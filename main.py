import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
DATA_URL = "https://raw.githubusercontent.com/jeongwon164/data/refs/heads/main/stroke.csv"
RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "stroke_lab_notes.json"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
)

# 한국어 뜻은 의도적으로 빈칸으로 둡니다.
COLUMNS = [
    "id",
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
    "stroke",
]


def load_notes():
    """내_기록 JSON 파일에서 사용자가 작성한 내용을 불러옵니다."""
    if not RECORD_FILE.exists():
        return {"meanings": {}, "source": ""}
    try:
        with RECORD_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"meanings": {}, "source": ""}
        return {
            "meanings": data.get("meanings", {}),
            "source": data.get("source", ""),
        }
    except (OSError, json.JSONDecodeError):
        return {"meanings": {}, "source": ""}


def save_notes(meanings, source):
    """입력한 뜻과 출처를 UTF-8 JSON으로 저장합니다."""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    data = {"meanings": meanings, "source": source}
    with RECORD_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@st.cache_data
def load_data():
    # UTF-8 인코딩으로 CSV를 읽습니다.
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🧠 뇌졸중 예측 실습실")
st.caption("데이터를 살펴보고, 각 열의 의미를 정리하며, 뇌졸중 예측 실습을 준비하는 공간입니다.")

try:
    df = load_data()
except Exception as exc:
    st.error("데이터를 불러오지 못했습니다. 인터넷 연결과 CSV 주소를 확인해 주세요.")
    st.exception(exc)
    st.stop()

notes = load_notes()

# 사용자가 지정한 열 이름을 우선 표시하고, 데이터에 추가 열이 있어도 함께 보여 줍니다.
columns_for_table = list(COLUMNS)
for col in df.columns:
    if col not in columns_for_table:
        columns_for_table.append(col)

# ------------------------------------------------------------
# 요약 카드
# ------------------------------------------------------------
total_people = len(df)
column_count = len(df.columns)
stroke_count = int((pd.to_numeric(df["stroke"], errors="coerce") == 1).sum()) if "stroke" in df else 0
stroke_rate = (stroke_count / total_people * 100) if total_people else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("전체 사람 수", f"{total_people:,}명")
c2.metric("열 개수", f"{column_count:,}개")
c3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
c4.metric("stroke가 1인 비율", f"{stroke_rate:.2f}%")

st.divider()

# ------------------------------------------------------------
# 열 설명 표와 교재 기록 입력
# ------------------------------------------------------------
st.subheader("1. 열 이름과 데이터 살펴보기")
st.write("‘우리말 뜻’은 교재를 참고해 직접 입력할 수 있습니다. 입력한 내용은 `내_기록/stroke_lab_notes.json`에 저장됩니다.")

rows = []
for col in columns_for_table:
    if col in df.columns:
        value_kinds = int(df[col].nunique(dropna=True))
        missing_count = int(df[col].isna().sum())
    else:
        value_kinds = 0
        missing_count = 0
    rows.append({
        "열 이름": col,
        "우리말 뜻": notes.get("meanings", {}).get(col, ""),
        "값의 종류": value_kinds,
        "빈 값 개수": missing_count,
    })

overview_df = pd.DataFrame(rows)
st.dataframe(overview_df, use_container_width=True, hide_index=True)

with st.expander("우리말 뜻과 데이터 출처 입력 / 수정", expanded=True):
    with st.form("notes_form"):
        st.markdown("**열 이름별 우리말 뜻**")
        updated_meanings = {}
        existing_meanings = notes.get("meanings", {})
        meaning_cols = st.columns(2)
        for idx, col in enumerate(columns_for_table):
            with meaning_cols[idx % 2]:
                updated_meanings[col] = st.text_input(
                    col,
                    value=str(existing_meanings.get(col, "")),
                    key=f"meaning_{col}",
                    placeholder="교재를 보고 우리말 뜻 입력",
                )
        st.markdown("**데이터 출처**")
        updated_source = st.text_area(
            "교재에 적힌 데이터 출처를 입력하세요.",
            value=str(notes.get("source", "")),
            height=100,
            placeholder="데이터 출처를 여기에 입력",
        )
        submitted = st.form_submit_button("기록 저장", type="primary")
        if submitted:
            try:
                save_notes(updated_meanings, updated_source)
                st.success("기록을 저장했습니다. 새로고침해도 다시 불러옵니다.")
                st.rerun()
            except OSError as exc:
                st.error(f"기록 파일을 저장하지 못했습니다: {exc}")

st.subheader("2. 데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

