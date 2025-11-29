import os
import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px

# ----------------------------------------------------
#  PAGE CONFIG  (RTL + TITLE)
# ----------------------------------------------------
st.set_page_config(
    page_title="Absher Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"  # Sidebar يفتح على الكمبيوتر
)

# ----------------------------------------------------
#  CUSTOM CSS — Absher Theme + Mobile Fix
# ----------------------------------------------------
st.markdown("""
<style>

html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
    font-family: "Tajawal", sans-serif;
}

header, footer {visibility: hidden;}

.main-title {
    font-size: 32px;
    font-weight: 800;
    color: #0B3D0B;
}

.sub-title {
    font-size: 18px;
    color: #444;
}

.sidebar .sidebar-content {
    background-color: #0B3D0B;
}

.report-box {
    border-radius: 10px;
    padding: 18px;
    background: #F7FFF7;
    border-right: 6px solid #009900;
}

/* MOBILE FIX: Sidebar مخفي، المحتوى ياخذ كامل الشاشة */
@media (max-width: 768px) {
    .css-1d391kg {  /* container Sidebar */
        width: 0 !important;
    }
    .css-1v3fvcr {  /* container الصفحة */
        margin-left: 0 !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
#  HEADER
# ----------------------------------------------------
st.markdown("""
<div style="text-align:right;">
    <div class="main-title">منصة أبشر — نظام التنبؤ الأمني</div>
    <div class="sub-title">تحليل الأنشطة و كشف السلوكيات الشاذة باستخدام الذكاء الاصطناعي</div>
</div>
<br>
""", unsafe_allow_html=True)

# ----------------------------------------------------
#  تحديد مجلد الملفات بالنسبة للكود
# ----------------------------------------------------
BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "model_iforest.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
thresholds = json.load(open(os.path.join(BASE_DIR, "thresholds.json")))
feature_cols = list(pd.read_csv(os.path.join(BASE_DIR, "features_abshar.csv"), nrows=1).columns)

# ----------------------------------------------------
#  SIDEBAR
# ----------------------------------------------------
st.sidebar.image(os.path.join(BASE_DIR, "AbsherTuwaig.png"), width=120)
st.sidebar.markdown("### لوحة التحكم")

page = st.sidebar.radio(
    "",
    ["التحليل", "شرح المخاطر", "معلومات النموذج"],
    index=0
)

# ----------------------------------------------------
#  PAGE 1 — FULL ANALYSIS
# ----------------------------------------------------
if page == "التحليل":

    st.markdown("##  رفع ملف وتحليل كامل للأنشطة")

    uploaded = st.file_uploader(" ارفع ملف CSV للأنشطة", type="csv")

    if uploaded:
        df = pd.read_csv(uploaded)

        # APPLY MODEL
        x = scaler.transform(df[feature_cols])
        scores = model.decision_function(x)

        df["score"] = scores
        df["risk_level"] = [
            "عالي" if s < thresholds["high_risk"]
            else "مراجعة" if s < thresholds["review"]
            else "طبيعي"
            for s in scores
        ]

        st.success("✔ تم تحليل البيانات بنجاح")

        # STAT CARDS
        st.markdown("### ملخص سريع للحالات")
        col1, col2, col3 = st.columns(3)

        col1.metric("🟥 عالي الخطورة", str(sum(df["risk_level"] == "عالي")))
        col2.metric("🟧 يحتاج مراجعة", str(sum(df["risk_level"] == "مراجعة")))
        col3.metric("🟩 طبيعي", str(sum(df["risk_level"] == "طبيعي")))

        st.markdown("---")

        # PIE CHART
        st.markdown("### 📊 توزيع مستويات الخطورة")
        pie = px.pie(
            df,
            names="risk_level",
            title="نسبة مستويات الخطورة",
            color="risk_level",
            color_discrete_map={
                "عالي": "red",
                "مراجعة": "orange",
                "طبيعي": "green"
            }
        )
        st.plotly_chart(pie, use_container_width=True)

        # BAR CHART
        st.markdown("### 📈 درجات المخاطر حسب النشاط")
        bar = px.bar(
            df,
            x=df.index,
            y="score",
            color="risk_level",
            title="درجات المخاطر لكل نشاط",
            color_discrete_map={
                "عالي": "red",
                "مراجعة": "orange",
                "طبيعي": "green"
            }
        )
        st.plotly_chart(bar, use_container_width=True)

        # RAW DATA
        st.markdown("### 📄 البيانات التفصيلية")
        st.dataframe(df)

        # DOWNLOAD
        st.download_button(
            "⬇ تنزيل النتائج",
            df.to_csv(index=False),
            file_name="results.csv",
            mime="text/csv"
        )

    else:
        st.warning("الرجاء رفع ملف CSV لبدء التحليل.")

# ----------------------------------------------------
#  PAGE 2 — RISK EXPLANATION
# ----------------------------------------------------
elif page == "شرح المخاطر":
    st.markdown("""
    ###  كيف يتم حساب مستوى الخطورة؟
    يعتمد النظام على نموذج **Isolation Forest** لتحديد الشذوذ السلوكي.

    #### 🟥 عالي:
    - خطر مباشر  
    - سلوك خارج الأنماط المعتادة  

    #### 🟧 يحتاج مراجعة:
    - سلوك غير معتاد قليلاً  
    - يحتاج تحقق يدوي  

    #### 🟩 طبيعي:
    - ضمن السلوك العادي  
    """)

# ----------------------------------------------------
#  PAGE 3 — MODEL INFO
# ----------------------------------------------------
elif page == "معلومات النموذج":
    st.markdown(f"""
    ### ℹ معلومات حول نموذج التنبؤ
    - النوع: Isolation Forest  
    - عدد الميزات: {len(feature_cols)}  
    - مدرب على بيانات الأنشطة السلوكية  
    """)

# ----------------------------------------------------
#  FOOTER
# ----------------------------------------------------
st.markdown("""
<br><br>
<div style='text-align:center; opacity:0.6; font-size:13px;'>
    © 2025 — وزارة الداخلية السعودية — منصة أبشر  
</div>
""", unsafe_allow_html=True)
