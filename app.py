# ---------------- IMPORTS ----------------
import streamlit as st
import pickle
import docx
import PyPDF2
import re
import plotly.express as px

# ---------------- LOAD MODELS ----------------
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# ---------------- SKILLS ----------------
SKILLS_DB = {
    "Tech": [
        "python", "java", "c++", "sql", "pandas", "numpy",
        "machine learning", "deep learning", "nlp",
        "scikit-learn", "tensorflow", "keras", "pytorch",
        "data analysis", "data science",
        "business analysis", "business intelligence"
    ],
    "Cloud": [
        "aws", "azure", "gcp", "docker", "kubernetes"
    ],
    "Tools": [
        "power bi", "tableau", "excel", "git", "github",
        "power query", "dax"
    ],
    "Soft": [
        "communication", "teamwork", "leadership",
        "problem solving", "critical thinking",
        "collaboration", "adaptability", "time management"
    ]
}

ROLE_SKILLS = {
    "Data Science": [
        "python", "machine learning", "pandas", "numpy",
        "scikit-learn", "data analysis"
    ],
    "Data Analyst": [
        "sql", "excel", "power bi", "data analysis", "business intelligence"
    ],
    "Web Developer": [
        "html", "css", "javascript", "react"
    ]
}

# ---------------- CLEAN TEXT ----------------
def cleanResume(txt):
    txt = re.sub(r'http\S+', ' ', txt)
    txt = re.sub(r'RT|cc', ' ', txt)
    txt = re.sub(r'#\S+', ' ', txt)
    txt = re.sub(r'@\S+', ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt

# ---------------- FILE HANDLING ----------------
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "".join([p.extract_text() or "" for p in reader.pages])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except:
        return file.read().decode('latin-1')

def handle_file_upload(file):
    ext = file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(file)
    elif ext == 'docx':
        return extract_text_from_docx(file)
    elif ext == 'txt':
        return extract_text_from_txt(file)
    else:
        raise ValueError("Unsupported file type")

# ---------------- SKILL EXTRACTION ----------------
def extract_skills(text):
    text = text.lower().replace("-", " ").replace(".", "")
    categorized = {}

    for category, skills in SKILLS_DB.items():
        found = []
        for skill in skills:
            if skill in text or f"{skill} skills" in text:
                found.append(skill)
        categorized[category] = list(set(found))

    return categorized

# ---------------- SKILL GAP ----------------
def get_skill_gap(role, categorized_skills):
    user_skills = [s for v in categorized_skills.values() for s in v]
    required = ROLE_SKILLS.get(role, [])
    return [s for s in required if s not in user_skills]

# ---------------- MODEL ----------------
def pred(text):
    vec = tfidf.transform([cleanResume(text)]).toarray()
    return le.inverse_transform(svc_model.predict(vec))[0]

# ---------------- JOB MATCH ----------------
def job_match_score(resume, jd):
    from sklearn.metrics.pairwise import cosine_similarity
    r = tfidf.transform([cleanResume(resume)])
    j = tfidf.transform([cleanResume(jd)])
    return round(cosine_similarity(r, j)[0][0] * 100, 2)

# ---------------- HIGHLIGHT ----------------
def highlight_skills(text, skills):
    for s in skills:
        text = re.sub(
            rf"\b({re.escape(s)})\b",
            r"<mark style='background-color:#22c55e'>\1</mark>",
            text,
            flags=re.IGNORECASE
        )
    return text

# ---------------- UI ----------------
def main():
    st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

    # 🔥 Modern UI
    st.markdown("""
    <style>
    .stApp {background: linear-gradient(135deg, #0f172a, #020617); color: white;}
    h1 {text-align:center;color:#38bdf8;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>🚀 AI Resume Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray;'>Smart Resume Insights</p>", unsafe_allow_html=True)

    file = st.file_uploader("📄 Upload Resume", type=["pdf", "docx", "txt"])

    if file:
        try:
            text = handle_file_upload(file)
            st.success("✅ Resume processed successfully!")

            if st.checkbox("Show Resume Text"):
                st.text_area("", text, height=200)

            # Prediction
            role = pred(text)

            skills = extract_skills(text)
            all_skills = [s for v in skills.values() for s in v]

            total_possible = sum(len(v) for v in SKILLS_DB.values())
            score = round((len(all_skills) / total_possible) * 100, 2)

            gap = get_skill_gap(role, skills)

            col1, col2 = st.columns(2)

            with col1:
                st.success(f"🎯 Role: {role}")
                st.metric("📊 Resume Score", f"{score}%")

            with col2:
                st.info("🛠 Skills")
                st.write(", ".join(all_skills) if all_skills else "None")

            st.markdown("---")

            # Skills breakdown
            st.subheader("🧠 Skills Breakdown")
            for cat, s in skills.items():
                st.write(f"**{cat}:** {', '.join(s) if s else 'None'}")

            # Skill gaps
            st.subheader("⚠️ Skill Gaps")
            if gap:
                st.write("❌ " + ", ".join(gap))
            else:
                st.success("Perfect match!")

            # Plotly chart
            st.subheader("📊 Skills Distribution")
            counts = {k: len(v) for k, v in skills.items()}

            fig = px.bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                color=list(counts.keys()),
                text=list(counts.values())
            )

            fig.update_layout(
                plot_bgcolor="#0f172a",
                paper_bgcolor="#0f172a",
                font=dict(color="white")
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Job match
            st.subheader("🎯 Job Match Analyzer")
            jd = st.text_area("Paste Job Description")

            if st.button("🚀 Analyze Match"):
                if jd.strip() == "":
                    st.warning("Enter job description")
                else:
                    match = job_match_score(text, jd)
                    st.metric("Match Score", f"{match}%")

                    if match > 75:
                        st.success("🔥 Strong Match")
                    elif match > 50:
                        st.warning("⚠️ Moderate Match")
                    else:
                        st.error("❌ Low Match")

            # Highlight
            if st.checkbox("📄 Show Highlighted Resume"):
                st.markdown(highlight_skills(text, all_skills), unsafe_allow_html=True)

        except Exception as e:
            st.error(str(e))

if __name__ == "__main__":
    main()