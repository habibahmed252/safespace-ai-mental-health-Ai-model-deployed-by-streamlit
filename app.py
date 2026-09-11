"""
SafeSpace AI — Mental Health Text Analysis
Designed & developed by Eng. Habiba Ahmed Talat.

Model artifact: XGBoost + TF-IDF + LabelEncoder.
The saved model is English-trained, so Arabic input is translated to English
before inference. Voice input is converted to text and follows the same path.
"""

import html
import io
import re
import warnings
from datetime import datetime

import joblib
import streamlit as st

warnings.filterwarnings("ignore")

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

try:
    import speech_recognition as sr
except Exception:
    sr = None


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="SafeSpace AI",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =============================================================================
# SESSION STATE
# =============================================================================
DEFAULTS = {
    "language": "en",
    "theme": "light",
    "text_input": "",
    "results": None,
    "analysis_history": [],
    "voice_transcript": "",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

lang = st.session_state.language
T = None


# =============================================================================
# CONTENT
# =============================================================================
TEXT = {
    "en": {
        "disclaimer": "For informational and educational support only. Not a medical diagnosis.",
        "brand": "SafeSpace AI",
        "subtitle": "Understanding your unspoken words",
        "creator": "Designed & developed by Eng. Habiba Ahmed Talat",
        "about": "About SafeSpace AI",
        "about_desc": "SafeSpace AI uses an English-trained NLP classifier to estimate which of seven mental-health categories is most consistent with the submitted text. The result is a model prediction for reflection and screening-style analysis — not a clinical diagnosis.",
        "accuracy": "94.2% Model Accuracy",
        "model": "XGBoost + TF-IDF",
        "model_section": "Model Information",
        "model_type": "Model architecture",
        "features": "Feature extraction",
        "classes": "Prediction classes",
        "training_language": "Training language",
        "input_title": "Share Your Thoughts",
        "input_hint": "Write in English or Arabic. The interface language does not limit the language of your message.",
        "placeholder": "Express your thoughts freely here...",
        "voice": "Voice input",
        "voice_hint": "Speak in English or Arabic and convert your voice into text.",
        "transcribe": "Convert voice to text",
        "analyze": "Analyze Text",
        "clear": "Clear",
        "detected": "Detected Mental State",
        "confidence": "Confidence",
        "breakdown": "Class Probabilities Breakdown",
        "response": "Empathetic AI Response",
        "guidance": "Personalized Guidance",
        "selfcare": "Self-Care Micro-Steps",
        "emergency": "URGENT SUPPORT",
        "emergency_text": "If you may be in immediate danger or having thoughts of suicide, contact local emergency services or a trusted person now.",
        "processing": "Analyzing your text...",
        "translation": "Arabic detected — translating the message for the English-trained model.",
        "translation_failed": "I couldn't translate the Arabic text. Please try again or enter English.",
        "voice_missing": "Voice recognition is unavailable because its dependency is missing.",
        "voice_failed": "I couldn't understand the recording. Please try again in a quieter place.",
        "voice_network": "Speech recognition could not reach its recognition service.",
        "empty": "Please write something or convert a voice recording first.",
        "history": "Session History",
        "history_note": "Previous analyses are kept only in this active session and are not written to a database or file.",
        "view": "View analysis",
        "clear_history": "Clear history",
        "developer": "Chat with Developer",
        "developer_note": "Have feedback, ideas, or a question about SafeSpace AI?",
        "theme": "Dark theme",
        "light": "Light",
        "dark": "Dark",
        "language": "AR",
        "footer": "Built with Streamlit · Designed & developed by Eng. Habiba Ahmed Talat",
        "privacy": "Session-only processing. This application does not intentionally store your messages outside the active session.",
    },
    "ar": {
        "disclaimer": "للاستخدام الإرشادي والتعليمي فقط، وليس تشخيصًا طبيًا.",
        "brand": "SafeSpace AI",
        "subtitle": "فهم ما بين السطور",
        "creator": "تصميم وتطوير م. حبيبة أحمد طلعت",
        "about": "عن SafeSpace AI",
        "about_desc": "يستخدم SafeSpace AI نموذجًا لمعالجة اللغة الطبيعية مدرّبًا باللغة الإنجليزية لتقدير الفئة النفسية الأكثر توافقًا مع النص من بين سبع فئات. النتيجة توقع من النموذج لأغراض الفهم والتحليل وليست تشخيصًا طبيًا.",
        "accuracy": "دقة النموذج 94.2%",
        "model": "XGBoost + TF-IDF",
        "model_section": "معلومات النموذج",
        "model_type": "بنية النموذج",
        "features": "استخراج الخصائص",
        "classes": "فئات التوقع",
        "training_language": "لغة التدريب",
        "input_title": "شارك أفكارك",
        "input_hint": "اكتب بالعربي أو بالإنجليزي. لغة الواجهة لا تحدد لغة الرسالة.",
        "placeholder": "عبّر عن أفكارك ومشاعرك هنا بحرية...",
        "voice": "الإدخال الصوتي",
        "voice_hint": "اتكلم بالعربي أو بالإنجليزي وحوّل صوتك إلى نص.",
        "transcribe": "تحويل الصوت إلى نص",
        "analyze": "تحليل النص",
        "clear": "مسح",
        "detected": "الحالة المتوقعة",
        "confidence": "درجة الثقة",
        "breakdown": "توزيع احتمالات الفئات",
        "response": "استجابة AI داعمة",
        "guidance": "توجيهات شخصية",
        "selfcare": "خطوات بسيطة للعناية بالنفس",
        "emergency": "دعم فوري مطلوب",
        "emergency_text": "لو في خطر فوري أو أفكار انتحارية، تواصل الآن مع خدمات الطوارئ المحلية أو شخص تثق به.",
        "processing": "جاري تحليل النص...",
        "translation": "تم اكتشاف العربية — يتم ترجمة الرسالة للنموذج المدرب باللغة الإنجليزية.",
        "translation_failed": "لم أستطع ترجمة النص العربي. جرّب مرة أخرى أو اكتب بالإنجليزية.",
        "voice_missing": "الإدخال الصوتي غير متاح لأن مكون التعرف على الكلام غير مثبت.",
        "voice_failed": "لم أستطع فهم التسجيل. جرّب مرة أخرى في مكان أهدأ.",
        "voice_network": "تعذر الوصول إلى خدمة التعرف على الكلام.",
        "empty": "اكتب نصًا أو حوّل تسجيلًا صوتيًا إلى نص أولًا.",
        "history": "سجل الجلسة",
        "history_note": "التحليلات السابقة موجودة داخل الجلسة الحالية فقط ولا يتم كتابتها في قاعدة بيانات أو ملف.",
        "view": "عرض التحليل",
        "clear_history": "مسح السجل",
        "developer": "تواصل مع المطوّرة",
        "developer_note": "عندك ملاحظة أو فكرة أو سؤال عن SafeSpace AI؟",
        "theme": "الوضع الداكن",
        "light": "فاتح",
        "dark": "داكن",
        "language": "EN",
        "footer": "تم تطويره باستخدام Streamlit · تصميم وتطوير م. حبيبة أحمد طلعت",
        "privacy": "معالجة داخل الجلسة فقط. لا يقوم التطبيق بحفظ رسائلك خارج الجلسة الحالية بشكل مقصود.",
    },
}

CLASS_AR = {
    "Anxiety": "القلق",
    "Bipolar": "ثنائي القطب",
    "Depression": "الاكتئاب",
    "Normal": "طبيعي",
    "Personality disorder": "اضطراب الشخصية",
    "Stress": "الضغط والتوتر",
    "Suicidal": "أفكار انتحارية",
}

GUIDANCE = {
    "Anxiety": {
        "en": ("Your anxiety is valid. Try grounding yourself in the present moment.", ["Name 5 things you can see.", "Take slow breaths for one minute.", "Take a short gentle walk."]),
        "ar": ("مشاعرك مهمة. حاولي ترجعي انتباهك للحظة الحالية بدل ما يفضل ذهنك في دائرة القلق.", ["سمي 5 حاجات شايفاها حولك.", "خدي نفس ببطء لمدة دقيقة.", "اعملي مشية بسيطة وهادية."]),
    },
    "Stress": {
        "en": ("Stress can be a signal that your mind and body need a pause.", ["Write down your top three stressors.", "Take a 15-minute break.", "Choose one small task you can control today."]),
        "ar": ("التوتر ممكن يكون إشارة إن جسمك وذهنك محتاجين وقفة وراحة.", ["اكتبي أهم 3 حاجات مسببة للتوتر.", "خدي استراحة 15 دقيقة.", "اختاري خطوة صغيرة واحدة تقدري تتحكمي فيها النهارده."]),
    },
    "Depression": {
        "en": ("What you feel matters, but a model prediction cannot define you.", ["Drink some water and get a little daylight.", "Message someone you trust.", "Choose one tiny achievable activity."]),
        "ar": ("إحساسك مهم، لكن نتيجة النموذج لا تحدد شخصيتك أو قيمتك.", ["اشربي مياه واقعدي شوية في ضوء النهار.", "كلمي شخص تثقي فيه.", "اختاري نشاط صغير جدًا تقدري تعمليه."]),
    },
    "Normal": {
        "en": ("Your input is most consistent with the Normal category in this model.", ["Keep routines that support wellbeing.", "Notice what is working for you.", "Stay connected with people you trust."]),
        "ar": ("النص أقرب لفئة الطبيعي حسب النموذج الحالي.", ["حافظي على العادات اللي بتدعم راحتك.", "خدي بالك من الحاجات اللي بتساعدك.", "خلي التواصل مع الناس الموثوقين جزءًا من حياتك."]),
    },
    "Bipolar": {
        "en": ("This is a model category, not a diagnosis. A clinician is needed for proper assessment.", ["Track noticeable mood or sleep changes.", "Avoid major decisions during intense mood changes.", "Consider professional assessment if concerns persist."]),
        "ar": ("دي فئة في النموذج وليست تشخيصًا. التقييم الصحيح يحتاج متخصصًا.", ["سجلي التغيرات الواضحة في المزاج أو النوم.", "تجنبي القرارات الكبيرة أثناء التغيرات الشديدة في المزاج.", "فكري في تقييم متخصص لو القلق مستمر."]),
    },
    "Personality disorder": {
        "en": ("This model label is not a clinical diagnosis and should not be used to label yourself.", ["Treat the result as a signal for reflection only.", "Notice recurring patterns without judging yourself.", "Seek professional assessment for persistent concerns."]),
        "ar": ("التصنيف ده من النموذج مش تشخيص طبي، ومينفعش نستخدمه للحكم على نفسك.", ["اعتبري النتيجة مجرد إشارة للتفكير.", "لاحظي الأنماط المتكررة بدون لوم لنفسك.", "استشيري متخصصًا لو في مشكلة مستمرة."]),
    },
    "Suicidal": {
        "en": ("This category requires extra caution. The model cannot determine your safety. If you may act on suicidal thoughts, seek immediate human help.", ["Contact a trusted person now.", "Move away from anything you could use to hurt yourself.", "Contact local emergency/crisis services immediately if danger is imminent."]),
        "ar": ("الفئة دي تحتاج حذر شديد. النموذج لا يستطيع تحديد مدى أمانك. لو ممكن تتصرفي بناءً على أفكار انتحارية، اطلبي مساعدة بشرية فورًا.", ["كلمي شخص تثقي فيه الآن.", "ابعدي عن أي شيء ممكن تستخدميه لإيذاء نفسك.", "تواصلي فورًا مع الطوارئ أو خدمات الأزمات المحلية لو في خطر قريب."]),
    },
}


# =============================================================================
# MODEL
# =============================================================================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


def is_arabic(text: str) -> bool:
    arabic = len(re.findall(r"[\u0600-\u06FF]", text or ""))
    latin = len(re.findall(r"[A-Za-z]", text or ""))
    return arabic >= 2 and arabic >= max(2, int(latin * 0.15))


def translate_to_english(text: str):
    if not is_arabic(text):
        return text, False
    if GoogleTranslator is None:
        raise RuntimeError("deep-translator is not installed")
    translated = GoogleTranslator(source="auto", target="en").translate(text)
    if not translated or not translated.strip():
        raise RuntimeError("Empty translation")
    return translated, True


def speech_to_text(audio_bytes: bytes):
    if sr is None:
        return None, "missing"
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)

        candidates = []
        for locale in ("en-US", "ar-EG"):
            try:
                result = recognizer.recognize_google(audio, language=locale, show_all=True)
                if isinstance(result, dict):
                    alternatives = result.get("alternative", [])
                    if alternatives:
                        best = max(alternatives, key=lambda x: float(x.get("confidence", 0)))
                        candidates.append((float(best.get("confidence", 0)), best.get("transcript", "")))
                elif isinstance(result, str) and result.strip():
                    candidates.append((0.0, result))
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                return None, "network"

        if not candidates:
            return None, "unknown"
        return max(candidates, key=lambda x: x[0])[1], None
    except Exception:
        return None, "unknown"


def safe_text(value):
    return html.escape(str(value))


def save_history(raw_text, result):
    item = {
        "text": raw_text,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"].copy(),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }
    # Do not duplicate an identical analysis on the same rerun.
    if not st.session_state.analysis_history or st.session_state.analysis_history[-1]["text"] != raw_text:
        st.session_state.analysis_history.append(item)


# =============================================================================
# THEME / LANGUAGE CONTROLS
# =============================================================================
lang = st.session_state.language
T = TEXT[lang]

dark = st.session_state.theme == "dark"
if dark:
    bg = "#0D1624"
    surface = "#142235"
    surface2 = "#182A40"
    input_bg = "#102033"
    text = "#F4F8FC"
    muted = "#AAB8C8"
    line = "#29415B"
    navy = "#F4F8FC"
    accent_bg = "#173A50"
else:
    bg = "#F4FBFD"
    surface = "#FFFFFF"
    surface2 = "#F7FCFE"
    input_bg = "#FFFFFF"
    text = "#132238"
    muted = "#637386"
    line = "#D9EAF0"
    navy = "#071A35"
    accent_bg = "#EFF9FC"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700;800&display=swap');
:root {{
  --bg:{bg}; --surface:{surface}; --surface2:{surface2}; --input:{input_bg};
  --text:{text}; --muted:{muted}; --line:{line}; --navy:{navy}; --accent-bg:{accent_bg};
  --blue:#1689A8; --blue2:#57C7D8; --danger:#D94B4B;
}}
.stApp {{ background:var(--bg); color:var(--text); }}
[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ max-width:1450px; padding-top:.65rem; padding-bottom:2rem; }}
* {{ font-family:'Inter',sans-serif; }}
h1,h2,h3,h4 {{ font-family:'Playfair Display',serif !important; color:var(--navy) !important; }}
.topbar {{ height:38px; background:#12365F; color:#fff; border-radius:0 0 12px 12px; display:flex; align-items:center; justify-content:center; font-size:10.5px; letter-spacing:.15px; margin:-1rem -1rem 1rem -1rem; }}
.header-row {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:.8rem; }}
.brand {{ display:flex; align-items:center; gap:10px; font-weight:800; color:var(--navy); font-size:16px; }}
.brand-mark {{ width:34px; height:34px; border-radius:11px; background:linear-gradient(135deg,#DDF6FB,#BFE9F1); display:flex; align-items:center; justify-content:center; box-shadow:0 6px 14px rgba(22,137,168,.12); }}
.creator-name {{ text-align:center; font-family:'Playfair Display',serif; font-size:17px; font-weight:700; color:var(--blue); margin-top:8px; }}
.hero {{ text-align:center; margin:.1rem 0 1.2rem; }}
.hero h1 {{ font-size:56px; line-height:1; margin:.1rem 0 .35rem; letter-spacing:-1.5px; }}
.hero-sub {{ font-size:17px; color:var(--text); opacity:.85; }}
.control-label {{ color:var(--muted); font-size:11px; font-weight:700; margin-bottom:2px; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:18px; box-shadow:0 10px 30px rgba(20,80,105,.07); padding:21px; margin-bottom:16px; }}
.section-title {{ font-size:19px; font-weight:800; margin:8px 0 7px; color:var(--navy); }}
.section-sub {{ color:var(--muted); font-size:12px; line-height:1.55; margin-bottom:10px; }}
.about-title {{ font-size:18px; font-weight:800; color:var(--navy); margin-bottom:6px; }}
.about-text {{ color:var(--text); opacity:.86; font-size:13px; line-height:1.6; }}
.metric-pill {{ display:inline-flex; background:linear-gradient(135deg,#2B9DB8,#5BCBD9); color:white; padding:10px 15px; border-radius:12px; font-size:12px; font-weight:800; box-shadow:0 8px 18px rgba(35,154,179,.2); }}
.model-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:15px; }}
.model-item {{ background:var(--surface2); border:1px solid var(--line); border-radius:13px; padding:13px; }}
.model-label {{ font-size:10px; color:var(--muted); font-weight:700; text-transform:uppercase; letter-spacing:.5px; }}
.model-value {{ margin-top:5px; font-size:13px; font-weight:800; color:var(--navy); }}
.input-card {{ background:var(--surface2); border:1px solid var(--line); border-radius:16px; padding:15px; }}
div[data-testid="stTextArea"] textarea {{ background:var(--input) !important; color:var(--text) !important; border:1px solid #8CC9D8 !important; border-radius:13px !important; min-height:165px; }}
div[data-testid="stTextArea"] textarea::placeholder {{ color:var(--muted) !important; }}
.stButton > button, .stLinkButton > a {{ border-radius:11px !important; min-height:43px !important; font-weight:800 !important; border:1px solid #0D3158 !important; }}
.stButton > button[kind="primary"] {{ background:#0B315C !important; color:#fff !important; }}
.stButton > button:not([kind="primary"]), .stLinkButton > a {{ background:var(--surface) !important; color:var(--text) !important; border-color:var(--line) !important; }}
.stButton > button:hover, .stLinkButton > a:hover {{ transform:translateY(-1px); }}
.voice-box {{ background:var(--accent-bg); border:1px solid var(--line); border-radius:13px; padding:11px; margin:10px 0; }}
.voice-caption {{ color:var(--muted); font-size:12px; line-height:1.5; }}
.result-card {{ background:linear-gradient(145deg,#EFFBFD,#E7F7FB); border:1px solid #D4EAF0; border-radius:15px; padding:18px; text-align:center; }}
.result-label {{ font-size:12px; color:#617486; }}
.result-value {{ font-size:30px; font-weight:800; color:#071A35; margin-top:4px; }}
.result-confidence {{ font-size:13px; color:#27687A; font-weight:800; margin-top:5px; }}
.ai-card {{ background:var(--surface); border:1px solid var(--line); border-radius:15px; padding:17px; box-shadow:0 8px 22px rgba(24,80,105,.06); }}
.ai-title {{ font-size:16px; font-weight:800; margin-bottom:7px; color:var(--navy); }}
.ai-copy {{ color:var(--text); opacity:.87; line-height:1.6; font-size:13px; }}
.prob-row {{ margin:8px 0 13px; }}
.prob-head {{ display:flex; justify-content:space-between; gap:10px; font-size:13px; color:var(--text); margin-bottom:5px; }}
.prob-track {{ height:9px; background:#DDECF1; border-radius:20px; overflow:hidden; }}
.prob-fill {{ height:100%; border-radius:20px; background:linear-gradient(90deg,#2D91AF,#62CFD6); }}
.prob-fill.danger {{ background:linear-gradient(90deg,#D44848,#F08C8C); }}
.emergency {{ background:#FFF1F1; border:1px solid #E4A4A4; border-radius:14px; padding:14px; color:#6F2020; margin-top:12px; }}
.history-item {{ background:var(--surface); border:1px solid var(--line); border-radius:15px; padding:15px 17px; margin:9px 0; }}
.history-head {{ display:flex; justify-content:space-between; gap:10px; align-items:center; }}
.history-class {{ font-weight:800; color:var(--navy); }}
.history-time {{ font-size:11px; color:var(--muted); }}
.dev-card {{ background:linear-gradient(135deg,#0D3158,#17577A); color:#fff; border-radius:18px; padding:18px 20px; margin-top:16px; box-shadow:0 12px 30px rgba(13,49,88,.2); }}
.dev-card * {{ color:#fff !important; }}
.footer {{ text-align:center; color:var(--muted); font-size:11px; padding:18px 0 4px; }}
@media(max-width:900px) {{ .hero h1 {{font-size:42px}} .model-grid {{grid-template-columns:1fr 1fr}} }}
@media(max-width:520px) {{ .model-grid {{grid-template-columns:1fr}} }}
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# HEADER CONTROLS
# =============================================================================
st.markdown(f'<div class="topbar">{safe_text(T["disclaimer"])}</div>', unsafe_allow_html=True)

h1, h2, h3 = st.columns([4.7, 1.1, 1.1], gap="small")
with h1:
    st.markdown('<div class="brand"><div class="brand-mark">💙</div> SafeSpace AI</div>', unsafe_allow_html=True)
with h2:
    if st.toggle(T["language"], value=(lang == "ar"), key="language_toggle") != (lang == "ar"):
        st.session_state.language = "ar" if lang == "en" else "en"
        st.rerun()
with h3:
    if st.toggle("◐", value=dark, key="theme_toggle", help=T["theme"]) != dark:
        st.session_state.theme = "dark" if not dark else "light"
        st.rerun()


direction = "rtl" if lang == "ar" else "ltr"

st.markdown(
    f'''<div dir="{direction}" class="hero">
        <h1>SafeSpace AI</h1>
        <div class="hero-sub">{safe_text(T["subtitle"])}</div>
        <div class="creator-name">{safe_text(T["creator"])}</div>
    </div>''',
    unsafe_allow_html=True,
)

# =============================================================================
# ABOUT / MODEL INFORMATION — SEPARATE SECTION
# =============================================================================
st.markdown(
    f'''<div dir="{direction}" class="card">
        <div class="about-title">{safe_text(T["about"])}</div>
        <div class="about-text">{safe_text(T["about_desc"])}</div>
        <div style="margin-top:12px"><span class="metric-pill">{safe_text(T["accuracy"])}</span></div>
    </div>''',
    unsafe_allow_html=True,
)

st.markdown(f'<div dir="{direction}" class="section-title">{safe_text(T["model_section"])}</div>', unsafe_allow_html=True)
st.markdown(
    f'''<div dir="{direction}" class="card" style="padding:14px">
        <div class="model-grid">
            <div class="model-item"><div class="model-label">{safe_text(T["model_type"])}</div><div class="model-value">XGBoost Classifier</div></div>
            <div class="model-item"><div class="model-label">{safe_text(T["features"])}</div><div class="model-value">TF-IDF · 20,000 features</div></div>
            <div class="model-item"><div class="model-label">{safe_text(T["classes"])}</div><div class="model-value">7 categories</div></div>
            <div class="model-item"><div class="model-label">{safe_text(T["training_language"])}</div><div class="model-value">English</div></div>
        </div>
    </div>''',
    unsafe_allow_html=True,
)

# =============================================================================
# INPUT AREA
# =============================================================================
left, right = st.columns([1.75, 1.0], gap="large")

with left:
    st.markdown(f'<div dir="{direction}" class="section-title">{safe_text(T["input_title"])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div dir="{direction}" class="section-sub">{safe_text(T["input_hint"])}</div>', unsafe_allow_html=True)

    st.text_area(
        "Text",
        placeholder=T["placeholder"],
        height=165,
        label_visibility="collapsed",
        key="text_input",
    )

    st.markdown(
        f'''<div dir="{direction}" class="voice-box"><div class="voice-caption"><b>{safe_text(T["voice"])}</b> · {safe_text(T["voice_hint"])}</div></div>''',
        unsafe_allow_html=True,
    )

    audio = st.audio_input(T["voice"])
    b1, b2 = st.columns([1.35, 1])
    with b1:
        transcribe = st.button(T["transcribe"], use_container_width=True)
    with b2:
        clear = st.button(T["clear"], use_container_width=True)

    if clear:
        st.session_state.text_input = ""
        st.session_state.voice_transcript = ""
        st.session_state.results = None
        st.rerun()

    if transcribe:
        if audio is None:
            st.warning(T["empty"])
        else:
            transcript, err = speech_to_text(audio.getvalue())
            if err == "missing":
                st.error(T["voice_missing"])
            elif err == "network":
                st.error(T["voice_network"])
            elif err:
                st.warning(T["voice_failed"])
            else:
                st.session_state.voice_transcript = transcript
                st.session_state.text_input = transcript
                st.rerun()

    if st.session_state.voice_transcript:
        st.caption(st.session_state.voice_transcript)

    analyze = st.button(T["analyze"], use_container_width=True, type="primary")

with right:
    st.markdown(
        f'''<div dir="{direction}" class="ai-card" style="min-height:205px">
            <div class="ai-title">💙 {safe_text(T["response"])}</div>
            <div class="ai-copy">{safe_text("Write a thought or use your microphone to begin." if lang == "en" else "اكتبي فكرة أو استخدمي الميكروفون للبدء.")}</div>
        </div>''',
        unsafe_allow_html=True,
    )

# =============================================================================
# ANALYSIS
# =============================================================================
if analyze:
    raw_text = st.session_state.text_input.strip()
    if not raw_text:
        st.warning(T["empty"])
    else:
        try:
            with st.spinner(T["processing"]):
                english_text, translated = translate_to_english(raw_text)
                if translated:
                    st.info(T["translation"])

                pipeline = load_model()
                vectorizer = pipeline["vectorizer"]
                model = pipeline["model"]
                label_encoder = pipeline["label_encoder"]

                features = vectorizer.transform([english_text])
                pred_encoded = int(model.predict(features)[0])
                probabilities = model.predict_proba(features)[0]
                labels = list(label_encoder.classes_)
                predicted_label = str(label_encoder.inverse_transform([pred_encoded])[0])
                prob_dict = {str(labels[i]): float(probabilities[i]) for i in range(len(labels))}

                result = {
                    "prediction": predicted_label,
                    "confidence": float(prob_dict[predicted_label]),
                    "probabilities": prob_dict,
                    "translated": translated,
                }
                st.session_state.results = result
                save_history(raw_text, result)
        except Exception as exc:
            st.error(f"{safe_text('Analysis error: ' + str(exc))}")

# =============================================================================
# RESULTS
# =============================================================================
results = st.session_state.results
if results:
    pred = results["prediction"]
    display_pred = CLASS_AR.get(pred, pred) if lang == "ar" else pred
    confidence = results["confidence"]
    suicidal_prob = results["probabilities"].get("Suicidal", 0.0)

    st.markdown('<div style="height:5px"></div>', unsafe_allow_html=True)
    r1, r2 = st.columns([1.25, 1.0], gap="large")

    with r1:
        st.markdown(
            f'''<div dir="{direction}" class="result-card"><div class="result-label">{safe_text(T["detected"])}</div><div class="result-value">{safe_text(display_pred)}</div><div class="result-confidence">{safe_text(T["confidence"])}: {confidence:.1%}</div></div>''',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div dir="{direction}" class="section-title">{safe_text(T["breakdown"])}</div>', unsafe_allow_html=True)
        for label, probability in sorted(results["probabilities"].items(), key=lambda x: x[1], reverse=True):
            shown = CLASS_AR.get(label, label) if lang == "ar" else label
            danger = " danger" if label == "Suicidal" else ""
            st.markdown(
                f'''<div dir="{direction}" class="prob-row"><div class="prob-head"><span>{safe_text(shown)}</span><span>{probability:.0%}</span></div><div class="prob-track"><div class="prob-fill{danger}" style="width:{max(0,min(100,probability*100)):.2f}%"></div></div></div>''',
                unsafe_allow_html=True,
            )

    with r2:
        consolation, steps = GUIDANCE.get(pred, GUIDANCE["Normal"])[lang]
        st.markdown(
            f'''<div dir="{direction}" class="ai-card"><div class="ai-title">💙 {safe_text(T["response"])}</div><div class="ai-copy">{safe_text(consolation)}</div><br><div class="ai-title">{safe_text(T["guidance"])}</div><div class="ai-copy">{safe_text(consolation)}</div><br><div class="ai-title">🛡️ {safe_text(T["selfcare"])}</div><div class="ai-copy">{"<br>".join("• " + safe_text(x) for x in steps)}</div></div>''',
            unsafe_allow_html=True,
        )
        if suicidal_prob >= 0.25:
            st.markdown(
                f'''<div dir="{direction}" class="emergency"><div class="ai-title">🚨 {safe_text(T["emergency"])}</div><div style="font-size:12px;line-height:1.55">{safe_text(T["emergency_text"])}</div></div>''',
                unsafe_allow_html=True,
            )

    st.caption(f'{safe_text("Analysis time" if lang == "en" else "وقت التحليل")}: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

# =============================================================================
# SESSION HISTORY
# =============================================================================
if st.session_state.analysis_history:
    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    st.markdown(f'<div dir="{direction}" class="section-title">{safe_text(T["history"])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div dir="{direction}" class="section-sub">{safe_text(T["history_note"])}</div>', unsafe_allow_html=True)

    hc1, hc2 = st.columns([4.5, 1])
    with hc2:
        if st.button(T["clear_history"], use_container_width=True):
            st.session_state.analysis_history = []
            st.session_state.results = None
            st.rerun()

    for item in reversed(st.session_state.analysis_history):
        display_class = CLASS_AR.get(item["prediction"], item["prediction"]) if lang == "ar" else item["prediction"]
        st.markdown(
            f'''<div dir="{direction}" class="history-item"><div class="history-head"><span class="history-class">{safe_text(display_class)}</span><span class="history-time">{safe_text(item["timestamp"])}</span></div><div style="margin-top:5px;color:var(--muted);font-size:12px">{safe_text(T["confidence"])}: <b>{item["confidence"]:.1%}</b></div></div>''',
            unsafe_allow_html=True,
        )
        with st.expander(T["view"]):
            st.write(item["text"])
            for label, p in sorted(item["probabilities"].items(), key=lambda x: x[1], reverse=True):
                shown = CLASS_AR.get(label, label) if lang == "ar" else label
                st.write(f"**{shown}:** {p:.1%}")

# =============================================================================
# DEVELOPER CONTACT
# =============================================================================
st.markdown(
    f'''<div dir="{direction}" class="dev-card"><div style="font-size:18px;font-weight:800">{safe_text(T["developer"])}</div><div style="font-size:12px;opacity:.88;margin-top:5px">{safe_text(T["developer_note"])}</div></div>''',
    unsafe_allow_html=True,
)
st.link_button(
    "💬 " + ("Chat with Habiba on WhatsApp" if lang == "en" else "تحدثي مع حبيبة على واتساب"),
    "https://wa.me/qr/LG7JYBZDF2WMF1",
    use_container_width=True,
)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown(
    f'''<div dir="{direction}" class="footer"><b>SafeSpace AI</b><br>{safe_text(T["footer"])}<br>{safe_text(T["privacy"])}</div>''',
    unsafe_allow_html=True,
)
