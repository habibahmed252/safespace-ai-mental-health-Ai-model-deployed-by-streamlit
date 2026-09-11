"""SafeSpace AI — professional Streamlit application.

Designed & developed by Eng. Habiba Ahmed Talat.

The deployed model is an English-trained TF-IDF + XGBoost classifier.
Arabic messages are translated to English before inference. User wording is
not grammar-checked or rejected; spelling, slang, punctuation and informal
writing are accepted as normal input.
"""

import html
import io
import re
import warnings
from datetime import datetime

import joblib
import nltk
from scipy.sparse import hstack, csr_matrix  # Required because the saved model contains nltk.stem.PorterStemmer.
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


# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SafeSpace AI",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -----------------------------------------------------------------------------
# SESSION STATE — NOTHING IS PERSISTED OUTSIDE THE CURRENT SESSION
# -----------------------------------------------------------------------------
for key, default in {
    "language": "en",
    "theme": "light",
    "text_input": "",
    "results": None,
    "analysis_history": [],
    "voice_transcript": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# -----------------------------------------------------------------------------
# TRANSLATIONS
# -----------------------------------------------------------------------------
UI = {
    "en": {
        "disclaimer": "For informational and educational support only. Not a medical diagnosis.",
        "brand": "SafeSpace AI",
        "subtitle": "Understanding your unspoken words",
        "creator": "Designed & developed by Eng. Habiba Ahmed Talat",
        "about": "About SafeSpace AI",
        "about_desc": "SafeSpace AI uses an English-trained NLP classifier to estimate which of seven categories is most consistent with the submitted text. The output is a model prediction for reflection and screening-style analysis — not a clinical diagnosis.",
        "accuracy": "Model Accuracy: 94.2%",
        "model_section": "Model Information",
        "model_arch": "Model architecture",
        "features": "Feature extraction",
        "classes": "Prediction classes",
        "language": "Training language",
        "input": "Share Your Thoughts",
        "input_hint": "Write naturally in English or Arabic. Spelling mistakes, slang, informal wording and punctuation are accepted.",
        "placeholder": "Express your thoughts freely here...",
        "voice": "Voice input",
        "voice_hint": "Speak in English or Arabic and convert your voice into text.",
        "transcribe": "Convert voice to text",
        "analyze": "Analyze Text",
        "clear": "Clear",
        "empty": "Please write something or record your voice first.",
        "processing": "Analyzing your text...",
        "translation": "Arabic detected — translating the message for the English-trained model.",
        "translation_error": "Arabic translation is temporarily unavailable. You can still use English input, or try again in a moment.",
        "voice_missing": "Voice recognition is unavailable because its dependency is not installed.",
        "voice_unknown": "I couldn't understand the recording. Please try again.",
        "voice_network": "Speech recognition could not reach its recognition service.",
        "detected": "Detected Mental State",
        "confidence": "Confidence",
        "breakdown": "Class Probabilities Breakdown",
        "response": "Empathetic AI Response",
        "guidance": "Personalized Guidance",
        "selfcare": "Self-Care Micro-Steps",
        "emergency": "URGENT SUPPORT",
        "emergency_text": "If you may be in immediate danger or having thoughts of suicide, contact local emergency services or a trusted person now.",
        "history": "Session History",
        "history_note": "Previous analyses stay only in this active browser session and are not written to a database or file.",
        "view": "View analysis",
        "clear_history": "Clear history",
        "developer": "Chat with Developer",
        "developer_note": "Have feedback, ideas, or a question about SafeSpace AI?",
        "theme": "Dark theme",
        "whatsapp": "Chat with Habiba on WhatsApp",
        "footer": "Built with Streamlit · Designed & developed by Eng. Habiba Ahmed Talat",
        "privacy": "Session-only processing. Your messages are not intentionally stored outside the active session.",
    },
    "ar": {
        "disclaimer": "للاستخدام الإرشادي والتعليمي فقط، وليس تشخيصًا طبيًا.",
        "brand": "SafeSpace AI",
        "subtitle": "فهم ما بين السطور",
        "creator": "تصميم وتطوير م. حبيبة أحمد طلعت",
        "about": "عن SafeSpace AI",
        "about_desc": "يستخدم SafeSpace AI نموذجًا لمعالجة اللغة الطبيعية مدرّبًا باللغة الإنجليزية لتقدير الفئة الأكثر توافقًا مع النص من بين سبع فئات. النتيجة توقع من النموذج لأغراض الفهم والتحليل وليست تشخيصًا طبيًا.",
        "accuracy": "دقة النموذج: 94.2%",
        "model_section": "معلومات النموذج",
        "model_arch": "بنية النموذج",
        "features": "استخراج الخصائص",
        "classes": "فئات التوقع",
        "language": "لغة التدريب",
        "input": "شاركي أفكارك",
        "input_hint": "اكتبي بطريقتك الطبيعية بالعربي أو بالإنجليزي. الأخطاء الإملائية والاختصارات والكتابة العامية وعلامات الترقيم مسموحة.",
        "placeholder": "عبّري عن أفكارك ومشاعرك هنا بحرية...",
        "voice": "الإدخال الصوتي",
        "voice_hint": "اتكلمي بالعربي أو بالإنجليزي وحوّلي صوتك إلى نص.",
        "transcribe": "تحويل الصوت إلى نص",
        "analyze": "تحليل النص",
        "clear": "مسح",
        "empty": "اكتبي نصًا أو سجلي صوتك أولًا.",
        "processing": "جاري تحليل النص...",
        "translation": "تم اكتشاف العربية — يتم ترجمة الرسالة للنموذج المدرب بالإنجليزية.",
        "translation_error": "ترجمة العربية غير متاحة مؤقتًا. جرّبي مرة أخرى بعد قليل أو استخدمي الإنجليزية.",
        "voice_missing": "الإدخال الصوتي غير متاح لأن مكوّن التعرف على الكلام غير مثبت.",
        "voice_unknown": "لم أستطع فهم التسجيل. جرّبي مرة أخرى.",
        "voice_network": "تعذر الوصول إلى خدمة التعرف على الكلام.",
        "detected": "الحالة المتوقعة",
        "confidence": "درجة الثقة",
        "breakdown": "توزيع احتمالات الفئات",
        "response": "استجابة AI داعمة",
        "guidance": "توجيهات شخصية",
        "selfcare": "خطوات بسيطة للعناية بالنفس",
        "emergency": "دعم فوري مطلوب",
        "emergency_text": "لو في خطر فوري أو أفكار انتحارية، تواصلي الآن مع خدمات الطوارئ المحلية أو شخص تثقين به.",
        "history": "سجل الجلسة",
        "history_note": "التحليلات السابقة موجودة داخل الجلسة الحالية فقط ولا يتم حفظها في قاعدة بيانات أو ملف.",
        "view": "عرض التحليل",
        "clear_history": "مسح السجل",
        "developer": "تواصلي مع المطوّرة",
        "developer_note": "عندك ملاحظة أو فكرة أو سؤال عن SafeSpace AI؟",
        "theme": "الوضع الداكن",
        "whatsapp": "تحدثي مع حبيبة على واتساب",
        "footer": "تم تطويره باستخدام Streamlit · تصميم وتطوير م. حبيبة أحمد طلعت",
        "privacy": "معالجة داخل الجلسة فقط. لا يتم حفظ رسائلك خارج الجلسة الحالية بشكل مقصود.",
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
        "en": ("Your anxiety is valid. A model prediction does not define you.", ["Name five things you can see.", "Take slow breaths for one minute.", "Take a short gentle walk."]),
        "ar": ("مشاعرك مهمة، ونتيجة النموذج لا تحددك.", ["سمي خمس حاجات شايفاها حولك.", "خدي نفس ببطء لمدة دقيقة.", "اعملي مشية بسيطة وهادية."]),
    },
    "Stress": {
        "en": ("Stress can be a signal that your mind and body need a pause.", ["Write down your top three stressors.", "Take a 15-minute break.", "Choose one small task you can control today."]),
        "ar": ("التوتر ممكن يكون إشارة إن جسمك وذهنك محتاجين وقفة.", ["اكتبي أهم 3 حاجات مسببة للتوتر.", "خدي استراحة 15 دقيقة.", "اختاري خطوة صغيرة تقدري تتحكمي فيها النهارده."]),
    },
    "Depression": {
        "en": ("What you feel matters, but a model prediction cannot define your worth.", ["Drink water and get a little daylight.", "Message someone you trust.", "Choose one tiny achievable activity."]),
        "ar": ("إحساسك مهم، لكن نتيجة النموذج لا تحدد قيمتك.", ["اشربي مياه واقعدي شوية في ضوء النهار.", "كلمي شخص تثقي فيه.", "اختاري نشاطًا صغيرًا جدًا تقدري تعمليه."]),
    },
    "Normal": {
        "en": ("Your text is most consistent with the Normal category in this model.", ["Keep routines that support your wellbeing.", "Notice what is working for you.", "Stay connected with people you trust."]),
        "ar": ("النص أقرب لفئة الطبيعي حسب النموذج الحالي.", ["حافظي على العادات اللي بتدعم راحتك.", "خدي بالك من الحاجات اللي بتساعدك.", "خلي التواصل مع الناس الموثوقين جزءًا من حياتك."]),
    },
    "Bipolar": {
        "en": ("This is a model category, not a diagnosis. Proper assessment requires a clinician.", ["Track noticeable mood or sleep changes.", "Avoid major decisions during intense mood changes.", "Consider professional assessment if concerns persist."]),
        "ar": ("دي فئة في النموذج وليست تشخيصًا. التقييم الصحيح يحتاج متخصصًا.", ["سجلي التغيرات الواضحة في المزاج أو النوم.", "تجنبي القرارات الكبيرة أثناء التغيرات الشديدة.", "فكري في تقييم متخصص لو القلق مستمر."]),
    },
    "Personality disorder": {
        "en": ("This model label is not a clinical diagnosis and should not be used to label yourself.", ["Treat the result as a reflection signal only.", "Notice recurring patterns without judging yourself.", "Seek professional assessment for persistent concerns."]),
        "ar": ("التصنيف ده من النموذج مش تشخيص طبي ومينفعش نستخدمه للحكم على نفسك.", ["اعتبري النتيجة مجرد إشارة للتفكير.", "لاحظي الأنماط المتكررة بدون لوم لنفسك.", "استشيري متخصصًا لو المشكلة مستمرة."]),
    },
    "Suicidal": {
        "en": ("This category requires extra caution. The model cannot determine your safety. If you may act on suicidal thoughts, seek immediate human help.", ["Contact a trusted person now.", "Move away from anything you could use to hurt yourself.", "Contact local emergency or crisis services if danger is imminent."]),
        "ar": ("الفئة دي تحتاج حذر شديد. النموذج لا يستطيع تحديد مدى أمانك. لو ممكن تتصرفي بناءً على أفكار انتحارية، اطلبي مساعدة بشرية فورًا.", ["كلمي شخص تثقي فيه الآن.", "ابعدي عن أي شيء ممكن تستخدميه لإيذاء نفسك.", "تواصلي مع الطوارئ أو خدمات الأزمات المحلية لو في خطر قريب."]),
    },
}


# -----------------------------------------------------------------------------
# MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    # The model artifact contains a PorterStemmer object, hence nltk is a real
    # runtime dependency even though app.py does not call a tokenizer directly.
    return joblib.load("model.pkl")


# -----------------------------------------------------------------------------
# INPUT HANDLING
# -----------------------------------------------------------------------------
def contains_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def normalize_user_text(text: str) -> str:
    """Light normalization only. No grammar/spelling correction is performed."""
    text = (text or "").replace("\u200f", "").replace("\u200e", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def translate_if_needed(text: str):
    """Translate Arabic/mixed Arabic text without rejecting informal writing."""
    if not contains_arabic(text):
        return text, False
    if GoogleTranslator is None:
        return None, False
    translated = GoogleTranslator(source="auto", target="en").translate(text)
    return normalize_user_text(translated), True


def speech_to_text(audio_bytes: bytes):
    if sr is None:
        return None, "missing"
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)

        # Try both languages and keep the stronger recognized result.
        candidates = []
        for locale in ("en-US", "ar-EG"):
            try:
                result = recognizer.recognize_google(audio, language=locale, show_all=True)
                if isinstance(result, dict):
                    for alt in result.get("alternative", []):
                        transcript = alt.get("transcript", "").strip()
                        if transcript:
                            candidates.append((float(alt.get("confidence", 0)), transcript))
                elif isinstance(result, str) and result.strip():
                    candidates.append((0.0, result.strip()))
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                return None, "network"

        if not candidates:
            return None, "unknown"
        return max(candidates, key=lambda x: x[0])[1], None
    except Exception:
        return None, "unknown"


def run_prediction(raw_text: str):
    text = normalize_user_text(raw_text)
    if not text:
        raise ValueError("empty")

    model_pack = load_model()
    english_text, translated = translate_if_needed(text)
    if contains_arabic(text) and english_text is None:
        raise RuntimeError("translation_unavailable")

    vectorizer = model_pack["vectorizer"]
    model = model_pack["model"]
    label_encoder = model_pack["label_encoder"]

    # Never grammar-check or reject the user's writing. The trained vectorizer
    # simply receives the normalized text (or its English translation).
    # The saved XGBoost model expects 20,002 features:
    # 20,000 TF-IDF features + 2 numeric text features used during training.
    # The original vectorizer intentionally exposes only the 20,000 TF-IDF columns.
    tfidf_features = vectorizer.transform([english_text])

    # These two features match the model's extra feature positions:
    # f20000 = character count, f20001 = word count.
    text_length = len(english_text)
    word_count = len(re.findall(r"\b\w+\b", english_text, flags=re.UNICODE))

    extra_features = csr_matrix([[text_length, word_count]], dtype=tfidf_features.dtype)
    features = hstack([tfidf_features, extra_features], format="csr")

    # Safety check so a future model/vectorizer mismatch gives a useful message.
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is not None and features.shape[1] != expected_features:
        raise ValueError(
            f"Model expects {expected_features} features, but the application created "
            f"{features.shape[1]} features."
        )

    probabilities = model.predict_proba(features)[0]
    encoded_prediction = int(model.predict(features)[0])
    prediction = str(label_encoder.inverse_transform([encoded_prediction])[0])

    classes = list(label_encoder.classes_)
    prob_dict = {str(classes[i]): float(probabilities[i]) for i in range(len(classes))}

    return {
        "prediction": prediction,
        "confidence": float(prob_dict[prediction]),
        "probabilities": prob_dict,
        "translated": translated,
    }


def save_history(text, result):
    item = {
        "text": text,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"].copy(),
        "timestamp": datetime.now().strftime("%H:%M:%S"),
    }
    st.session_state.analysis_history.append(item)


def esc(value):
    return html.escape(str(value))


# -----------------------------------------------------------------------------
# THEME
# -----------------------------------------------------------------------------
lang = st.session_state.language
T = UI[lang]
dark = st.session_state.theme == "dark"

if dark:
    colors = {
        "bg": "#0D1624", "surface": "#142235", "surface2": "#182A40",
        "input": "#102033", "text": "#F4F8FC", "muted": "#AAB8C8",
        "line": "#29415B", "navy": "#F4F8FC", "accent_bg": "#173A50",
    }
else:
    colors = {
        "bg": "#F4FBFD", "surface": "#FFFFFF", "surface2": "#F7FCFE",
        "input": "#FFFFFF", "text": "#132238", "muted": "#637386",
        "line": "#D9EAF0", "navy": "#071A35", "accent_bg": "#EFF9FC",
    }

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700;800&display=swap');
:root {{
 --bg:{colors['bg']}; --surface:{colors['surface']}; --surface2:{colors['surface2']};
 --input:{colors['input']}; --text:{colors['text']}; --muted:{colors['muted']};
 --line:{colors['line']}; --navy:{colors['navy']}; --accent-bg:{colors['accent_bg']};
 --blue:#1689A8; --blue2:#57C7D8;
}}
.stApp {{background:var(--bg); color:var(--text);}}
[data-testid="stHeader"] {{background:transparent;}}
.block-container {{max-width:1450px; padding-top:.65rem; padding-bottom:2rem;}}
* {{font-family:'Inter',sans-serif;}}
h1,h2,h3,h4 {{font-family:'Playfair Display',serif !important; color:var(--navy) !important;}}
.topbar {{height:38px; background:#12365F; color:#fff; border-radius:0 0 12px 12px; display:flex; align-items:center; justify-content:center; font-size:10.5px; margin:-1rem -1rem 1rem;}}
.brand {{display:flex; align-items:center; gap:10px; font-weight:800; color:var(--navy); font-size:16px;}}
.brand-mark {{width:34px;height:34px;border-radius:11px;background:linear-gradient(135deg,#DDF6FB,#BFE9F1);display:flex;align-items:center;justify-content:center;}}
.hero {{text-align:center; margin:.1rem 0 1.2rem;}}
.hero h1 {{font-size:56px;line-height:1;margin:.1rem 0 .35rem;letter-spacing:-1.5px;}}
.hero-sub {{font-size:17px;color:var(--text);opacity:.85;}}
.creator-name {{font-family:'Playfair Display',serif;font-size:18px;font-weight:800;color:#1689A8;margin-top:8px;}}
.card {{background:var(--surface);border:1px solid var(--line);border-radius:18px;box-shadow:0 10px 30px rgba(20,80,105,.07);padding:21px;margin-bottom:16px;}}
.section-title {{font-size:19px;font-weight:800;margin:10px 0 7px;color:var(--navy);}}
.section-sub {{color:var(--muted);font-size:12px;line-height:1.55;margin-bottom:10px;}}
.about-title {{font-size:18px;font-weight:800;color:var(--navy);margin-bottom:6px;}}
.about-text {{color:var(--text);opacity:.88;font-size:13px;line-height:1.6;}}
.metric-pill {{display:inline-flex;background:linear-gradient(135deg,#2B9DB8,#5BCBD9);color:white;padding:10px 15px;border-radius:12px;font-size:12px;font-weight:800;}}
.model-grid {{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;}}
.model-item {{background:var(--surface2);border:1px solid var(--line);border-radius:13px;padding:13px;}}
.model-label {{font-size:10px;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.5px;}}
.model-value {{margin-top:5px;font-size:13px;font-weight:800;color:var(--navy);}}
div[data-testid="stTextArea"] textarea {{background:var(--input) !important;color:var(--text) !important;border:1px solid #8CC9D8 !important;border-radius:13px !important;min-height:165px;}}
div[data-testid="stTextArea"] textarea::placeholder {{color:var(--muted) !important;}}
.stButton > button, .stLinkButton > a {{border-radius:11px !important;min-height:43px !important;font-weight:800 !important;}}
.stButton > button[kind="primary"] {{background:#0B315C !important;color:#fff !important;}}
.stButton > button:not([kind="primary"]), .stLinkButton > a {{background:var(--surface) !important;color:var(--text) !important;border-color:var(--line) !important;}}
.voice-box {{background:var(--accent-bg);border:1px solid var(--line);border-radius:13px;padding:11px;margin:10px 0;}}
.voice-caption {{color:var(--muted);font-size:12px;line-height:1.5;}}
.result-card {{background:linear-gradient(145deg,#EFFBFD,#E7F7FB);border:1px solid #D4EAF0;border-radius:15px;padding:18px;text-align:center;}}
.result-label {{font-size:12px;color:#617486;}}
.result-value {{font-size:30px;font-weight:800;color:#071A35;margin-top:4px;}}
.result-confidence {{font-size:13px;color:#27687A;font-weight:800;margin-top:5px;}}
.ai-card {{background:var(--surface);border:1px solid var(--line);border-radius:15px;padding:17px;box-shadow:0 8px 22px rgba(24,80,105,.06);}}
.ai-title {{font-size:16px;font-weight:800;margin-bottom:7px;color:var(--navy);}}
.ai-copy {{color:var(--text);opacity:.88;line-height:1.6;font-size:13px;}}
.prob-row {{margin:8px 0 13px;}}
.prob-head {{display:flex;justify-content:space-between;gap:10px;font-size:13px;color:var(--text);margin-bottom:5px;}}
.prob-track {{height:9px;background:#DDECF1;border-radius:20px;overflow:hidden;}}
.prob-fill {{height:100%;border-radius:20px;background:linear-gradient(90deg,#2D91AF,#62CFD6);}}
.prob-fill.danger {{background:linear-gradient(90deg,#D44848,#F08C8C);}}
.emergency {{background:#FFF1F1;border:1px solid #E4A4A4;border-radius:14px;padding:14px;color:#6F2020;margin-top:12px;}}
.history-item {{background:var(--surface);border:1px solid var(--line);border-radius:15px;padding:15px 17px;margin:9px 0;}}
.history-head {{display:flex;justify-content:space-between;gap:10px;align-items:center;}}
.history-class {{font-weight:800;color:var(--navy);}}
.history-time {{font-size:11px;color:var(--muted);}}
.dev-card {{background:linear-gradient(135deg,#0D3158,#17577A);color:#fff;border-radius:18px;padding:18px 20px;margin-top:16px;box-shadow:0 12px 30px rgba(13,49,88,.2);}}
.dev-card * {{color:#fff !important;}}
.footer {{text-align:center;color:var(--muted);font-size:11px;padding:18px 0 4px;}}
@media(max-width:900px) {{.hero h1{{font-size:42px}}.model-grid{{grid-template-columns:1fr 1fr}}}}
@media(max-width:520px) {{.model-grid{{grid-template-columns:1fr}}}}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# HEADER
# -----------------------------------------------------------------------------
st.markdown(f'<div class="topbar">{esc(T["disclaimer"])}</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns([5.5, 1.0, 1.0])
with c1:
    st.markdown('<div class="brand"><div class="brand-mark">💙</div> SafeSpace AI</div>', unsafe_allow_html=True)
with c2:
    if st.button("AR" if lang == "en" else "EN", use_container_width=True, key="language_btn"):
        st.session_state.language = "ar" if lang == "en" else "en"
        st.rerun()
with c3:
    if st.button("☾" if not dark else "☀", use_container_width=True, key="theme_btn", help=T["theme"]):
        st.session_state.theme = "dark" if not dark else "light"
        st.rerun()


direction = "rtl" if lang == "ar" else "ltr"
st.markdown(
    f'<div dir="{direction}" class="hero"><h1>SafeSpace AI</h1><div class="hero-sub">{esc(T["subtitle"])}</div><div class="creator-name">{esc(T["creator"])}</div></div>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# ABOUT
# -----------------------------------------------------------------------------
st.markdown(
    f'<div dir="{direction}" class="card"><div class="about-title">{esc(T["about"])}</div><div class="about-text">{esc(T["about_desc"])}</div><div style="margin-top:12px"><span class="metric-pill">{esc(T["accuracy"])}</span></div></div>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# MODEL INFORMATION — SEPARATE, CLEAN SECTION
# -----------------------------------------------------------------------------
st.markdown(f'<div dir="{direction}" class="section-title">{esc(T["model_section"])}</div>', unsafe_allow_html=True)
st.markdown(
    f'''<div dir="{direction}" class="card" style="padding:14px"><div class="model-grid">
    <div class="model-item"><div class="model-label">{esc(T["model_arch"])}</div><div class="model-value">XGBoost Classifier</div></div>
    <div class="model-item"><div class="model-label">{esc(T["features"])}</div><div class="model-value">TF-IDF · 20,000 features</div></div>
    <div class="model-item"><div class="model-label">{esc(T["classes"])}</div><div class="model-value">7 categories</div></div>
    <div class="model-item"><div class="model-label">{esc(T["language"])}</div><div class="model-value">English</div></div>
    </div></div>''',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
left, right = st.columns([1.7, 1.0], gap="large")

with left:
    st.markdown(f'<div dir="{direction}" class="section-title">{esc(T["input"])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div dir="{direction}" class="section-sub">{esc(T["input_hint"])}</div>', unsafe_allow_html=True)

    st.text_area(
        "Text",
        placeholder=T["placeholder"],
        height=165,
        label_visibility="collapsed",
        key="text_input",
    )

    st.markdown(f'<div dir="{direction}" class="voice-box"><div class="voice-caption"><b>{esc(T["voice"])}</b> · {esc(T["voice_hint"])}</div></div>', unsafe_allow_html=True)
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
            transcript, error = speech_to_text(audio.getvalue())
            if error == "missing":
                st.error(T["voice_missing"])
            elif error == "network":
                st.error(T["voice_network"])
            elif error:
                st.warning(T["voice_unknown"])
            else:
                st.session_state.voice_transcript = transcript
                st.session_state.text_input = transcript
                st.rerun()

    if st.session_state.voice_transcript:
        st.caption(st.session_state.voice_transcript)

    analyze = st.button(T["analyze"], use_container_width=True, type="primary")

with right:
    starter = "Write a thought or use your microphone to begin." if lang == "en" else "اكتبي فكرة أو استخدمي الميكروفون للبدء."
    st.markdown(f'<div dir="{direction}" class="ai-card" style="min-height:205px"><div class="ai-title">💙 {esc(T["response"])}</div><div class="ai-copy">{esc(starter)}</div></div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# ANALYSIS
# -----------------------------------------------------------------------------
if analyze:
    raw_text = normalize_user_text(st.session_state.text_input)
    if not raw_text:
        st.warning(T["empty"])
    else:
        try:
            with st.spinner(T["processing"]):
                result = run_prediction(raw_text)
            if result["translated"]:
                st.info(T["translation"])
            st.session_state.results = result
            save_history(raw_text, result)
        except RuntimeError as exc:
            if str(exc) == "translation_unavailable":
                st.error(T["translation_error"])
            else:
                st.error("The analysis service is temporarily unavailable. Please try again.")
        except Exception as exc:
            # Show a useful but non-sensitive error. No grammar-related input is rejected.
            st.error(f"Analysis error: {esc(exc)}")


# -----------------------------------------------------------------------------
# RESULTS
# -----------------------------------------------------------------------------
results = st.session_state.results
if results:
    prediction = results["prediction"]
    shown_prediction = CLASS_AR.get(prediction, prediction) if lang == "ar" else prediction
    confidence = results["confidence"]
    suicidal_prob = results["probabilities"].get("Suicidal", 0.0)

    r1, r2 = st.columns([1.25, 1.0], gap="large")
    with r1:
        st.markdown(f'<div dir="{direction}" class="result-card"><div class="result-label">{esc(T["detected"])}</div><div class="result-value">{esc(shown_prediction)}</div><div class="result-confidence">{esc(T["confidence"])}: {confidence:.1%}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div dir="{direction}" class="section-title">{esc(T["breakdown"])}</div>', unsafe_allow_html=True)
        for label, probability in sorted(results["probabilities"].items(), key=lambda x: x[1], reverse=True):
            shown = CLASS_AR.get(label, label) if lang == "ar" else label
            danger = " danger" if label == "Suicidal" else ""
            width = max(0, min(100, probability * 100))
            st.markdown(f'<div dir="{direction}" class="prob-row"><div class="prob-head"><span>{esc(shown)}</span><span>{probability:.0%}</span></div><div class="prob-track"><div class="prob-fill{danger}" style="width:{width:.2f}%"></div></div></div>', unsafe_allow_html=True)

    with r2:
        consolation, steps = GUIDANCE.get(prediction, GUIDANCE["Normal"])[lang]
        steps_html = "<br>".join("• " + esc(step) for step in steps)
        st.markdown(f'<div dir="{direction}" class="ai-card"><div class="ai-title">💙 {esc(T["response"])}</div><div class="ai-copy">{esc(consolation)}</div><br><div class="ai-title">{esc(T["guidance"])}</div><div class="ai-copy">{esc(consolation)}</div><br><div class="ai-title">🛡️ {esc(T["selfcare"])}</div><div class="ai-copy">{steps_html}</div></div>', unsafe_allow_html=True)
        if suicidal_prob >= 0.25:
            st.markdown(f'<div dir="{direction}" class="emergency"><div class="ai-title">🚨 {esc(T["emergency"])}</div><div style="font-size:12px;line-height:1.55">{esc(T["emergency_text"])}</div></div>', unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# HISTORY
# -----------------------------------------------------------------------------
if st.session_state.analysis_history:
    st.markdown(f'<div dir="{direction}" class="section-title">{esc(T["history"])}</div>', unsafe_allow_html=True)
    st.markdown(f'<div dir="{direction}" class="section-sub">{esc(T["history_note"])}</div>', unsafe_allow_html=True)

    _, clear_col = st.columns([5, 1])
    with clear_col:
        if st.button(T["clear_history"], use_container_width=True, key="clear_history"):
            st.session_state.analysis_history = []
            st.session_state.results = None
            st.rerun()

    for item in reversed(st.session_state.analysis_history):
        shown = CLASS_AR.get(item["prediction"], item["prediction"]) if lang == "ar" else item["prediction"]
        st.markdown(f'<div dir="{direction}" class="history-item"><div class="history-head"><span class="history-class">{esc(shown)}</span><span class="history-time">{esc(item["timestamp"])}</span></div><div style="margin-top:5px;color:var(--muted);font-size:12px">{esc(T["confidence"])}: <b>{item["confidence"]:.1%}</b></div></div>', unsafe_allow_html=True)
        with st.expander(T["view"], expanded=False):
            st.write(item["text"])
            for label, probability in sorted(item["probabilities"].items(), key=lambda x: x[1], reverse=True):
                shown_label = CLASS_AR.get(label, label) if lang == "ar" else label
                st.write(f"**{shown_label}:** {probability:.1%}")


# -----------------------------------------------------------------------------
# DEVELOPER CONTACT
# -----------------------------------------------------------------------------
st.markdown(f'<div dir="{direction}" class="dev-card"><div style="font-size:18px;font-weight:800">{esc(T["developer"])}</div><div style="font-size:12px;opacity:.88;margin-top:5px">{esc(T["developer_note"])}</div></div>', unsafe_allow_html=True)
st.link_button(T["whatsapp"], "https://wa.me/qr/LG7JYBZDF2WMF1", use_container_width=True)


# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown(f'<div dir="{direction}" class="footer">{esc(T["footer"])}<br>{esc(T["privacy"])}</div>', unsafe_allow_html=True)
