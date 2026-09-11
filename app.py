"""
SafeSpace AI — Mental Health Text Analysis
Author: Habiba Ahmed Talat

The bundled model is an English XGBoost + TF-IDF classifier.
Arabic input is translated to English before inference.
Voice input is transcribed with Google Speech Recognition (English/Arabic).
"""

import html
import re
import warnings
from datetime import datetime

import joblib
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

# Optional runtime dependencies are imported here so the app can show a useful
# message instead of crashing if one of them is missing.
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
# DESIGN SYSTEM
# =============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --navy:#071A35;
    --blue:#1689A8;
    --blue2:#57C7D8;
    --ice:#EAF8FC;
    --card:#FFFFFF;
    --line:#D9EAF0;
    --text:#132238;
    --muted:#637386;
    --danger:#D94B4B;
    --danger-bg:#FFF1F1;
}

.stApp {
    background:
      radial-gradient(circle at 8% 8%, rgba(87,199,216,.16), transparent 25%),
      radial-gradient(circle at 92% 28%, rgba(22,137,168,.10), transparent 25%),
      #F4FBFD;
    color:var(--text);
}

[data-testid="stHeader"] { background:transparent; }
.block-container { max-width: 1500px; padding-top: .7rem; padding-bottom: 2rem; }

* { font-family:'Inter', sans-serif; }
h1,h2,h3,h4 { font-family:'Playfair Display', serif !important; color:var(--navy) !important; }

.topbar {
    height:42px;
    background:#12365F;
    color:white;
    border-radius:0 0 12px 12px;
    display:flex;
    align-items:center;
    justify-content:center;
    font-size:11px;
    letter-spacing:.2px;
    margin:-1rem -1rem 1.1rem -1rem;
}

.brand-row {
    display:flex;
    align-items:center;
    justify-content:space-between;
    margin-bottom:1.2rem;
}
.brand {
    display:flex;
    gap:10px;
    align-items:center;
    font-weight:700;
    color:var(--navy);
    font-size:15px;
}
.brand-mark {
    width:30px;height:30px;border-radius:50%;
    background:#E7F7FB;
    display:flex;align-items:center;justify-content:center;
    font-size:17px;
}

.lang-box {
    display:flex; align-items:center; gap:8px;
    color:#1C2C3D; font-size:13px; font-weight:600;
}
.lang-pill {
    padding:5px 10px; border-radius:20px;
    background:#DDEAF1; color:#17324D;
}

.hero { text-align:center; margin: .3rem 0 1.4rem; }
.hero h1 { font-size:52px; line-height:1.0; margin-bottom:.35rem; }
.hero-sub { font-size:18px; color:#1D2E3F; }
.creator { font-size:15px; margin-top:.35rem; color:#24384B; }

.card {
    background:rgba(255,255,255,.90);
    border:1px solid var(--line);
    border-radius:18px;
    box-shadow:0 10px 30px rgba(24,80,105,.08);
    padding:22px;
    margin-bottom:16px;
}
.about-card { min-height:150px; }
.about-title { font-size:18px; font-weight:700; margin-bottom:6px; }
.about-text { color:#283B4D; font-size:13px; line-height:1.55; }
.metric-pill {
    display:inline-flex;
    background:linear-gradient(135deg,#2B9DB8,#5BCBD9);
    color:white; padding:11px 16px; border-radius:12px;
    font-size:13px; font-weight:700;
    box-shadow:0 8px 18px rgba(35,154,179,.25);
}

.section-title { font-size:19px; font-weight:700; margin:8px 0 10px; }
.section-sub { color:var(--muted); font-size:12px; margin-bottom:10px; }

.input-card {
    background:#F7FCFE;
    border:1px solid #D7EAF0;
    border-radius:16px;
    padding:16px;
}
textarea {
    border-radius:12px !important;
    border:1px solid #8CC9D8 !important;
}
div[data-testid="stTextArea"] textarea { background:#FFFFFF !important; min-height:155px; }

.stButton > button {
    border-radius:10px;
    border:1px solid #0D3158;
    background:#0B315C;
    color:white;
    font-weight:700;
    min-height:44px;
}
.stButton > button:hover { background:#124579; color:white; border-color:#124579; }

.result-card {
    background:linear-gradient(145deg,#EFFBFD,#E7F7FB);
    border:1px solid #D4EAF0;
    border-radius:15px;
    padding:18px;
    text-align:center;
}
.result-label { font-size:12px; color:#617486; }
.result-value { font-size:30px; font-weight:800; color:var(--navy); margin-top:3px; }
.result-confidence { font-size:13px; color:#27687A; font-weight:700; margin-top:5px; }

.ai-card {
    background:white;
    border:1px solid var(--line);
    border-radius:15px;
    padding:17px;
    box-shadow:0 8px 22px rgba(24,80,105,.06);
}
.ai-title { font-size:17px; font-weight:800; margin-bottom:8px; color:var(--navy); }
.ai-copy { color:#34495A; line-height:1.55; font-size:13px; }

.prob-row { margin:8px 0 13px; }
.prob-head { display:flex; justify-content:space-between; font-size:13px; margin-bottom:5px; }
.prob-track { height:9px; background:#DDECF1; border-radius:20px; overflow:hidden; }
.prob-fill { height:100%; border-radius:20px; background:linear-gradient(90deg,#2D91AF,#62CFD6); }
.prob-fill.danger { background:linear-gradient(90deg,#D44848,#F08C8C); }

.emergency {
    background:var(--danger-bg);
    border:1px solid #E4A4A4;
    border-radius:14px;
    padding:14px;
    color:#6F2020;
}
.emergency-title { font-weight:800; font-size:15px; margin-bottom:5px; }
.emergency-text { font-size:12px; line-height:1.5; }

.voice-box {
    background:#EFF9FC;
    border:1px solid #D2EAF0;
    border-radius:13px;
    padding:11px;
    margin-top:10px;
}
.voice-caption { font-size:12px; color:#597081; margin-bottom:5px; }

.footer {
    text-align:center; color:#6A7B89; font-size:11px;
    padding:15px 0 5px;
}

[dir="rtl"] .hero,
[dir="rtl"] .card,
[dir="rtl"] .ai-card,
[dir="rtl"] .result-card { text-align:right; }

[dir="rtl"] .prob-head { direction:rtl; }
[dir="rtl"] .brand-row { direction:rtl; }

@media (max-width: 900px) {
    .hero h1 { font-size:40px; }
    .block-container { padding-left:.7rem; padding-right:.7rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# CONTENT
# =============================================================================
TEXT = {
    "en": {
        "disclaimer": "For informational and educational support only. Not a medical diagnosis.",
        "brand": "SafeSpace AI",
        "subtitle": "Understanding your unspoken words",
        "creator": "Created by Eng / Habiba Ahmed Talat",
        "about": "About the Model",
        "about_desc": (
            "SafeSpace AI analyzes the language in a user's text and estimates "
            "which of seven trained mental-health categories is most consistent "
            "with the input. The prediction is a screening-style model output, "
            "not a clinical diagnosis."
        ),
        "accuracy": "Model Accuracy: 94.2%",
        "model": "XGBoost + TF-IDF NLP",
        "input_title": "Share Your Thoughts",
        "input_hint": "Write in English or Arabic. Your language does not need to match the interface.",
        "placeholder": "Express your thoughts freely here...",
        "voice": "Voice input",
        "voice_hint": "Speak in English or Arabic, then transcribe it into the text box.",
        "transcribe": "Transcribe voice",
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
        "translation": "Arabic text detected — translating for the English-trained model...",
        "translation_failed": "Arabic translation could not be completed. Please try again.",
        "voice_missing": "Voice dependencies are not installed. Add SpeechRecognition to requirements.txt.",
        "voice_failed": "I couldn't understand the recording. Please try again in a quieter place.",
        "voice_network": "Speech recognition could not reach the recognition service.",
        "empty": "Please write something or transcribe a voice recording first.",
        "timestamp": "Analysis time",
        "privacy": "Text is processed for this session and is not intentionally stored by this application.",
    },
    "ar": {
        "disclaimer": "للاستخدام الإرشادي والتعليمي فقط، وليس تشخيصًا طبيًا.",
        "brand": "SafeSpace AI",
        "subtitle": "فهم ما بين السطور",
        "creator": "تطوير م. حبيبة أحمد طلعت",
        "about": "عن النموذج",
        "about_desc": (
            "يحلل SafeSpace AI اللغة المستخدمة في النص ويقدّر الفئة النفسية "
            "الأكثر توافقًا مع المدخل من بين سبع فئات مدرّب عليها النموذج. "
            "النتيجة تقدير لنموذج فحص وليست تشخيصًا طبيًا."
        ),
        "accuracy": "دقة النموذج: 94.2%",
        "model": "XGBoost + TF-IDF NLP",
        "input_title": "شارك أفكارك",
        "input_hint": "اكتب بالعربي أو بالإنجليزي، ومش لازم لغة الكتابة تطابق لغة الواجهة.",
        "placeholder": "عبّر عن أفكارك ومشاعرك هنا بحرية...",
        "voice": "الإدخال الصوتي",
        "voice_hint": "اتكلم بالعربي أو بالإنجليزي، وبعدها حوّل التسجيل إلى نص.",
        "transcribe": "تحويل الصوت إلى نص",
        "analyze": "تحليل النص",
        "clear": "مسح",
        "detected": "الحالة النفسية المتوقعة",
        "confidence": "درجة الثقة",
        "breakdown": "توزيع احتمالات الفئات",
        "response": "استجابة AI داعمة",
        "guidance": "توجيهات شخصية",
        "selfcare": "خطوات بسيطة للعناية بالنفس",
        "emergency": "دعم فوري مطلوب",
        "emergency_text": "لو في خطر فوري أو أفكار انتحارية، تواصل الآن مع خدمات الطوارئ المحلية أو شخص تثق به.",
        "processing": "جاري تحليل النص...",
        "translation": "تم اكتشاف نص عربي — يتم ترجمته للنموذج المدرب باللغة الإنجليزية...",
        "translation_failed": "تعذرت ترجمة النص العربي. جرّبي مرة أخرى.",
        "voice_missing": "مكونات الإدخال الصوتي غير مثبتة. أضيفي SpeechRecognition إلى requirements.txt.",
        "voice_failed": "لم أستطع فهم التسجيل. جرّبي مرة أخرى في مكان أهدأ.",
        "voice_network": "تعذر الوصول إلى خدمة التعرف على الكلام.",
        "empty": "اكتبي نصًا أو حوّلي تسجيلًا صوتيًا إلى نص أولًا.",
        "timestamp": "وقت التحليل",
        "privacy": "يتم التعامل مع النص خلال الجلسة ولا يقوم التطبيق بحفظه بشكل مقصود.",
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
        "en": ("Your anxiety is valid. Try grounding yourself in the present moment.",
               ["Name 5 things you can see.", "Take slow breaths for one minute.", "Take a short gentle walk."]),
        "ar": ("مشاعرك مهمة. حاولي ترجعي انتباهك للحظة الحالية بدل ما يفضل ذهنك في دائرة القلق.",
               ["سمي 5 حاجات شايفاها حولك.", "خدي نفس ببطء لمدة دقيقة.", "اعملي مشية بسيطة وهادية."]),
    },
    "Stress": {
        "en": ("Stress can be a signal that your mind and body need a pause.",
               ["Write down your top three stressors.", "Take a 15-minute break.", "Choose one small task you can control today."]),
        "ar": ("التوتر ممكن يكون إشارة إن جسمك وذهنك محتاجين وقفة وراحة.",
               ["اكتبي أهم 3 حاجات مسببة للتوتر.", "خدي استراحة 15 دقيقة.", "اختاري خطوة صغيرة واحدة تقدري تتحكمي فيها النهارده."]),
    },
    "Depression": {
        "en": ("What you feel matters, but a model prediction cannot define you.",
               ["Drink some water and get a little daylight.", "Message someone you trust.", "Choose one tiny achievable activity."]),
        "ar": ("إحساسك مهم، لكن نتيجة النموذج لا تحدد شخصيتك أو قيمتك.",
               ["اشربي مياه واقعدي شوية في ضوء النهار.", "كلمي شخص تثقي فيه.", "اختاري نشاط صغير جدًا تقدري تعمليه."]),
    },
    "Normal": {
        "en": ("Your input is most consistent with the Normal category in this model.",
               ["Keep your routines that support wellbeing.", "Notice what is working for you.", "Stay connected with people you trust."]),
        "ar": ("النص أقرب لفئة الطبيعي حسب النموذج الحالي.",
               ["حافظي على العادات اللي بتدعم راحتك.", "خدي بالك من الحاجات اللي بتساعدك.", "خلي التواصل مع الناس الموثوقين جزءًا من حياتك."]),
    },
    "Bipolar": {
        "en": ("This is a model category, not a diagnosis. A clinician is needed for proper assessment.",
               ["Track noticeable mood or sleep changes.", "Avoid making major decisions during intense mood changes.", "Consider professional assessment if concerns persist."]),
        "ar": ("دي فئة في النموذج وليست تشخيصًا. التقييم الصحيح يحتاج متخصصًا.",
               ["سجلي التغيرات الواضحة في المزاج أو النوم.", "تجنبي القرارات الكبيرة أثناء التغيرات الشديدة في المزاج.", "فكري في تقييم متخصص لو القلق مستمر."]),
    },
    "Personality disorder": {
        "en": ("This model label is not a clinical diagnosis and should not be used to label yourself.",
               ["Treat the result as a signal for reflection only.", "Notice recurring patterns without judging yourself.", "Seek professional assessment for persistent concerns."]),
        "ar": ("التصنيف ده من النموذج مش تشخيص طبي، ومينفعش نستخدمه للحكم على نفسك.",
               ["اعتبري النتيجة مجرد إشارة للتفكير.", "لاحظي الأنماط المتكررة بدون لوم لنفسك.", "استشيري متخصصًا لو في مشكلة مستمرة."]),
    },
    "Suicidal": {
        "en": ("This category requires extra caution. The model cannot determine your safety. If you may act on suicidal thoughts, seek immediate human help.",
               ["Contact a trusted person now.", "Move away from anything you could use to hurt yourself.", "Contact local emergency/crisis services immediately if danger is imminent."]),
        "ar": ("الفئة دي تحتاج حذر شديد. النموذج لا يستطيع تحديد مدى أمانك. لو ممكن تتصرفي بناءً على أفكار انتحارية، اطلبي مساعدة بشرية فورًا.",
               ["كلمي شخص تثقي فيه الآن.", "ابعدي عن أي شيء ممكن تستخدميه لإيذاء نفسك.", "تواصلي فورًا مع الطوارئ أو خدمات الأزمات المحلية لو في خطر قريب."]),
    },
}


# =============================================================================
# SESSION STATE / LANGUAGE
# =============================================================================
if "language" not in st.session_state:
    st.session_state.language = "en"
if "text_input" not in st.session_state:
    st.session_state.text_input = ""
if "results" not in st.session_state:
    st.session_state.results = None
if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""

lang = st.session_state.language
T = TEXT[lang]

# =============================================================================
# HEADER
# =============================================================================
st.markdown(
    f'<div class="topbar">{html.escape(T["disclaimer"])}</div>',
    unsafe_allow_html=True,
)

c1, c2 = st.columns([5, 1])
with c1:
    st.markdown(
        '<div class="brand-row"><div class="brand"><div class="brand-mark">💙</div> SafeSpace AI</div></div>',
        unsafe_allow_html=True,
    )
with c2:
    current_is_ar = lang == "ar"
    selected = st.toggle("AR", value=current_is_ar, key="language_toggle")
    new_lang = "ar" if selected else "en"
    if new_lang != lang:
        st.session_state.language = new_lang
        st.rerun()

direction = "rtl" if lang == "ar" else "ltr"

st.markdown(
    f"""
<div dir="{direction}" class="hero">
  <h1>SafeSpace AI</h1>
  <div class="hero-sub">{html.escape(T["subtitle"])}</div>
  <div class="creator">{html.escape(T["creator"])}</div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# ABOUT
# =============================================================================
st.markdown(
    f"""
<div dir="{direction}" class="card about-card">
  <div class="about-title">{html.escape(T["about"])}</div>
  <div class="about-text">{html.escape(T["about_desc"])}</div>
  <br>
  <span class="metric-pill">{html.escape(T["accuracy"])}</span>
  <span style="margin-left:8px;color:#52697A;font-size:12px;">{html.escape(T["model"])}</span>
</div>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# MODEL
# =============================================================================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


def is_arabic(text: str) -> bool:
    letters = re.findall(r"[\u0600-\u06FF]", text or "")
    latin = re.findall(r"[A-Za-z]", text or "")
    return len(letters) > 0 and len(letters) >= max(2, len(latin) * 0.15)


def translate_to_english(text: str):
    if not is_arabic(text):
        return text, False
    if GoogleTranslator is None:
        return None, True
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(text)
        return translated, True
    except Exception:
        return None, True


def speech_to_text(audio_bytes: bytes):
    if sr is None:
        return None, None, "missing"
    try:
        recognizer = sr.Recognizer()
        # st.audio_input returns WAV/PCM data that SpeechRecognition can read.
        import io
        with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
            audio = recognizer.record(source)

        candidates = []
        for locale in ("en-US", "ar-EG"):
            try:
                result = recognizer.recognize_google(
                    audio, language=locale, show_all=True
                )
                if isinstance(result, dict):
                    alts = result.get("alternative", [])
                    if alts:
                        best = max(
                            alts,
                            key=lambda x: float(x.get("confidence", 0.0))
                        )
                        candidates.append(
                            (float(best.get("confidence", 0.0)), best.get("transcript", ""), locale)
                        )
                elif isinstance(result, str):
                    candidates.append((0.0, result, locale))
            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                return None, None, "network"

        if not candidates:
            return None, None, "unknown"

        confidence, transcript, locale = max(candidates, key=lambda x: x[0])
        return transcript, locale, None
    except Exception:
        return None, None, "unknown"


# =============================================================================
# INPUT
# =============================================================================
left, right = st.columns([1.9, 1.0], gap="large")

with left:
    st.markdown(
        f'<div dir="{direction}" class="section-title">{html.escape(T["input_title"])}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div dir="{direction}" class="section-sub">{html.escape(T["input_hint"])}</div>',
        unsafe_allow_html=True,
    )

    text_input = st.text_area(
        "Text",
        value=st.session_state.text_input,
        height=165,
        placeholder=T["placeholder"],
        label_visibility="collapsed",
        key="text_area_widget",
    )
    st.session_state.text_input = text_input

    st.markdown(
        f"""
<div dir="{direction}" class="voice-box">
  <div class="voice-caption"><b>{html.escape(T["voice"])}</b> — {html.escape(T["voice_hint"])}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    audio = st.audio_input(T["voice"])

    b1, b2 = st.columns([1.4, 1])
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
            transcript, locale, err = speech_to_text(audio.getvalue())
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
        st.info(st.session_state.voice_transcript)

    analyze = st.button(T["analyze"], use_container_width=True, type="primary")

with right:
    # Empty-state card keeps the layout close to the reference image.
    st.markdown(
        f"""
<div dir="{direction}" class="ai-card">
  <div class="ai-title">{html.escape(T["response"])}</div>
  <div class="ai-copy">
    {"Write a thought or use your microphone to begin." if lang == "en" else "اكتبي فكرة أو استخدمي الميكروفون للبدء."}
  </div>
</div>
""",
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
        with st.spinner(T["processing"]):
            english_text, translated = translate_to_english(raw_text)

            if translated:
                if english_text is None:
                    st.error(T["translation_failed"])
                    st.stop()
                st.info(T["translation"])

            try:
                pipeline = load_model()
                vectorizer = pipeline["vectorizer"]
                model = pipeline["model"]
                label_encoder = pipeline["label_encoder"]

                features = vectorizer.transform([english_text])
                pred_encoded = int(model.predict(features)[0])
                probabilities = model.predict_proba(features)[0]

                # Use the actual LabelEncoder from the saved artifact.
                labels = list(label_encoder.classes_)
                predicted_label = str(label_encoder.inverse_transform([pred_encoded])[0])

                prob_dict = {
                    str(labels[i]): float(probabilities[i])
                    for i in range(len(labels))
                }

                st.session_state.results = {
                    "prediction": predicted_label,
                    "confidence": float(prob_dict[predicted_label]),
                    "probabilities": prob_dict,
                    "translated": translated,
                }
            except Exception as exc:
                st.error(f"Analysis error: {exc}")
                st.stop()

# =============================================================================
# RESULTS
# =============================================================================
results = st.session_state.results

if results:
    pred = results["prediction"]
    display_pred = CLASS_AR.get(pred, pred) if lang == "ar" else pred
    confidence = results["confidence"]

    # Safety: do not claim certainty. Use the highest model probability only.
    suicidal_prob = results["probabilities"].get("Suicidal", 0.0)

    r1, r2 = st.columns([1.25, 1.0], gap="large")

    with r1:
        st.markdown(
            f"""
<div dir="{direction}" class="result-card">
  <div class="result-label">{html.escape(T["detected"])}</div>
  <div class="result-value">{html.escape(display_pred)}</div>
  <div class="result-confidence">{html.escape(T["confidence"])}: {confidence:.1%}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div dir="{direction}" class="section-title">{html.escape(T["breakdown"])}</div>',
            unsafe_allow_html=True,
        )

        for label, probability in sorted(
            results["probabilities"].items(), key=lambda x: x[1], reverse=True
        ):
            shown = CLASS_AR.get(label, label) if lang == "ar" else label
            danger = " danger" if label == "Suicidal" else ""
            st.markdown(
                f"""
<div dir="{direction}" class="prob-row">
  <div class="prob-head">
    <span>{html.escape(shown)}</span>
    <span>{probability:.0%}</span>
  </div>
  <div class="prob-track">
    <div class="prob-fill{danger}" style="width:{max(0,min(100,probability*100)):.2f}%"></div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    with r2:
        g = GUIDANCE.get(pred, GUIDANCE["Normal"])
        consolation, steps = g[lang]

        st.markdown(
            f"""
<div dir="{direction}" class="ai-card">
  <div class="ai-title">💙 {html.escape(T["response"])}</div>
  <div class="ai-copy">{html.escape(consolation)}</div>
  <br>
  <div class="ai-title">{html.escape(T["guidance"])}</div>
  <div class="ai-copy">{html.escape(g[lang][0])}</div>
  <br>
  <div class="ai-title">🛡️ {html.escape(T["selfcare"])}</div>
  <div class="ai-copy">
    {"<br>".join("• " + html.escape(x) for x in steps)}
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        if suicidal_prob >= 0.25:
            st.markdown(
                f"""
<div dir="{direction}" class="emergency">
  <div class="emergency-title">🚨 {html.escape(T["emergency"])}</div>
  <div class="emergency-text">{html.escape(T["emergency_text"])}</div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.caption(
        f'{T["timestamp"]}: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    )

# =============================================================================
# FOOTER
# =============================================================================
st.markdown(
    f"""
<div dir="{direction}" class="footer">
  <b>SafeSpace AI</b> · {html.escape(T["disclaimer"])}<br>
  {html.escape(T["privacy"])}
</div>
""",
    unsafe_allow_html=True,
)
