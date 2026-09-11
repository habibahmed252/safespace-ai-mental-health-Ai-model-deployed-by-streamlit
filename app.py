```python
"""SafeSpace AI — bilingual mental-health text analysis app."""

import html
import os
import re
import warnings
from datetime import datetime

import joblib
import streamlit as st
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="SafeSpace AI", page_icon="💙", layout="wide")

for key, default in {
    "language": "en",
    "theme": "light",
    "text_input": "",
    "results": None,
    "analysis_history": [],
    "support_response": None,
    "show_ai_chat": False,
    "ai_messages": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


UI = {
    "en": {
        "disclaimer": "For informational and educational support only. Not a medical diagnosis.",
        "subtitle": "Understanding your unspoken words",
        "creator": "Designed & developed by Eng. Habiba Ahmed Talat",
        "about": "About SafeSpace AI",
        "about_desc": "A text-analysis tool that uses a machine-learning model to identify the category that best matches the language in a message. Results are not a clinical diagnosis.",
        "model_section": "About the Model",
        "input": "Share Your Thoughts",
        "input_hint": "Write naturally in English or Arabic. Spelling mistakes, slang, informal wording and punctuation are accepted.",
        "placeholder": "Express your thoughts freely here...",
        "translate": "Translate to English",
        "translate_done": "Translated to English — the model can analyze the translated text.",
        "translate_error": "Translation is temporarily unavailable. Please try again.",
        "analyze": "Analyze Text",
        "clear": "Clear",
        "empty": "Please write something first.",
        "processing": "Analyzing your text...",
        "detected": "Detected Mental State",
        "confidence": "Confidence",
        "breakdown": "Class Probabilities Breakdown",
        "response": "AI Support",
        "response_wait": "Your supportive AI response will appear here after analysis.",
        "guidance": "What may help",
        "selfcare": "Practical next steps",
        "sources": "Evidence-based guidance",
        "emergency": "URGENT SUPPORT",
        "emergency_text": "If you may be in immediate danger or having thoughts of suicide, contact local emergency services or a trusted person now. Do not stay alone.",
        "history": "Session History",
        "history_note": "Previous analyses stay only in this active session and are not intentionally saved to a database or file.",
        "view": "View analysis",
        "clear_history": "Clear history",
        "developer": "Chat with Developer",
        "developer_note": "Have feedback, ideas, or a question about SafeSpace AI?",
        "whatsapp": "Chat with Habiba on WhatsApp",
        "footer": "Built with Streamlit · Designed & developed by Eng. Habiba Ahmed Talat",
        "privacy": "Session-only processing. Your messages are not intentionally stored outside the active session.",
        "ai_unavailable": "The AI support message is unavailable right now. The practical guidance below is still available.",
        "model_ai": "AI support uses an optional generative model. If no API key is configured, SafeSpace AI uses a built-in supportive fallback instead.",
    },
    "ar": {
        "disclaimer": "للاستخدام الإرشادي والتعليمي فقط، وليس تشخيصًا طبيًا.",
        "subtitle": "فهم ما بين السطور",
        "creator": "تصميم وتطوير م. حبيبة أحمد طلعت",
        "about": "عن SafeSpace AI",
        "about_desc": "أداة لتحليل النصوص باستخدام نموذج تعلم آلي لتحديد الفئة الأقرب للغة الموجودة في الرسالة. النتيجة ليست تشخيصًا طبيًا.",
        "model_section": "About the Model",
        "input": "اكتبي اللي جواكي",
        "input_hint": "اكتبي بطريقتك الطبيعية بالعربي أو بالإنجليزي. الأخطاء الإملائية والاختصارات والكتابة العامية وعلامات الترقيم مسموحة.",
        "placeholder": "اكتبي أفكارك ومشاعرك هنا بحرية...",
        "translate": "ترجمة للإنجليزية",
        "translate_done": "تمت الترجمة للإنجليزية — النص جاهز للتحليل بالنموذج.",
        "translate_error": "الترجمة غير متاحة مؤقتًا. جربي مرة تانية.",
        "analyze": "حللي النص",
        "clear": "مسح",
        "empty": "اكتبي نص الأول.",
        "processing": "جاري تحليل النص...",
        "detected": "الحالة المتوقعة",
        "confidence": "درجة الثقة",
        "breakdown": "توزيع احتمالات الفئات",
        "response": "AI Support",
        "response_wait": "رد داعم بالذكاء الاصطناعي هيظهر هنا بعد التحليل.",
        "guidance": "إيه ممكن يساعد",
        "selfcare": "خطوات عملية",
        "sources": "إرشادات مبنية على مصادر موثوقة",
        "emergency": "دعم فوري مطلوب",
        "emergency_text": "لو في خطر فوري أو أفكار انتحارية، كلمي شخص تثقي فيه وخدمات الطوارئ المحلية فورًا، ومتفضليش لوحدك.",
        "history": "سجل الجلسة",
        "history_note": "التحليلات السابقة موجودة داخل الجلسة الحالية فقط ولا يتم حفظها عمدًا في قاعدة بيانات أو ملف.",
        "view": "عرض التحليل",
        "clear_history": "مسح السجل",
        "developer": "تواصلي مع المطوّرة",
        "developer_note": "عندك ملاحظة أو فكرة أو سؤال عن SafeSpace AI؟",
        "whatsapp": "تحدثي مع حبيبة على واتساب",
        "footer": "تم تطويره باستخدام Streamlit · تصميم وتطوير م. حبيبة أحمد طلعت",
        "privacy": "معالجة داخل الجلسة فقط. لا يتم حفظ رسائلك خارج الجلسة الحالية بشكل مقصود.",
        "ai_unavailable": "رد الـAI مش متاح دلوقتي، لكن النصائح العملية موجودة تحت.",
        "model_ai": "رد الـAI بيستخدم موديل توليدي اختياري. لو مفيش API key، التطبيق بيستخدم رد داعم جاهز بدلًا منه.",
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
        "en": (
            "Anxiety can feel overwhelming, but small grounding steps can reduce the intensity.",
            [
                "Slow your breathing for one minute.",
                "Use the 5-4-3-2-1 grounding exercise.",
                "Reduce caffeine if it makes your anxiety worse.",
                "If anxiety keeps interfering with daily life, consider speaking with a mental-health professional.",
            ],
        ),
        "ar": (
            "القلق ممكن يبقى تقيل جدًا، بس خطوات صغيرة للتهدئة ممكن تقلل حدته.",
            [
                "هدي نفسك وخدي نفس ببطء لمدة دقيقة.",
                "جربي تمرين 5-4-3-2-1 للـgrounding.",
                "قللي الكافيين لو بتحسي إنه بيزود القلق.",
                "لو القلق معطل حياتك بشكل مستمر، فكري تكلمي متخصص صحة نفسية.",
            ],
        ),
        "sources": ["NIMH — Psychotherapies", "NIMH — Caring for Your Mental Health"],
    },
    "Stress": {
        "en": (
            "Stress is a signal to pause and identify what is actually under your control.",
            [
                "Write down the top three things stressing you.",
                "Take a short break and return to one small task.",
                "Protect regular sleep, movement and meals.",
                "If stress stays intense or disrupts your life, seek professional support.",
            ],
        ),
        "ar": (
            "التوتر ممكن يكون إشارة إنك محتاجة توقفي وتحددي إيه فعلًا تحت سيطرتك.",
            [
                "اكتبي أهم 3 حاجات مضغوطة بسببها.",
                "خدي بريك صغير وارجعي لمهمة واحدة بس.",
                "حاولي تحافظي على النوم والحركة والأكل المنتظم.",
                "لو التوتر شديد ومستمر أو معطل حياتك، اطلبي دعم متخصص.",
            ],
        ),
        "sources": ["WHO — Doing What Matters in Times of Stress", "NIMH — Caring for Your Mental Health"],
    },
    "Depression": {
        "en": (
            "Feeling low does not mean you have to handle everything alone. Tiny actions and human connection matter.",
            [
                "Pick one very small task for today.",
                "Get some daylight and gentle movement if you can.",
                "Message someone you trust instead of isolating yourself.",
                "If symptoms persist or affect daily life, talk with a qualified professional.",
            ],
        ),
        "ar": (
            "الإحساس التقيل مش معناه إنك لازم تشيلي كل حاجة لوحدك. الخطوات الصغيرة والتواصل مع الناس مهمين.",
            [
                "اختاري مهمة صغيرة جدًا تعمليها النهارده.",
                "اتعرضي شوية لضوء النهار وحاولي تتحركي حركة بسيطة.",
                "ابعتي لشخص تثقي فيه بدل ما تعزلي نفسك.",
                "لو الأعراض مستمرة أو مأثرة على حياتك، كلمي متخصص مؤهل.",
            ],
        ),
        "sources": ["WHO — Depression", "NIMH — My Mental Health: Do I Need Help?"],
    },
    "Bipolar": {
        "en": (
            "This model label is not a diagnosis. Consistent routines and professional care are especially important when mood changes are significant.",
            [
                "Keep sleep, meals and daily activities as regular as possible.",
                "Track noticeable mood and sleep changes.",
                "Take prescribed treatment as directed and keep appointments.",
                "Avoid alcohol or drugs if they worsen mood or sleep.",
            ],
        ),
        "ar": (
            "دي فئة في النموذج وليست تشخيصًا. انتظام الروتين والرعاية المتخصصة مهمين خصوصًا مع التغيرات الكبيرة في المزاج.",
            [
                "حافظي قدر الإمكان على انتظام النوم والأكل والروتين.",
                "سجلي التغيرات الواضحة في المزاج والنوم.",
                "التزمي بالعلاج الموصوف ومواعيد المتابعة.",
                "ابعدي عن الكحول أو المخدرات لو بتزود مشاكل المزاج أو النوم.",
            ],
        ),
        "sources": ["NIMH — Bipolar Disorder"],
    },
    "Personality disorder": {
        "en": (
            "This is a model category, not a diagnosis. Skills that improve emotional regulation and relationships can be useful with professional guidance.",
            [
                "Notice emotional triggers without judging yourself.",
                "Pause before reacting when emotions are intense.",
                "Practice grounding or mindfulness skills.",
                "Consider evidence-based psychotherapy; DBT is one approach used for borderline personality disorder.",
            ],
        ),
        "ar": (
            "دي فئة من النموذج ومش تشخيص. مهارات تنظيم المشاعر والعلاقات ممكن تساعد مع التوجيه المتخصص.",
            [
                "لاحظي الحاجات اللي بتشغّل مشاعرك من غير ما تلومي نفسك.",
                "خدي وقفة قبل رد الفعل وقت الانفعال الشديد.",
                "جربي مهارات الـgrounding أو الـmindfulness.",
                "فكري في العلاج النفسي المبني على الدليل؛ DBT من الأساليب المستخدمة في اضطراب الشخصية الحدّية.",
            ],
        ),
        "sources": ["NIMH — Borderline Personality Disorder"],
    },
    "Normal": {
        "en": (
            "The model did not find strong evidence for one of the other categories in this message. That does not measure your overall mental health.",
            [
                "Keep routines that support your wellbeing.",
                "Stay connected with people you trust.",
                "Notice what helps you feel balanced.",
                "Ask for help whenever your wellbeing changes or you feel overwhelmed.",
            ],
        ),
        "ar": (
            "النموذج ما لاقاش دليل قوي على واحدة من الفئات التانية في الرسالة دي، وده مش مقياس لصحتك النفسية بشكل عام.",
            [
                "حافظي على العادات اللي بتدعم راحتك.",
                "خلي التواصل مع الناس الموثوقين موجود.",
                "خدي بالك من الحاجات اللي بتخليكي متوازنة.",
                "اطلبي مساعدة لو حالتك النفسية اتغيرت أو حسيتي إنك مش قادرة.",
            ],
        ),
        "sources": ["NIMH — Caring for Your Mental Health"],
    },
    "Suicidal": {
        "en": (
            "This result needs caution. A text classifier cannot determine whether you are safe. If you might act on suicidal thoughts, get human help now.",
            [
                "Tell a trusted person exactly what is happening and stay with them.",
                "Move away from medicines, weapons or other means you could use to hurt yourself.",
                "Contact local emergency or crisis services if danger is immediate.",
                "A mental-health professional can help create a safety plan and connect you with treatment.",
            ],
        ),
        "ar": (
            "النتيجة دي محتاجة حذر شديد. نموذج النصوص مش يقدر يحدد إذا كنتِ بأمان. لو ممكن تتصرفي بناءً على أفكار انتحارية، اطلبي مساعدة بشرية دلوقتي.",
            [
                "قولي لشخص تثقي فيه بوضوح إيه اللي بيحصل وخليكي معاه.",
                "ابعدي عن الأدوية أو الأسلحة أو أي وسيلة ممكن تستخدميها لإيذاء نفسك.",
                "لو الخطر قريب، تواصلي فورًا مع الطوارئ أو خدمات الأزمات المحلية.",
                "متخصص الصحة النفسية يقدر يساعد في عمل safety plan وتوصيلك للعلاج المناسب.",
            ],
        ),
        "sources": ["WHO — Suicide", "NIMH — 5 Action Steps to Help Someone Having Thoughts of Suicide"],
    },
}


SOURCE_URLS = {
    "WHO — Suicide": "https://www.who.int/ar/news-room/questions-and-answers/item/suicide",
    "WHO — Doing What Matters in Times of Stress": "https://www.who.int/publications/i/item/9789240003927",
    "WHO — Depression": "https://www.who.int/news-room/fact-sheets/detail/depression",
    "NIMH — Psychotherapies": "https://www.nimh.nih.gov/health/topics/psychotherapies",
    "NIMH — Caring for Your Mental Health": "https://www.nimh.nih.gov/health/topics/caring-for-your-mental-health",
    "NIMH — My Mental Health: Do I Need Help?": "https://www.nimh.nih.gov/health/publications/my-mental-health-do-i-need-help",
    "NIMH — Bipolar Disorder": "https://www.nimh.nih.gov/health/publications/bipolar-disorder",
    "NIMH — Borderline Personality Disorder": "https://www.nimh.nih.gov/health/publications/borderline-personality-disorder",
    "NIMH — 5 Action Steps to Help Someone Having Thoughts of Suicide": "https://www.nimh.nih.gov/health/publications/5-action-steps-to-help-someone-having-thoughts-of-suicide",
}


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("model.pkl")


def contains_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def normalize_user_text(text: str) -> str:
    text = (text or "").replace("\u200f", "").replace("\u200e", "")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def translate_to_english(text: str):
    text = normalize_user_text(text)

    if not text:
        return ""

    if not contains_arabic(text):
        return text

    if GoogleTranslator is None:
        raise RuntimeError("translation_unavailable")

    translated = GoogleTranslator(source="auto", target="en").translate(text)
    return normalize_user_text(translated)


def translate_input():
    current = normalize_user_text(st.session_state.get("text_input", ""))

    if not current:
        return

    try:
        st.session_state.text_input = translate_to_english(current)
        st.session_state.translation_notice = True
    except Exception:
        st.session_state.translation_notice = False
        st.session_state.translation_error = True


def run_prediction(raw_text: str):
    text = normalize_user_text(raw_text)

    if not text:
        raise ValueError("empty")

    model_pack = load_model()
    english_text = translate_to_english(text)

    vectorizer = model_pack["vectorizer"]
    model = model_pack["model"]
    label_encoder = model_pack["label_encoder"]

    tfidf_features = vectorizer.transform([english_text])

    text_length = len(english_text)
    word_count = len(re.findall(r"\b\w+\b", english_text, flags=re.UNICODE))

    extra_features = csr_matrix(
        [[text_length, word_count]],
        dtype=tfidf_features.dtype,
    )

    features = hstack(
        [tfidf_features, extra_features],
        format="csr",
    )

    expected_features = getattr(model, "n_features_in_", None)

    if expected_features is not None and features.shape[1] != expected_features:
        raise ValueError(
            f"Model expects {expected_features} features, "
            f"but the application created {features.shape[1]} features."
        )

    probabilities = model.predict_proba(features)[0]

    encoded_prediction = int(model.predict(features)[0])

    prediction = str(
        label_encoder.inverse_transform([encoded_prediction])[0]
    )

    classes = list(label_encoder.classes_)

    prob_dict = {
        str(classes[i]): float(probabilities[i])
        for i in range(len(classes))
    }

    return {
        "prediction": prediction,
        "confidence": float(prob_dict[prediction]),
        "probabilities": prob_dict,
        "english_text": english_text,
        "translated": english_text != text,
    }


def get_openai_key():
    key = os.getenv("OPENAI_API_KEY")

    if key:
        return key

    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def generate_ai_support(user_text: str, prediction: str, lang: str):
    if prediction == "Suicidal":
        if lang == "ar":
            return (
                "أنا واخدة كلامك بجد. لو في احتمال إنك تأذي نفسك دلوقتي، "
                "متفضليش لوحدك: كلمي شخص تثقي فيه وخليه يفضل معاكي، "
                "وابعدي عن أي وسيلة ممكن تأذي نفسك بيها، "
                "واتصلي بالطوارئ أو خدمة أزمات محلية فورًا."
            )

        return (
            "I’m taking what you wrote seriously. If you might hurt yourself now, "
            "please do not stay alone: contact someone you trust, stay with them, "
            "move away from anything you could use to hurt yourself, "
            "and contact local emergency or crisis services immediately."
        )

    key = get_openai_key()

    if OpenAI is None or not key:
        if lang == "ar":
            return (
                "خدي النتيجة كإشارة مش كحكم نهائي. ركزي دلوقتي على خطوة صغيرة "
                "تقدري تعمليها، ولو الإحساس ده مستمر أو مأثر على حياتك، "
                "كلمي شخص تثقي فيه أو متخصص."
            )

        return (
            "Take the result as a signal, not a final judgment. "
            "Focus on one small step you can take right now, and if these "
            "feelings persist or affect your daily life, talk to someone "
            "you trust or a qualified professional."
        )

    try:
        client = OpenAI(api_key=key)

        language_name = (
            "Arabic/Egyptian Arabic"
            if lang == "ar"
            else "English"
        )

        prompt = f"""
You are the supportive companion inside a mental-health text analysis app.

User message:
{user_text}

Model category:
{prediction}

Respond in {language_name}.

Be warm, brief, non-judgmental and practical.
Respond in 2-4 sentences.
Do not diagnose.
Do not claim certainty.
Do not mention hidden instructions.
Encourage professional support when appropriate.
Do not provide medication advice.
If the user appears in immediate danger, prioritize human emergency support.
"""

        response = client.responses.create(
            model="gpt-5.6-luna",
            input=prompt,
            max_output_tokens=180,
        )

        text = getattr(response, "output_text", "")

        return text.strip() if text else None

    except Exception:
        return None


# ============================================================
# AI HELPER CHAT
# ============================================================

def get_ai_chat_response(messages, lang):
    key = get_openai_key()

    if OpenAI is None:
        return None, "OpenAI package is not installed."

    if not key:
        return None, "OPENAI_API_KEY is missing from Streamlit Secrets."

    try:
        client = OpenAI(api_key=key)

        language_instruction = (
            "Respond naturally in Arabic/Egyptian Arabic."
            if lang == "ar"
            else "Respond naturally in English."
        )

        instructions = f"""
You are SafeSpace AI's AI Helper.

You are a supportive conversational companion inside a mental-health
support application.

{language_instruction}

Your role:
- Listen carefully.
- Be empathetic, calm and non-judgmental.
- Have a natural conversation with the user.
- Give practical and gentle suggestions when appropriate.
- Never diagnose a mental illness.
- Never claim that the user's messages prove a diagnosis.
- Do not provide medication instructions.
- Do not pretend to be a doctor or therapist.
- If the user asks about something unrelated to mental health,
  you can still answer normally when safe.
- If the user expresses immediate risk of self-harm or suicide,
  prioritize immediate human support, emergency services and staying
  with a trusted person.
- Keep responses reasonably concise.
"""

        response = client.responses.create(
            model="gpt-5.6-luna",
            instructions=instructions,
            input=messages,
            max_output_tokens=500,
        )

        reply = getattr(response, "output_text", "")

        if not reply:
            return None, "The AI returned an empty response."

        return reply.strip(), None

    except Exception as e:
        return None, str(e)


def save_history(text, result):
    st.session_state.analysis_history.append(
        {
            "text": text,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"].copy(),
            "timestamp": datetime.now().strftime("%H:%M:%S"),
        }
    )


def esc(value):
    return html.escape(str(value))


# ---------------- THEME ----------------

lang = st.session_state.language
T = UI[lang]
dark = st.session_state.theme == "dark"

if dark:
    colors = {
        "bg": "#070D16",
        "surface": "#0D1726",
        "surface2": "#111F32",
        "input": "#0A1422",
        "text": "#F4F8FC",
        "muted": "#9EADBF",
        "line": "#253A52",
        "navy": "#FFFFFF",
        "accent_bg": "#10263A",
    }
else:
    colors = {
        "bg": "#F4FBFD",
        "surface": "#FFFFFF",
        "surface2": "#F7FCFE",
        "input": "#FFFFFF",
        "text": "#132238",
        "muted": "#637386",
        "line": "#D9EAF0",
        "navy": "#071A35",
        "accent_bg": "#EFF9FC",
    }


st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700;800&display=swap');

:root{{
--bg:{colors['bg']};
--surface:{colors['surface']};
--surface2:{colors['surface2']};
--input:{colors['input']};
--text:{colors['text']};
--muted:{colors['muted']};
--line:{colors['line']};
--navy:{colors['navy']};
--accent-bg:{colors['accent_bg']};
--blue:#38B8D0;
--blue2:#73DCE6;
}}

.stApp{{background:var(--bg);color:var(--text);}}

[data-testid="stHeader"]{{background:transparent;}}

.block-container{{
max-width:1450px;
padding-top:.65rem;
padding-bottom:2rem;
}}

*{{font-family:'Inter',sans-serif;}}

h1,h2,h3,h4{{
font-family:'Playfair Display',serif !important;
color:var(--navy) !important;
}}

.topbar{{
height:38px;
background:#12365F;
color:#fff;
border-radius:0 0 12px 12px;
display:flex;
align-items:center;
justify-content:center;
font-size:10.5px;
margin:-1rem -1rem 1rem;
}}

.brand{{
display:flex;
align-items:center;
gap:10px;
font-weight:800;
color:var(--navy);
font-size:16px;
}}

.brand-mark{{
width:36px;
height:36px;
border-radius:11px;
background:linear-gradient(135deg,#DDF6FB,#BFE9F1);
display:flex;
align-items:center;
justify-content:center;
}}

.hero{{
text-align:center;
margin:.1rem 0 1.2rem;
}}

.hero h1{{
font-size:56px;
line-height:1;
margin:.1rem 0 .35rem;
letter-spacing:-1.5px;
}}

.hero-sub{{
font-size:17px;
color:var(--text);
opacity:.85;
}}

.creator-name{{
font-family:'Playfair Display',serif;
font-size:18px;
font-weight:800;
color:#38B8D0;
margin-top:8px;
}}

.card,.ai-card{{
background:var(--surface);
border:1px solid var(--line);
border-radius:18px;
box-shadow:0 10px 30px rgba(20,80,105,.10);
padding:21px;
margin-bottom:16px;
}}

.section-title{{
font-size:19px;
font-weight:800;
margin:10px 0 7px;
color:var(--navy);
}}

.section-sub{{
color:var(--muted);
font-size:12px;
line-height:1.55;
margin-bottom:10px;
}}

.about-title{{
font-size:18px;
font-weight:800;
color:var(--navy);
margin-bottom:6px;
}}

.about-text{{
color:var(--text);
opacity:.88;
font-size:13px;
line-height:1.6;
}}

.model-grid{{
display:grid;
grid-template-columns:repeat(4,1fr);
gap:10px;
}}

.model-item{{
background:var(--surface2);
border:1px solid var(--line);
border-radius:13px;
padding:13px;
}}

.model-label{{
font-size:10px;
color:var(--muted);
font-weight:700;
text-transform:uppercase;
letter-spacing:.5px;
}}

.model-value{{
margin-top:5px;
font-size:13px;
font-weight:800;
color:var(--navy);
}}

div[data-testid="stTextArea"] textarea{{
background:var(--input) !important;
color:var(--text) !important;
border:1px solid #5FC5D5 !important;
border-radius:13px !important;
min-height:165px;
}}

div[data-testid="stTextArea"] textarea::placeholder{{
color:var(--muted) !important;
}}

.stButton>button,.stLinkButton>a{{
border-radius:11px !important;
min-height:45px !important;
font-weight:800 !important;
}}

.stButton>button[kind="primary"]{{
background:#0B315C !important;
color:#fff !important;
border:0 !important;
}}

.stButton>button:not([kind="primary"]),.stLinkButton>a{{
background:var(--surface) !important;
color:var(--text) !important;
border-color:var(--line) !important;
}}

.control-button button{{
background:#0B315C !important;
color:#fff !important;
border:0 !important;
box-shadow:0 5px 16px rgba(11,49,92,.22) !important;
}}

.result-card{{
background:linear-gradient(145deg,#EFFBFD,#E7F7FB);
border:1px solid #D4EAF0;
border-radius:15px;
padding:18px;
text-align:center;
}}

.result-label{{
font-size:12px;
color:#617486;
}}

.result-value{{
font-size:30px;
font-weight:800;
color:#071A35;
margin-top:4px;
}}

.result-confidence{{
font-size:13px;
color:#27687A;
font-weight:800;
margin-top:5px;
}}

.ai-title{{
font-size:16px;
font-weight:800;
margin-bottom:7px;
color:var(--navy);
}}

.ai-copy{{
color:var(--text);
opacity:.9;
line-height:1.65;
font-size:13px;
}}

.chat-icon{{
width:42px;
height:42px;
border-radius:50%;
background:linear-gradient(135deg,#2AA8C2,#0B315C);
display:flex;
align-items:center;
justify-content:center;
color:#fff;
font-size:21px;
margin-bottom:9px;
box-shadow:0 7px 18px rgba(20,100,130,.2);
}}

.ai-helper-card{{
background:linear-gradient(145deg,var(--surface),var(--surface2));
border:1px solid #67C9D7;
border-radius:18px;
padding:18px;
margin-bottom:16px;
box-shadow:0 10px 30px rgba(20,80,105,.10);
}}

.ai-helper-header{{
display:flex;
align-items:center;
gap:12px;
margin-bottom:8px;
}}

.ai-helper-icon{{
width:46px;
height:46px;
border-radius:50%;
background:linear-gradient(135deg,#2AA8C2,#0B315C);
display:flex;
align-items:center;
justify-content:center;
font-size:23px;
}}

.ai-helper-name{{
font-size:16px;
font-weight:800;
color:var(--navy);
}}

.ai-helper-sub{{
font-size:11px;
color:var(--muted);
margin-top:2px;
}}

.chat-box{{
background:var(--surface2);
border:1px solid var(--line);
border-radius:15px;
padding:12px;
margin:10px 0;
}}

.prob-row{{
margin:8px 0 13px;
}}

.prob-head{{
display:flex;
justify-content:space-between;
gap:10px;
font-size:13px;
color:var(--text);
margin-bottom:5px;
}}

.prob-track{{
height:9px;
background:#DDECF1;
border-radius:20px;
overflow:hidden;
}}

.prob-fill{{
height:100%;
border-radius:20px;
background:linear-gradient(90deg,#2D91AF,#62CFD6);
}}

.prob-fill.danger{{
background:linear-gradient(90deg,#D44848,#F08C8C);
}}

.guidance-card{{
background:var(--surface2);
border:1px solid var(--line);
border-radius:15px;
padding:15px;
margin-top:12px;
}}

.guidance-card ul{{
margin:7px 0 0 18px;
padding:0;
}}

.guidance-card li{{
margin:7px 0;
color:var(--text);
font-size:13px;
line-height:1.45;
}}

.emergency{{
background:#3A151A;
border:1px solid #B95B64;
border-radius:14px;
padding:14px;
color:#FFDDE0;
margin-top:12px;
}}

.emergency .ai-title{{
color:#FFDDE0 !important;
}}

.history-item{{
background:var(--surface);
border:1px solid var(--line);
border-radius:15px;
padding:15px 17px;
margin:9px 0;
}}

.history-head{{
display:flex;
justify-content:space-between;
gap:10px;
align-items:center;
}}

.history-class{{
font-weight:800;
color:var(--navy);
}}

.history-time{{
font-size:11px;
color:var(--muted);
}}

.dev-card{{
background:linear-gradient(135deg,#0D3158,#17577A);
color:#fff;
border-radius:18px;
padding:18px 20px;
margin-top:16px;
box-shadow:0 12px 30px rgba(13,49,88,.2);
}}

.dev-card *{{
color:#fff !important;
}}

.footer{{
text-align:center;
color:var(--muted);
font-size:11px;
padding:18px 0 4px;
}}

details[data-testid="stExpander"]{{
background:var(--surface);
border:1px solid var(--line);
border-radius:15px;
}}

@media(max-width:900px){{
.hero h1{{font-size:42px}}
.model-grid{{grid-template-columns:1fr 1fr}}
}}

@media(max-width:520px){{
.model-grid{{grid-template-columns:1fr}}
}}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------- HEADER ----------------

st.markdown(
    f'<div class="topbar">{esc(T["disclaimer"])}</div>',
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([5.2, 1.2, 1.2])

with c1:
    st.markdown(
        '<div class="brand"><div class="brand-mark">💙</div> SafeSpace AI</div>',
        unsafe_allow_html=True,
    )

with c2:
    st.markdown('<div class="control-button">', unsafe_allow_html=True)

    if st.button(
        "🌐  AR" if lang == "en" else "🌐  EN",
        use_container_width=True,
        key="language_btn",
    ):
        st.session_state.language = "ar" if lang == "en" else "en"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="control-button">', unsafe_allow_html=True)

    if st.button(
        "🌙  Dark" if not dark else "☀️  Light",
        use_container_width=True,
        key="theme_btn",
    ):
        st.session_state.theme = "dark" if not dark else "light"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


direction = "rtl" if lang == "ar" else "ltr"

st.markdown(
    f'<div dir="{direction}" class="hero">'
    f'<h1>SafeSpace AI</h1>'
    f'<div class="hero-sub">{esc(T["subtitle"])}</div>'
    f'<div class="creator-name">{esc(T["creator"])}</div>'
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f'<div dir="{direction}" class="card">'
    f'<div class="about-title">{esc(T["about"])}</div>'
    f'<div class="about-text">{esc(T["about_desc"])}</div>'
    f"</div>",
    unsafe_allow_html=True,
)


# ---------------- MODEL INFO ----------------

with st.expander(T["model_section"], expanded=False):
    pack = load_model()
    model = pack["model"]
    vectorizer = pack["vectorizer"]
    encoder = pack["label_encoder"]
    params = model.get_params()
    classes = list(encoder.classes_)

    tfidf_count = len(vectorizer.vocabulary_)
    expected = getattr(model, "n_features_in_", "N/A")

    extra = (
        max(0, expected - tfidf_count)
        if isinstance(expected, int)
        else 2
    )

    model_text = """
    <div class="model-grid">
      <div class="model-item"><div class="model-label">Model</div><div class="model-value">XGBoost Classifier</div></div>
      <div class="model-item"><div class="model-label">Feature extractor</div><div class="model-value">TF-IDF / {vec}</div></div>
      <div class="model-item"><div class="model-label">TF-IDF features</div><div class="model-value">{tfidf:,}</div></div>
      <div class="model-item"><div class="model-label">Total input features</div><div class="model-value">{expected}</div></div>
      <div class="model-item"><div class="model-label">Prediction classes</div><div class="model-value">{n} categories</div></div>
      <div class="model-item"><div class="model-label">Training language</div><div class="model-value">English</div></div>
      <div class="model-item"><div class="model-label">N-gram range</div><div class="model-value">{ngram}</div></div>
      <div class="model-item"><div class="model-label">Max features</div><div class="model-value">{maxf}</div></div>
    </div>

    <div style="margin-top:14px;font-weight:800;color:var(--navy)">
        Prediction categories
    </div>

    <div style="margin-top:8px;color:var(--text);line-height:1.8;font-size:13px">
        {cats}
    </div>

    <div style="margin-top:14px;font-weight:800;color:var(--navy)">
        Model input construction
    </div>

    <div style="margin-top:7px;color:var(--text);font-size:13px;line-height:1.6">
        {tfidf:,} TF-IDF features + character count + word count = {expected} input features.
    </div>

    <div style="margin-top:14px;font-weight:800;color:var(--navy)">
        Key XGBoost settings
    </div>

    <div style="margin-top:7px;color:var(--text);font-size:12px;line-height:1.7">
        n_estimators: {est} · max_depth: {depth} · learning_rate: {lr} ·
        subsample: {sub} · colsample_bytree: {col}
    </div>

    <div style="margin-top:14px;color:var(--muted);font-size:12px;line-height:1.6">
        {note}
    </div>
    """

    model_text = model_text.format(
        vec=type(vectorizer).__name__,
        tfidf=tfidf_count,
        expected=expected,
        n=len(classes),
        ngram=esc(vectorizer.ngram_range),
        maxf=esc(vectorizer.max_features),
        cats=esc(" · ".join(classes)),
        est=esc(params.get("n_estimators")),
        depth=esc(params.get("max_depth")),
        lr=esc(params.get("learning_rate")),
        sub=esc(params.get("subsample")),
        col=esc(params.get("colsample_bytree")),
        note=esc(T["model_ai"]),
    )

    st.markdown(
        f'<div dir="{direction}" class="card" style="margin:0;padding:16px">'
        f"{model_text}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------- INPUT ----------------

def clear_input():
    st.session_state.text_input = ""
    st.session_state.results = None
    st.session_state.support_response = None


def reset_translation_flags():
    st.session_state.translation_notice = False
    st.session_state.translation_error = False


left, right = st.columns([1.7, 1.0], gap="large")


with left:
    st.markdown(
        f'<div dir="{direction}" class="section-title">{esc(T["input"])}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div dir="{direction}" class="section-sub">{esc(T["input_hint"])}</div>',
        unsafe_allow_html=True,
    )

    tr1, tr2 = st.columns([1.35, 1])

    with tr1:
        st.button(
            "🌐  " + T["translate"],
            use_container_width=True,
            key="translate_btn",
            on_click=translate_input,
        )

    with tr2:
        st.button(
            T["clear"],
            use_container_width=True,
            key="clear_btn",
            on_click=clear_input,
        )

    st.text_area(
        "Text",
        placeholder=T["placeholder"],
        height=180,
        label_visibility="collapsed",
        key="text_input",
    )

    if st.session_state.get("translation_notice"):
        st.success(T["translate_done"])
        st.session_state.translation_notice = False

    if st.session_state.get("translation_error"):
        st.error(T["translate_error"])
        st.session_state.translation_error = False

    analyze = st.button(
        T["analyze"],
        use_container_width=True,
        type="primary",
        key="analyze_btn",
    )


# ============================================================
# AI HELPER CARD
# ============================================================

with right:

    st.markdown(
        f"""
        <div dir="{direction}" class="ai-helper-card">
            <div class="ai-helper-header">
                <div class="ai-helper-icon">🤖</div>
                <div>
                    <div class="ai-helper-name">
                        AI Helper — Here for You
                    </div>
                    <div class="ai-helper-sub">
                        Talk freely with your AI support companion
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "💬  Open AI Chat" if not st.session_state.show_ai_chat else "✕  Close AI Chat",
        use_container_width=True,
        key="ai_helper_btn",
    ):
        st.session_state.show_ai_chat = not st.session_state.show_ai_chat
        st.rerun()


# ============================================================
# AI CHAT
# ============================================================

if st.session_state.show_ai_chat:

    st.markdown(
        f"""
        <div dir="{direction}" class="ai-helper-card">
            <div class="ai-helper-header">
                <div class="ai-helper-icon">🤖</div>
                <div>
                    <div class="ai-helper-name">AI Helper</div>
                    <div class="ai-helper-sub">
                        Your conversation stays in this active session.
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Display previous messages
    for message in st.session_state.ai_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # New user message
    user_message = st.chat_input(
        "Tell me what's on your mind..."
        if lang == "en"
        else "احكيلي إيه اللي جواكي..."
    )

    if user_message:

        user_message = normalize_user_text(user_message)

        st.session_state.ai_messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_message)

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
                if lang == "en"
                else "بفكر في رد مناسب..."
            ):

                ai_reply, ai_error = get_ai_chat_response(
                    st.session_state.ai_messages,
                    lang,
                )

            if ai_reply:

                st.markdown(ai_reply)

                st.session_state.ai_messages.append(
                    {
                        "role": "assistant",
                        "content": ai_reply,
                    }
                )

            else:
                if "OPENAI_API_KEY" in str(ai_error):
                    st.error(
                        "OPENAI_API_KEY is missing. Add it to Streamlit Secrets."
                    )
                else:
                    st.error(
                        "AI chat is temporarily unavailable. "
                        "Please try again."
                    )


# ---------------- ANALYSIS ----------------

if analyze:

    raw_text = normalize_user_text(
        st.session_state.text_input
    )

    if not raw_text:
        st.warning(T["empty"])

    else:

        try:

            with st.spinner(T["processing"]):

                result = run_prediction(raw_text)

            st.session_state.results = result

            st.session_state.support_response = generate_ai_support(
                raw_text,
                result["prediction"],
                lang,
            )

            save_history(
                raw_text,
                result,
            )

        except RuntimeError as exc:

            if str(exc) == "translation_unavailable":
                st.error(T["translate_error"])

            else:
                st.error(
                    "The analysis service is temporarily unavailable."
                )

        except Exception as exc:

            st.error(
                f"Analysis error: {esc(exc)}"
            )


# ---------------- RESULTS ----------------

results = st.session_state.results

if results:

    prediction = results["prediction"]

    shown_prediction = (
        CLASS_AR.get(prediction, prediction)
        if lang == "ar"
        else prediction
    )

    confidence = results["confidence"]

    suicidal_prob = results["probabilities"].get(
        "Suicidal",
        0.0,
    )

    r1, r2 = st.columns(
        [1.25, 1.0],
        gap="large",
    )

    with r1:

        st.markdown(
            f"""
            <div dir="{direction}" class="result-card">
                <div class="result-label">
                    {esc(T["detected"])}
                </div>

                <div class="result-value">
                    {esc(shown_prediction)}
                </div>

                <div class="result-confidence">
                    {esc(T["confidence"])}: {confidence:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f'<div dir="{direction}" class="section-title">'
            f'{esc(T["breakdown"])}'
            f"</div>",
            unsafe_allow_html=True,
        )

        for label, probability in sorted(
            results["probabilities"].items(),
            key=lambda x: x[1],
            reverse=True,
        ):

            shown = (
                CLASS_AR.get(label, label)
                if lang == "ar"
                else label
            )

            danger = " danger" if label == "Suicidal" else ""

            width = max(
                0,
                min(
                    100,
                    probability * 100,
                ),
            )

            st.markdown(
                f"""
                <div dir="{direction}" class="prob-row">
                    <div class="prob-head">
                        <span>{esc(shown)}</span>
                        <span>{probability:.0%}</span>
                    </div>

                    <div class="prob-track">
                        <div class="prob-fill{danger}"
                             style="width:{width:.2f}%">
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with r2:

        support = st.session_state.support_response

        if support:

            st.markdown(
                f"""
                <div dir="{direction}" class="ai-card">
                    <div class="chat-icon">💬</div>

                    <div class="ai-title">
                        {esc(T["response"])}
                    </div>

                    <div class="ai-copy">
                        {esc(support)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.warning(T["ai_unavailable"])

        consolation, steps = GUIDANCE.get(
            prediction,
            GUIDANCE["Normal"],
        )[lang]

        steps_html = "".join(
            f"<li>{esc(step)}</li>"
            for step in steps
        )

        st.markdown(
            f"""
            <div dir="{direction}" class="guidance-card">

                <div class="ai-title">
                    {esc(T["guidance"])}
                </div>

                <div class="ai-copy">
                    {esc(consolation)}
                </div>

                <div class="ai-title" style="margin-top:12px">
                    🛡️ {esc(T["selfcare"])}
                </div>

                <ul>
                    {steps_html}
                </ul>

            </div>
            """,
            unsafe_allow_html=True,
        )

        if suicidal_prob >= 0.25:

            st.markdown(
                f"""
                <div dir="{direction}" class="emergency">

                    <div class="ai-title">
                        🚨 {esc(T["emergency"])}
                    </div>

                    <div style="font-size:12px;line-height:1.6">
                        {esc(T["emergency_text"])}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div style="margin-top:10px;
                        color:var(--muted);
                        font-size:11px;
                        font-weight:700">
                {esc(T["sources"])}
            </div>
            """,
            unsafe_allow_html=True,
        )

        for src in GUIDANCE.get(
            prediction,
            GUIDANCE["Normal"],
        )["sources"]:

            st.markdown(
                f'<a href="{SOURCE_URLS[src]}" target="_blank" '
                f'style="font-size:11px;color:#38B8D0;text-decoration:none">'
                f'↗ {esc(src)}</a>',
                unsafe_allow_html=True,
            )


# ---------------- HISTORY ----------------

if st.session_state.analysis_history:

    st.markdown(
        f'<div dir="{direction}" class="section-title">'
        f'{esc(T["history"])}'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div dir="{direction}" class="section-sub">'
        f'{esc(T["history_note"])}'
        f"</div>",
        unsafe_allow_html=True,
    )

    if st.button(
        T["clear_history"],
        use_container_width=True,
        key="clear_history",
    ):

        st.session_state.analysis_history = []
        st.session_state.results = None
        st.session_state.support_response = None

        st.rerun()

    for item in reversed(
        st.session_state.analysis_history
    ):

        shown = (
            CLASS_AR.get(
                item["prediction"],
                item["prediction"],
            )
            if lang == "ar"
            else item["prediction"]
        )

        st.markdown(
            f"""
            <div dir="{direction}" class="history-item">

                <div class="history-head">

                    <span class="history-class">
                        {esc(shown)}
                    </span>

                    <span class="history-time">
                        {esc(item["timestamp"])}
                    </span>

                </div>

                <div style="margin-top:5px;
                            color:var(--muted);
                            font-size:12px">

                    {esc(T["confidence"])}:
                    <b>{item["confidence"]:.1%}</b>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(
            T["view"],
            expanded=False,
        ):

            st.write(item["text"])

            for label, probability in sorted(
                item["probabilities"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):

                shown_label = (
                    CLASS_AR.get(label, label)
                    if lang == "ar"
                    else label
                )

                st.write(
                    f"**{shown_label}:** {probability:.1%}"
                )


# ---------------- DEVELOPER ----------------

st.markdown(
    f"""
    <div dir="{direction}" class="dev-card">

        <div style="font-size:18px;font-weight:800">
            {esc(T["developer"])}
        </div>

        <div style="font-size:12px;
                    opacity:.88;
                    margin-top:5px">

            {esc(T["developer_note"])}

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)

st.link_button(
    T["whatsapp"],
    "https://wa.me/qr/LG7JYBZDF2WMF1",
    use_container_width=True,
)

st.markdown(
    f"""
    <div dir="{direction}" class="footer">
        {esc(T["footer"])}
        <br>
        {esc(T["privacy"])}
    </div>
    """,
    unsafe_allow_html=True,
)
```

### مهم جدًا قبل الـDeploy

إنتِ **مش محتاجة تضيفي `from openai import OpenAI`** لأن الكود بتاعك أصلًا فيه:

```python
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
```

وكمان الـAPI Key **مش بيتحط في الكود**. طالما موجود عندك في Streamlit Secrets بالشكل ده:

```toml
OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxx"
```

يبقى تمام.

بعد الـDeploy، المفروض تشوفي في الكارت اليمين:

**🤖 AI Helper — Here for You**
**💬 Open AI Chat**

تضغطي **Open AI Chat** → يظهر مربع المحادثة → تكتبي → الـAI يرد → وتكملي معاه في نفس الـchat.
