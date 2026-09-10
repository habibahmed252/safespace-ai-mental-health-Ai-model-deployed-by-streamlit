"""
SafeSpace AI - Streamlit Web Application
Mental Health Text Analysis with Personalized Guidance
Author: Habiba Ahmed Talat
"""

import streamlit as st
import joblib
import numpy as np
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="SafeSpace AI",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for luxury design
st.markdown("""
    <style>
    :root {
        --primary: #0F172A;
        --secondary: #0284C7;
        --light-bg: #FAFAFA;
    }
    
    body {
        background-color: var(--light-bg);
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif;
    }
    
    .main {
        background-color: white;
    }
    
    h1, h2, h3 {
        font-family: 'Playfair Display', 'Syne', serif;
        color: var(--primary);
    }
    
    .stMarkdown {
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONTENT DICTIONARY (MULTILINGUAL)
# ============================================================================

CONTENT = {
    'en': {
        'title': 'SafeSpace AI',
        'subtitle': 'Understanding Your Unspoken Words',
        'creator': 'Created by Eng / Habiba Ahmed Talat',
        'disclaimer': 'This tool is for guidance only • Not a replacement for professional medical diagnosis',
        'about_description': 'SafeSpace AI is an intelligent companion designed to help you decode the emotions hidden within your words. Often, our thoughts carry feelings we might not fully notice—like quiet stress, underlying anxiety, or unexpressed depression. By combining advanced Natural Language Processing (NLP) with compassionate AI, this platform offers a reflective mirror for your mind, empowering you to understand yourself better.',
        'model_accuracy': 'Model Accuracy: 94.2%',
        'model_type': 'XGBoost + TF-IDF NLP',
        'placeholder': 'Express your thoughts freely here...',
        'analyze_btn': 'Analyze Text',
        'detected_state': 'Detected Mental State',
        'confidence': 'Confidence Level',
        'guidance': 'Personalized Guidance',
        'self_care': 'Self-Care Steps',
        'detailed_analysis': 'Detailed Analysis',
        'emergency_title': '⚠️ URGENT SUPPORT NEEDED',
        'emergency_msg': 'If you are experiencing suicidal thoughts, please reach out for help immediately.',
        'us_support': 'US: 988 (National Suicide Prevention Lifeline)',
        'uk_support': 'UK: 116 123 (Samaritans)',
        'crisis_support': 'Crisis Text Line: Text HOME to 741741',
        'classes': ['Anxiety', 'Bipolar', 'Depression', 'Normal', 'Personality disorder', 'Stress', 'Suicidal']
    },
    'ar': {
        'title': 'SafeSpace AI',
        'subtitle': 'فَهْمُ ما بَيْنَ السُّطور',
        'creator': 'من تطوير م.حبيبة أحمد طلعت',
        'disclaimer': 'هذا الأداة للإرشاد فقط • لا تحل محل التشخيص الطبي المتخصص',
        'about_description': 'تطبيق SafeSpace AI هو رفيقك الذكي المصمم لمساعدتك في استكشاف المشاعر المخبأة بين كلماتك. كثيراً ما نحمل في أفكارنا مشاعر قد لا نلاحظها بوضوح—كالضغط الخفي، أو القلق المتراكم. يجمع التطبيق بين تقنيات معالجة اللغة الطبيعية والذكاء الاصطناعي الداعم.',
        'model_accuracy': 'دقة النموذج: 94.2%',
        'model_type': 'معالجة لغة طبيعية متقدمة',
        'placeholder': 'عَبّر عن أفكارك ومشاعرك هنا بحرية...',
        'analyze_btn': 'تحليل النص',
        'detected_state': 'الحالة المكتشفة',
        'confidence': 'مستوى الثقة',
        'guidance': 'التوجيهات الشخصية',
        'self_care': 'خطوات العناية بالذات',
        'detailed_analysis': 'التحليل التفصيلي',
        'emergency_title': '⚠️ تحتاج إلى دعم فوري',
        'emergency_msg': 'إذا كنت تعاني من أفكار انتحارية، يرجى الاتصال فوراً',
        'us_support': 'الولايات المتحدة: 988',
        'uk_support': 'بريطانيا: 116 123',
        'crisis_support': 'أرسل HOME إلى 741741',
        'classes': ['القلق', 'ثنائي القطب', 'الاكتئاب', 'طبيعي', 'اضطراب الشخصية', 'الضغط', 'أفكار انتحارية']
    }
}

# ============================================================================
# MODEL LOADING (CACHED)
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model pipeline."""
    try:
        model = joblib.load('model.pkl')
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None

# ============================================================================
# PERSONALIZED GUIDANCE
# ============================================================================

GUIDANCE_EN = {
    'Anxiety': {
        'consolation': "Your anxiety is valid, and many people experience similar feelings. You are stronger than you think. Right now, your mind is just trying to protect you, even if it feels overwhelming. Let's ground you in this present moment.",
        'technique': "Try the 4-7-8 Breathing Technique: Breathe in for 4 counts, hold for 7, exhale for 8. Repeat 4 times.",
        'steps': [
            "Step 1: Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste",
            "Step 2: Write down three anxious thoughts and challenge one with evidence",
            "Step 3: Take a 10-minute gentle walk while listening to calming music"
        ]
    },
    'Stress': {
        'consolation': "Stress is your mind and body asking for attention. You don't have to handle everything alone. Taking time to rest is not weakness—it's wisdom.",
        'technique': "Try the 'worry dump': spend 10 minutes writing everything that stresses you, then read it aloud.",
        'steps': [
            "Step 1: Identify your top 3 stressors and pick one small action today",
            "Step 2: Take a 15-minute break: step outside, stretch, or relax",
            "Step 3: Evening ritual: write your accomplishments and what you'll release"
        ]
    },
    'Depression': {
        'consolation': "What you're feeling is real, and it doesn't define your worth. This feeling is temporary, even if it doesn't feel that way right now. You matter.",
        'technique': "Start with micro-actions: even 2 minutes of sunlight, one glass of water, or a single text to someone you trust counts.",
        'steps': [
            "Step 1: Do one small action today (open curtains, drink water, text someone)",
            "Step 2: Practice self-compassion: speak to yourself as you would a dear friend",
            "Step 3: Schedule one small activity you enjoy for tomorrow at a specific time"
        ]
    },
    'Normal': {
        'consolation': "Your emotional baseline is healthy and balanced. This is wonderful. Continue nurturing your mental wellbeing with consistency.",
        'technique': "Maintain your wellness routine and practice preventive self-care.",
        'steps': [
            "Step 1: Reflect on what's supporting your wellbeing and protect those practices",
            "Step 2: Practice gratitude: write 3 things you appreciate about today",
            "Step 3: Share your positivity: connect with someone or practice kindness"
        ]
    },
    'Suicidal': {
        'consolation': "Your life has value and meaning. Crisis passes, and with support, perspective shifts. You deserve to live and be helped.",
        'technique': "CALL IMMEDIATELY: Reach out to someone you trust or a crisis service right now.",
        'steps': [
            "Step 1: CALL NOW: 988 (US) / 116 123 (UK) / Text HOME to 741741",
            "Step 2: Tell someone close to you—right now",
            "Step 3: Seek professional mental health support immediately"
        ]
    }
}

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.markdown("## ⚙️ Settings")
language = st.sidebar.radio("Language / اللغة", ['English', 'العربية'], key='language')
lang_code = 'en' if language == 'English' else 'ar'
content = CONTENT[lang_code]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About the Model")
st.sidebar.markdown(f"""
- **Accuracy**: 94.2%
- **Model**: XGBoost Classifier
- **Features**: TF-IDF Vectorizer (20,000)
- **Classes**: 7 mental health categories
- **Training Data**: Multi-source mental health texts
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🆘 Crisis Support")
st.sidebar.markdown(f"""
If you need immediate help:

**{content['us_support']}**  
**{content['uk_support']}**  
**{content['crisis_support']}**
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**SafeSpace AI** © 2024  
Created by Habiba Ahmed Talat  
All rights reserved
""")

# ============================================================================
# MAIN UI
# ============================================================================

# Header
st.markdown(f"# {content['title']}")
st.markdown(f"### {content['subtitle']}")
st.markdown(f"_{content['creator']}_")
st.markdown(f"> **{content['disclaimer']}**")

st.divider()

# About Card
st.markdown("### 💚 About SafeSpace AI")
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(content['about_description'])
with col2:
    st.metric("Model Accuracy", "94.2%")
    st.metric("Model", "XGBoost NLP")

st.divider()

# Input Section
st.markdown("### 📝 Share Your Thoughts")
text_input = st.text_area(
    content['placeholder'],
    height=150,
    placeholder=content['placeholder'],
    key='text_input'
)

# Analyze Button
analyze_btn = st.button(
    content['analyze_btn'],
    use_container_width=True,
    type="primary"
)

# ============================================================================
# ANALYSIS & RESULTS
# ============================================================================

if analyze_btn or 'results' in st.session_state:
    if not text_input.strip():
        st.warning("Please enter some text to analyze")
    else:
        try:
            # Load model
            model_pipeline = load_model()
            
            if model_pipeline is None:
                st.error("Model not available")
            else:
                # Predict
                vectorizer = model_pipeline['vectorizer']
                model = model_pipeline['model']
                label_encoder = model_pipeline['label_encoder']
                
                # Transform text
                features = vectorizer.transform([text_input])
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                
                # Map to class names
                class_names = content['classes']
                prob_dict = {
                    class_names[i]: float(prob)
                    for i, prob in enumerate(probabilities)
                }
                
                # Store results
                st.session_state.results = {
                    'prediction': class_names[prediction],
                    'confidence': float(prob_dict[class_names[prediction]]),
                    'probabilities': prob_dict,
                    'suicidal_prob': float(prob_dict[class_names[6]])
                }
                
                # Display Results
                results = st.session_state.results
                
                # Crisis Alert
                if results['suicidal_prob'] > 0.25:
                    st.markdown("---")
                    st.error(f"""
                    ### {content['emergency_title']}
                    {content['emergency_msg']}
                    
                    - {content['us_support']}
                    - {content['uk_support']}
                    - {content['crisis_support']}
                    """)
                    st.markdown("---")
                
                # Primary Result
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("### 🎯 " + content['detected_state'])
                    st.markdown(f"## {results['prediction']}")
                with col2:
                    st.metric(
                        content['confidence'],
                        f"{results['confidence']:.1%}",
                        delta=f"{results['confidence']:.2%}"
                    )
                
                # Probability Chart
                st.markdown("### 📊 " + content['detailed_analysis'])
                prob_sorted = dict(sorted(results['probabilities'].items(), key=lambda x: x[1], reverse=True))
                st.bar_chart(prob_sorted)
                
                # Personalized Guidance
                if results['prediction'] in GUIDANCE_EN:
                    guidance = GUIDANCE_EN[results['prediction']]
                    
                    st.markdown("### 💙 " + content['guidance'])
                    
                    with st.container(border=True):
                        st.markdown("#### 💭 " + ("Consolation & Reassurance" if lang_code == 'en' else "التطمين والطمأنة"))
                        st.markdown(f"*{guidance['consolation']}*")
                        
                        st.markdown("#### 🌬️ Grounding Technique")
                        st.markdown(guidance['technique'])
                        
                        st.markdown(f"#### 🛡️ {content['self_care']}")
                        for step in guidance['steps']:
                            st.markdown(f"- {step}")
                
                # Data timestamp
                st.caption(f"Analysis performed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
---
**Disclaimer**: This tool is for educational and guidance purposes only. 
It is not a substitute for professional medical diagnosis or treatment. 
If you are experiencing mental health concerns, please consult with a qualified healthcare provider.

**Privacy**: Your text is analyzed locally and is not stored or transmitted anywhere.
""")
