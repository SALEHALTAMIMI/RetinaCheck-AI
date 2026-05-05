import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# --- 1. قاموس اللغات ---
LANGUAGES = {
    "English": {
        "title": "RetinaCheck AI: Early Diagnosis System",
        "sub": "AI-powered system to assist doctors in detecting retinal diseases.",
        "upload": "Please upload a fundus image...",
        "orig": "Original Image",
        "proc": "Medical Filter (Ben Graham)",
        "detecting": "AI analyzing and processing...",
        "safe": "Healthy",
        "danger": "Infected",
        "conf": "Confidence Level",
        "note": "Note: This is a technical analysis tool, the final decision belongs to the specialist.",
        "student": "Student: SALEH SABRI ALTAMIMI",
        "supervisor": "Supervisor: Prof. Dr. Yusuf ÖZCAN",
        "auto_msg": "System detected: ",
        "steps": "Processing Steps"
    },
    "Türkçe": {
        "title": "RetinaCheck AI: Erken Teşhis Sistemi",
        "sub": "Doktorlara retina hastalıklarını tespit etmede yardımcı olan yapay zeka sistemi.",
        "upload": "Lütfen fundus görüntüsünü yükleyin...",
        "orig": "Orijinal Görüntü",
        "proc": "Tıbbi Filtre (Ben Graham)",
        "detecting": "Yapay zeka analiz ediyor ve işliyor...",
        "safe": "Sağlıklı",
        "danger": "Enfekte",
        "conf": "Güven Seviyesi",
        "note": "Not: Bu teknik bir analiz aracıdır, nihai karar uzmana aittir.",
        "student": "Öğrenci: SALEH SABRI ALTAMIMI",
        "supervisor": "Danışman: Prof. Dr. Yusuf ÖZCAN",
        "auto_msg": "Sistem algıladı: ",
        "steps": "İşleme Adımları"
    },
    "العربية": {
        "title": "RetinaCheck AI: نظام التشخيص المبكر",
        "sub": "نظام يعتمد على الذكاء الاصطناعي لمساعدة الأطباء في كشف أمراض الشبكية.",
        "upload": "يرجى رفع صورة قاع العين...",
        "orig": "الصورة الأصلية",
        "proc": "الفلتر الطبي (Ben Graham)",
        "detecting": "جاري التحليل والمعالجة الذكية...",
        "safe": "سليم",
        "danger": "مصاب",
        "conf": "نسبة التأكد",
        "note": "تنبيه: هذه أداة تحليل تقنية، القرار النهائي يعود للطبيب المختص.",
        "student": "الطالب: صالح صبري التميمي",
        "supervisor": "المشرف: Prof. Dr. Yusuf ÖZCAN",
        "auto_msg": "النظام اكتشف: ",
        "steps": "خطوات المعالجة"
    }
}

# --- 2. إعدادات الصفحة والتصميم ---
st.set_page_config(page_title="RetinaCheck AI", page_icon="👁️", layout="wide")

# اختيار اللغة
selected_lang = st.sidebar.selectbox("🌐 Language / Dil / اللغة", ["English", "Türkçe", "العربية"])
L = LANGUAGES[selected_lang]

# تصميم CSS مخصص
st.markdown(f"""
    <style>
    .main {{ background-color: #f8f9fa; }}
    .stAlert {{ border-radius: 10px; }}
    .footer {{ position: fixed; bottom: 0; width: 100%; text-align: center; color: #6c757d; font-size: 14px; padding: 10px; background: white; }}
    {"div[data-testid='stBlock'] {direction: rtl; text-align: right;}" if selected_lang == "العربية" else ""}
    </style>
    """, unsafe_allow_html=True)

# --- 3. المحرك الذكي (Smart Preprocessor) ---
def is_already_processed(img):
    # تحليل إحصائي: صور Ben Graham تكون رمادية ومنحرفة معيارياً بشكل منخفض
    std_dev = np.std(img)
    return std_dev < 35 # عتبة تقديرية للصور الرمادية المعالجة

def smart_preprocess(image_pil, size=512):
    steps = []  # قائمة لتخزين الخطوات
    img = np.array(image_pil)
    steps.append("تحويل الصورة إلى مصفوفة NumPy")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    steps.append("تحويل من RGB إلى BGR")
    
    # فحص تلقائي
    if is_already_processed(img):
        status = "Already Processed / İşlenmiş / معالجة مسبقاً"
        steps.append("فحص: الصورة معالجة مسبقاً، سيتم توحيد الحجم فقط")
        # فقط توحيد الحجم والقص
        img = cv2.resize(img, (size, size))
        steps.append("تغيير حجم الصورة إلى 512x512")
        return img, status, steps
    
    # المعالجة الكاملة للصور الخام
    status = "Raw Image / Ham Görüntü / صورة خام"
    steps.append("فحص: الصورة خام، سيتم المعالجة الكاملة")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    steps.append("تحويل إلى صورة رمادية")
    mask = gray > 10
    if mask.any(): 
        img = img[np.ix_(mask.any(1), mask.any(0))]
        steps.append("قص الصورة لإزالة الحواف السوداء")
    img = cv2.resize(img, (size, size))
    steps.append("تغيير حجم الصورة إلى 512x512")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    steps.append("تحويل إلى فضاء LAB")
    l, a, b = cv2.split(lab)
    steps.append("فصل القنوات L, A, B")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    steps.append("تطبيق CLAHE على قناة L لتحسين التباين")
    img = cv2.merge((l, a, b))
    steps.append("دمج القنوات مرة أخرى")
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    steps.append("تحويل إلى BGR")
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0,0), size/30), -4, 128)
    steps.append("تطبيق فلتر Ben Graham (addWeighted مع Gaussian Blur)")
    mask_circ = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask_circ, (size//2, size//2), int(size/2.1), 1, -1)
    img = cv2.bitwise_and(img, img, mask=mask_circ)
    steps.append("تطبيق قناع دائري لتركيز على المنطقة المركزية")
    
    return img, status, steps

# --- 4. واجهة العرض الرئيسية ---
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("https://cdn.pau.edu.tr/BIYS/logo/PAUlogoTR.png", width=120)
with col_title:
    st.title(L["title"])
    st.write(f"**{L['student']}** | **{L['supervisor']}**")

st.write(L["sub"])

# تحميل الموديل
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# رفع الصورة
uploaded_file = st.file_uploader(L["upload"], type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    
    # المعالجة والذكاء التلقائي
    with st.spinner(L["detecting"]):
        processed_img, detection_status, steps = smart_preprocess(original_image)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"### {L['orig']}")
            st.image(original_image, use_container_width=True)
        with c2:
            st.markdown(f"### {L['proc']}")
            display_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            st.image(display_rgb, use_container_width=True)
            st.caption(f"{L['auto_msg']} {detection_status}")

        # عرض خطوات المعالجة
        st.markdown(f"### {L['steps']}")
        for i, step in enumerate(steps, 1):
            st.write(f"{i}. {step}")

        # التنبؤ
        results = model.predict(processed_img, imgsz=512)
        probs = results[0].probs
        prediction = "Infected" if probs.top1 == 1 else "Healthy"
        confidence = probs.top1conf.item()

        # النتيجة النهائية
        st.divider()
        if prediction == "Infected":
            st.error(f"## {L['danger']} ({confidence:.2%})")
        else:
            st.success(f"## {L['safe']} ({confidence:.2%})")
        
        st.progress(confidence)

st.markdown(f"<div class='footer'>{L['note']}</div>", unsafe_allow_html=True)