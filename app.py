import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
from peft import PeftModel
import os
import gdown
import zipfile

# ===================== 基本页面配置 =====================
st.set_page_config(
    page_title="6405 Group 16 Project",
    page_icon="🤖",
    layout="wide"
)

# ===================== 全局样式美化（CSS） =====================
st.markdown("""
<style>
/* 整体背景 & 字体 */
.stApp {
    background: radial-gradient(circle at top left, #f5f7ff 0, #ffffff 40%, #fdf2ff 100%);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* 让主区域稍微窄一点，更像 dashboard */
.block-container {
    padding-top: 4.2rem;
    padding-bottom: 2rem;
}

/* 主标题 & 副标题 */
.main-title {
    font-size: 2.1rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.subtitle {
    font-size: 1rem;
    color: #555;
}

/* 通用卡片 */
.nice-card {
    padding: 1.1rem 1.3rem;
    border-radius: 0.9rem;
    border: 1px solid #e5e5ef;
    background-color: rgba(255,255,255,0.96);
    box-shadow: 0 18px 40px rgba(15, 23, 42, 0.10);
    margin-bottom: 1.2rem;
}

/* 结果 badge */
.result-badge {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    background-color: #e5f4ff;
    color: #0050b3;
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 0.35rem;
}

/* 置信度文本 */
.confidence {
    font-size: 0.9rem;
    color: #666;
    margin-top: 0.25rem;
}

/* sidebar 说明文字样式 */
.sidebar-title {
    font-weight: 600;
    font-size: 0.95rem;
    margin-top: 0.4rem;
}
.sidebar-hint {
    font-size: 0.85rem;
    color: #888;
}

/* 小号分割线 */
.soft-divider {
    margin: 0.8rem 0 0.6rem 0;
    border-top: 1px dashed #ddd;
}
</style>
""", unsafe_allow_html=True)

# ===================== 顶部封面区域 =====================
header_col_logo, header_col_text = st.columns([1, 5])

with header_col_logo:
    st.image("assets/pingu1.jpg", width=135)

with header_col_text:
    st.markdown(
        '<div class="main-title">🤖 6405 Group 16 · Online Prediction Platform</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">BERT & RoBERTa · Sentiment Analysis · News Topic Classification · Natural Language Inference</div>',
        unsafe_allow_html=True
    )

# 使用卡片包一层“使用说明”
st.markdown("""
<div class="nice-card">
    <b>How to use:</b>
    <ol style="margin-top: 0.4rem; padding-left: 1.1rem;">
        <li>Use the <b>left sidebar</b> to choose a <b>task</b> and <b>model</b>.</li>
        <li>Enter your text (or <i>Premise + Hypothesis</i> for NLI).</li>
        <li>Click <b>Start Prediction</b> to see the result and training metrics.</li>
    </ol>
</div>
""", unsafe_allow_html=True)

# ===================== 模型路径与配置 =====================
MODEL_PATHS = {
    "BERT_SentimentAnalysis": {"path": "model/bert_base_sentiment", "num_labels": 2, "adapter": True},
    "ROBERTA_SentimentAnalysis": {"path": "model/roberta_base_sentiment", "num_labels": 2, "adapter": True},
    "BERT_News": {
        "path": "model/bert_news",
        "num_labels": 4,
        "adapter": False,
        "gdrive_url": "https://drive.google.com/uc?id=1RgFH1aDaNaQkVC9MKZPq511NPpomNXNH"
    },
    "ROBERTA_News": {
        "path": "model/roberta_news",
        "num_labels": 4,
        "adapter": False,
        "gdrive_url": "https://drive.google.com/uc?id=1k65ZZY4M0GdArxztFeKslOBF3k69e54_"
    },
    "BERT_NLI": {
        "path": "model/bert_nli",
        "num_labels": 3,
        "adapter": False,
        "gdrive_url": "https://drive.google.com/uc?id=1IoMeB3Cqg_D8C-SKGIezuOxD4eBMJexV"
    },
    "ROBERTA_NLI": {
        "path": "model/roberta_nli",
        "num_labels": 3,
        "adapter": False,
        "gdrive_url": "https://drive.google.com/uc?id=1gyP1I1cf2ztaT21L0YbeHeiVuIgzdOsK"
    }
}

# ===================== 模型加载（缓存） =====================
@st.cache_resource
def load_models():
    models = {}
    tokenizers = {}

    for name, cfg in MODEL_PATHS.items():
        path = cfg["path"]
        num_labels = cfg["num_labels"]
        use_adapter = cfg.get("adapter", False)

        # 如果文件夹不存在或为空，自动下载并解压 zip
        if not os.path.exists(path) or len(os.listdir(path)) == 0:
            if "gdrive_url" in cfg:
                os.makedirs(path, exist_ok=True)
                zip_file = os.path.join(path, "model.zip")
                gdown.download(cfg["gdrive_url"], zip_file, quiet=False)
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(path)
                os.remove(zip_file)

        # 选择基础模型
        if "BERT" in name:
            base_model_name = "bert-base-uncased"
        elif "ROBERTA" in name:
            base_model_name = "facebook/roberta-base"

        # 加载基础模型
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=num_labels
        )

        # 特殊：BERT_NLI 的存储方式不一样
        if path == "model/bert_nli":
            specific_path = os.path.join(path, "bert_nli")
            models[name] = AutoModelForSequenceClassification.from_pretrained(specific_path)
            tokenizers[name] = AutoTokenizer.from_pretrained(specific_path)
            continue

        # 如果是 adapter 模型
        if use_adapter:
            model = PeftModel.from_pretrained(base_model, path, is_trainable=False)
        else:
            model = base_model

        model.eval()
        models[name] = model
        tokenizers[name] = AutoTokenizer.from_pretrained(base_model_name)

    return models, tokenizers, MODEL_PATHS

# ===================== 图片加载函数 =====================
def load_confusion_matrix(model_name):
    img_path = f"metrics/confusion_{model_name}.png"
    img = Image.open(img_path)
    return img

def load_overall_performance(model_name):
    img_path = f"metrics/overall_{model_name}.png"
    img = Image.open(img_path)
    return img

def load_categories_performance(model_name):
    img_path = f"metrics/categories_{model_name}.png"
    img = Image.open(img_path)
    return img

# ===================== 加载所有模型（只执行一次） =====================
models, tokenizers, model_names = load_models()

# ===================== Sidebar：任务 & 模型选择 =====================
st.sidebar.markdown("### ⚙️ Task & Model")

task_selected = st.sidebar.radio(
    "Choose a task:",
    ["Sentiment Analysis", "News Topic Categorization", "Natural Language Inference(NLI)"]
)

st.sidebar.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<p class="sidebar-title">🚶 Quick Steps</p>'
    '<ol class="sidebar-hint" style="padding-left:1.1rem;">'
    '<li>Select a task</li>'
    '<li>Select a model</li>'
    '<li>Enter text</li>'
    '<li>Click <b>Start Prediction</b></li>'
    '</ol>',
    unsafe_allow_html=True
)

# ===================== 主体：根据任务显示输入区域 =====================
model_options = ["BERT", "ROBERTA"]

if task_selected == "Sentiment Analysis":
    st.subheader("🧠 Sentiment Analysis")

    model_selected = st.selectbox("Select Model:", model_options)
    user_input = st.text_area(
        "Enter text for sentiment analysis:",
        value="",
        placeholder="Please enter a sentence with emotional connotations.",
        key="sentiment_input"
    )

elif task_selected == "News Topic Categorization":
    st.subheader("📰 News Topic Categorization")

    model_selected = st.selectbox("Select Model:", model_options)
    user_input = st.text_area(
        "Enter text for news topic categorization:",
        value="",
        placeholder="Please enter a sentence belonging to 'World', 'Sports', 'Business', or 'Sci/Tech'.",
        key="news_input"
    )

elif task_selected == "Natural Language Inference(NLI)":
    st.subheader("🔗 Natural Language Inference (NLI)")

    model_selected = st.selectbox("Select Model:", model_options)
    premise = st.text_area(
        "Premise:",
        value="",
        placeholder="Enter the first sentence (Premise)",
        key="premise_input"
    )
    hypothesis = st.text_area(
        "Hypothesis:",
        value="",
        placeholder="Enter the second sentence (Hypothesis)",
        key="hypothesis_input"
    )

# ===================== 统一预测按钮 =====================
submit = st.button("🚀 Start Prediction")

if submit:
    # 先根据任务准备 model_key、tokenizer、model、inputs、result_map
    if task_selected == "Sentiment Analysis":
        if not user_input:
            st.warning("⚠️ Please enter text for Sentiment Analysis.")
            st.stop()
        model_key = "BERT_SentimentAnalysis" if model_selected == "BERT" else "ROBERTA_SentimentAnalysis"
        model_name = model_key
        result_map = {0: "Negative", 1: "Positive"}
        tokenizer = tokenizers[model_key]
        model = models[model_key]
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

    elif task_selected == "News Topic Categorization":
        if not user_input:
            st.warning("⚠️ Please enter text for News Topic Categorization.")
            st.stop()
        model_key = "BERT_News" if model_selected == "BERT" else "ROBERTA_News"
        model_name = model_key
        result_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        tokenizer = tokenizers[model_key]
        model = models[model_key]
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

    elif task_selected == "Natural Language Inference(NLI)":
        try:
            premise
            hypothesis
        except NameError:
            st.warning("⚠️ Please provide both Premise and Hypothesis for NLI.")
            st.stop()

        if not premise or not hypothesis:
            st.warning("⚠️ Please enter both Premise and Hypothesis for NLI.")
            st.stop()

        model_key = "BERT_NLI" if model_selected == "BERT" else "ROBERTA_NLI"
        model_name = model_key
        result_map = {0: "Entailment", 1: "Contradiction", 2: "Neutral"}
        tokenizer = tokenizers[model_key]
        model = models[model_key]
        inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True)

    else:
        st.warning("⚠️ Unknown task selected.")
        st.stop()

    # ===================== 推理 =====================
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).item()
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        confidence = probs[predictions] * 100

    label_text = result_map[predictions]

    # ===================== 预测结果显示（卡片） =====================
    st.markdown("### 🔍 Prediction Result")
    st.markdown(
        f"""
        <div class="nice-card">
            <div class="result-badge">{task_selected}</div>
            <div style="font-size:1.05rem; margin-top:0.15rem;">
                <b>{model_selected}</b> predicts:
                <span style="color:#111; font-weight:700;">{label_text}</span>
            </div>
            <div class="confidence">
                Confidence: <b>{confidence:.1f}%</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ===================== 评估结果（Tabs） =====================
    st.markdown("### 📈 Training & Evaluation")

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Metrics per Epoch", "Classification Report"])

    # Tab 1: 混淆矩阵
    with tab1:
        conf_matrix_img_path = f"metrics/confusion_{model_name}.png"
        if os.path.exists(conf_matrix_img_path):
            st.markdown("**Confusion Matrix**")
            conf_matrix_img = load_confusion_matrix(model_name)
            st.image(conf_matrix_img, width=450)
        else:
            st.info("No confusion matrix image found for this model.")

    # Tab 2: 训练指标
    with tab2:
        overall_img_path = f"metrics/overall_{model_name}.png"
        if os.path.exists(overall_img_path):
            st.markdown("**Training Performance Metrics per Epoch**")
            overall_img = load_overall_performance(model_name)
            st.image(overall_img, use_container_width=True)
        else:
            st.info("No overall performance image found for this model.")

    # Tab 3: 分类报告
    with tab3:
        categories_img_path = f"metrics/categories_{model_name}.png"
        if os.path.exists(categories_img_path):
            st.markdown("**Classification Report**")
            categories_img = load_categories_performance(model_name)
            st.image(categories_img, use_container_width=True)
        else:
            st.info("No classification report image found for this model.")

# ===================== 底部说明 =====================
st.markdown("---")

st.markdown("""
**Model Information**

- **BERT** is trained based on *google-bert/bert-base-uncased*.  
- **RoBERTa** is trained based on *FacebookAI/roberta-base*.  

Deployed using **Streamlit**.  

**Authors: NTU EEE 6405 Group 16**  
- Zeng Jiabo  
- Fu Wanting  
- Hou Xinyu  
- Wang Di  
- Wang Jianyu  
- Xie Debin  

*(Sorted by first letter of surname)*  
""")

col1, col2 = st.columns([1, 5])
with col1:
    st.image("assets/pingu6.jpg", width=150)
with col2:
    st.write("Thank you for using~ 🎉")
