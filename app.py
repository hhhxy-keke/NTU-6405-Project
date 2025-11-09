import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
from peft import PeftModel
import os
import gdown
import zipfile


st.set_page_config(
    page_title="6405 Group 16 Project",
    layout="wide"
)
st.title("🤖 6405 Group 16: Online Prediction Platform for BERT and its Variant Models")
st.write("Please select a model, enter text, and view the prediction results and the model's training performance metrics.")

MODEL_PATHS = {
    "BERT_SentimentAnalysis": {"path": "model/bert_base_sentiment", "num_labels": 2, "adapter": True},
    "BERT_News": {
        "path": "model/bert_news",
        "num_labels": 4,
        "adapter": False,
        "gdrive_url": "https://drive.google.com/uc?id=1RgFH1aDaNaQkVC9MKZPq511NPpomNXNH"
    }
}

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

        # 如果是 adapter 模型
        if use_adapter:
            model = PeftModel.from_pretrained(base_model, path, is_trainable=False)
        else:
            model = base_model

        model.eval()
        models[name] = model
        tokenizers[name] = AutoTokenizer.from_pretrained(base_model_name)

    return models, tokenizers, MODEL_PATHS

# 加载图片（PNG）
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


# 加载
models, tokenizers, model_names = load_models()

# 用户输入与模型选择
st.sidebar.subheader("Task Selection")
task_selected = st.sidebar.radio(
    "Choose a task:",
    ["Sentiment Analysis", "News Topic Categorization"]
)

# 根据任务显示对应的模型选择和输入框
if task_selected == "Sentiment Analysis":
    st.subheader("Sentiment Analysis")

    # 模型选择
    model_options = ["BERT", "ROBERTA"]
    model_selected = st.selectbox("Select Sentiment Model:", model_options)

    # 文本输入
    user_input = st.text_area(
        "Enter text for sentiment analysis:",
        value="",  # 初始为空
        placeholder="Please enter a sentence with emotional connotations.",
        key="sentiment_input"
    )

elif task_selected == "News Topic Categorization":
    st.subheader("News Topic Categorization")

    # 模型选择
    model_options = ["BERT", "ROBERTA"]
    model_selected = st.selectbox("Select News Model:", model_options)

    # 文本输入
    user_input = st.text_area(
        "Enter text for news topic categorization:",
        value="",
        placeholder="Please enter a sentence belonging to 'World', 'Sports', 'Business', or 'Sci/Tech'.",
        key="news_input"
    )

# 统一预测按钮
submit = st.button("Start Prediction")

if submit and user_input:
    # 根据选择加载模型
    if task_selected == "Sentiment Analysis":
        model_key = "BERT_SentimentAnalysis" if model_selected == "BERT" else "ROBERTA_SentimentAnalysis"
        result_map = {0: "Negative", 1: "Positive"}
    else:  # News
        model_key = "BERT_News" if model_selected == "BERT" else "ROBERTA_News"
        result_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

    model = models[model_key]
    tokenizer = tokenizers[model_key]
    model_name = model_key

    # 预测
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).item()

    # 显示结果
    st.subheader("Prediction Result")
    st.success(f"{model_selected} Prediction: {result_map[predictions]}")

    # 显示混淆矩阵
    st.subheader(f"{model_selected} Evaluation Results")


    #  混淆矩阵单独一行
    conf_matrix_img = load_confusion_matrix(model_name)
    st.image(conf_matrix_img, use_column_width=True)
    if os.path.exists(conf_matrix_img):
        st.markdown("**Confusion Matrix:**")
        st.image(conf_matrix_img, use_column_width=True)

    #  训练指标和分类报告并排展示
    cols = st.columns(2)

    # 左列：训练指标
    with cols[0]:
        overall_img = load_overall_performance(model_name)
        if os.path.exists(overall_img):
            st.markdown("**Training Performance Metrics per Epoch:**")
            st.image(overall_img, use_column_width=True)

    # 右列：分类报告
    with cols[1]:
        categories_img = load_categories_performance(model_name)
        if os.path.exists(categories_img):
            st.markdown("**Classification Report:**")
            st.image(categories_img, use_column_width=True)



st.markdown("---")

st.markdown("""
BERT is trained based on google-bert/bert-base-uncased.<br>
ROBERTA is trained based on FacebookAI/roberta-base.<br>
Deploy using @Streamlit.<br>
Authors: NTU EEE 6405 Group 16: Zeng Jiabo, Fu Wanting, Hou Xinyu, Wang Di, Wang Jianyu, Xie Debin (Sort by first letter of surname)
""", unsafe_allow_html=True)
