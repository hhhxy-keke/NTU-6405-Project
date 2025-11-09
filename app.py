import streamlit as st
import torch
from sympy.physics.control.control_plots import plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
from peft import PeftModel

st.set_page_config(
    page_title="6405 Group 16 Project",
    layout="wide"
)
st.title("🤖6405 Group 16: Online Prediction Platform for BERT and its Variant Models")
st.write("Please select a model, enter text, and view the prediction results and the model's training performance metrics.")


MODEL_PATHS = {
    "BERT_SentimentAnalysis": "model/bert_base_sentiment",
}

BASE_MODEL = "bert-base-uncased"

# 加载模型函数（缓存，避免重复加载）
@st.cache_resource
def load_models():
    models = {}
    tokenizers = {}
    for name, adapter_path in MODEL_PATHS.items():
        base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
        model.eval()  # 推理模式

        models[name] = model
        tokenizers[name] = AutoTokenizer.from_pretrained(BASE_MODEL)
    return models, tokenizers, MODEL_PATHS


# 加载混淆矩阵图片（PNG）
def load_confusion_matrix(model_name):
    img_path = f"metrics/confusion_{model_name}.png"
    img = Image.open(img_path)
    return img

# 加载
models, tokenizers, model_names = load_models()

# 用户输入与模型选择
with st.sidebar:  # 侧边栏放输入控件
    st.subheader("Sentiment Analysis")
    sentiment_models = ["BERT", "ROBERTA"]
    sentiment_model_selected = st.selectbox("Select Sentiment Model:", sentiment_models)
    sentiment_input = st.text_area(
        "Enter text for sentiment analysis:",
        "Please enter a sentence with emotional connotations."
    )

    st.subheader("News Topic Categorization")
    news_models = ["BERT", "ROBERTA"]
    news_model_selected = st.selectbox("Select News Model:", news_models)
    news_input = st.text_area(
        "Enter text for news topic categorization:",
        "Please enter a sentence belonging to 'World', 'Sports', 'Business', or 'Sci/Tech'."
    )


    submit = st.button("Start Predicting")


# 模型预测与结果展示
if submit:
    # 情感分析预测
    if sentiment_input and sentiment_model_selected:
        model = models["BERT_SentimentAnalysis"] if sentiment_model_selected == "BERT" else models["ROBERTA_SentimentAnalysis"]
        tokenizer = tokenizers["BERT_SentimentAnalysis"] if sentiment_model_selected == "BERT" else tokenizers["ROBERTA_SentimentAnalysis"]

        inputs = tokenizer(sentiment_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1).item()

        st.subheader("Sentiment Analysis Prediction")
        result_map = {0: "负面", 1: "正面"}  # 根据模型标签调整
        st.success(f"{sentiment_model_selected} 预测结果: {result_map[predictions]}")

        # 混淆矩阵展示
        st.subheader(f"{sentiment_model_selected} 情感分析混淆矩阵")
        conf_matrix_img = load_confusion_matrix("BERT_SentimentAnalysis")  # 或 ROBERTA 的图片路径
        st.image(conf_matrix_img, use_column_width=True)

    # 新闻分类预测
    if news_input and news_model_selected:
        model = models["BERT_News"] if news_model_selected == "BERT" else models["ROBERTA_News"]
        tokenizer = tokenizers["BERT_News"] if news_model_selected == "BERT" else tokenizers["ROBERTA_News"]

        inputs = tokenizer(news_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1).item()

        st.subheader("News Topic Categorization Prediction")
        topic_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}  # 按你的标签映射
        st.success(f"{news_model_selected} 预测结果: {topic_map[predictions]}")

        # 混淆矩阵展示
        st.subheader(f"{news_model_selected} 新闻分类混淆矩阵")
        conf_matrix_img = load_confusion_matrix("BERT_News")  # 或 ROBERTA 的图片路径
        st.image(conf_matrix_img, use_column_width=True)


# # 2️⃣ 展示总体性能对比表格
# metrics_file = "metrics/metrics.csv"
# if os.path.exists(metrics_file):
#     df = pd.read_csv(metrics_file)
#     st.bar_chart(df.set_index("model")["accuracy"])

st.markdown("---")
st.write("BERT is trained on google-bert/bert-base-uncased.\n"
         "ROBERTA is trained on FacebookAI/roberta-base. \n"
         "Deploy using Streamlit \n"
         "Authors: NTU EEE 6405 Group 16: Zeng Jiabo, Fu Wanting, Hou Xinyu, Wang Di, Wang Jianyu, Xie Debin (Sort by first letter of surname)")