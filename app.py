import streamlit as st
import torch
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
    "BERT_SentimentAnalysis": {"path": "model/bert_base_sentiment", "num_labels": 2},
    "ROBERTA_SentimentAnalysis": {"path": "model/roberta_base_sentiment", "num_labels": 2},
    "BERT_News": {"path": "model/bert_news", "num_labels": 4},
    "ROBERTA_News": {"path": "model/roberta_news", "num_labels": 4},
}

@st.cache_resource
def load_models():
    models = {}
    tokenizers = {}
    for name, info in MODEL_PATHS.items():
        adapter_path = info["path"]
        num_labels = info["num_labels"]

        # 根据模型名选择基座
        if "BERT" in name:
            base_model_name = "bert-base-uncased"
        elif "ROBERTA" in name:
            base_model_name = "facebook/roberta-base"

        # 加载基座模型 + Adapter
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name, num_labels=num_labels
        )
        model = PeftModel.from_pretrained(base_model, adapter_path, is_trainable=False)
        model.eval()  # 推理模式

        models[name] = model
        tokenizers[name] = AutoTokenizer.from_pretrained(base_model_name)

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
        model_name = "BERT_SentimentAnalysis" if sentiment_model_selected == "BERT" else "ROBERTA_SentimentAnalysis"

        inputs = tokenizer(sentiment_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1).item()

        st.subheader("Sentiment Analysis Prediction")
        result_map = {0: "Negative", 1: "Positive"}  # 根据模型标签调整
        st.success(f"{sentiment_model_selected} Prediction Results: {result_map[predictions]}")

        # 混淆矩阵展示
        st.subheader(f"{sentiment_model_selected} Confusion Matrix (Sentiment Analysis)")
        conf_matrix_img = load_confusion_matrix(model_name)
        st.image(conf_matrix_img, use_column_width=True)

    # 新闻分类预测
    if news_input and news_model_selected:
        model = models["BERT_News"] if news_model_selected == "BERT" else models["ROBERTA_News"]
        tokenizer = tokenizers["BERT_News"] if news_model_selected == "BERT" else tokenizers["ROBERTA_News"]
        model_name = "BERT_AGNews" if sentiment_model_selected == "BERT" else "ROBERTA_AGNews"

        inputs = tokenizer(news_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1).item()

        st.subheader("News Topic Categorization Prediction")
        topic_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        st.success(f"{news_model_selected} Prediction Results: {topic_map[predictions]}")

        # 混淆矩阵展示
        st.subheader(f"{news_model_selected} Confusion Matrix (News Topic Categorization)")
        conf_matrix_img = load_confusion_matrix(model_name)
        st.image(conf_matrix_img, use_column_width=True)


# # 2️⃣ 展示总体性能对比表格
# metrics_file = "metrics/metrics.csv"
# if os.path.exists(metrics_file):
#     df = pd.read_csv(metrics_file)
#     st.bar_chart(df.set_index("model")["accuracy"])

st.markdown("---")
st.markdown("""
BERT is trained based on google-bert/bert-base-uncased.
ROBERTA is trained based on FacebookAI/roberta-base.
Deploy using @Streamlit.
Authors: NTU EEE 6405 Group 16: Zeng Jiabo, Fu Wanting, Hou Xinyu, Wang Di, Wang Jianyu, Xie Debin (Sort by first letter of surname)
""")