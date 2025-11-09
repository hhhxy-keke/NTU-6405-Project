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
st.title("🤖 NLP 模型可视化与在线预测平台")
st.write("请选择一个模型，输入文本，查看预测结果和模型表现指标。")


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
        base_model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=3
        )
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path
        )
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
    st.subheader("User Inputs")
    user_input = st.text_area("请输入文本:", "这是一个测试句子")
    selected_model = st.selectbox("选择模型:", list(model_names.keys()))
    submit = st.button("运行预测")


# 模型预测与结果展示
if submit and user_input:
    # 获取选中的模型和分词器
    model = models[selected_model]
    tokenizer = tokenizers[selected_model]

    # 模型推理
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).item()  # 假设是分类任务

    # 显示结果（根据你的任务类型调整，如情感分析返回正面/负面）
    st.subheader("预测结果")
    result_map = {0: "负面", 1: "中性", 2: "正面"}
    st.success(f"模型预测: {result_map[predictions]}")

    # 显示模型性能图表（混淆矩阵等）
    st.subheader("模型性能分析")
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"{selected_model} 混淆矩阵")
        conf_matrix_img = load_confusion_matrix(selected_model)
        st.image(conf_matrix_img, use_column_width=True)

    with col2:
        st.write("模型准确率对比")
        # 假设提前计算了各模型的准确率
        accuracy_data = {
            "情感分析模型": 0.89,
            "文本分类模型": 0.85,
            "命名实体识别": 0.92,
            "关键词提取": 0.81,
            "文本摘要": 0.78,
            "机器翻译": 0.87
        }
        # 绘制柱状图
        fig, ax = plt.subplots()
        ax.bar(accuracy_data.keys(), accuracy_data.values())
        plt.xticks(rotation=45)
        plt.ylim(0, 1.0)
        st.pyplot(fig)

# # 2️⃣ 展示总体性能对比表格
# metrics_file = "metrics/metrics.csv"
# if os.path.exists(metrics_file):
#     df = pd.read_csv(metrics_file)
#     st.bar_chart(df.set_index("model")["accuracy"])

st.markdown("---")
st.write("模型基于Colab训练，使用Streamlit部署 | 联系作者：xxx")