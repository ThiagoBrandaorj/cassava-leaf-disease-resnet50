import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download
import keras

# ---------------- Métrica customizada (necessária para o load) ----------------
@keras.saving.register_keras_serializable(package="Custom")
class MeanRecall(tf.keras.metrics.Metric):
    def __init__(self, num_classes=5, name="mean_recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.tp = self.add_weight(shape=(num_classes,), initializer="zeros", name="tp")
        self.fn = self.add_weight(shape=(num_classes,), initializer="zeros", name="fn")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_class = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        tp_list, fn_list = [], []
        for i in range(self.num_classes):
            label = tf.fill(tf.shape(y_true), i)
            tp_list.append(tf.reduce_sum(tf.cast((y_true == label) & (y_pred_class == label), tf.float32)))
            fn_list.append(tf.reduce_sum(tf.cast((y_true == label) & (y_pred_class != label), tf.float32)))
        self.tp.assign_add(tf.stack(tp_list))
        self.fn.assign_add(tf.stack(fn_list))

    def result(self):
        return tf.reduce_mean(self.tp / (self.tp + self.fn + 1e-7))

    def reset_state(self):
        self.tp.assign(tf.zeros(self.num_classes))
        self.fn.assign(tf.zeros(self.num_classes))

    def get_config(self):
        base = super().get_config()
        base.update({"num_classes": self.num_classes})
        return base


# ---------------- Config ----------------
IMG_SIZE = 336
REPO_ID  = "seu-usuario/seu-repo-no-hf"   # <-- ajuste
FILENAME = "cassava_model.keras"

CLASS_NAMES = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mottle (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy",
]

CLASS_INFO = {
    "Cassava Bacterial Blight (CBB)":     "Doença bacteriana que causa murcha e necrose nas folhas.",
    "Cassava Brown Streak Disease (CBSD)": "Doença viral com manchas amareladas e apodrecimento das raízes.",
    "Cassava Green Mottle (CGM)":          "Mosaico verde, com manchas verde-claras nas folhas.",
    "Cassava Mosaic Disease (CMD)":        "Mosaico da mandioca, folhas deformadas e coloração irregular.",
    "Healthy":                             "Folha saudável, sem sinais de doença.",
}


# ---------------- Carregamento (cacheado) ----------------
@st.cache_resource(show_spinner="Baixando e carregando o modelo...")
def load_model_and_preprocess():
    # Lê o token dos Secrets do Streamlit Cloud (ou variável de ambiente local)
    hf_token = st.secrets.get("HF_TOKEN")
    if not hf_token:
        st.error(
            "Token do Hugging Face (HF_TOKEN) não configurado. "
            "Adicione nos Secrets do Streamlit Cloud ou como variável de ambiente."
        )
        st.stop()
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, token=hf_token)
    model = tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects={"MeanRecall": MeanRecall},
    )
    preprocess = tf.keras.Sequential([
        tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE, crop_to_aspect_ratio=True),
        tf.keras.layers.Lambda(tf.keras.applications.resnet.preprocess_input),
    ])
    return model, preprocess


model, preprocess = load_model_and_preprocess()


# ---------------- Predição ----------------
def predict(img_pil: Image.Image):
    img_rgb = img_pil.convert("RGB")
    x = tf.convert_to_tensor(np.array(img_rgb), dtype=tf.float32)
    x = tf.expand_dims(x, 0)
    x = preprocess(x)
    pred = model.predict(x, verbose=0)[0]
    return pred


# ---------------- Interface ----------------
st.set_page_config(page_title="Cassava Leaf Disease Classifier", page_icon="🌱", layout="centered")

st.title("🌱 Classificador de Doenças da Mandioca")
st.markdown(
    "Envie uma foto de uma folha de mandioca. O modelo prevê a classe entre "
    "**4 doenças + folha saudável**, usando ResNet50 (transfer learning)."
)

uploaded = st.file_uploader("Escolha uma imagem (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img_pil = Image.open(uploaded)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img_pil, caption="Imagem enviada", use_container_width=True)

    with st.spinner("Classificando..."):
        probs = predict(img_pil)

    top_idx  = int(np.argmax(probs))
    top_name = CLASS_NAMES[top_idx]
    top_conf = float(probs[top_idx])

    with col2:
        st.success(f"### {top_name}")
        st.metric("Confiança", f"{top_conf*100:.2f}%")
        st.caption(CLASS_INFO[top_name])

    st.subheader("Probabilidades por classe")
    scores = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    # Ordena do maior para o menor
    scores_sorted = dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
    st.bar_chart(scores_sorted)

    with st.expander("Ver detalhes numéricos"):
        for name, p in scores_sorted.items():
            st.write(f"**{name}**: {p*100:.2f}%")