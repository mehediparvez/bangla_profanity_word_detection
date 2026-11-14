import os
import sys
import gradio as gr
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import load_trained_model, LABEL_MAP
from src.predict import predict_text

MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        "bangla-bert-base",
        "run_1",
        "best_model",
    ),
)

model = None
tokenizer = None


def load():
    global model, tokenizer
    if model is None:
        model, tokenizer = load_trained_model(MODEL_DIR)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()


def classify(text):
    if not text or not text.strip():
        return "Please enter a Bangla comment.", ""

    load()
    result = predict_text(text.strip(), model, tokenizer)

    label = result["label"]
    confidence = result["confidence"]
    probs = result["probabilities"]

    output_lines = [
        f"**Prediction:** {label.upper()}",
        f"**Confidence:** {confidence:.1%}",
        "",
        "**Probabilities:**",
        f"- Non-toxic: {probs['non-toxic']:.1%}",
        f"- Toxic: {probs['toxic']:.1%}",
    ]

    emoji = "🟢" if label == "non-toxic" else "🔴"
    header = f"{emoji} {label.upper()} ({confidence:.1%})"

    return header, "\n".join(output_lines)


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        label="Enter a Bangla comment",
        placeholder="এখানে বাংলা মন্তব্য লিখুন...",
        lines=3,
    ),
    outputs=[
        gr.Textbox(label="Result"),
        gr.Markdown(label="Details"),
    ],
    title="Bangla Toxic Comment Detector",
    description=(
        "Detects whether a Bangla comment is toxic or non-toxic "
        "using a fine-tuned BanglaBERT model."
    ),
    examples=[
        ["এটি একটি সুন্দর দেশ"],
        ["তুই একটা গাধা"],
        ["আজকের আবহাওয়া খুব সুন্দর"],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
