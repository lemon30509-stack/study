import gradio as gr
import torch
from PIL import Image
from torchvision import models, transforms


CLASSES = ["NORMAL", "PNEUMONIA"]
CKPT = "outputs/classifier/best_classifier.pth"


def build_model():
    m = models.resnet18(weights=None)
    in_f = m.fc.in_features
    m.fc = torch.nn.Sequential(
        torch.nn.Linear(in_f, 256),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(256, 2),
    )
    return m


device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model().to(device)
try:
    model.load_state_dict(torch.load(CKPT, map_location=device))
except FileNotFoundError:
    pass
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image: Image.Image):
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0]
    pred = int(torch.argmax(p).item())
    return {"NORMAL": float(p[0]), "PNEUMONIA": float(p[1])}, f"预测: {CLASSES[pred]}"


app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="上传胸部X光图像"),
    outputs=[gr.Label(label="类别概率"), gr.Textbox(label="结果")],
    title="基于GAN少样本增强的肺炎识别系统",
    description="上传单张X光图像，输出正常/肺炎概率。",
)

if __name__ == "__main__":
    app.launch()
