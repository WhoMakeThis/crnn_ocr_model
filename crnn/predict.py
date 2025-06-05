import os
import csv
import torch
from PIL import Image
from model import CRNN
from dataset import CHARS
import torchvision.transforms as transforms
from tqdm import tqdm
from difflib import SequenceMatcher

# 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 문자 사전
IDX2CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
IDX2CHAR[0] = ""  # CTC blank

# ✅ 전처리 - 학습 이미지 크기와 동일하게
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((50, 150)),  # 변경됨
    transforms.ToTensor()
])

# 디코딩 함수
def decode_prediction(preds):
    preds = preds.permute(1, 0, 2)  # [T, B, C] -> [B, T, C]
    preds = torch.argmax(preds, dim=2)
    preds = preds[0].detach().cpu().numpy().tolist()

    decoded = []
    prev = -1
    for p in preds:
        if p != prev and p != 0:
            decoded.append(IDX2CHAR.get(p, ""))
        prev = p
    return ''.join(decoded)

# 문자 단위 정확도 계산
def char_accuracy(pred, truth):
    correct = sum(p == t for p, t in zip(pred, truth))
    return correct / max(len(truth), 1)

# 유사도 계산
def calc_similarity(pred, truth):
    return SequenceMatcher(None, pred, truth).ratio()

# ✅ 모델 로드 - height=50로 맞춤
model = CRNN(50, 1, len(CHARS) + 1, 256).to(DEVICE)
model.load_state_dict(torch.load("best_crnn_model.pth", map_location=DEVICE))
model.eval()

# 테스트 이미지
test_dir = "dataset/captcha_images_split/test"
image_paths = sorted([
    os.path.join(test_dir, fname)
    for fname in os.listdir(test_dir)
    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
])

results = []

for img_path in tqdm(image_paths, desc="Predicting"):
    img = Image.open(img_path).convert("L")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(img)
        pred_text = decode_prediction(preds)

    filename = os.path.basename(img_path)
    ground_truth = os.path.splitext(filename)[0]
    is_correct = (pred_text == ground_truth)
    acc = round(char_accuracy(pred_text, ground_truth), 3)
    similarity = round(calc_similarity(pred_text, ground_truth), 3)

    results.append({
        "filename": filename,
        "ground_truth": ground_truth,
        "prediction": pred_text,
        "is_correct": is_correct,
        "char_accuracy": acc,
        "similarity": similarity
    })

# CSV 저장
csv_path = "crnn_prediction_results_with_accuracy.csv"
with open(csv_path, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "filename", "ground_truth", "prediction", "is_correct", "char_accuracy", "similarity"
    ])
    writer.writeheader()
    writer.writerows(results)

print(f"✅ 예측 결과가 {csv_path}에 저장되었습니다.")
