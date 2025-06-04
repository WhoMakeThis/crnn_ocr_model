import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm

# ---------- 설정 ----------
TOTAL_IMAGES = 50000
CAPTCHA_LENGTH = 5
CHARS = string.ascii_uppercase + string.digits
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

# ✅ Flask에서 사용한 폰트로 통일
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(SCRIPT_DIR, "DejaVuSans-Bold.ttf")

# ✅ Flask에서 사용하는 CAPTCHA 이미지 크기와 동일하게
IMAGE_SIZE = (150, 50)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "captcha_images_split")

# ---------- CAPTCHA 이미지 생성 ----------
def generate_random_text(length=CAPTCHA_LENGTH):
    return ''.join(random.choices(CHARS, k=length))

def create_captcha_image(text):
    width, height = IMAGE_SIZE
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 36)

    x = 5
    for char in text:
        y_offset = random.randint(0, 10)
        angle = random.randint(-30, 30)

        char_img = Image.new('RGBA', (40, 40), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)
        char_draw.text((0, 0), char, font=font, fill=(0, 0, 0))

        rotated = char_img.rotate(angle, expand=1)
        img.paste(rotated, (x, y_offset), rotated)
        x += 25

    # 랜덤 선
    for _ in range(5):
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.line((x1, y1, x2, y2), fill=color, width=1)

    # 랜덤 점
    for _ in range(100):
        x_dot, y_dot = random.randint(0, width), random.randint(0, height)
        color = tuple(random.randint(0, 255) for _ in range(3))
        draw.point((x_dot, y_dot), fill=color)

    # 블러 처리
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return img

# ---------- 메인 ----------
if __name__ == "__main__":
    print("📦 CAPTCHA 이미지 생성 시작!")

    # 폴더 생성
    for split in SPLIT_RATIOS:
        os.makedirs(os.path.join(OUTPUT_DIR, split), exist_ok=True)

    # 랜덤 텍스트 생성 및 셔플
    all_texts = [generate_random_text() for _ in range(TOTAL_IMAGES)]
    random.shuffle(all_texts)

    train_end = int(TOTAL_IMAGES * SPLIT_RATIOS["train"])
    val_end = train_end + int(TOTAL_IMAGES * SPLIT_RATIOS["val"])

    for idx, text in enumerate(tqdm(all_texts, desc="Generating")):
        if idx < train_end:
            split = "train"
        elif idx < val_end:
            split = "val"
        else:
            split = "test"

        img = create_captcha_image(text)
        save_path = os.path.join(OUTPUT_DIR, split, f"{text}.png")
        img.save(save_path)

    print("✅ CAPTCHA 데이터셋 생성 완료!")