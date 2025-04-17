import os
import random
import string
import platform
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from tqdm import tqdm

# ---------- 설정 ----------
TOTAL_IMAGES = 50000  # 생성할 총 이미지 수
CAPTCHA_LENGTH = 5
CHARS = string.ascii_uppercase + string.digits
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

# 운영체제에 맞는 기본 경로 설정
if platform.system() == "Windows":
    FONT_PATH = "C:/Users/ADMIN/AppData/Local/Microsoft/Windows/Fonts/dejavu-sans.bold_0.ttf"  # Windows 사용자 폰트 경로
elif platform.system() == "Linux":
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
elif platform.system() == "Darwin":  # macOS
    FONT_PATH = "/usr/local/share/fonts/DejaVuSans-Bold.ttf"  # 직접 설치한 경우 경로 예시
else:
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# 폰트 존재 여부 확인
if not os.path.exists(FONT_PATH):
    raise FileNotFoundError(f"Font file not found at: {FONT_PATH}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "captcha_images_split")
IMAGE_SIZE = (180, 60)

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

    # 블러
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    return img

# ---------- 메인 ----------
if __name__ == "__main__":
    print("📦 CAPTCHA 이미지 생성 시작!")

    # 폴더 생성
    for split in SPLIT_RATIOS:
        split_path = os.path.join(OUTPUT_DIR, split)
        os.makedirs(split_path, exist_ok=True)

    # 이미지 생성
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
