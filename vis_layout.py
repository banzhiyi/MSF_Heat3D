import cv2
import numpy as np
from pathlib import Path

def draw_zoom_figure(
    img,
    bbox,              # (x, y, w, h)
    ellipse=True,
    scale=3,           # 放大倍数（决定右图宽度）
    spacing=20         # 左右间距
):
    """
    输入一张图，输出论文风格：
    左：原图+蓝框
    右：等高放大图+红椭圆
    """

    x, y, w, h = bbox
    H, W, _ = img.shape

    # ===== 1. 画左侧蓝框 =====
    img_left = img.copy()
    cv2.rectangle(img_left, (x, y), (x+w, y+h), (255, 0, 0), 3)

    # ===== 2. 裁剪 =====
    crop = img[y:y+h, x:x+w]

    # ===== 3. resize 到“同高度” =====
    new_h = H
    new_w = int(w * (H / h)) * scale   # 横向放大

    zoom = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # ===== 4. 画红色椭圆 =====
    if ellipse:
        center = (new_w // 2, new_h // 2)
        axes = (new_w // 3, new_h // 3)
        cv2.ellipse(zoom, center, axes, 30, 0, 360, (0, 0, 255), 5)

    # ===== 5. 拼接 =====
    canvas = np.ones((H, W + spacing + new_w, 3), dtype=np.uint8) * 255

    canvas[:, :W] = img_left
    canvas[:, W + spacing:] = zoom

    return canvas


# ===== 使用示例 =====
INPUT_PATH = Path('/home/sy/Downloads/最新可视化图保存/Houston数据集/2DCNN')
if INPUT_PATH.is_dir():
    candidates = sorted(
        p for p in INPUT_PATH.iterdir()
        if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    )
    if not candidates:
        raise FileNotFoundError(f'目录中未找到可读取的图片文件: {INPUT_PATH}')
    img_path = candidates[0]
else:
    img_path = INPUT_PATH

img = cv2.imread(str(img_path))
if img is None:
    raise FileNotFoundError(f'图片读取失败，请检查路径是否正确: {img_path}')

# bbox: 你要放大的区域
bbox = (300, 100, 120, 120)

result = draw_zoom_figure(img, bbox)

cv2.imwrite('result.png', result)
