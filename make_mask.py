from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw


# =========================
# 路径
# =========================

XML_PATH = Path(r"D:\GRAZPEDWRI\annotations_train1.xml")

IMAGE_PATH = Path(r"D:\VC Code\Pytorch_test\data\images\0208_1044966620_01_WRI-L1_F003.png")


MASK_DIR = Path(r"D:\VC Code\Pytorch_test\data\masks")
MASK_DIR.mkdir(parents=True, exist_ok=True)

MASK_PATH = MASK_DIR / IMAGE_PATH.name


# =========================
# 目标图片
# =========================

TARGET_NAME = IMAGE_PATH.name


# =========================
# 读取 XML
# =========================

tree = ET.parse(XML_PATH)
root = tree.getroot()

target = None

for image in root.findall(".//image"):
    if image.get("name") == TARGET_NAME:
        target = image
        break

if target is None:
    raise ValueError(f"没有找到目标图片: {TARGET_NAME}")


width = int(target.get("width"))
height = int(target.get("height"))

print("找到目标图片")
print("原始尺寸:", width, "x", height)


# =========================
# 创建空 Mask
# =========================

mask = Image.new("L", (width, height), 0)
draw = ImageDraw.Draw(mask)


# =========================
# 处理 Polygon
# =========================

polygon_count = 0

for polygon in target.findall("polygon"):

    points_text = polygon.get("points")

    points = []

    for point in points_text.split(";"):
        x, y = point.split(",")
        points.append((float(x), float(y)))

    draw.polygon(points, fill=255)

    polygon_count += 1


print("Polygon 数量:", polygon_count)


# =========================
# 保存原始尺寸 Mask
# =========================

mask.save(MASK_PATH)

print("Mask 已保存:")
print(MASK_PATH)


# =========================
# 显示统计
# =========================

mask_array = np.array(mask)

print("Mask shape:", mask_array.shape)
print("Mask dtype:", mask_array.dtype)
print("Mask min:", mask_array.min())
print("Mask max:", mask_array.max())
print("前景像素:", np.sum(mask_array > 0))