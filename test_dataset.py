from dataset import XrayDataset
import matplotlib.pyplot as plt


# =========================
# 创建 Dataset
# =========================

dataset = XrayDataset(
    "data/images",
    "data/masks"
)


# =========================
# 基本信息
# =========================

print("数据集数量:", len(dataset))


# =========================
# 读取第一个样本
# =========================

image, mask = dataset[0]


print("\n===== Image =====")
print("shape:", image.shape)
print("dtype:", image.dtype)
print("min:", image.min())
print("max:", image.max())


print("\n===== Mask =====")
print("shape:", mask.shape)
print("dtype:", mask.dtype)
print("min:", mask.min())
print("max:", mask.max())


# =========================
# 可视化
# =========================

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image[0], cmap="gray")
plt.title("X-ray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask[0], cmap="gray")
plt.title("Mask")
plt.axis("off")

plt.tight_layout()
plt.show()