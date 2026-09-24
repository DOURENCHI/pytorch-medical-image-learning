from dataset import XrayDataset
from torch.utils.data import DataLoader


# =========================
# 1. 创建 Dataset
# =========================

dataset = XrayDataset(
    "data/images",
    "data/masks"
)


# =========================
# 2. 创建 DataLoader
# =========================

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)


# =========================
# 3. 查看 Dataset
# =========================

print("Dataset 数量:", len(dataset))


# =========================
# 4. 从 DataLoader 取一批数据
# =========================

for images, masks in loader:

    print("Images shape:", images.shape)
    print("Masks shape:", masks.shape)

    break