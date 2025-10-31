# debug_eval.py — 在项目根目录运行： python debug_eval.py
import numpy as np
import torch
from scipy.io import loadmat
from dataset import prepare_dataset
from s2vnet_model import S2VNet
import os

# --- 配置（与 demo.py 一致） ---
class Args: pass
args = Args()
args.dataset = 'Indian'
args.patches = 7
args.batch_size = 64
args.gpu_id = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# --- 加载数据（会打印和构造 loader） ---
label_train_loader, label_test_loader, label_true_loader, band, height, width, num_classes, label, total_pos_true = prepare_dataset(args)
# 🔍 检查数据加载器中的类别分布
print("\n================= 数据加载检查 =================")
all_train_labels = []
all_test_labels = []
all_true_labels = []

for _, y in label_train_loader:
    all_train_labels.extend(y.numpy())
for _, y in label_test_loader:
    all_test_labels.extend(y.numpy())
for _, y in label_true_loader:
    all_true_labels.extend(y.numpy())

print(f"Train label unique: {np.unique(all_train_labels)}")
print(f"Test  label unique: {np.unique(all_test_labels)}")
print(f"True  label unique: {np.unique(all_true_labels)}")

# 🔍 如果发现 0 类缺失，提示用户
if 0 not in np.unique(all_train_labels):
    print("⚠️ [警告] 训练集中未包含类别 0！")
if 0 not in np.unique(all_test_labels):
    print("⚠️ [警告] 测试集中未包含类别 0！")
print("================================================\n")

print("num_classes(from prepare_dataset):", num_classes)
print("label_true_loader batch_size:", args.batch_size)

# --- 加载模型（用你要检查的权重） ---
model = S2VNet(band, num_classes, args.patches).to(device)
weight_paths = ['./results/Indian_s2vnet_p7_96.03_epoch485.pkl',
                './results/Indian_s2vnet_p7_95.83_epoch415.pkl']

for weight_path in weight_paths:
    if not os.path.exists(weight_path):
        print("weight not found:", weight_path)
        continue

    print("\n=== Testing weight:", weight_path, " ===")
    ckpt = torch.load(weight_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # collect preds and trues
    preds = []
    trues = []
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(label_true_loader):
            batch_data = batch_data.to(device)
            # forward
            re_unmix_nonlinear, re_unmix, batch_pred, *_ = model(batch_data)
            _, pred = batch_pred.topk(1, 1, True, True)
            pp = pred.squeeze().cpu().numpy()
            preds = np.append(preds, pp)
            trues = np.append(trues, batch_target.numpy())

    preds = preds.astype(int)
    trues = trues.astype(int)
    print("pred shape:", preds.shape, "true shape:", trues.shape)
    print("unique true labels:", np.unique(trues))
    print("true counts (label:count):")
    u,c = np.unique(trues, return_counts=True)
    for uu,cc in zip(u,c): print(uu, cc)
    print("pred counts (label:count):")
    up,cp = np.unique(preds, return_counts=True)
    for uu,cc in zip(up,cp): print(uu, cc)

    # raw agreement
    raw_agree = np.mean(preds == trues)
    print("raw agreement (pred == true):", raw_agree)

    # agreement with pred+1 (if trues are 1-based)
    agree_plus = np.mean((preds + 1) == trues)
    print("agreement with (pred+1)==true:", agree_plus)

    # agreement with pred-1 (if preds were 1-based)
    agree_minus = np.mean((preds - 1) == trues)
    print("agreement with (pred-1)==true:", agree_minus)

    # === compute confusion ignoring background label 0 ===
    mask = trues != 0
    print("valid pixels (trues != 0):", mask.sum(), "/", len(trues))
    trues_valid = trues[mask]
    preds_valid = preds[mask]

    # if trues are 1..C, convert to 0..C-1
    if trues_valid.max() == num_classes:
        print("Detected trues max == num_classes, converting trues_valid -= 1")
        trues_valid = trues_valid - 1

    # ensure preds in 0..num_classes-1
    print("preds_valid unique:", np.unique(preds_valid))
    print("trues_valid unique:", np.unique(trues_valid))

    # build confusion matrix
    from sklearn.metrics import confusion_matrix
    labels = list(range(num_classes))
    cm = confusion_matrix(trues_valid, preds_valid, labels=labels)
    print("confusion matrix shape:", cm.shape)
    OA = np.trace(cm) / np.sum(cm)
    per_class = np.diag(cm) / (cm.sum(axis=1) + 1e-12)
    AA = np.mean(per_class)
    print("OA (ignore background):", OA)
    print("AA mean (ignore background):", AA)
    print("per-class acc:", per_class)