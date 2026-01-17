import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import savemat
from torch import optim
from s2vnet_model import S2VNet
from vheat3d_model import MSF_Heat3D
from utils import AvgrageMeter, accuracy, output_metric, NonZeroClipper, print_args
from dataset import prepare_dataset
import numpy as np
import time
import os
import json
import pandas as pd
from datetime import datetime
import subprocess
from ViT import ViT
from HybridSN import HybridSN
from morphFormer import MorphFormer
from ssftt import SSFTT
from HSI3DCNN import HSI3DCNN
from DSNet import DSNet
from MASSFormer import MASSFormer
from SiT import SiT
from HSI2DCNN import HSI2DCNN
from MambaHSI.MambaHSI import MambaHSIClassifier
from VisionMamba.VMamba import VisionMambaClassifier
from fvcore.nn import FlopCountAnalysis
from S2Mamba import S2Mamba
from vHeat import S2VHeat
# 参数配置
parser = argparse.ArgumentParser("HSI")
parser.add_argument('--fix_random', action='store_true', default=True, help='fix randomness')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Berlin', 'Augsburg', 'Houston'], default='Indian', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--model_name', choices=['s2vnet','vHeat', 'MSF_Heat3D','HybridSN','ViT','MorphFormer','SSFTT','HSI3DCNN','DSNet','MASSFormer','SiT', 'HSI2DCNN', 'MambaHSI', 'VMamba', 'S2Mamba'], default='s2vnet', help='S2VNet')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patches', type=int, default=7, help='number of patches')
parser.add_argument('--epoches', type=int, default=500, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
args = parser.parse_args()

def get_git_branch_name():
    """获取当前 Git 分支名"""
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
    except Exception:
        branch = "unknown_branch"
        print("⚠️ 未检测到 Git 分支信息，结果将保存在 unknown_branch 目录中。")
    return branch

# 🆕 创建实验结果目录
def create_experiment_dir(dataset_name):
    """创建实验目录结构"""
    branch_name = get_git_branch_name()  # 获取当前 Git 分支名
    base_dir = f'./results/{branch_name}/{dataset_name}'  # ← 多一层分支目录
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


# 🆕 保存实验配置
def save_experiment_config(args, experiment_dir):
    """保存实验配置到JSON文件"""
    config = {
        'dataset': args.dataset,
        'model_name': args.model_name,
        'batch_size': args.batch_size,
        'patches': args.patches,
        'epoches': args.epoches,
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'weight_decay': args.weight_decay,
        'test_freq': args.test_freq,
        'seed': args.seed,
        'gpu_id': args.gpu_id,
        'fix_random': args.fix_random,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    config_path = os.path.join(experiment_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"📄 实验配置已保存至: {config_path}")


# 🆕 初始化训练日志
def init_training_log(experiment_dir):
    """初始化训练日志CSV文件"""
    log_path = os.path.join(experiment_dir, 'training_log.csv')
    # 如果文件不存在，创建并写入表头
    if not os.path.exists(log_path):
        pd.DataFrame(columns=[
            'epoch', 'train_loss', 'train_acc', 'val_OA', 'val_AA', 'val_Kappa',
            'learning_rate', 'timestamp'
        ]).to_csv(log_path, index=False)
    return log_path


# 🆕 记录训练日志
def log_training_epoch(log_path, epoch, train_loss, train_acc, val_OA, val_AA, val_Kappa, lr):
    """记录单个epoch的训练结果"""
    log_data = {
        'epoch': epoch + 1,
        'train_loss': float(train_loss),
        'train_acc': float(train_acc),
        'val_OA': float(val_OA) if val_OA is not None else None,
        'val_AA': float(val_AA) if val_AA is not None else None,
        'val_Kappa': float(val_Kappa) if val_Kappa is not None else None,
        'learning_rate': float(lr),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # 读取现有日志并追加新数据
    if not os.path.exists(log_path):
        # 第一次创建文件，写入表头
        pd.DataFrame(columns=log_data.keys()).to_csv(log_path, index=False)

        # 追加新行
    with open(log_path, 'a') as f:
        pd.DataFrame([log_data]).to_csv(f, header=False, index=False)


# 🆕 保存最终测试结果
def save_final_results(experiment_dir, OA, AA, Kappa, class_AA, training_time, best_epoch=None):
    """保存最终测试结果到JSON文件"""
    if class_AA is None:
        class_AA = []
    results = {
        'dataset': args.dataset,
        'final_OA': float(OA),
        'final_AA': float(AA),
        'final_Kappa': float(Kappa),
        'class_wise_AA': [float(x) for x in class_AA],
        'training_time_seconds': float(training_time),
        'best_epoch': best_epoch,
        'test_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    results_path = os.path.join(experiment_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"📊 最终结果已保存至: {results_path}")
    return results_path


# 训练阶段（保持不变）
def train_epoch(model, train_loader, criterion, optimizer, device):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        optimizer.zero_grad()
        if args.model_name == 's2vnet':
            re_unmix_nonlinear, re_unmix, batch_pred, edm_var_1, edm_var_2, feature_abu, edm_per = model(batch_data)

            band = re_unmix.shape[1] // 2  # 2 represents the number of decoder layer
            output_linear = re_unmix[:, 0:band] + re_unmix[:, band:band * 2]
            re_unmix = re_unmix_nonlinear + output_linear

            # 以下是S2Vnet特有的多损失函数
            # compute kl loss
            kl_div = -0.5 * (edm_var_2 + 1 - edm_var_1 ** 2 - edm_var_2.exp())
            kl_div = kl_div.sum() / batch_pred.shape[0]
            kl_div = torch.max(kl_div, torch.tensor(0).cuda())

            # compute tv loss
            edm_per_diff = edm_per[1:, :] - edm_per[:(edm_per.shape[0] - 1), :]
            edm_per_diff = edm_per_diff.abs()
            loss_tv = edm_per_diff.mean()  # endmember tv_loss

            b_x, h_x, w_x = feature_abu.shape[0], feature_abu.shape[-2], feature_abu.shape[-1]
            h_tv = torch.pow((feature_abu[:, :, 1:, :] - feature_abu[:, :, :h_x - 1, :]), 2).sum()
            w_tv = torch.pow((feature_abu[:, :, :, 1:] - feature_abu[:, :, :, :w_x - 1]), 2).sum()
            loss_tv_abu = (h_tv + w_tv) / (b_x * 2 * h_x * w_x)  # abundance tv_loss

            sad_loss = torch.mean(torch.acos(torch.sum(batch_data * re_unmix, dim=1) /
                                             (torch.norm(re_unmix, dim=1, p=2) * torch.norm(batch_data, dim=1,
                                                                                            p=2) + 1e-5)))
            loss = criterion(batch_pred, batch_target) + sad_loss + 0.01 * kl_div + 0.01 * loss_tv + 0.01 * loss_tv_abu
        elif args.model_name == 'vheat3d':
            # 🆕 vheat3d 只需要分类损失
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)
        elif args.model_name == 'MSF_Heat3D':
            # 只需要分类损失
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)
        else:
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre


# 验证阶段，只计算分类损失和SAD损失；不计算梯度，节省内存；用于模型选择和早停
def valid_epoch(model, valid_loader, criterion, optimizer, device):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        if args.model_name == 's2vnet':
            re_unmix_nonlinear, re_unmix, batch_pred, edm_var_1, edm_var_2, _, _ = model(batch_data)

            band = re_unmix.shape[1] // 2  # 2 represents the number of decoder layer
            output_linear = re_unmix[:, 0:band] + re_unmix[:, band:band * 2]
            re_unmix = re_unmix_nonlinear + output_linear

            sad_loss = torch.mean(torch.acos(torch.sum(batch_data * re_unmix, dim=1) /
                                             (torch.norm(re_unmix, dim=1, p=2) * torch.norm(batch_data, dim=1, p=2))))
            loss = criterion(batch_pred, batch_target) + sad_loss
        elif args.model_name == 'vheat3d':
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)
        elif args.model_name == 'MSF_Heat3D':
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)
        else:
            batch_pred = model(batch_data)
            loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return tar, pre


# 测试阶段，仅进行前向传播获取预测结果；支持分块推理处理大尺寸数据
def test_epoch(model, test_loader, device):
    pre = []  # 存放预测结果
    tar = []  # 存放真实标签

    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)

        # 模型前向传播
        if args.model_name == 's2vnet':
            re_unmix_nonlinear, re_unmix, batch_pred, edm_var_1, edm_var_2, _, _ = model(batch_data)
        elif args.model_name == 'vheat3d':
            batch_pred = model(batch_data)  # 🆕 vheat3d 直接输出分类结果
        elif args.model_name == 'MSF_Heat3D':
            batch_pred = model(batch_data)  # 🆕 vheat3d 直接输出分类结果
        else:
            batch_pred = model(batch_data)

        # 获取预测类别
        _, pred = batch_pred.topk(1, 1, True, True)
        #pp = pred.squeeze()
        # 🆕 修复：确保 pred 保持正确的维度
        if pred.dim() > 1:
            pp = pred.squeeze()
        else:
            pp = pred

        # 🆕 确保 pp 是张量而不是标量
        if pp.dim() == 0:  # 如果是标量
            pp = pp.unsqueeze(0)  # 重新添加批次维度

        # 收集结果
        pre.extend(pp.detach().cpu().numpy().tolist())
        tar.extend(batch_target.detach().cpu().numpy().tolist())

    pre = np.array(pre)
    tar = np.array(tar)

    return pre, tar


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    # 设置 GPU 环境
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        device = torch.device("cuda")
    else:
        print("⚠️ 未检测到 CUDA，使用 CPU 运行")
        device = torch.device("cpu")

    print("✅ 当前使用设备:", device)

    if args.fix_random:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

    # 🆕 创建实验目录和保存配置
    experiment_dir = create_experiment_dir(args.dataset)
    save_experiment_config(args, experiment_dir)

    ## prepare dataset
    # 🆕 对所有数据集统一调用
    label_train_loader, label_test_loader, label_true_loader, band, height, width, num_classes, label, total_pos_true = prepare_dataset(args)

    # create model
    if args.model_name == 's2vnet':
        model = S2VNet(band, num_classes, args.patches)
    elif args.model_name == 'MSF_Heat3D':
        model = MSF_Heat3D(band, num_classes, args.patches, dataset_name=args.dataset,)
    elif args.model_name == 'HybridSN':
        # 与其他模型相同的接口: (band, num_classes, patches)
        model = HybridSN(band, num_classes, args.patches)
    elif args.model_name == "DSNet":
        model = DSNet(band, num_classes, args.patches)
    elif args.model_name == "MASSFormer":
        model = MASSFormer(band, num_classes, args.patches)
    elif args.model_name == "SiT":
        model = SiT(band, num_classes, args.patches)
    elif args.model_name == "HSI2DCNN":
        model = HSI2DCNN(band, num_classes, args.patches)
    elif args.model_name == "vHeat":
        model = S2VHeat(band, num_classes, args.patches)
    elif args.model_name == "MambaHSI":
        model = MambaHSIClassifier(
            band=band,
            num_classes=num_classes,
            hidden_dim=128,
            mamba_type="both",
            token_num=4,
            group_num=4,
            use_residual=True,
            use_att=True,
            pool="gap",
        )
    elif args.model_name == "VMamba":
        model = VisionMambaClassifier(
            band=band,
            num_classes=num_classes,
            img_size=args.patches,  # 与当前 patch 裁剪尺寸一致
            embed_dim=128,
            depth=8,
            d_state=8,
            drop_path_rate=0.6,
            if_abs_pos_embed=True,
            if_rope=False,
            if_cls_token=True,
            use_middle_cls_token=True,
        )
    elif args.model_name == "S2Mamba":
        model = S2Mamba(
            patch=args.patches,  # patch 尺寸
            in_chans=band,  # HSI 波段数
            num_classes=num_classes,  # 类别数
            depths=[1],  # 先用最小配置跑通
            dims=[64],  # hidden dim
            d_state=16,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
        )
    elif args.model_name == 'ViT':
        # 这里 img_size 使用 HSI patch 的空间尺寸 args.patches
        # vit_patch_size 可以先设为 1，表示整个 HSI patch 视作一个 "token 网格"
        model = ViT(
            band=band,
            num_classes=num_classes,
            img_size=args.patches,
            vit_patch_size=1,   # 若想划分更多 patch，可改为 2, 3 等，需保证能整除 img_size
            embed_dim=192,
            depth=6,
            num_heads=3,
        )
    elif args.model_name == 'MorphFormer':
        # 这里 patch_size 使用 args.patches，与数据预处理保持一致
        model = MorphFormer(
            band=band,
            num_classes=num_classes,
            patch_size=args.patches,
            fm=16,  # 可按原始代码调整
            hsi_only=False  # 若原实现始终采用 False，则保持 False
        )
    elif args.model_name == 'SSFTT':
        model = SSFTT(
            band=band,
            num_classes=num_classes,
            patch_size=args.patches,
            num_tokens=4,  # 可按论文或需要调整
            dim=64,
            depth=1,
            heads=8,
            mlp_dim=128,
            dropout=0.1,
            emb_dropout=0.1,
        )
    elif args.model_name == 'HSI3DCNN':
        # 标准 3D-CNN 模型
        model = HSI3DCNN(
            band=band,
            num_classes=num_classes,
            patch_size=args.patches
        )
    else:
        raise KeyError("{} model is unknown.".format(args.model_name))
    model = model.to(device)
    # \* 统计并打印当前模型参数量（单位：K）
    total_params = sum(p.numel() for p in model.parameters())
    total_params_k = total_params / 1e3
    print("Model Name: {}".format(args.model_name))
    print("Total Params: {:.2f}K ({:,} parameters)".format(total_params_k, total_params))
    # \* 统计并打印当前模型计算量（MACs/FLOPs）
    # 构造一个与训练数据一致的单个样本形状的 dummy 输入
    if args.model_name == 'MSF_Heat3D':
        dummy_input = torch.randn(1, band, args.patches, args.patches, device=device)
    elif args.model_name == 's2vnet':
        # 根据 S2VNet 实际输入格式调整，这里假设为 (B, band, H, W)
        dummy_input = torch.randn(1, band, args.patches, args.patches, device=device)
    else:
        dummy_input = torch.randn(1, band, args.patches, args.patches, device=device)

    model.eval()
    # VMamba 会触发 mamba_ssm 的 Triton layer norm 编译，fvcore trace 阶段可能直接报错
    if args.model_name == "VMamba":
        print("Skip MACs/FLOPs for VMamba (Triton kernel is not trace-friendly under fvcore).")
    else:
        try:
            with torch.no_grad():
                flops_analyzer = FlopCountAnalysis(model, (dummy_input,))
                macs = flops_analyzer.total()  # 单位: FLOPs (\~= MACs)
            macs_g = macs / 1e6
            print("Total MACs: {:.2f} M".format(macs_g))
        except Exception as e:
            print(f"Skip MACs/FLOPs due to analysis error: {type(e).__name__}: {e}")
    # criterion
    criterion = nn.CrossEntropyLoss().to(device)
    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # 🆕 修改：只有 s2vnet 需要应用 NonZeroClipper
    if args.model_name == 's2vnet':
        apply_nonegative = NonZeroClipper()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)

    # 🆕 初始化训练日志
    log_path = init_training_log(experiment_dir)

    # -------------------------------------------------------------------------------
    if args.flag_test == 'test':
        print("🚀 Start testing...")
        model.eval()
        ts_start = time.time()

        # ✅ 自动选择模型权重路径（根据当前数据集）
        branch_name = get_git_branch_name()
        weight_dir = f'./results/{branch_name}/{args.dataset}'
        # 查找最新或最优模型
        weight_files = [f for f in os.listdir(weight_dir) if f.endswith('.pkl')]
        if not weight_files:
            raise FileNotFoundError(f"❌ 没有在 {weight_dir} 中找到权重文件。")
        weight_files.sort(key=lambda x: os.path.getmtime(os.path.join(weight_dir, x)), reverse=True)
        weight_path = os.path.join(weight_dir, weight_files[0])
        print(f"✅ 自动加载最新权重: {weight_path}")
        if not os.path.exists(weight_path):
            # 若找不到best模型，可改成你之前保存的具体文件名
            weight_path = 'results/vheat3d-未优化/Berlin/Berlin_Heat3D_Pipeline_p5_72.15_epoch460_2025-11-22-1904.pkl'
        print(f"Loading weights from: {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location=device))

        # ✅ 更新：检查新的内存映射文件路径
        large_file_path = f'./results/memmap/{args.dataset}_true.npy'
        preds_all = []
        tar_t = []

        if os.path.exists(large_file_path):
            print("⚠️ 检测到分块数据文件，启用分块推理模式...")
            # 🚀 更新：直接使用内存映射文件，不需要分块加载
            print(f"🧩 加载内存映射文件: {large_file_path}")
            x_true_memmap = np.load(large_file_path, mmap_mode='r')
            total_samples = x_true_memmap.shape[0]
            print(f"🧩 总样本数: {total_samples}")

            # 设置推理批次大小（可以比训练时大一些，因为只做前向传播）
            inference_batch_size = min(128, total_samples)

            # 分批推理
            for start_idx in range(0, total_samples, inference_batch_size):
                end_idx = min(start_idx + inference_batch_size, total_samples)
                batch_num = (start_idx // inference_batch_size) + 1
                total_batches = (total_samples + inference_batch_size - 1) // inference_batch_size

                print(f"🧩 正在推理第 {batch_num}/{total_batches} 批: [{start_idx}:{end_idx}]")

                # 获取当前批次数据
                batch_data = x_true_memmap[start_idx:end_idx]
                x_tensor = torch.from_numpy(batch_data.transpose(0, 3, 1, 2)).float().to(device)

                with torch.no_grad():
                    if args.model_name == 's2vnet':
                        _, _, batch_pred, _, _, _, _ = model(x_tensor)
                    elif args.model_name == 'vheat3d':
                        batch_pred = model(x_tensor)  # 🆕 vheat3d 直接输出分类结果
                    elif args.model_name == 'MSF_Heat3D':
                        batch_pred = model(x_tensor)  # 直接输出分类结果
                    else:
                        batch_pred = model(x_tensor)

                    preds = torch.argmax(batch_pred, dim=1).cpu().numpy()
                preds_all.append(preds)

            pre_u = np.concatenate(preds_all, axis=0)
            print(f"✅ 分块推理完成，共得到 {len(pre_u)} 个预测结果。")

            # 加载真实标签
            for _, targets in label_true_loader:
                tar_t.extend(targets.numpy())
            tar_t = np.array(tar_t)

        else:
            print("✅ 未检测到内存映射文件，使用 DataLoader 直接推理...")
            with torch.no_grad():
                pre_u, tar_t = test_epoch(model, label_true_loader, device)

        # ✅ 构造预测矩阵并保存
        prediction_matrix = np.zeros((height, width), dtype=float)
        for i in range(total_pos_true.shape[0]):
            prediction_matrix[total_pos_true[i, 0], total_pos_true[i, 1]] = pre_u[i] + 1

        # 🚀 更新：保存文件名包含数据集名称
        output_mat_path = os.path.join(experiment_dir, f'{args.dataset}_classification_map.mat')
        savemat(output_mat_path, {'P': prediction_matrix, 'label': label})

        # ✅ 计算 OA、AA、Kappa
        pre_t = np.array(pre_u)
        OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_t, pre_t, num_classes, args.dataset,'test')

        # 🆕 保存最终测试结果
        save_final_results(experiment_dir, OA2, AA_mean2, Kappa2, AA2, 0)

        # ✅ 打印与保存结果
        print("**************************************************")
        print("Final result:")
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
        print("AA for each class:", AA2)
        print("**************************************************")

        # 🚀 生成唯一文件名，防止被覆盖
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_json_path = os.path.join(experiment_dir, f'{args.dataset}_test_metrics_{timestamp}.json')

        # 🚀 将指标打包为字典
        metrics = {
            "dataset": args.dataset,
            "Overall_Accuracy": float(OA2),
            "Average_Accuracy": float(AA_mean2),
            "Kappa": float(Kappa2),
            "Class_Wise_AA": [float(x) for x in AA2],
            "timestamp": timestamp
        }

        # 🚀 保存为 JSON 文件
        with open(results_json_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        print(f"📊 测试指标已保存为 JSON 文件：{results_json_path}")
        ts_end = time.time()
        ts_time = ts_end - ts_start
        print("TS Time (total test): {:.4f} s".format(ts_time))

    else:
        print("start training")
        tic = time.time()
        min_val_obj, best_OA = 0.5, 0
        best_epoch = 0
        best_AA = 0
        best_Kappa = 0
        best_each_AA = None

        for epoch in range(args.epoches):
            # train model
            model.train()
            train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, criterion, optimizer, device)
            scheduler.step()
            OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t, num_classes, 'train')

            # 🆕 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']

            # 初始化验证指标为None（非验证轮次）
            val_OA, val_AA, val_Kappa = None, None, None

            print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
                  .format(epoch + 1, train_obj, train_acc))
            # 🆕 只有 s2vnet 需要应用 NonZeroClipper
            if args.model_name == 's2vnet':
                model.unmix_decoder.apply(apply_nonegative)  # regularize unmix decoder

            if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
                model.eval()
                tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, optimizer, device)
                OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v, num_classes, 'train')
                print("OA: {:.4f} AA: {:.4f} Kappa: {:.4f}"
                      .format(OA2, AA_mean2, Kappa2))
                print("*************************")

                # 更新验证指标
                val_OA, val_AA, val_Kappa = OA2, AA_mean2, Kappa2

                if OA2 > min_val_obj and epoch > 10:
                    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
                    model_save_path = os.path.join(
                        experiment_dir,
                        f"{args.dataset}_{args.model_name}_p{args.patches}_{round(OA2 * 100, 2)}_epoch{epoch}_{timestamp}.pkl"
                    )
                    torch.save(model.state_dict(), model_save_path)

                    min_val_obj = OA2
                    best_epoch = epoch
                    best_OA = OA2
                    best_AA = AA_mean2
                    best_Kappa = Kappa2
                    best_each_AA = AA2

            # 🆕 记录训练日志（每个epoch都记录）
            log_training_epoch(log_path, epoch, train_obj, train_acc, val_OA, val_AA, val_Kappa, current_lr)

        toc = time.time()
        training_time = toc - tic
        print("Running Time: {:.2f}".format(training_time))
        print("**************************************************")

        # 🆕 保存最终训练结果
        save_final_results(experiment_dir, best_OA, best_AA, best_Kappa, best_each_AA, training_time, best_epoch)

        if best_OA == 0:
            model_save_path = os.path.join(experiment_dir,
                                           f'{args.dataset}_{args.model_name}_p{args.patches}_'
                                           f'{round(OA2 * 100, 2)}_epoch{epoch}.pkl')
            torch.save(model.state_dict(), model_save_path)

    print("**************************************************")
    if args.flag_test == 'test':
        print("🧪 Test finished.")
        print("Final result:")
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
        print("AA for each class:", AA2)
        print(f"Results saved in: {experiment_dir}")
    else:
        print("🏋️ Training finished.")
        print("Best Epoch: {:03d} | Best OA: {:.4f} | Best AA: {:.4f} | Best Kappa: {:.4f}"
              .format(best_epoch, best_OA, best_AA, best_Kappa))
        print("AA for each class:", best_each_AA)
        print(f"Results saved in: {experiment_dir}")

    print("**************************************************")
    print("Parameter:")
    print_args(vars(args))


if __name__ == '__main__':
    main()
