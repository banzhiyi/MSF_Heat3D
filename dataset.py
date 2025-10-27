import torch
import torch.utils.data as Data
from scipy.io import loadmat
import numpy as np
import os


def prepare_dataset(args, samples_type='ratio'):
    # prepare data
    if args.dataset == 'Indian':
        data = loadmat('./data/IndianPine.mat')
        TR = data['TR']
        TE = data['TE']
        input = data['input']  # (145,145,200)
    elif args.dataset == 'Berlin':
        data = loadmat('./data/Berlin/data_HS_LR.mat')
        data_train = loadmat('./data/Berlin/TrainImage.mat')
        data_test = loadmat('./data/Berlin/TestImage.mat')
        TR = data_train['TrainImage']
        TE = data_test['TestImage']
        input = data['data_HS_LR']  # (1723,476,244)
    elif args.dataset == 'Augsburg':
        data = loadmat('./data/Augsburg/data_HS_LR.mat')
        data_train = loadmat('./data/Augsburg/TrainImage.mat')
        data_test = loadmat('./data/Augsburg/TestImage.mat')
        TR = data_train['TrainImage']
        TE = data_test['TestImage']
        input = data['data_HS_LR']  # (332,485,180)
    else:
        raise ValueError("Unknown dataset")

    label = TR + TE
    num_classes = np.max(TR)
    # train data change to the ratio of train samples
    if samples_type == 'ratio':
        training_ratio = 1  # range from 0 to 1, e.g. training_ratio=0.5 means 50% training samples.
        print('Train data change to the ratio of train samples: {}'.format(training_ratio))
        train_idx, TR = split_train_data_clssnum(TR, num_classes, training_ratio)

    # normalize data by band norm
    input_normalize = np.zeros(input.shape)
    for i in range(input.shape[2]):
        input_max = np.max(input[:, :, i])
        input_min = np.min(input[:, :, i])
        input_normalize[:, :, i] = (input[:, :, i] - input_min) / (input_max - input_min)
    # data size
    height, width, band = input.shape
    print("height={0},width={1},band={2}".format(height, width, band))
    # -------------------------------------------------------------------------------
    # obtain train and test data
    total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = chooose_train_and_test_point(
        TR, TE, label, num_classes)
    mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)

    # 🚀 关键修改：智能选择数据加载策略
    x_train_band, x_test_band, x_true_band = train_and_test_data_optimized(
        mirror_image, band, total_pos_train, total_pos_test,
        patch=args.patches, true_point=total_pos_true, dataset_name=args.dataset
    )

    y_train, y_test, y_true = train_and_test_label(number_train, number_test, num_classes, number_true)

    # 🚀 关键修改：统一的数据加载器创建
    label_train_loader, label_test_loader, label_true_loader = create_universal_dataloaders(
        x_train_band, x_test_band, x_true_band, y_train, y_test, y_true, args.batch_size, args.dataset
    )

    print(f"✅ Dataset band number = {band}")
    return label_train_loader, label_test_loader, label_true_loader, band, height, width, num_classes, label, total_pos_true


# 🚀 新增：统一的数据加载器创建函数
def create_universal_dataloaders(x_train_band, x_test_band, x_true_band, y_train, y_test, y_true, batch_size,
                                 dataset_name):
    """统一的数据加载器创建，兼容所有数据集"""

    # 训练集：总是使用原始Tensor方式（确保性能）
    if isinstance(x_train_band, np.ndarray):
        x_train = torch.from_numpy(x_train_band.transpose(0, 3, 1, 2)).float()
    else:
        # 如果是文件路径，加载为数组
        x_train_band_data = np.load(x_train_band, mmap_mode='r')
        x_train = torch.from_numpy(x_train_band_data.transpose(0, 3, 1, 2)).float()

    y_train_tensor = torch.from_numpy(y_train).long()
    train_dataset = Data.TensorDataset(x_train, y_train_tensor)
    label_train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"✅ 训练集加载完成: {len(train_dataset)} 个样本")

    # 测试集：根据数据集大小选择策略
    test_samples = len(y_test)
    if test_samples < 10000:  # 小数据集使用原始方式
        if isinstance(x_test_band, np.ndarray):
            x_test = torch.from_numpy(x_test_band.transpose(0, 3, 1, 2)).float()
        else:
            x_test_band_data = np.load(x_test_band, mmap_mode='r')
            x_test = torch.from_numpy(x_test_band_data.transpose(0, 3, 1, 2)).float()

        y_test_tensor = torch.from_numpy(y_test).long()
        test_dataset = Data.TensorDataset(x_test, y_test_tensor)
        print(f"✅ 测试集加载完成 (原始方式): {len(test_dataset)} 个样本")
    else:  # 大数据集使用内存优化方式
        test_dataset = UniversalDataset(x_test_band, y_test, dataset_name + '_test')
        print(f"✅ 测试集加载完成 (内存优化): {len(test_dataset)} 个样本")

    label_test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 全图数据：根据数据集大小选择策略
    true_samples = len(y_true) if y_true is not None else 0
    if true_samples < 50000:  # 小数据集使用原始方式
        if isinstance(x_true_band, np.ndarray):
            x_true = torch.from_numpy(x_true_band.transpose(0, 3, 1, 2)).float()
        else:
            x_true_band_data = np.load(x_true_band, mmap_mode='r')
            x_true = torch.from_numpy(x_true_band_data.transpose(0, 3, 1, 2)).float()

        y_true_tensor = torch.from_numpy(y_true).long()
        true_dataset = Data.TensorDataset(x_true, y_true_tensor)
        print(f"✅ 全图数据加载完成 (原始方式): {len(true_dataset)} 个样本")
    else:  # 大数据集使用内存优化方式
        true_dataset = UniversalDataset(x_true_band, y_true, dataset_name + '_true')
        print(f"✅ 全图数据加载完成 (内存优化): {len(true_dataset)} 个样本")

    label_true_loader = Data.DataLoader(true_dataset, batch_size=batch_size, shuffle=False)

    return label_train_loader, label_test_loader, label_true_loader


# 🚀 新增：统一的数据集类
class UniversalDataset(Data.Dataset):
    """统一的数据集类，自动处理数组和文件路径"""

    def __init__(self, data_source, labels, identifier):
        self.identifier = identifier
        self.labels = torch.from_numpy(labels).long()

        # 自动检测数据类型
        if isinstance(data_source, np.ndarray):
            # 直接使用数组
            self.data = data_source
            self.use_memmap = False
            print(f"📁 {identifier}: 使用数组数据，形状: {self.data.shape}")
        elif isinstance(data_source, str) and os.path.exists(data_source):
            # 使用内存映射文件
            self.data_path = data_source
            self.use_memmap = True
            self.data = np.load(data_source, mmap_mode='r')
            print(f"📁 {identifier}: 使用内存映射文件，形状: {self.data.shape}")
        else:
            raise ValueError(f"不支持的数据源类型: {type(data_source)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.use_memmap:
            # 内存映射文件：复制数据确保可写
            sample = self.data[idx].copy()
        else:
            # 普通数组：直接访问
            sample = self.data[idx]

        # 转换维度: (H, W, C) -> (C, H, W)
        sample_tensor = torch.from_numpy(sample.transpose(2, 0, 1)).float()
        label = self.labels[idx]

        return sample_tensor, label


# 🚀 新增：内存优化的数据加载器创建函数
def create_memory_efficient_dataloaders(x_train_band, x_test_band, x_true_band, y_train, y_test, y_true, batch_size,
                                        dataset_name):
    """为大数据集创建内存优化的数据加载器"""

    # 训练集通常较小，可以直接加载
    if isinstance(x_train_band, np.ndarray):
        x_train = torch.from_numpy(x_train_band.transpose(0, 3, 2, 1)).float()
        y_train_tensor = torch.from_numpy(y_train).long()
        train_dataset = Data.TensorDataset(x_train, y_train_tensor)
        label_train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        # 如果训练集也很大，使用自定义数据集
        train_dataset = MemmapDataset(x_train_band, y_train, dataset_name + '_train')
        label_train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 🚀 测试集使用内存映射加载
    test_dataset = MemmapDataset(x_test_band, y_test, dataset_name + '_test')
    label_test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 🚀 全图数据使用内存映射加载
    true_dataset = MemmapDataset(x_true_band, y_true, dataset_name + '_true')
    label_true_loader = Data.DataLoader(true_dataset, batch_size=batch_size, shuffle=False)

    return label_train_loader, label_test_loader, label_true_loader


# 🚀 新增：内存映射数据集类
class MemmapDataset(Data.Dataset):
    """处理大型数据集的记忆映射数据集类"""

    def __init__(self, data_path_or_array, labels, identifier):
        self.identifier = identifier

        if isinstance(data_path_or_array, str) and os.path.exists(data_path_or_array):
            # 从内存映射文件加载
            self.data = np.load(data_path_or_array, mmap_mode='r')
            print(f"📁 {identifier}: 使用内存映射加载，形状: {self.data.shape}")
        elif isinstance(data_path_or_array, np.ndarray):
            # 小数据集直接使用数组
            self.data = data_path_or_array
            print(f"📁 {identifier}: 直接加载数组，形状: {self.data.shape}")
        else:
            # 其他情况，尝试作为路径处理
            self.data = np.load(data_path_or_array, mmap_mode='r')
            print(f"📁 {identifier}: 使用内存映射加载，形状: {self.data.shape}")

        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 🚀 关键：只在需要时加载单个样本，而不是整个数据集
        if isinstance(self.data, np.memmap):
            # 内存映射文件，直接索引
            sample = self.data[idx].copy()
        else:
            # 普通数组
            sample = self.data[idx]

        # 转换维度: (H, W, C) -> (C, H, W)
        sample_tensor = torch.from_numpy(sample.transpose(2, 0, 1)).float()
        label = self.labels[idx]

        return sample_tensor, label


# 🚀 修改：优化的数据准备函数
def train_and_test_data_optimized(mirror_image, band, train_point, test_point, patch=5, true_point=None,
                                  dataset_name='unknown', chunk_size=10000):
    """内存优化的数据准备函数"""

    # 创建结果目录
    os.makedirs('./results/memmap', exist_ok=True)

    # ========== 1️⃣ 训练数据 ==========
    train_samples = train_point.shape[0]
    train_memory_gb = train_samples * patch * patch * band * 4 / (1024 ** 3)

    if train_memory_gb > 2:  # 训练集大于2GB时使用内存映射
        print(f"⚠️ 训练数据较大 ({train_memory_gb:.2f} GB)，启用内存映射...")
        x_train_path = f'./results/memmap/{dataset_name}_train.npy'
        x_train = create_memmap_data(x_train_path, mirror_image, train_point, patch, band, '训练集')
    else:
        x_train = create_array_data(mirror_image, train_point, patch, band, '训练集')

    # ========== 2️⃣ 测试数据 ==========
    test_samples = test_point.shape[0]
    test_memory_gb = test_samples * patch * patch * band * 4 / (1024 ** 3)

    if test_memory_gb > 1:  # 测试集大于1GB时使用内存映射
        print(f"⚠️ 测试数据较大 ({test_memory_gb:.2f} GB)，启用内存映射...")
        x_test_path = f'./results/memmap/{dataset_name}_test.npy'
        x_test = create_memmap_data(x_test_path, mirror_image, test_point, patch, band, '测试集', chunk_size)
    else:
        x_test = create_array_data(mirror_image, test_point, patch, band, '测试集')

    # ========== 3️⃣ 全图数据 ==========
    if true_point is not None and true_point.size > 0:
        true_samples = true_point.shape[0]
        true_memory_gb = true_samples * patch * patch * band * 4 / (1024 ** 3)

        if true_memory_gb > 1:  # 全图数据大于1GB时使用内存映射
            print(f"⚠️ 全图数据较大 ({true_memory_gb:.2f} GB)，启用内存映射...")
            x_true_path = f'./results/memmap/{dataset_name}_true.npy'
            x_true = create_memmap_data(x_true_path, mirror_image, true_point, patch, band, '全图数据', chunk_size)
        else:
            x_true = create_array_data(mirror_image, true_point, patch, band, '全图数据')
    else:
        x_true = None

    print("**************************************************")
    return x_train, x_test, x_true


def create_memmap_data(file_path, mirror_image, points, patch, band, data_name, chunk_size=10000):
    """创建内存映射数据文件"""

    if os.path.exists(file_path):
        print(f"🔁 {data_name}: 复用现有内存映射文件 {file_path}")
        return file_path

    total_points = points.shape[0]
    print(f"🔄 {data_name}: 创建内存映射文件，样本数: {total_points}")

    # 创建内存映射文件
    memmap_file = np.lib.format.open_memmap(
        file_path, mode='w+', dtype=np.float32,
        shape=(total_points, patch, patch, band)
    )

    # 分块处理
    num_chunks = (total_points + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_points)
        print(f"  处理 {data_name} 第 {i + 1}/{num_chunks} 块: [{start}:{end}]")

        # 处理当前块
        chunk_data = np.zeros((end - start, patch, patch, band), dtype=np.float32)
        for k in range(end - start):
            chunk_data[k] = gain_neighborhood_pixel(mirror_image, points, start + k, patch)

        # 写入内存映射文件
        memmap_file[start:end] = chunk_data
        del chunk_data  # 及时释放内存

    print(f"✅ {data_name}: 内存映射文件创建完成: {file_path}")
    return file_path


def create_array_data(mirror_image, points, patch, band, data_name):
    """创建普通数组数据（用于小数据集）"""
    total_points = points.shape[0]
    print(f"🔄 {data_name}: 创建数组数据，样本数: {total_points}")

    data = np.zeros((total_points, patch, patch, band), dtype=np.float32)
    for i in range(total_points):
        data[i] = gain_neighborhood_pixel(mirror_image, points, i, patch)

    print(f"✅ {data_name}: 数组数据创建完成，形状: {data.shape}")
    return data


# 以下函数保持不变（split_train_data_clssnum, chooose_train_and_test_point,
# mirror_hsi, gain_neighborhood_pixel, train_and_test_label）
# 为了简洁，这里省略重复代码，您保留原有的这些函数即可

# split dataset by training set ratio
def split_train_data_clssnum(gt, num_classes, train_num_ratio):
    train_idx = []

    TR = np.zeros_like(gt)
    for i in range(num_classes):
        idx = np.argwhere(gt == i + 1)
        samplesCount = len(idx)
        sample_num = np.ceil(train_num_ratio * samplesCount).astype('int32')
        train_idx.append(idx[: sample_num])

        for j in range(sample_num):
            TR[idx[j, 0], idx[j, 1]] = i + 1

    train_idx = np.concatenate(train_idx, axis=0)
    return train_idx, TR


# 定位训练和测试样本
def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}

    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data == (i + 1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]
    total_pos_train = total_pos_train.astype(int)

    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data == (i + 1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]
    total_pos_test = total_pos_test.astype(int)

    for i in range(num_classes + 1):
        each_class = []
        each_class = np.argwhere(true_data == i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes + 1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true


# 边界拓展：镜像
def mirror_hsi(height, width, band, input_normalize, patch=5):
    padding = patch // 2
    mirror_hsi = np.zeros((height + 2 * padding, width + 2 * padding, band), dtype=float)
    mirror_hsi[padding:(padding + height), padding:(padding + width), :] = input_normalize
    for i in range(padding):
        mirror_hsi[padding:(height + padding), i, :] = input_normalize[:, padding - i - 1, :]
    for i in range(padding):
        mirror_hsi[padding:(height + padding), width + padding + i, :] = input_normalize[:, width - 1 - i, :]
    for i in range(padding):
        mirror_hsi[i, :, :] = mirror_hsi[padding * 2 - i - 1, :, :]
    for i in range(padding):
        mirror_hsi[height + padding + i, :, :] = mirror_hsi[height + padding - 1 - i, :, :]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0], mirror_hsi.shape[1], mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi


# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i, 0]
    y = point[i, 1]
    temp_image = mirror_image[x:(x + patch), y:(y + patch), :]
    return temp_image


# 标签y_train, y_test
def train_and_test_label(number_train, number_test, num_classes, number_true=None):
    y_train = []
    y_test = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print("y_train: shape = {} ,type = {}".format(y_train.shape, y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape, y_test.dtype))

    if number_true != None:
        y_true = []
        for i in range(num_classes + 1):
            for j in range(number_true[i]):
                y_true.append(i)
        y_true = np.array(y_true)
        print("y_true: shape = {} ,type = {}".format(y_true.shape, y_true.dtype))
        print("**************************************************")
        return y_train, y_test, y_true
    else:
        print("**************************************************")
        return y_train, y_test
