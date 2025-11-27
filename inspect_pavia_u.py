import scipy.io
import numpy as np
import h5py
import os


def analyze_pavia_mat_file(file_path):
    """
    详细分析Pavia.mat文件的内容结构
    """
    print("=" * 60)
    print("Houston.mat 文件结构分析")
    print("=" * 60)

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return None

    print(f"分析文件: {file_path}")
    print(f"文件大小: {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")

    try:
        # 首先尝试用scipy.io.loadmat加载
        print("\n1. 使用 scipy.io.loadmat 加载...")
        mat_data = scipy.io.loadmat(file_path)
        file_type = "传统MAT格式"
        print("✓ 使用scipy.io.loadmat加载成功")

    except NotImplementedError:
        # 如果是HDF5格式
        print("\n1. 使用 h5py 加载 (HDF5格式)...")
        mat_data = h5py.File(file_path, 'r')
        file_type = "HDF5格式"
        print("✓ 使用h5py加载成功")

    except Exception as e:
        print(f"加载失败: {e}")
        return None

    print(f"文件格式: {file_type}")

    # 分析文件内容
    print(f"\n2. 文件内容分析")
    print("-" * 40)

    if file_type == "传统MAT格式":
        analyze_traditional_mat(mat_data)
    else:
        analyze_hdf5_mat(mat_data)

    return mat_data


def analyze_traditional_mat(mat_data):
    """分析传统MAT格式文件"""
    print("文件中的变量:")
    for key in mat_data.keys():
        if not key.startswith('__'):  # 过滤系统变量
            value = mat_data[key]
            if isinstance(value, np.ndarray):
                print(f"  {key}: {value.shape} | {value.dtype} | 数值范围: [{value.min():.2f}, {value.max():.2f}]")

                # 详细分析主要数据
                if value.ndim == 3 and (value.shape[2] > 10):  # 可能是高光谱数据
                    print(f"    → 可能为高光谱数据: {value.shape[2]} 个波段")
                    analyze_hyperspectral_data(value, key)
                elif value.ndim == 2 and value.dtype in [np.uint8, np.int32, np.int64]:
                    print(f"    → 可能为标签数据")
                    analyze_label_data(value, key)
            else:
                print(f"  {key}: {type(value)}")


def analyze_hdf5_mat(mat_data):
    """分析HDF5格式文件"""
    print("文件中的数据集:")

    def print_h5_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: {obj.shape} | {obj.dtype}")
            # 如果是小数据集，可以显示一些统计信息
            if obj.size < 10000:  # 避免处理过大数组
                data = obj[()]
                if isinstance(data, np.ndarray):
                    print(f"    数值范围: [{data.min():.2f}, {data.max():.2f}]")

                    # 分析数据类型
                    if data.ndim == 3 and (data.shape[2] > 10):
                        print(f"    → 可能为高光谱数据: {data.shape[2]} 个波段")
                    elif data.ndim == 2 and data.dtype in [np.uint8, np.int32, np.int64]:
                        print(f"    → 可能为标签数据")
                        unique_vals = np.unique(data)
                        print(f"    唯一值: {unique_vals}")
                        print(f"    类别数: {len(unique_vals)}")

    mat_data.visititems(print_h5_structure)


def analyze_hyperspectral_data(data, key):
    """分析高光谱数据"""
    print(f"\n3. 高光谱数据详细分析 - {key}")
    print("-" * 40)
    print(f"  数据形状: {data.shape} (高度×宽度×波段数)")
    print(f"  数据类型: {data.dtype}")
    print(f"  数值范围: [{data.min():.2f}, {data.max():.2f}]")

    # 统计信息
    print(f"  统计信息:")
    print(f"    均值: {data.mean():.2f}")
    print(f"    标准差: {data.std():.2f}")
    print(f"    中位数: {np.median(data):.2f}")

    # 波段信息
    if data.ndim == 3:
        print(f"  波段数: {data.shape[2]}")
        print(f"  空间尺寸: {data.shape[0]} × {data.shape[1]}")


def analyze_label_data(data, key):
    """分析标签数据"""
    print(f"\n4. 标签数据详细分析 - {key}")
    print("-" * 40)
    print(f"  数据形状: {data.shape}")
    print(f"  数据类型: {data.dtype}")

    unique_vals, counts = np.unique(data, return_counts=True)
    print(f"  唯一值: {unique_vals}")
    print(f"  类别数: {len(unique_vals)}")

    print(f"  各类别像素统计:")
    total_pixels = data.size
    for val, count in zip(unique_vals, counts):
        percentage = (count / total_pixels) * 100
        print(f"    类别 {val}: {count:8d} 像素 ({percentage:6.2f}%)")


def check_data_consistency(mat_data):
    """检查数据一致性"""
    print(f"\n5. 数据一致性检查")
    print("-" * 40)

    # 寻找可能的数据和标签对
    data_candidates = []
    label_candidates = []

    if isinstance(mat_data, dict):
        # 传统MAT格式
        for key in mat_data.keys():
            if not key.startswith('__'):
                value = mat_data[key]
                if isinstance(value, np.ndarray):
                    if value.ndim == 3 and value.shape[2] > 10:
                        data_candidates.append((key, value.shape))
                    elif value.ndim == 2 and value.dtype in [np.uint8, np.int32, np.int64]:
                        label_candidates.append((key, value.shape))
    else:
        # HDF5格式
        for key in mat_data.keys():
            obj = mat_data[key]
            if isinstance(obj, h5py.Dataset):
                if obj.ndim == 3 and obj.shape[2] > 10:
                    data_candidates.append((key, obj.shape))
                elif obj.ndim == 2 and obj.dtype in [np.uint8, np.int32, np.int64]:
                    label_candidates.append((key, obj.shape))

    print("找到的数据候选:")
    for name, shape in data_candidates:
        print(f"  {name}: {shape}")

    print("找到的标签候选:")
    for name, shape in label_candidates:
        print(f"  {name}: {shape}")

    # 检查尺寸匹配
    if data_candidates and label_candidates:
        data_shape = data_candidates[0][1][:2]  # 高和宽
        label_shape = label_candidates[0][1]

        if data_shape == label_shape:
            print(f"✓ 数据和标签尺寸匹配: {data_shape}")
        else:
            print(f"✗ 数据和标签尺寸不匹配: 数据{data_shape} vs 标签{label_shape}")


def get_dataset_summary(mat_data):
    """生成数据集摘要"""
    print(f"\n6. 数据集摘要")
    print("-" * 40)

    if isinstance(mat_data, dict):
        # 传统MAT格式摘要
        data_vars = [key for key in mat_data.keys() if not key.startswith('__')]
        print(f"变量数量: {len(data_vars)}")
        for var in data_vars:
            value = mat_data[var]
            if isinstance(value, np.ndarray):
                print(f"  {var}: {value.shape} | {value.dtype}")
    else:
        # HDF5格式摘要
        datasets = [key for key in mat_data.keys() if isinstance(mat_data[key], h5py.Dataset)]
        print(f"数据集数量: {len(datasets)}")
        for ds in datasets:
            obj = mat_data[ds]
            print(f"  {ds}: {obj.shape} | {obj.dtype}")


# 主执行程序
if __name__ == "__main__":
    # 文件路径
    file_path = "data/Houston.mat"

    # 执行分析
    mat_data = analyze_pavia_mat_file(file_path)

    if mat_data is not None:
        check_data_consistency(mat_data)
        get_dataset_summary(mat_data)

    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)

    # 如果成功加载，提供使用建议
    if mat_data is not None:
        print("\n使用建议:")
        print("1. 根据分析结果确定主要数据变量名")
        print("2. 检查是否需要数据预处理（归一化等）")
        print("3. 根据标签数据确定类别数量")
        print("4. 设计合适的数据划分策略")