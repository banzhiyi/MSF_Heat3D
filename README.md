VHEAT3D 项目使用指南
环境配置
1. 基础环境

    Python 3.9

    安装依赖包：

bash

pip install -r requirements.txt

数据准备
1. 下载数据集

下载项目所需的数据集文件。
2. 目录设置

    在项目根目录下创建 data 文件夹：

bash

mkdir data

    将数据集文件解压后放入 data 文件夹，确保文件路径为：

text

data/IndianPine.mat

项目结构
主要代码文件

    vheat3d_model.py - 模型定义文件

    demo.py - 演示代码

    dataset.py - 数据集处理模块

    util.py - 工具函数模块

使用方法
训练模型
bash

# 添加执行权限
chmod +x run_train.sh

# 运行训练脚本
./run_train.sh

测试模型
bash

# 添加执行权限
chmod +x run_test.sh

# 运行测试脚本
./run_test.sh
