#!/bin/bash
# py_runc_env.sh

# 设置YML文件名
YML_NAME="py_runc_env.yml"

# 检查YAML文件是否存在
if [ ! -f "$YML_NAME" ]; then
    echo "错误: YAML文件 '$YML_NAME' 未找到，请确保文件在当前目录。"
    exit 1
fi

# 从YAML提取环境名称（增强格式兼容性）
ENV_NAME=$(grep -E '^name:' "$YML_NAME" | sed -E 's/^name:[[:space:]]*//')
if [ -z "$ENV_NAME" ]; then
    echo "错误: 无法从 $YML_NAME 中解析环境名称，请检查YAML格式。"
    exit 1
fi

# 检查Conda是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: Conda未安装，请先安装Conda。"
    exit 1
fi

# 获取Conda路径并验证
CONDA_PATH=$(dirname "$(dirname "$(which conda)")")
if [ ! -d "$CONDA_PATH" ]; then
    echo "错误: 无法定位Conda安装路径，请检查Conda配置。"
    exit 1
fi

# Conda版本检查（支持--prune的最低版本）
REQUIRED_VERSION="4.9"
CONDA_VERSION=$(conda --version | awk '{print $2}')
if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$CONDA_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "错误: Conda版本需≥$REQUIRED_VERSION，当前版本: $CONDA_VERSION"
    exit 1
fi

# 初始化Conda Shell
eval "$("$CONDA_PATH/bin/conda" shell.bash hook 2> /dev/null)"
if [ $? -ne 0 ] && [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
    . "$CONDA_PATH/etc/profile.d/conda.sh"
fi

# 环境管理逻辑
if conda info --envs | grep -wq "$ENV_NAME"; then
    echo "发现已存在环境 '$ENV_NAME'，开始清理旧环境..."
    if ! conda env remove --name "$ENV_NAME" --yes; then
        echo "错误: 无法删除环境 '$ENV_NAME'，请检查权限或依赖问题。"
        exit 1
    fi
fi

# 创建新环境
echo "正在创建全新环境 '$ENV_NAME'..."
if ! conda env create -f "$YML_NAME"; then
    echo "错误: 环境创建失败，请检查 $YML_NAME 文件配置。"
    exit 1
fi

# 环境激活验证
if ! conda activate "$ENV_NAME"; then
    echo "错误: 无法激活环境 '$ENV_NAME'，请手动排查问题。"
    exit 1
fi
echo "环境 '$ENV_NAME' 已成功激活。"

# 设置HF镜像端点（按需启用）
# export HF_ENDPOINT=https://hf-mirror.com

# 执行提示
echo -e "\n操作成功！建议通过以下命令手动激活环境："
echo -e "source \"$CONDA_PATH/etc/profile.d/conda.sh\" && conda activate $ENV_NAME\n"
