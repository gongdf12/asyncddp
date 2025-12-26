#!/bin/sh
set -eu

log() { printf '%s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"; }

if [ -z "${CONDA_PREFIX:-}" ]; then
  die "CONDA_PREFIX is not set. Activate a conda env first (e.g. 'conda activate <env>')."
fi

# Prefer mamba if available.
if [ -n "${CONDA_CMD:-}" ]; then
  CONDA_BIN="$CONDA_CMD"
elif command -v mamba >/dev/null 2>&1; then
  CONDA_BIN=mamba
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN=conda
else
  die "Cannot find conda (or mamba) on PATH."
fi

need_cmd "$CONDA_BIN"

conda_pkg_installed() {
  pkg="$1"
  "$CONDA_BIN" list 2>/dev/null | awk -v p="$pkg" '($1==p){found=1} END{exit found?0:1}'
}

conda_pkg_version() {
  pkg="$1"
  "$CONDA_BIN" list "$pkg" 2>/dev/null | awk -v p="$pkg" '($1==p){print $2; exit 0}'
}

conda_install() {
  channel="$1"; shift
  log "> Installing: $* (channel: $channel)"
  "$CONDA_BIN" install -y -c "$channel" "$@"
}

log "> CONDA_PREFIX=$CONDA_PREFIX"
log "> Using conda frontend: $CONDA_BIN"

# 1) NCCL (from nvidia channel)
if conda_pkg_installed nccl; then
  log "> conda package already present: nccl $(conda_pkg_version nccl || true)"
else
  conda_install nvidia nccl
fi

# 2) FlatBuffers (force 1.12.0 as requested)
fb_ver="$(conda_pkg_version flatbuffers || true)"
if [ "$fb_ver" = "1.12.0" ]; then
  log "> conda package already present: flatbuffers $fb_ver"
else
  if [ -n "$fb_ver" ]; then
    log "> flatbuffers version is $fb_ver; switching to 1.12.0"
  fi
  conda_install conda-forge "flatbuffers=1.12.0"
fi

# 3) Boost (conda-forge)
if conda_pkg_installed boost; then
  log "> conda package already present: boost $(conda_pkg_version boost || true)"
else
  conda_install conda-forge boost
fi

# 4) OpenMPI 4.* (conda-forge)
ompi_ver="$(conda_pkg_version openmpi || true)"
case "${ompi_ver:-}" in
  4.*)
    log "> conda package already present: openmpi $ompi_ver"
    ;;
  "")
    conda_install conda-forge "openmpi=4.*"
    ;;
  *)
    log "> openmpi version is $ompi_ver; switching to 4.*"
    conda_install conda-forge "openmpi=4.*"
    ;;
esac

log "> Versions (conda list):"
"$CONDA_BIN" list 2>/dev/null | grep -E '^(nccl|flatbuffers|boost|openmpi)\s' || true

# Verify headers/libs exist under this conda env.
NCCL_H="$CONDA_PREFIX/include/nccl.h"
FB_H="$CONDA_PREFIX/include/flatbuffers/flatbuffers.h"

[ -f "$NCCL_H" ] || die "Missing NCCL header: $NCCL_H"
[ -f "$FB_H" ] || die "Missing FlatBuffers header: $FB_H"

NCCL_LIBDIR=""
for d in "$CONDA_PREFIX/lib" "$CONDA_PREFIX/lib64"; do
  if [ -d "$d" ] && ls "$d"/libnccl.so* >/dev/null 2>&1; then
    NCCL_LIBDIR="$d"
    break
  fi
done

[ -n "$NCCL_LIBDIR" ] || die "Missing NCCL shared library under $CONDA_PREFIX/lib or $CONDA_PREFIX/lib64 (expected libnccl.so*)"

log "> Found: $NCCL_H"
log "> Found: $FB_H"
log "> Found: $NCCL_LIBDIR/libnccl.so*"
# 4. 升级构建工具 (防止 Python 3.13 下的兼容性问题)
echo "> 升级构建工具: setuptools, wheel..."
pip install --upgrade setuptools wheel

# 自动定位 Torch 路径
TORCH_PATH=$(python -c 'import torch; import os; print(os.path.dirname(torch.__file__))')

# 1. 加入头文件搜索路径 (系统级)
export CPATH="$TORCH_PATH/include:$TORCH_PATH/include/torch/csrc/api/include:${CPATH:-}"

# 2. 加入库文件链接路径 (系统级)
export LIBRARY_PATH="$TORCH_PATH/lib:${LIBRARY_PATH:-}"

# 3. 加入动态库运行路径 (运行时)
export LD_LIBRARY_PATH="$TORCH_PATH/lib:${LD_LIBRARY_PATH:-}"

# 💡 验证是否设置成功
echo "CPATH is set to: $CPATH"
# Env vars for building BlueFog with NCCL.
export BLUEFOG_WITH_NCCL=1
export BLUEFOG_NCCL_LINK=SHARED
export BLUEFOG_NCCL_HOME="$CONDA_PREFIX"
export BLUEFOG_NCCL_INCLUDE="$CONDA_PREFIX/include"
export BLUEFOG_NCCL_LIB="$NCCL_LIBDIR"
# 2. 获取 Torch 的库路径 (这是最关键的一步)


# 2. 获取 Torch 路径（增加错误检查）
# 使用 python 命令获取路径，如果 torch 没安装，这里会报错并退出

# --- 2. 动态获取 Torch 路径 ---
if ! python -c "import torch" &> /dev/null; then
    echo "ERROR: PyTorch not found in the current python environment."
    exit 1
fi

# 获取 Torch 的 C++ 头文件路径
# TORCH_INC=$(python -c 'import torch; from torch.utils.cpp_extension import include_paths; print(":".join(include_paths()))')
# # 获取 Torch 的库文件路径 (包含 libtorch_python.so)
# TORCH_LIB=$(python -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
# Help compilers find headers/libs.
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH:-}"
export LD_LIBRARY_PATH="$NCCL_LIBDIR:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="$(python -c 'import torch; from torch.utils.cpp_extension import include_paths; print(":".join(include_paths()))'):${CPLUS_INCLUDE_PATH:-}"
# 5. 执行 pip 安装
# 2. 合并设置头文件路径 (包含 Torch 和 Conda 环境路径)
# export CPLUS_INCLUDE_PATH="${TORCH_INC}:$CONDA_PREFIX/include:${CPLUS_INCLUDE_PATH:-}"

# # 3. 设置库文件路径 (必须包含 Torch 的 lib 目录，否则链接时会找不到 libtorch_python.so)
# export LIBRARY_PATH="$TORCH_LIB:$NCCL_LIBDIR:${LIBRARY_PATH:-}"
# export LD_LIBRARY_PATH="$TORCH_LIB:$NCCL_LIBDIR:${LD_LIBRARY_PATH:-}"

# 4. 执行安装
#echo #echo "> Running pip install with torch library path: $TORCH_LIB
if [ ! -f "setup.py" ]; then
    echo "ERROR: setup.py not found in current directory."
    echo "Please run this script from the project root directory."
    exit 1
fi

echo "> Running pip install..."

python -m pip install -e . --no-build-isolation --no-cache-dir
