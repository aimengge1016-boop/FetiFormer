# iTransformer（Windows 复现指南）

> 适用场景：你在 Windows（PowerShell）下复现本仓库的脚本结果，或至少把训练/测试流程跑通。

## 0. 进入项目根目录（重要）

你的工作区里存在两层 `iTransformer-main`。请确保**进入包含 `run.py` 的那一层**：

```powershell
cd .\iTransformer-main
ls
# 你应该能看到：run.py / requirements.txt / scripts / model / data_provider ...
```

## 1. Python 版本要求（重要）

本仓库的 `requirements.txt` 固定了：

- `torch==2.0.0`
- `numpy==1.23.5`

它们**不支持 Python 3.11**。因此请使用 **Python 3.10（推荐）** 或 **Python 3.9**。

### 方案 A：conda（推荐）

```powershell
conda create -n itransformer python=3.10 -y
conda activate itransformer
python --version
```

### 方案 B：py launcher + venv（你已安装 Python 3.10 时可用）

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

## 2. 安装依赖

### 2.1 安装 PyTorch

建议从 PyTorch 官方选择对应 CUDA/CPU 的安装命令，然后在当前环境里执行。

### 2.2 安装其余 Python 依赖

为了避免 “torch 安装渠道不同导致 `pip -r` 失败”，仓库提供了一个不包含 torch 的版本：

```powershell
pip install -r .\requirements_notorch.txt
```

如果你明确知道自己能用 `pip install -r requirements.txt` 安装 torch，也可以直接用原始文件：

```powershell
pip install -r .\requirements.txt
```

## 3. 准备数据集（放置目录必须匹配脚本）

作者提供的数据集打包下载链接见主 README（`README.md`）：
- Google Drive / 百度网盘

下载并解压后，将数据放到如下结构（路径大小写保持一致）：

```text
iTransformer-main/
  dataset/
    traffic/
      traffic.csv
    electricity/
      electricity.csv
    weather/
      weather.csv
    exchange_rate/
      exchange_rate.csv
    ETT-small/
      ETTh1.csv
      ETTh2.csv
      ETTm1.csv
      ETTm2.csv
    PEMS/
      PEMS03.npz
      PEMS04.npz
      PEMS07.npz
      PEMS08.npz
    solar/
      solar_AL.txt
```

### CSV 格式要求（避免最常见的报错）

对 `custom` / ETT 类 CSV（例如 `traffic.csv`, `electricity.csv`）：

- **必须有一列名为 `date`**
- 其它列为时间序列变量（数值）
- `target` 默认是 `OT`（只有 `features=S` / `MS` 才会用到）；`features=M` 时会使用所有变量列

你可以用下面的脚本做快速自检：

```powershell
python .\tools\check_dataset.py --root .\dataset
```

## 4. 跑一个最小可复现示例（PowerShell）

Windows 下不能直接 `bash xxx.sh`，但 `.sh` 本质只是 `python run.py ...` 的参数集合。

仓库已补充了 PowerShell 脚本（见 `scripts/windows/`），例如 Traffic：

```powershell
.\scripts\windows\run_traffic_itransformer.ps1 -Gpu 0 -TrainEpochs 10
```

训练结束后，你会看到这些输出：

- `./checkpoints/<setting>/checkpoint.pth`
- `./results/<setting>/metrics.npy`（以及 `pred.npy`, `true.npy`）
- 根目录追加写入 `result_long_term_forecast.txt`（mse/mae）

## 5. Smoke Test（无真实数据也能验证环境）

如果你暂时还没下载数据，可以先跑一个会自动生成小 CSV 的 smoke test：

```powershell
python .\tools\smoke_test.py
```

它会在 `./dataset/_smoke/` 下生成数据并训练 1 个 epoch，用于验证：
- 依赖是否安装正确
- 训练/测试流程是否能完整跑通


