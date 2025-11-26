# 虚拟环境设置指南

## 重要提示

**所有 `pip install` 操作必须安装到虚拟环境中，而不是系统 Python！**

## 自动设置（推荐）

### 方法 1：使用 setup 脚本（自动创建虚拟环境并安装依赖）

**Windows PowerShell:**
```powershell
.\scripts\setup.ps1
```

**Linux/Mac Bash:**
```bash
bash scripts/setup.sh
```

这些脚本会：
1. 自动检测并使用 `uv`、`pdm` 或 `venv` 创建虚拟环境
2. 使用虚拟环境中的 `pip` 安装所有依赖
3. 虚拟环境默认位置：`.venv/`

### 方法 2：使用快速启动脚本（自动检查并设置）

**Windows PowerShell:**
```powershell
.\scripts\run_qwen.ps1
```

**Linux/Mac Bash:**
```bash
bash scripts/run_qwen.sh
```

这些脚本会：
1. 检查虚拟环境是否存在
2. 如果不存在，自动创建并安装依赖
3. 然后运行 Qwen 微调

### 方法 3：使用 Python 安装脚本

```bash
python install_deps.py
```

这个脚本会：
1. 检查虚拟环境是否存在
2. 使用虚拟环境中的 `pip` 安装依赖
3. 如果虚拟环境不存在，会提示你先运行 setup 脚本

## 手动设置

### 1. 创建虚拟环境

**Windows:**
```powershell
python -m venv .venv
```

**Linux/Mac:**
```bash
python3 -m venv .venv
```

### 2. 激活虚拟环境

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows CMD:**
```cmd
.venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### 3. 安装依赖（现在 pip 指向虚拟环境）

```bash
pip install -r requirements.txt
```

### 4. 验证安装位置

```bash
# 检查 pip 位置（应该指向虚拟环境）
which pip  # Linux/Mac
where pip   # Windows

# 检查 Python 位置（应该指向虚拟环境）
which python  # Linux/Mac
where python  # Windows
```

## 常见问题

### Q: 如何确认我在虚拟环境中？

**A:** 检查命令提示符：
- Windows PowerShell: `(.venv) PS C:\...>`
- Linux/Mac: `(.venv) $`

或者检查环境变量：
```bash
echo $VIRTUAL_ENV  # Linux/Mac
echo $env:VIRTUAL_ENV  # Windows PowerShell
```

### Q: 如果我不小心在系统 Python 中安装了包怎么办？

**A:** 
1. 激活虚拟环境
2. 重新运行 `pip install -r requirements.txt`（会安装到虚拟环境）
3. 系统包可以保留，不会影响虚拟环境

### Q: 如何删除虚拟环境重新开始？

**A:**
```bash
# 删除虚拟环境目录
rm -rf .venv  # Linux/Mac
Remove-Item -Recurse -Force .venv  # Windows PowerShell

# 然后重新运行 setup 脚本
```

### Q: 虚拟环境在哪里？

**A:** 默认在项目根目录的 `.venv/` 文件夹中。

## 最佳实践

1. **始终使用 setup 脚本**：它们确保依赖安装到正确的位置
2. **运行前检查虚拟环境**：使用 `run_qwen_finetune.py` 会自动检查
3. **不要使用 `sudo pip install`**：这会安装到系统 Python
4. **使用虚拟环境中的 Python**：运行脚本时使用 `.venv/bin/python` 或 `.venv/Scripts/python.exe`

## 验证安装

运行以下命令验证所有依赖都已正确安装到虚拟环境：

```bash
# 激活虚拟环境后
python -c "import torch; import transformers; import peft; print('✓ 所有依赖已安装')"
```

