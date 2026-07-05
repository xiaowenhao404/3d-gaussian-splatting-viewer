# 3D-Reconstruction · 基于 3D Gaussian Splatting 的交互式三维重建与可视化系统

> 计算机视觉课程设计。从一组照片重建出三维高斯模型，并在浏览器中以可拖拽旋转 / 自由翻滚 / 第一人称漫游的方式实时查看。

![status](https://img.shields.io/badge/status-v1.0-brightgreen) ![license](https://img.shields.io/badge/license-MIT-blue)

---

## 一、这个系统做什么

一句话：**一堆照片 → 三维模型 → 网页里实时拖着看**。

- **选内置数据集 / 上传自己的照片** → 一键发起三维重建
- **可视化进度条**：SfM（求相机位姿）→ 训练 → 导出 三阶段实时进度（SSE 推送）
- **浏览器内三种查看方式**：环视（绕物体）/ 自由翻滚 / 漫游（WASD 第一人称，适合室内）
- **产物管理**：结果可存到根目录或自定义路径；历史 `.ply/.splat/.ksplat` 可再导入直接看（无需 GPU）
- **引擎/参数对比页**：并排对比两个模型 + 训练步数、MCMC 正则项的消融实验数据

### 三个核心概念（先搞懂再用）

| 概念 | 是什么 | 好比 |
|---|---|---|
| **数据集 (Dataset)** | 一组照片（原料，还没重建） | 一叠从不同角度拍的照片 |
| **模型 (Model)** | 重建完成的 3D 成品（`.ply` 文件） | 照片变成的可旋转 3D 物体 |
| **重建任务 (Job)** | 把数据集训练成模型的过程，需几分钟 | 照片 → 3D 的"加工"过程 |

> 数据集不能直接看，要先"重建"成模型；模型才能拖着看。

---

## 二、快速开始

### 前置环境（本项目已在此环境验证）
- **GPU**：NVIDIA 独显（本项目在 RTX 4070 Laptop 8GB 上验证），CUDA 11.8
- **Python 3.10**（gsplat 预编译 wheel 仅提供 cp310）
- **Node.js ≥ 18**
- **COLMAP**：Windows 官方二进制（已放于 `third_party/colmap_dist/`）

### 一键启动（推荐）
双击项目根目录的 **`start.bat`** —— 自动启动后端 + 前端并打开浏览器。

停止：关掉弹出的两个黑窗口即可。

### 手动启动
```bash
# 后端 (:8000)
cd backend
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000 --app-dir .

# 前端 (:5173)
cd frontend
npm run dev
```
打开 http://localhost:5173

---

## 三、从零安装（换机部署）

### 后端
```powershell
# 1. 安装 uv 并创建 Python 3.10 环境
pip install uv
uv venv backend/.venv --python 3.10

# 2. 安装 PyTorch cu118（约 2.5GB）
uv pip install --python backend/.venv/Scripts/python.exe `
  torch==2.4.1+cu118 torchvision==0.19.1+cu118 `
  --index-url https://download.pytorch.org/whl/cu118

# 3. 安装 gsplat 预编译 wheel（零编译，关键！）
uv pip install --python backend/.venv/Scripts/python.exe `
  "https://github.com/nerfstudio-project/gsplat/releases/download/v1.5.2/gsplat-1.5.2%2Bpt24cu118-cp310-cp310-win_amd64.whl"

# 4. 其余依赖
uv pip install --python backend/.venv/Scripts/python.exe `
  fastapi "uvicorn[standard]" python-multipart aiofiles pydantic-settings `
  pillow "numpy<2.0" packaging setuptools

# 5. COLMAP：下载 Windows 二进制解压到 third_party/colmap_dist/
#    https://github.com/colmap/colmap/releases (colmap-x64-windows-cuda.zip)
```
> **关键点**：只用 `gsplat` 包本体 + 我们自研的精简训练脚本（`backend/train/`），**不装 nerfstudio / tiny-cuda-nn / fused-ssim**，彻底避开 Windows 上的 CUDA 编译地狱。

### 前端
```bash
cd frontend && npm install
```

### 数据集（可选，用于演示重建）
```bash
# Tanks&Temples + Deep Blending（含 train/truck/playroom 等，约 650MB）
# 解压到 datasets/ 下，后端自动扫描 datasets/*/*/images 结构
```

---

## 四、怎么用（三个页面）

### 🅰️「模型库」— 看现成 3D 模型
点任意模型卡片进入查看器：
- **左键拖** = 旋转 · **右键拖** = 平移 · **滚轮** = 缩放
- 底部三模式：**环视 / 自由翻滚 / 漫游**（漫游用 WASD 移动 + 拖拽转视角，适合室内房间）
- **剔除杂点** 滑块：拉高可洗掉边缘半透明碎片

### 🅱️「新建重建」— 重建成 3D
1. 选内置数据集，**或**上传自己拍的照片
2. 选训练步数：**3k**（仅测流程）/ **7k**（偏软）/ **30k**（清晰，推荐）
3. 可选自定义保存路径
4. 开始 → 看进度条 → 完成后进查看器

### 🅲️「引擎/参数对比」— 报告用
并排对比两个模型 + 训练步数/正则项的消融数据表。

### 拍照建议（上传自己的照片）
- 环绕一个物体均匀拍 **30–80 张**，重叠度高，避免运动模糊
- 背景放有纹理的物体（书/海报）帮助相机定位
- 避免大面积纯色墙面、强反光

---

## 五、技术栈与原理

| 层 | 技术 |
|---|---|
| 重建（SfM） | COLMAP（特征提取 + 匹配 + 增量重建 + 去畸变） |
| 重建（训练） | 自研精简 3DGS 训练器（gsplat 1.5.2 · MCMC 密度控制 · SH 颜色 · L1+SSIM · 正则项抑制漂浮） |
| 后端 | FastAPI + 单队列串行任务 + SSE 进度推送 |
| 前端 | React 18 + TypeScript + Vite + Three.js + `@mkkellogg/gaussian-splats-3d` + Tailwind |

系统架构、数学推导见 [`DEV_SPEC.md`](DEV_SPEC.md) 与研究报告 `.md`。

---

## 六、实测数据（RTX 4070 Laptop 8GB, tandt/train, data_factor=4）

**训练步数消融：**
| 步数 | PSNR | 耗时 | 质量 |
|---|---|---|---|
| 3k | 19.8 | ~35s | 仅测流程 |
| 7k | 21.9 | ~2min | 主体可辨，边缘偏软 |
| 30k | 26.1 | ~15min | 清晰，文字可读 |

**MCMC 正则项消融（30k）：**
| 配置 | 高斯数 | 体积 | 效果 |
|---|---|---|---|
| 无正则 | 287,301 | 68 MB | 大量飞散尖刺 |
| 有正则 | 226,174 | 50 MB | 残片显著减少 |

---

## 七、常见问题

- **画面全是尖刺/残片**：训练步数太低（用 30k）；已内置 MCMC 正则项抑制。
- **室内场景要"钻进去"才看得到**：切换到「漫游」模式，会从房间内部起步。
- **深色模型看不清**：背景已设中性深灰；可调剔除杂点滑块。
- **GitHub 推送超时**：本项目已配置 SSH 走 443 端口（`~/.ssh/config`）。

---

## 八、许可与引用

MIT License。第三方组件：
- **3D Gaussian Splatting**（INRIA）：非商业研究许可，仅用于学术课程
- **gsplat**（Apache-2.0）、**COLMAP**（BSD）、**GaussianSplats3D**（MIT）

学术使用请引用对应论文（见 DEV_SPEC §2.2）。
