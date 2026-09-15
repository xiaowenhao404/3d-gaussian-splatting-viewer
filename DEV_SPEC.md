# DEV_SPEC — 基于 3D Gaussian Splatting 的交互式三维重建与可视化系统

> 开发规范文档（Development Specification）
> 版本：v1.0 · 日期：2026-06-15
> 配套报告：`基于三维高斯泼溅与辐射场技术的高性能三维重建系统设计与实现研究报告.md`

---

## 0. 文档定位

本文件是**实现层**的开发蓝图，约定技术选型、系统架构、模块边界、API 契约、目录结构、环境配置、里程碑与验收标准。配套的研究报告负责**理论层**（3DGS 数学推导、性能对比）。两者一一对应：报告里出现的公式/指标，在本系统中都有对应的可运行模块。

阅读对象：实现者（你本人 + Claude Code）。一切实现以本文件为准；与研究报告冲突时，**以本文件为准**（报告中的 Vue/部分选型已在选型讨论中被修订）。

---

## 1. 项目目标与核心用户故事

### 1.1 一句话定义

一个**前后端分离**的 Web 系统：用户通过界面**选择内置数据集的一组图片**或**上传自己的一系列图片**，后端运行 SfM + 3D Gaussian Splatting 重建出显式高斯模型，前端在浏览器中以**可拖拽旋转/缩放的实时 3D 视口**呈现；重建产物可**保存到指定路径**并在之后**直接导入复看**。

### 1.2 核心用户故事（User Stories）

| 编号 | 角色 | 故事 | 验收要点 |
|---|---|---|---|
| US-1 | 用户 | 从内置数据集库选一组图片 → 一键发起重建 | 数据集列表可见，发起后进入任务 |
| US-2 | 用户 | 拖拽上传自己的一批图片/ZIP → 发起重建 | 支持多文件与 ZIP，上传有进度 |
| US-3 | 用户 | 重建过程中看到**可视化进度条**与阶段日志 | SfM / 训练 / 压缩 三阶段进度可见 |
| US-4 | 用户 | 重建完成后在浏览器里**拖拽旋转、平移、缩放**查看 3D 模型 | 鼠标交互流畅，FPS 可见 |
| US-5 | 用户 | 选择产物**保存到根目录或自定义路径** | 提供路径选择 UI，落盘成功 |
| US-6 | 用户 | 下次直接**导入历史 `.ply/.splat/.ksplat` 文件**进查看器，跳过重建 | 导入即可看，无需 GPU |
| US-7 | 用户（出彩项） | 对同一场景用 **gsplat vs Brush** 两种引擎重建并**对比** PSNR/速度/体积 | 生成对比表与并排视图 |

### 1.3 非目标（Out of Scope，明确不做）

- 不做用户账号/登录鉴权（报告里的 `auth.py` 删除，单机自用工具）。
- 不做分布式/多机训练、不做生产级高并发（单机单任务串行队列即可）。
- 不做移动端原生 App（浏览器响应式即可）。
- 不做动态场景（4D）/可变形高斯，只做静态场景。

---

## 2. 最终技术选型（已定稿）

> 以下为多轮讨论后的**定稿**，实现时不再更改大方向。每项附"为什么选它"。

### 2.1 选型总表

| 层级 | 选定技术 | 为什么 |
|---|---|---|
| **重建引擎（主）** | COLMAP + GLOMAP（SfM）→ **gsplat**（CUDA 训练，3DGS-MCMC） | 质量最高、最贴合报告；**直接用 `gsplat/examples/simple_trainer.py`，不装 nerfstudio，彻底绕开 tiny-cuda-nn 编译地狱** |
| **重建引擎（对比/出彩）** | **Brush**（Rust + WebGPU + Burn，无 CUDA） | 跨平台、零 CUDA；用于与 gsplat 做"质量/速度/配置成本"三维对比，作为报告亮点 |
| **后端框架** | **FastAPI**（Python 3.10+，`uv` 管理） | 异步、自带 OpenAPI 文档、SSE/WebSocket 友好 |
| **任务调度** | 进程内**单队列串行** + 子进程执行 + **SSE 进度推送** | 单机课设无需 Celery/Redis；串行避免显存抢占 |
| **前端框架** | **React 18 + TypeScript + Vite** | Web-3D 生态最大（R3F/drei）、参考最多、易出彩 |
| **3D 查看器** | **Three.js + `@mkkellogg/gaussian-splats-3d`** | 框架无关、对大场景排序/压缩/渐进加载最稳，内置 OrbitControls 拖拽，支持 `.ply/.splat/.ksplat` |
| **样式** | **Tailwind CSS** | 快速搭出专业 UI |
| **模型压缩** | PLY → **`.ksplat`**（mkkellogg 转换器，0/1/2 压缩级） | 无损裁剪 + 量化，体积降 70%~90%，前端加载快 |
| **可选清洗** | **PlayCanvas SuperSplat**（在线编辑器） | 裁切/去飞散噪点，导出干净 PLY/KSPLAT |
| **包管理** | Python: `uv`；前端: `pnpm`（或 npm） | 用户偏好 |

### 2.2 开源仓库清单与许可（实现时 clone/安装）

| 用途 | 仓库 | 许可 | 备注 |
|---|---|---|---|
| 训练引擎 | [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) | Apache-2.0 | 用 `examples/simple_trainer.py`；有 Windows 预编译 wheel |
| SfM（增量+MVS） | [colmap/colmap](https://github.com/colmap/colmap) | BSD | Windows 官方二进制，解压即用 |
| SfM（全局，提速） | [colmap/glomap](https://github.com/colmap/glomap) | BSD | 新版随 COLMAP 发布捆绑 |
| 对比引擎 | [ArthurBrussee/brush](https://github.com/ArthurBrussee/brush) | Apache-2.0 | Rust/WGPU，无 CUDA |
| 前端查看器 | [mkkellogg/GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) | MIT | npm: `@mkkellogg/gaussian-splats-3d` |
| 在线清洗（可选） | [playcanvas/supersplat](https://github.com/playcanvas/supersplat) | MIT | 可本地部署或用官方在线版 |

> ⚠️ **许可注意**：原始 INRIA 3DGS 算法为**非商业研究许可**；gsplat 是 Apache-2.0 的重实现，课设/学术使用安全。报告与代码中需注明引用。

---

## 3. 系统架构

### 3.1 总体架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                          浏览器 (Chrome 134+)                       │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────────────┐ │
│  │ 数据集/上传  │  │ 任务进度    │  │  3D 可拖拽视口 (Three.js +    │ │
│  │  选择页      │  │  SSE 进度条 │  │  gaussian-splats-3d Viewer)  │ │
│  └────────────┘  └────────────┘  └──────────────────────────────┘ │
│        React 18 + TS + Vite + Tailwind + axios/EventSource         │
└───────────────┬───────────────────────────────────┬──────────────┘
                │ REST (JSON)                         │ GET 静态模型 .ksplat
                ▼                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│                       FastAPI 后端 (localhost:8000)                 │
│  ┌──────────────┐  ┌───────────────┐  ┌────────────────────────┐  │
│  │ API 路由层    │  │ 任务调度器     │  │ 静态文件服务            │  │
│  │ datasets/    │→ │ 单队列串行     │  │ /models/*.ksplat        │  │
│  │ jobs/models/ │  │ + SSE 推送     │  │ /datasets/*/images      │  │
│  │ import/fs    │  └──────┬────────┘  └────────────────────────┘  │
│  └──────────────┘         │                                        │
│             ┌─────────────┼──────────────┐                         │
│             ▼             ▼              ▼                          │
│   ┌──────────────┐ ┌─────────────┐ ┌──────────────┐               │
│   │ sfm_service  │ │trainer_svc  │ │ compressor   │               │
│   │ COLMAP/GLOMAP│ │ gsplat/Brush│ │ PLY→.ksplat  │               │
│   └──────┬───────┘ └──────┬──────┘ └──────┬───────┘               │
└──────────┼────────────────┼───────────────┼──────────────────────┘
           ▼                ▼               ▼
   ┌─────────────────────────────────────────────────┐
   │  workspace/ (产物持久化, 默认根目录, 可自定义)     │
   │  tasks/{uuid}/{images,colmap,undistorted,output} │
   │  models.json (模型注册表)                         │
   └─────────────────────────────────────────────────┘
```

### 3.2 数据流（重建主链路）

```
图片来源(内置数据集 / 上传ZIP)
   │  ① 解压 + 校验 (file_handler)
   ▼
images/                          ← 原始/抽帧后图像
   │  ② COLMAP feature_extractor + matcher
   ▼
database.db                      ← SIFT 特征 + 匹配
   │  ③ GLOMAP mapper (全局 SfM, 提速)
   ▼
sparse/0/{cameras,images,points3D}  ← 相机位姿 + 稀疏点云
   │  ④ COLMAP image_undistorter
   ▼
undistorted/                     ← 去畸变图像 + 相机
   │  ⑤ gsplat simple_trainer.py mcmc (30k steps)
   ▼
output/point_cloud.ply           ← 高斯模型 (可达数百 MB)
   │  ⑥ compressor: PLY → .ksplat (level 2)
   ▼
output/model.ksplat              ← 压缩模型 (前端加载)
   │  ⑦ 注册进 models.json + 拷贝到 /models 静态目录
   ▼
前端 Viewer 拖拽渲染
```

### 3.3 关键设计原则

1. **查看器与重建解耦**：查看器是"永远能跑"的安全模块（纯前端，无 GPU 依赖）；重建是"重而有风险"的模块。答辩演示时，即使重建环境出问题，预置模型库也能保证有东西可看。
2. **产物即一等公民**：每个模型在 `models.json` 注册，带来源（bundled/generated/imported）、格式、路径、指标、缩略图。导入 = 注册一条记录。
3. **任务串行**：单 GPU，重建任务进队列串行执行，避免显存抢占（OOM）。
4. **路径可配**：输出根目录默认 `<project>/workspace`，每个任务可覆盖为自定义绝对路径（满足"存根目录/自选路径"）。

---

## 4. 后端设计

### 4.1 目录结构

```
backend/
├── pyproject.toml              # uv 项目定义
├── app/
│   ├── main.py                 # FastAPI 入口 + 静态挂载 + CORS
│   ├── config.py               # 路径/显存/超参 (pydantic-settings, frozen)
│   ├── api/
│   │   ├── datasets.py         # 内置数据集列表/详情
│   │   ├── jobs.py             # 发起重建/查询/SSE 进度/取消
│   │   ├── models.py           # 模型库 CRUD + 静态地址
│   │   ├── import_model.py     # 导入已有 ply/splat/ksplat
│   │   └── fs.py               # 路径选择/校验 (输出目录)
│   ├── core/
│   │   ├── scheduler.py        # 单队列串行调度 + 进度事件总线
│   │   ├── registry.py         # models.json 读写 (模型注册表)
│   │   └── events.py           # SSE 事件流 (asyncio.Queue)
│   ├── services/
│   │   ├── sfm_service.py      # 封装 COLMAP + GLOMAP 子进程
│   │   ├── trainer_service.py  # 封装 gsplat / Brush 子进程
│   │   ├── compressor.py       # PLY → .ksplat
│   │   └── metrics.py          # PSNR/SSIM/LPIPS 评测 (对比用)
│   └── utils/
│       ├── file_handler.py     # 异步分块上传 + ZIP 解压 + 校验
│       └── proc.py             # 子进程封装 + 实时 stdout 解析进度
└── scripts/
    ├── prepare_datasets.py     # 下载/抽帧/预训练内置数据集
    └── pretrain_gallery.py     # 批量预训练 → .ksplat (打包答辩用)
```

> 遵循全局 coding-style：单文件 200–400 行；超出则拆分（如 `sfm_service` 过大可拆 `colmap.py`/`glomap.py`）。配置用 `frozen` dataclass/pydantic。所有函数带 type hints，用 logger 不用 print。

### 4.2 任务生命周期（Job State Machine）

```
queued ──► running ──► succeeded
   │          │  └─(SfM)─►(train)─►(compress)─► 各阶段 progress 0~100
   │          └──► failed (记录 stderr 摘要)
   └──► canceled
```

每个 Job 的进度对象：

```jsonc
{
  "job_id": "uuid",
  "status": "running",
  "stage": "train",            // upload|sfm|undistort|train|compress|done
  "stage_progress": 47,         // 0-100, 当前阶段
  "overall_progress": 62,       // 0-100, 总体加权
  "engine": "gsplat",          // gsplat|brush
  "message": "iter 14000/30000  loss=0.031  PSNR=27.4",
  "started_at": "...", "eta_seconds": 320,
  "output_dir": "D:/.../workspace/tasks/uuid/output",
  "error": null
}
```

进度来源：`proc.py` 实时读取子进程 stdout，正则解析（COLMAP 的图像计数、gsplat 的 `iter x/30000`），换算为百分比，推入 `events.py` 的 SSE 队列。

### 4.3 REST API 契约

> Base: `http://localhost:8000/api`。所有响应 JSON；错误统一 `{ "detail": "..." }`。

#### 数据集

| Method | Path | 说明 | 返回 |
|---|---|---|---|
| GET | `/datasets` | 列出内置数据集（图片组） | `[{id,name,thumb,num_images,size,scene_type}]` |
| GET | `/datasets/{id}` | 数据集详情 + 样例图 | `{id,name,images:[...],...}` |

#### 重建任务

| Method | Path | 说明 |
|---|---|---|
| POST | `/jobs` | 发起重建。body: `{source:"dataset"|"upload", dataset_id?, upload_id?, engine:"gsplat"|"brush", output_dir?, params:{...}}` → `{job_id}` |
| POST | `/uploads` | 分块/整包上传图片或 ZIP → `{upload_id, num_images}` |
| GET | `/jobs` | 列出全部任务 |
| GET | `/jobs/{id}` | 查询单任务状态（轮询备用） |
| GET | `/jobs/{id}/events` | **SSE 进度流**（前端 EventSource 主用） |
| GET | `/jobs/{id}/logs` | 原始日志（debug） |
| DELETE | `/jobs/{id}` | 取消/删除任务 |

#### 模型库

| Method | Path | 说明 |
|---|---|---|
| GET | `/models` | 列出全部模型（bundled+generated+imported） |
| GET | `/models/{id}` | 模型元数据 + 静态 URL（`/static/models/{id}.ksplat`） |
| POST | `/models/import` | 导入已有文件。body: `{path:"D:/x.ply"}` 或 multipart 上传 → 注册并返回 `{id}` |
| POST | `/models/{id}/save-as` | 另存到自定义路径。body: `{target_path}` |
| DELETE | `/models/{id}` | 从注册表移除（可选删文件） |

#### 文件系统辅助

| Method | Path | 说明 |
|---|---|---|
| GET | `/fs/suggest-root` | 返回默认根目录（项目根 / workspace） |
| POST | `/fs/validate-path` | 校验自定义路径可写 → `{ok, writable, exists}` |
| GET | `/fs/browse?path=` | （可选）列目录，给路径选择 UI 用 |

#### 对比（出彩）

| Method | Path | 说明 |
|---|---|---|
| POST | `/compare` | 对同一来源用 gsplat+brush 各跑一次 → `{compare_id}` |
| GET | `/compare/{id}` | 返回两模型 id + 指标表 `{psnr,ssim,lpips,train_seconds,ply_mb}` |

### 4.4 模型注册表 `models.json`

```jsonc
{
  "models": [
    {
      "id": "garden_gsplat",
      "name": "Mip-NeRF360 Garden (gsplat)",
      "source": "bundled",          // bundled | generated | imported
      "engine": "gsplat",
      "format": "ksplat",
      "file": "models/garden_gsplat.ksplat",
      "abs_path": "D:/.../workspace/models/garden_gsplat.ksplat",
      "thumb": "thumbs/garden.jpg",
      "metrics": {"psnr": 27.2, "ssim": 0.82, "train_seconds": 540, "ply_mb": 312, "ksplat_mb": 48},
      "viewer_defaults": {"cameraUp":[0,-1,-0.6],"position":[-1,-4,6],"lookAt":[0,4,0]},
      "created_at": "2026-06-15T..."
    }
  ]
}
```

### 4.5 重建引擎封装细节

#### gsplat 主链路（`trainer_service.py`）

SfM（`sfm_service.py`，调用 COLMAP/GLOMAP 二进制）：

```bash
# ① 特征提取 (GPU SIFT)
colmap feature_extractor --database_path db.db --image_path images \
  --ImageReader.camera_model PINHOLE --SiftExtraction.use_gpu 1

# ② 匹配 (照片用 exhaustive；连续视频帧用 sequential)
colmap exhaustive_matcher --database_path db.db --SiftMatching.use_gpu 1

# ③ 全局 SfM (GLOMAP, 比增量 mapper 快 1~2 数量级)
glomap mapper --database_path db.db --image_path images --output_path sparse

# ④ 去畸变 (gsplat 需要)
colmap image_undistorter --image_path images --input_path sparse/0 \
  --output_path undistorted --output_type COLMAP
```

训练（gsplat，**不装 nerfstudio**）：

```bash
# MCMC 自适应密度控制, 30k 步; 大场景用 --data_factor 4 降分辨率防 OOM
python gsplat/examples/simple_trainer.py mcmc \
  --data_dir <task>/undistorted --data_factor 4 \
  --result_dir <task>/output --max_steps 30000 \
  --save_ply  # 末尾导出 point_cloud.ply
```

损失：$\mathcal{L}=(1-\lambda)\mathcal{L}_1+\lambda(1-\text{SSIM})$，$\lambda=0.2$（与报告一致）。

#### Brush 对比链路

```bash
# Brush 直接吃 COLMAP 数据 (sparse/ + images/), 无需 CUDA
brush --data-dir <task>  --total-steps 30000 --export-path <task>/output/brush.ply
# 或浏览器内: 加载同一数据集训练并截图对比
```

对比脚本 `metrics.py`：对两引擎输出在 hold-out 测试视角上算 PSNR/SSIM/LPIPS，记录训练墙钟时间与 PLY 体积，写入 `/compare/{id}`。

#### 压缩（`compressor.py`）

PLY → `.ksplat`：调用 mkkellogg 的转换（Node 脚本或前端转换器）。优先用其提供的 `@mkkellogg/gaussian-splats-3d` 的 `PlyLoader.loadFromURL` + `KSplatLoader` 在无头 Node 环境转换；压缩级 2，`splatAlphaRemovalThreshold` 默认 5，SH 阶数 3。

### 4.6 鲁棒性与防护（写进实现）

| 风险 | 规避 |
|---|---|
| **显存 OOM** | `--data_factor 2/4` 限分辨率；图片数 >150 张时提示降采样；任务串行 |
| **无纹理/强反光匹配失败** | SfM 失败时返回明确错误 + 拍摄建议（背景垫有纹理物体） |
| **gsplat wheel 不匹配** | 降级走 WSL2（见 §7.2）；config 里可切换 `python` 路径 |
| **大 PLY 传输慢** | 强制 `.ksplat` 后再给前端；渐进加载 |
| **路径越界/不可写** | `/fs/validate-path` 预校验；拒绝写系统目录 |

---

## 5. 前端设计

### 5.1 目录结构

```
frontend/
├── package.json
├── vite.config.ts
├── tailwind.config.js
├── index.html
└── src/
    ├── main.tsx
    ├── App.tsx                 # 路由
    ├── api/
    │   ├── client.ts           # axios 实例 + 类型
    │   └── sse.ts              # EventSource 封装
    ├── store/
    │   └── useStore.ts         # zustand 全局状态 (当前模型/任务)
    ├── pages/
    │   ├── GalleryPage.tsx     # 数据集库 + 模型库 (入口)
    │   ├── ReconstructPage.tsx # 上传/选数据集 + 发起 + 进度
    │   ├── ViewerPage.tsx      # 3D 拖拽视口
    │   └── ComparePage.tsx     # gsplat vs brush 并排 (出彩)
    └── components/
        ├── UploadZone.tsx      # 拖拽上传 (react-dropzone)
        ├── DatasetCard.tsx
        ├── ModelCard.tsx
        ├── JobProgress.tsx     # SSE 进度条 + 阶段 + 日志
        ├── PathPicker.tsx      # 保存路径选择 (调 /fs)
        ├── SplatViewer.tsx     # ★ 核心: 封装 gaussian-splats-3d
        └── ViewerControls.tsx  # alpha 阈值/点云模式/正交 等
```

### 5.2 核心组件 `SplatViewer.tsx`（拖拽视口）

职责：把 `@mkkellogg/gaussian-splats-3d` 的命令式 `Viewer` 用 React 组件包起来。

- `useEffect` 中 `new GaussianSplats3D.Viewer({...})`，挂到 `ref` 容器；卸载时 `viewer.dispose()` 防显存泄漏。
- props：`modelUrl`、`format`、`viewerDefaults`（cameraUp/position/lookAt）。
- 交互：内置 OrbitControls —— **左键拖拽旋转、右键平移、滚轮缩放**（满足"可拖拽旋转"核心诉求）。
- 暴露方法：`setAlphaCutoff(v)`（去飞散噪点）、`togglePointCloud()`、`toggleOrtho()`、`getStats()`（FPS / 高斯数 / 排序时延，用于报告截图）。
- 自适应：`ResizeObserver` 重排视口。

> 备选轻量路线（若时间紧）：`@react-three/fiber` + `@react-three/drei` 的 `<Splat/>`，几行即可渲染；但大场景稳健性不如 mkkellogg，默认走 mkkellogg。

### 5.3 页面流

```
GalleryPage ──选数据集/上传──► ReconstructPage ──发起──► JobProgress(SSE)
     │                                                        │完成
     │导入历史文件 ─────────────────────────────────────────┐  │
     ▼                                                       ▼  ▼
ViewerPage (拖拽 3D)  ◄────────────────────────────── 自动跳转/手动进入
     │
     └──► ComparePage (并排 gsplat vs brush + 指标表)
```

### 5.4 保存路径 UI（满足 US-5/US-6）

- 发起重建时，`PathPicker` 默认填 `/fs/suggest-root`（项目根/workspace），用户可改为绝对路径；`/fs/validate-path` 实时校验绿/红。
- 模型详情页有"另存为…"，调 `/models/{id}/save-as`。
- "导入已有文件"：`PathPicker` 选本地路径 **或** 拖拽上传 → `/models/import` → 直接跳 ViewerPage。

---

## 6. 数据集方案

### 6.1 内置数据集（打包进系统，保证答辩稳）

| 数据集 | 用途 | 体积 | 下载 |
|---|---|---|---|
| **NeRF Synthetic**（Lego/Ship/Drums 选 2-3 个） | 小、快、经典；训练 2-3 分钟出结果，适合现场重建演示 | 小（~1GB） | [yenchenlin/nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) 的 `download_example_data.sh` |
| **Mip-NeRF 360**（Garden + Bonsai） | 大场景、出彩；体现无边界重建质量 | 大（~10GB，可只取 2 场景） | [360_v2.zip](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) |
| 用户自拍（示范） | US-2 演示；提供拍摄指南 | — | 现场/手机环绕拍 30-80 张 |

### 6.2 预训练产物（`scripts/pretrain_gallery.py`）

开发期**离线预训练** Garden / Bonsai / 1 个 Synthetic，转成 `.ksplat` 放入 `workspace/models/` 并写进 `models.json`。这样：
- Gallery 一打开就有高质量模型可拖拽看（不依赖现场重建成功）。
- 现场只需用小的 Synthetic 场景演示"上传→重建→看"的完整链路（2-3 分钟可完成）。

### 6.3 拍摄指南（写进前端帮助 + 报告）

- 环绕物体均匀拍 30–80 张，重叠度高，避免运动模糊。
- 背景放有纹理的物体（书/海报），帮助 SfM 匹配。
- 避免纯反光/纯色大平面。

---

## 7. 环境与部署

### 7.1 主环境：原生 Windows（推荐先走）

**前置**：NVIDIA 驱动 + CUDA 11.8/12.x（与所选 gsplat wheel 匹配）；Node ≥18；`uv`。

```powershell
# --- 后端 ---
# 1. COLMAP+GLOMAP: 下载官方 Windows 二进制, 解压, 把 bin 加入 PATH
#    https://github.com/colmap/colmap/releases  (含 glomap)
# 2. Python 环境 (uv)
uv venv --python 3.10
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
uv pip install gsplat   # 优先装匹配的预编译 wheel; 失败再看 §7.2
uv pip install fastapi uvicorn[standard] python-multipart aiofiles pydantic-settings pillow
git clone https://github.com/nerfstudio-project/gsplat  # 用其 examples/simple_trainer.py
# 3. 启动
uv run uvicorn app.main:app --reload --port 8000

# --- 前端 ---
pnpm install
pnpm dev   # http://localhost:5173, 代理 /api → :8000
```

> **关键**：只用 `gsplat` 包本体 + `examples/simple_trainer.py`，**不装 nerfstudio、不碰 tiny-cuda-nn**。COLMAP 用二进制不自己编译。这样 Windows 配置量最小。

### 7.2 保底环境：WSL2 Ubuntu（gsplat wheel 不匹配时）

```bash
# WSL2 中 CUDA 直通; Linux 下 gsplat/colmap 安装最稳
sudo apt install colmap         # 或 conda install -c conda-forge colmap glomap
uv pip install gsplat            # Linux wheel 齐全
# 后端跑在 WSL, Windows 浏览器访问 localhost:8000 (WSL2 自动转发)
# 注意: 输出路径用 /mnt/d/... 映射到 Windows 盘, "存根目录"功能仍可用
```

### 7.3 Brush（对比引擎）

```bash
# 装 Rust 后:
cargo install --git https://github.com/ArthurBrussee/brush
# 或下载 release 二进制; 浏览器版直接开 demo 页训练
```

---

## 8. 评测与对比方案（出彩 / 写进报告）

| 维度 | 指标 | 怎么测 |
|---|---|---|
| 重建质量 | PSNR / SSIM / LPIPS | hold-out 测试视角，`metrics.py` |
| 渲染性能 | 前端 FPS / 高斯数 / 排序时延 | Viewer 诊断面板（`I` 键） |
| 训练成本 | 墙钟时间 / 峰值显存 / PLY 体积 | 子进程计时 + `nvidia-smi` |
| 配置成本（定性） | 安装步骤数 / 平台支持 | gsplat(CUDA) vs Brush(WebGPU) 对比叙述 |

输出：一张 gsplat vs Brush 对比表 + ComparePage 并排视图截图，直接进报告"实验分析"章节。

---

## 9. 开发里程碑（2–4 周）

| 阶段 | 周 | 交付物 | 验收 |
|---|---|---|---|
| **M0 环境就绪** | W1 上 | COLMAP/gsplat 能在本机跑通官方样例；前端脚手架起来 | 命令行能训出一个 PLY 并被 mkkellogg demo 加载 |
| **M1 查看器闭环** | W1 下 | SplatViewer + Gallery + 导入历史文件 | US-4/US-6：导入 `.ply/.ksplat` 能拖拽看 |
| **M2 重建闭环** | W2 | jobs/SSE/sfm/trainer/compressor 串起来 | US-1/US-3：选数据集→进度条→出 .ksplat→自动可看 |
| **M3 上传+产物管理** | W3 上 | 上传、保存路径选择、models.json | US-2/US-5：上传重建、存自定义路径、再导入 |
| **M4 对比+打磨** | W3 下–W4 | Brush 对比、ComparePage、预训练 Gallery、UI 美化 | US-7：gsplat vs brush 指标表 + 并排 |
| **M5 报告对齐+答辩** | W4 | README、报告实验数据回填、演示脚本 | 端到端 Demo 顺滑，预置模型兜底 |

> 风险缓冲：M1（查看器）必须最先做完——它是答辩的"保命"模块。重建链路（M2）即使现场慢/偶发失败，也有预置模型库托底。

---

## 10. 项目总目录结构

```
3dgs-reconstruction-system/
├── DEV_SPEC.md                 # 本文件
├── README.md                   # 一键运行/部署说明
├── 基于三维高斯泼溅...研究报告.md   # 理论报告
├── backend/                    # §4.1
├── frontend/                   # §5.1
├── workspace/                  # 产物根目录 (默认, 可在 UI 改)
│   ├── uploads/
│   ├── tasks/{uuid}/{images,colmap,undistorted,output}
│   ├── models/                 # .ksplat + thumbs (静态服务)
│   └── models.json
├── third_party/                # gsplat / brush clone (或文档指明)
└── scripts/                    # 数据集准备 + 预训练 (§4.1)
```

---

## 11. 验收标准（Definition of Done）

- [ ] 浏览器打开即见 Gallery（含 ≥2 个预置可拖拽模型）。
- [ ] 选内置数据集 → 发起重建 → **进度条**走完 → 自动进查看器拖拽看。
- [ ] 拖拽上传一批图片/ZIP → 同上链路跑通。
- [ ] 重建产物可保存到**根目录**与**自定义路径**，二者均成功落盘。
- [ ] 重新导入历史 `.ply/.splat/.ksplat` → 直接拖拽查看（无需 GPU）。
- [ ] 3D 视口：左键转、右键移、滚轮缩放流畅；可调 alpha 去噪、看 FPS。
- [ ] （出彩）同场景 gsplat vs Brush 对比表 + 并排视图。
- [ ] README 能让另一台机器按文档跑起来。
- [ ] 代码符合全局规范（文件 200–400 行、type hints、logger、frozen config）。

---

## 12. 待确认 / 实现期再决策的小点

1. 进度推送用 **SSE**（默认，单向够用）还是 WebSocket？→ 先 SSE，需要双向交互再升级。
2. 状态管理用 **zustand**（默认，轻）还是 Redux？→ zustand。
3. Synthetic 数据集具体选哪 2-3 个场景？→ 实现时按训练速度/观感定（建议 Lego + Ship）。
4. `.ksplat` 转换放**后端 Node 子进程**还是**前端浏览器内**？→ 默认后端无头 Node，前端做兜底。
5. 是否本地部署 SuperSplat 做清洗？→ 可选，时间富余再加。

---

*本规范为活文档；实现中如遇与现实冲突，先在此更新再改代码。*
