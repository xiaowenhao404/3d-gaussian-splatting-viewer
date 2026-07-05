"""精简版 3DGS 训练脚本（只依赖 gsplat 本体 + torch + numpy + Pillow）。

参照 gsplat 官方 examples/simple_trainer.py 的核心逻辑实现，剥离
fused-ssim / viser / nerfview / tensorboard / 外观优化 等重依赖与可选项，
保留：SfM 初始化、MCMC 自适应密度控制、SH 颜色、L1+SSIM 损失、PLY 导出。

被后端 trainer_service 以子进程方式调用；进度通过 stdout 行 "step X/Y ..." 输出。

用法:
    python train_gsplat.py --data_dir <undistorted> --result_dir <out> \
        --max_steps 30000 --data_factor 4 --save_ply
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from gsplat import export_splats
from gsplat.rendering import rasterization
from gsplat.strategy import MCMCStrategy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colmap_io import CameraView, parse_colmap  # noqa: E402

C0 = 0.28209479177387814  # SH DC 系数


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / C0


def knn_scales(points: torch.Tensor, k: int = 4, chunk: int = 4096) -> torch.Tensor:
    """用分块 cdist 求每点第 k 近邻平均距离，作为初始尺度（避免 sklearn 依赖）。"""
    n = points.shape[0]
    out = torch.empty(n, device=points.device)
    for i in range(0, n, chunk):
        d = torch.cdist(points[i : i + chunk], points)  # [chunk, N]
        knn = d.topk(k, largest=False).values[:, 1:]    # 去掉自身
        out[i : i + chunk] = knn.mean(dim=-1)
    return out


def gaussian_window(size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(size, device=device) - size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(1)
    win = (g @ g.t()).unsqueeze(0).unsqueeze(0)  # [1,1,size,size]
    return win


def ssim(img1: torch.Tensor, img2: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """img1/img2: [B,C,H,W]，返回标量 SSIM。"""
    c = img1.shape[1]
    w = win.expand(c, 1, win.shape[-1], win.shape[-1])
    mu1 = F.conv2d(img1, w, groups=c)
    mu2 = F.conv2d(img2, w, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1 = F.conv2d(img1 * img1, w, groups=c) - mu1_sq
    sigma2 = F.conv2d(img2 * img2, w, groups=c) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, w, groups=c) - mu12
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2)
    )
    return s.mean()


def load_image(view: CameraView, device) -> torch.Tensor:
    img = Image.open(view.image_path).convert("RGB")
    if img.size != (view.width, view.height):
        img = img.resize((view.width, view.height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device)  # [H,W,3]


def build_splats(xyz: np.ndarray, rgb: np.ndarray, sh_degree: int, device):
    points = torch.from_numpy(xyz).float().to(device)
    rgbs = torch.from_numpy(rgb / 255.0).float().to(device)
    dist = knn_scales(points).clamp_min(1e-7)
    scales = torch.log(dist).unsqueeze(-1).repeat(1, 3)
    n = points.shape[0]
    quats = torch.rand((n, 4), device=device)
    opacities = torch.logit(torch.full((n,), 0.1, device=device))
    colors = torch.zeros((n, (sh_degree + 1) ** 2, 3), device=device)
    colors[:, 0, :] = rgb_to_sh(rgbs)
    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(points),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opacities),
        "sh0": torch.nn.Parameter(colors[:, :1, :]),
        "shN": torch.nn.Parameter(colors[:, 1:, :]),
    }).to(device)
    return splats


def make_optimizers(splats, scene_scale: float):
    lrs = {
        "means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
        "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20,
    }
    return {
        name: torch.optim.Adam([{"params": splats[name], "lr": lr}], eps=1e-15)
        for name, lr in lrs.items()
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("strategy", nargs="?", default="mcmc")  # 兼容子命令位置参数
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--data_factor", type=int, default=4)
    ap.add_argument("--cap_max", type=int, default=300000)  # 8GB 显存安全上限
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--ssim_lambda", type=float, default=0.2)
    ap.add_argument("--opacity_reg", type=float, default=0.01)  # MCMC 抑制漂浮
    ap.add_argument("--scale_reg", type=float, default=0.01)    # MCMC 抑制尖刺
    ap.add_argument("--save_ply", action="store_true")
    ap.add_argument("--disable_viewer", action="store_true")  # 兼容占位
    args = ap.parse_args()

    set_seed(42)
    device = "cuda"
    data_dir = Path(args.data_dir)
    result_dir = Path(args.result_dir)
    (result_dir / "ply").mkdir(parents=True, exist_ok=True)

    # ── 解析数据 ──
    sparse = data_dir / "sparse" / "0"
    views, xyz, rgb = parse_colmap(sparse, data_dir / "images", factor=args.data_factor)
    if len(views) == 0 or xyz.shape[0] == 0:
        print("ERROR: COLMAP 数据为空（无相机或无点云）", flush=True)
        return 1
    print(f"loaded {len(views)} views, {xyz.shape[0]} sfm points", flush=True)

    # 场景尺度（相机中心范围），用于 means 学习率缩放
    centers = np.stack([(-v.w2c[:3, :3].T @ v.w2c[:3, 3]) for v in views])
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max()) * 1.1
    scene_scale = max(scene_scale, 1e-3)

    # 预载图像到 CPU（按需移 GPU），8GB 显存友好
    images = [load_image(v, "cpu") for v in views]
    Ks = [torch.from_numpy(v.K).to(device) for v in views]
    w2cs = [torch.from_numpy(v.w2c).to(device) for v in views]

    splats = build_splats(xyz, rgb, args.sh_degree, device)
    optimizers = make_optimizers(splats, scene_scale)
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / args.max_steps)
    )
    # 高斯上限随步数缩放：短训练用更少但更充分优化的高斯，避免欠优化的尖刺/漂浮
    eff_cap = min(args.cap_max, max(80_000, args.max_steps * 20))
    strategy = MCMCStrategy(cap_max=eff_cap, verbose=False)
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state()
    win = gaussian_window(11, 1.5, device)

    order = list(range(len(views)))
    random.shuffle(order)
    ptr = 0
    t0 = time.time()
    for step in range(args.max_steps):
        if ptr >= len(order):
            random.shuffle(order)
            ptr = 0
        idx = order[ptr]; ptr += 1

        pixels = images[idx].to(device).unsqueeze(0)  # [1,H,W,3]
        v = views[idx]
        active_sh = min(args.sh_degree, step // 1000)
        colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N,K,3]

        renders, _, info = rasterization(
            means=splats["means"], quats=splats["quats"],
            scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]),
            colors=colors, viewmats=w2cs[idx][None], Ks=Ks[idx][None],
            width=v.width, height=v.height, sh_degree=active_sh,
            packed=False, rasterize_mode="classic", camera_model="pinhole",
        )
        renders = renders.clamp(0, 1)

        strategy.step_pre_backward(splats, optimizers, state, step, info)
        l1 = F.l1_loss(renders, pixels)
        ssim_loss = 1.0 - ssim(
            renders.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), win
        )
        loss = l1 * (1 - args.ssim_lambda) + ssim_loss * args.ssim_lambda
        # MCMC 正则项：抑制"又大又尖"的漂浮高斯（对齐官方 simple_trainer）。
        # 缺少 scale_reg 会放任高斯长成飞散尖刺，这是残片的训练源头。
        loss = loss + args.opacity_reg * torch.sigmoid(splats["opacities"]).abs().mean()
        loss = loss + args.scale_reg * torch.exp(splats["scales"]).abs().mean()
        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()
        strategy.step_post_backward(
            splats, optimizers, state, step, info,
            lr=means_sched.get_last_lr()[0],
        )

        if step % 50 == 0 or step == args.max_steps - 1:
            psnr = -10.0 * math.log10(F.mse_loss(renders, pixels).item() + 1e-10)
            print(
                f"step {step + 1}/{args.max_steps} loss={loss.item():.4f} "
                f"psnr={psnr:.2f} N={splats['means'].shape[0]}",
                flush=True,
            )

    # ── 导出 PLY（导出前剔除漂浮物/尖刺，提升可视质量） ──
    ply_path = result_dir / "ply" / f"point_cloud_{args.max_steps}.ply"
    if args.save_ply:
        with torch.no_grad():
            op = torch.sigmoid(splats["opacities"])               # 不透明度
            sc = torch.exp(splats["scales"]).amax(dim=1)          # 各高斯最大轴尺度
            keep = op > 0.05                                       # 剔除近全透明的飞散物
            if int(keep.sum()) > 1000:                            # 再剔除极端拉长的尖刺
                limit = torch.quantile(sc[keep].float(), 0.995) * 3.0
                keep = keep & (sc < limit)
            idx = keep.nonzero().squeeze(1)
            print(
                f"prune floaters: {splats['means'].shape[0]} -> {idx.numel()}",
                flush=True,
            )
            sel = lambda t: t[idx]  # noqa: E731
            export_splats(
                means=sel(splats["means"]), scales=sel(splats["scales"]),
                quats=sel(splats["quats"]), opacities=sel(splats["opacities"]),
                sh0=sel(splats["sh0"]), shN=sel(splats["shN"]),
                format="ply", save_to=str(ply_path),
            )
    elapsed = time.time() - t0
    print(f"DONE in {elapsed:.1f}s -> {ply_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
