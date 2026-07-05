"""从真实训练日志绘制 PSNR 随训练步数变化曲线（图6-10）。

解析训练日志中每 50 步的 "step X/30000 ... psnr=Z" 记录，绘制原始曲线 +
滑动平均趋势线，并标注 3k / 7k / 30k 三个消融锚点。输出 PNG 到根目录。

用法：
    python scripts/plot_psnr_curve.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LOG = Path(r"C:/Users/36774/AppData/Local/Temp/train30k_reg.log")  # 若不存在则回退
FALLBACK_LOG = Path("/tmp/train30k_reg.log")
OUT = ROOT / "fig_6_10_psnr_curve.png"

_STEP_RE = re.compile(r"step (\d+)/(\d+).*?psnr=([\d.]+)")


def parse_log(path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    psnrs: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _STEP_RE.search(line)
        if m:
            steps.append(int(m.group(1)))
            psnrs.append(float(m.group(3)))
    return steps, psnrs


def moving_avg(values: list[float], win: int = 15) -> list[float]:
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - win + 1)
        window = values[lo : i + 1]
        out.append(sum(window) / len(window))
    return out


def main() -> int:
    log = LOG if LOG.exists() else FALLBACK_LOG
    steps, psnrs = parse_log(log)
    if not steps:
        print(f"未从日志解析到 PSNR 记录: {log}")
        return 1

    smooth = moving_avg(psnrs, win=15)

    plt.figure(figsize=(8, 5), dpi=140)
    plt.plot(steps, psnrs, color="#9db4d0", lw=0.8, alpha=0.6, label="Per-step PSNR")
    plt.plot(steps, smooth, color="#2f6fb0", lw=2.2, label="Moving average (window=15)")

    # 消融锚点
    anchors = {3000: 19.5, 7000: 21.4, 30000: 26.1}
    ax, ay = list(anchors.keys()), list(anchors.values())
    plt.scatter(ax, ay, color="#d94f4f", zorder=5, s=45, label="Ablation checkpoints (3k/7k/30k)")
    for x, y in anchors.items():
        plt.annotate(f"{y:.1f}dB @ {x // 1000}k", (x, y),
                     textcoords="offset points", xytext=(6, -14), fontsize=9,
                     color="#d94f4f")

    plt.xlabel("Training Steps")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs Training Steps  (TandT-train, data_factor=4)")
    plt.grid(True, ls="--", alpha=0.35)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(OUT)
    print(f"[OK] 已保存 {OUT}  ({len(steps)} 个数据点)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
