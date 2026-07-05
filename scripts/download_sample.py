"""下载一个公开的 .splat 样例到 frontend/public/models/sample.splat。

用于 Phase 1 让前端查看器在无 GPU、无后端的情况下也能演示拖拽渲染。
样例文件较大（数 MB），已被 .gitignore 排除，不会进入版本库。

用法：
    python scripts/download_sample.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# Windows 控制台默认 GBK，强制 UTF-8 输出避免 emoji/中文崩溃
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

# 公开可用的 .splat 样例（按顺序尝试，任一成功即可）
CANDIDATE_URLS = [
    "https://media.reshot.ai/models/nike_next/model.splat",
    "https://huggingface.co/cakewalk/splat-data/resolve/main/train.splat",
]

# 部分源（如 HuggingFace）缺少 User-Agent 会返回 404
HEADERS = {"User-Agent": "Mozilla/5.0 (3DGS-Reconstruction sample fetcher)"}

OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "frontend"
    / "public"
    / "models"
    / "sample.splat"
)


def download(url: str, dst: Path) -> bool:
    """下载 url 到 dst，带简单进度。成功返回 True。"""
    try:
        print(f"尝试下载: {url}")
        req = urllib.request.Request(url, headers=HEADERS)  # noqa: S310
        with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
            total = int(resp.headers.get("Content-Length", 0))
            read = 0
            with open(dst, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 256)
                    if not chunk:
                        break
                    f.write(chunk)
                    read += len(chunk)
                    if total > 0:
                        pct = min(100, read * 100 // total)
                        sys.stdout.write(f"\r  进度 {pct}%")
                        sys.stdout.flush()
        print(f"\n[OK] 已保存到 {dst}  ({read / 1024 / 1024:.1f} MB)")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"\n  失败: {exc}")
        if dst.exists():
            dst.unlink(missing_ok=True)
        return False

def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists() and OUT_PATH.stat().st_size > 0:
        print(f"样例已存在，跳过: {OUT_PATH}")
        return 0
    for url in CANDIDATE_URLS:
        if download(url, OUT_PATH):
            return 0
    print(
        "[X] 所有候选源均下载失败。\n"
        "    可手动放置任意 .splat/.ply/.ksplat 到 frontend/public/models/sample.splat，\n"
        "    或直接在前端使用「导入本地文件」加载你自己的模型。"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
