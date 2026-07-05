import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { TrackballControls } from 'three/examples/jsm/controls/TrackballControls.js';
import { FlyControls } from 'three/examples/jsm/controls/FlyControls.js';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import type { SplatFormat, ViewerDefaults } from '@/types';

// 直接取库内真实枚举值（实际为 Splat=0, KSplat=1, Ply=2），避免硬编码出错
const FORMAT_ENUM: Record<SplatFormat, number> = {
  splat: GaussianSplats3D.SceneFormat.Splat,
  ksplat: GaussianSplats3D.SceneFormat.KSplat,
  ply: GaussianSplats3D.SceneFormat.Ply,
};

export interface ViewerStats {
  fps: number;
  splatCount: number;
}

/** 相机控制模式：环视(绕物体) / 自由翻滚 / 漫游(WASD第一人称) */
export type ControlMode = 'orbit' | 'tumble' | 'walk';

interface SplatViewerProps {
  /** 模型 URL（http 或 blob:） */
  url: string;
  format: SplatFormat;
  /** 不透明度剔除阈值 0-255，过滤边缘飞散噪点 */
  alphaThreshold?: number;
  /** 相机控制模式 */
  controlMode?: ControlMode;
  viewerDefaults?: ViewerDefaults;
  onStats?: (s: ViewerStats) => void;
  onLoaded?: () => void;
  onError?: (message: string) => void;
}

/**
 * 3DGS 拖拽查看器。封装 @mkkellogg/gaussian-splats-3d 的命令式 Viewer，
 * 并自管相机控制器以支持三种模式：
 *   - orbit：绕中心旋转，纵向限位防颠倒（默认，适合看单个物体）
 *   - tumble：任意方向自由翻滚（TrackballControls）
 *   - walk：WASD 第一人称漫游，从场景内部起步（适合室内房间）
 * 复用其渲染管线，不自写 splat 光栅化。
 */
export default function SplatViewer({
  url,
  format,
  alphaThreshold = 5,
  controlMode = 'orbit',
  viewerDefaults,
  onStats,
  onLoaded,
  onError,
}: Readonly<SplatViewerProps>) {
  const containerRef = useRef<HTMLDivElement>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const viewerRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const controlsRef = useRef<any>(null);
  const frameCenterRef = useRef(new THREE.Vector3());
  const frameRadiusRef = useRef(1);
  const [ready, setReady] = useState(false);

  // ── 创建查看器并加载场景（依赖 url/format/alpha；切换控制器不触发重载） ──
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    let disposed = false;
    let rafId = 0;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let viewer: any = null;
    setReady(false);

    try {
      viewer = new GaussianSplats3D.Viewer({
        rootElement: container,
        // 关闭共享内存可避免对 COOP/COEP 跨域隔离响应头的依赖
        sharedMemoryForWorkers: false,
        // 自管控制器（见下方 effect），以支持 Orbit/Trackball 切换
        useBuiltInControls: false,
        // GPU 排序在部分驱动/浏览器组合下静默失效导致 splat 不渲染，改用 WASM 排序更稳
        gpuAcceleratedSort: false,
        cameraUp: viewerDefaults?.cameraUp ?? [0, 1, 0],
        initialCameraPosition: viewerDefaults?.initialCameraPosition ?? [0, -1, 6],
        initialCameraLookAt: viewerDefaults?.initialCameraLookAt ?? [0, 0, 0],
      });
      viewerRef.current = viewer;

      viewer
        .addSplatScene(url, {
          format: FORMAT_ENUM[format],
          splatAlphaRemovalThreshold: alphaThreshold,
          showLoadingUI: true,
          progressiveLoad: false,
        })
        .then(() => {
          if (disposed) return;
          viewer.start();
          // 中性深灰背景，避免深色模型在纯黑底上看不清
          try {
            viewer.renderer?.setClearColor(new THREE.Color(0x3a4150), 1);
          } catch {
            /* 渲染器不可用时忽略 */
          }
          // 未显式指定相机位置时，按模型包围盒自动框住，确保任意模型都可见
          if (!viewerDefaults?.initialCameraPosition) {
            try {
              autoFrame();
            } catch (err) {
              console.error('[SplatViewer] autoFrame 失败:', err);
            }
          }
          setReady(true);
          onLoaded?.();
          runLoop();
        })
        .catch((e: unknown) => {
          if (!disposed) onError?.(e instanceof Error ? e.message : String(e));
        });
    } catch (e) {
      onError?.(e instanceof Error ? e.message : String(e));
    }

    /** 抗离群值的自动取景：采样高斯中心，取中位数为中心、95 分位距离为半径，
     *  忽略训练产生的远处"漂浮"高斯，避免包围盒被撑大导致相机过远而看不到模型。 */
    function autoFrame() {
      const mesh = viewer.splatMesh;
      const count: number = mesh?.getSplatCount?.() ?? 0;
      if (!count) return;
      const sampleN = Math.min(count, 20000);
      const stride = Math.max(1, Math.floor(count / sampleN));
      const xs: number[] = [];
      const ys: number[] = [];
      const zs: number[] = [];
      const tmp = new THREE.Vector3();
      for (let i = 0; i < count; i += stride) {
        mesh.getSplatCenter(i, tmp, true);
        xs.push(tmp.x);
        ys.push(tmp.y);
        zs.push(tmp.z);
      }
      const median = (a: number[]) => {
        const s = [...a].sort((p, q) => p - q);
        return s[Math.floor(s.length / 2)];
      };
      const center = new THREE.Vector3(median(xs), median(ys), median(zs));
      const dists = xs.map((_, i) =>
        Math.hypot(xs[i] - center.x, ys[i] - center.y, zs[i] - center.z),
      );
      dists.sort((p, q) => p - q);
      const radius = dists[Math.floor(dists.length * 0.95)] || 1;
      frameCenterRef.current.copy(center);
      frameRadiusRef.current = radius;

      const cam = viewer.camera;
      const fov = ((cam.fov ?? 60) * Math.PI) / 180;
      const dist = (radius / Math.sin(fov / 2)) * 1.3;
      const dir = new THREE.Vector3(0.5, 0.3, 1).normalize();
      cam.position.copy(center.clone().add(dir.multiplyScalar(dist)));
      cam.near = Math.max(dist / 1000, 0.01);
      cam.far = dist * 10;
      cam.lookAt(center);
      cam.updateProjectionMatrix?.();
    }

    function runLoop() {
      let frames = 0;
      let last = performance.now();
      let prev = last;
      const tick = () => {
        frames += 1;
        const now = performance.now();
        const delta = (now - prev) / 1000;
        prev = now;
        controlsRef.current?.update(delta); // FlyControls 需 delta；Orbit/Trackball 忽略
        if (now - last >= 1000) {
          let splatCount = 0;
          try {
            splatCount = viewer?.splatMesh?.getSplatCount?.() ?? 0;
          } catch {
            /* 某些版本无此方法，忽略 */
          }
          onStats?.({ fps: frames, splatCount });
          frames = 0;
          last = now;
        }
        rafId = requestAnimationFrame(tick);
      };
      rafId = requestAnimationFrame(tick);
    }

    return () => {
      disposed = true;
      cancelAnimationFrame(rafId);
      try {
        viewer?.dispose?.();
      } catch {
        /* ignore */
      }
      viewerRef.current = null;
      if (container) container.innerHTML = '';
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, format, alphaThreshold]);

  // ── 控制器：按 controlMode 切换 Orbit / Trackball / Fly(漫游)（不重载模型） ──
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!ready || !viewer?.camera || !viewer?.renderer) return;
    const dom = viewer.renderer.domElement;
    const center = frameCenterRef.current;
    const radius = frameRadiusRef.current;
    const cam = viewer.camera;
    let onResize: (() => void) | null = null;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let controls: any;

    if (controlMode === 'tumble') {
      controls = new TrackballControls(cam, dom);
      controls.rotateSpeed = 10.5;
      controls.zoomSpeed = 1.2;
      controls.panSpeed = 0.8;
      controls.staticMoving = true; // 关闭惯性，旋转更跟手
      controls.target.copy(center);
      controls.handleResize?.();
      onResize = () => controls.handleResize?.();
      window.addEventListener('resize', onResize);
    } else if (controlMode === 'walk') {
      // 第一人称漫游：把相机移到场景内部（中心），WASD 移动 + 拖拽转视角
      cam.position.copy(center);
      cam.near = Math.max(radius / 1000, 0.005);
      cam.far = radius * 20;
      cam.updateProjectionMatrix?.();
      controls = new FlyControls(cam, dom);
      controls.movementSpeed = radius * 0.6; // 移动速度随场景尺度自适应
      controls.rollSpeed = 0.6;
      controls.dragToLook = true; // 仅按住拖拽时转视角，避免乱飘
      controls.autoForward = false;
    } else {
      controls = new OrbitControls(cam, dom);
      controls.enableDamping = true;
      controls.dampingFactor = 0.1;
      controls.rotateSpeed = 0.5; // 降低默认过快的旋转速度
      controls.zoomSpeed = 0.8;
      controls.target.copy(center);
    }
    controls.update?.(0);
    controlsRef.current = controls;

    return () => {
      if (onResize) window.removeEventListener('resize', onResize);
      try {
        controls.dispose?.();
      } catch {
        /* ignore */
      }
      if (controlsRef.current === controls) controlsRef.current = null;
    };
  }, [controlMode, ready]);

  return <div ref={containerRef} className="absolute inset-0" />;
}
