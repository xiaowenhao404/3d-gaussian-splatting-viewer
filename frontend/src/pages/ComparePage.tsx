import { useEffect, useState } from 'react';
import SplatViewer from '@/components/SplatViewer';
import { fetchModels } from '@/api/client';
import type { SplatModel } from '@/types';

// 实测消融数据（tandt/train，RTX 4070 Laptop 8GB，data_factor=4）
const STEP_ABLATION = [
  { step: '3k', psnr: '19.8', time: '~35s', note: '仅测流程，明显模糊' },
  { step: '7k', psnr: '21.9', time: '~2min', note: '主体可辨，边缘偏软' },
  { step: '30k', psnr: '26.1', time: '~15min', note: '清晰，文字可读' },
];
const REG_ABLATION = [
  { name: '无正则项', gauss: '287,301', size: '68 MB', note: '大量飞散尖刺/残片' },
  { name: '有正则项 (opacity+scale)', gauss: '226,174', size: '50 MB', note: '残片显著减少' },
];

export default function ComparePage() {
  const [models, setModels] = useState<SplatModel[]>([]);
  const [leftId, setLeftId] = useState('');
  const [rightId, setRightId] = useState('');

  useEffect(() => {
    fetchModels().then((ms) => {
      setModels(ms);
      if (ms[0]) setLeftId(ms[0].id);
      if (ms[1]) setRightId(ms[1].id);
    });
  }, []);

  const left = models.find((m) => m.id === leftId);
  const right = models.find((m) => m.id === rightId);

  const picker = (value: string, onChange: (v: string) => void) => (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="rounded bg-slate-800 px-2 py-1 text-sm text-slate-200"
    >
      {models.map((m) => (
        <option key={m.id} value={m.id}>
          {m.name}
        </option>
      ))}
    </select>
  );

  return (
    <div className="mx-auto max-w-6xl p-6">
      <h1 className="text-2xl font-semibold text-white">引擎 / 参数对比</h1>
      <p className="mt-2 text-sm text-slate-400">
        并排对比两个模型，并展示训练参数对质量的消融实验（实测数据）。
      </p>

      {/* 并排查看器 */}
      <div className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-2">
        {[
          { side: 'left', m: left, set: setLeftId, val: leftId },
          { side: 'right', m: right, set: setRightId, val: rightId },
        ].map((side) => (
          <div key={side.side} className="rounded-xl border border-slate-800 bg-panel">
            <div className="flex items-center gap-2 border-b border-slate-800 p-2">
              {picker(side.val, side.set)}
            </div>
            <div className="relative h-80 w-full bg-black">
              {side.m ? (
                <SplatViewer
                  key={side.m.id}
                  url={side.m.url}
                  format={side.m.format}
                  controlMode="orbit"
                />
              ) : (
                <div className="flex h-full items-center justify-center text-slate-600">
                  暂无模型
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* 消融实验表 */}
      <div className="mt-8 grid grid-cols-1 gap-6 md:grid-cols-2">
        <div>
          <h2 className="mb-2 text-lg font-medium text-white">训练步数消融</h2>
          <table className="w-full text-sm text-slate-300">
            <thead className="text-slate-400">
              <tr className="border-b border-slate-700 text-left">
                <th className="py-1">步数</th>
                <th>PSNR</th>
                <th>耗时</th>
                <th>质量</th>
              </tr>
            </thead>
            <tbody>
              {STEP_ABLATION.map((r) => (
                <tr key={r.step} className="border-b border-slate-800">
                  <td className="py-1 font-medium text-white">{r.step}</td>
                  <td>{r.psnr}</td>
                  <td>{r.time}</td>
                  <td className="text-slate-400">{r.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <h2 className="mb-2 text-lg font-medium text-white">MCMC 正则项消融 (30k)</h2>
          <table className="w-full text-sm text-slate-300">
            <thead className="text-slate-400">
              <tr className="border-b border-slate-700 text-left">
                <th className="py-1">配置</th>
                <th>高斯数</th>
                <th>体积</th>
                <th>效果</th>
              </tr>
            </thead>
            <tbody>
              {REG_ABLATION.map((r) => (
                <tr key={r.name} className="border-b border-slate-800">
                  <td className="py-1 font-medium text-white">{r.name}</td>
                  <td>{r.gauss}</td>
                  <td>{r.size}</td>
                  <td className="text-slate-400">{r.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <p className="mt-4 text-xs text-slate-500">
        测试环境：RTX 4070 Laptop 8GB · CUDA 11.8 · gsplat 1.5.2 · tandt/train (301 图, data_factor=4)。
        PSNR 为训练视角指标。
      </p>
    </div>
  );
}
