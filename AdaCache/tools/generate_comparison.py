#!/usr/bin/env python3
import argparse
import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def _latest_metrics_file(root):
    candidates = []
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith('_metrics.json'):
                p = os.path.join(d, f)
                try:
                    candidates.append((p, os.path.getmtime(p)))
                except FileNotFoundError:
                    pass
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def collect_baseline(baseline_root):
    rows = []
    for name in os.listdir(baseline_root):
        m = re.search(r'_cot_(\d+)shot', name)
        if not m:
            continue
        shots = int(m.group(1))
        sub = os.path.join(baseline_root, name)
        mf = _latest_metrics_file(sub)
        if mf:
            js = read_json(mf)
            rows.append({
                'shots': shots,
                'acc': js.get('acc_numeric'),
                'em_string': js.get('em_string'),
                'total_time_s': js.get('total_time_s'),
                'avg_latency_s': js.get('avg_latency_s'),
                'avg_in_tokens': js.get('avg_in_tokens'),
                'avg_out_tokens': js.get('avg_out_tokens'),
            })
    rows.sort(key=lambda r: r['shots'])
    return rows

def collect_kv(kv_root):
    mf = _latest_metrics_file(kv_root)
    if not mf:
        return None
    js = read_json(mf)
    count = js.get('count') or js.get('n') or 0
    avg_latency = js.get('avg_latency_s')
    total_time = js.get('total_time_s')
    if total_time is None and avg_latency and count:
        total_time = avg_latency * count
    return {
        'acc': js.get('acc_numeric'),
        'em_string': js.get('em_string'),
        'avg_latency_s': avg_latency,
        'total_time_s': total_time,
        'avg_in_tokens': js.get('avg_in_tokens') or js.get('avg_kv_tokens'),
        'avg_out_tokens': js.get('avg_out_tokens'),
        'avg_shots': js.get('num_shots_mean') or js.get('avg_num_shots')
    }

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def _dataset_label_from_dir(outdir):
    return os.path.basename(outdir)

def plot_speedup_vs_shots(b_rows, kv, outdir):
    shots = [r['shots'] for r in b_rows]
    speed_total = []
    speed_lat = []
    for r in b_rows:
        bt = r.get('total_time_s')
        bl = r.get('avg_latency_s')
        speed_total.append(bt / kv['total_time_s'] if bt and kv['total_time_s'] else None)
        speed_lat.append(bl / kv['avg_latency_s'] if bl and kv['avg_latency_s'] else None)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(shots, speed_total, marker='o', label='Speedup (Total)')
    ax.plot(shots, speed_lat, marker='s', label='Speedup (Latency)')
    for x, y in zip(shots, speed_total):
        if y:
            ax.text(x, y, f'{y:.2f}x', fontsize=10, ha='center', va='bottom')
    for x, y in zip(shots, speed_lat):
        if y:
            ax.text(x, y, f'{y:.2f}x', fontsize=10, ha='center', va='top')
    ax.set_xlabel('Shots')
    ax.set_ylabel('Speedup (x)')
    ax.set_title(f'Speedup vs Shots ({_dataset_label_from_dir(outdir)})')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'speedup_vs_shots.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_speedup_bars(b_rows, kv, outdir):
    shots = [r['shots'] for r in b_rows]
    speed_total = []
    speed_lat = []
    for r in b_rows:
        bt = r.get('total_time_s')
        bl = r.get('avg_latency_s')
        speed_total.append(bt / kv['total_time_s'] if bt and kv['total_time_s'] else None)
        speed_lat.append(bl / kv['avg_latency_s'] if bl and kv['avg_latency_s'] else None)
    x = np.arange(len(shots))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11,6))
    ax.bar(x - w/2, speed_total, width=w, label='Total', color='#2ecc71')
    ax.bar(x + w/2, speed_lat, width=w, label='Latency', color='#3498db')
    ymax = 0
    for i, v in enumerate(speed_total):
        if v:
            ax.text(x[i]-w/2, min(v, v*0.98), f'{v:.2f}x', ha='center', va='bottom', fontsize=10)
            ymax = max(ymax, v)
    for i, v in enumerate(speed_lat):
        if v:
            ax.text(x[i]+w/2, min(v, v*0.98), f'{v:.2f}x', ha='center', va='bottom', fontsize=10)
            ymax = max(ymax, v)
    ax.set_ylim(0, ymax*1.15 if ymax else None)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in shots])
    ax.set_xlabel('Shots')
    ax.set_ylabel('Speedup (x)')
    ax.set_title(f'Speedup (Bars) ({_dataset_label_from_dir(outdir)})')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'speedup_bars.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_accuracy_vs_shots(b_rows, kv, outdir):
    shots = [r['shots'] for r in b_rows]
    accs = [r.get('acc') for r in b_rows]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(shots, accs, marker='o', label='Baseline Accuracy')
    if kv.get('acc') is not None:
        ax.axhline(y=kv['acc'], color='r', linestyle='--', label='KV Accuracy')
    ymin = 0
    ymax = max([a for a in accs if a is not None] + [kv.get('acc') or 0])
    ax.set_ylim(ymin, ymax*1.15 if ymax else 1)
    ax.set_xlabel('Shots')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Accuracy vs Shots ({_dataset_label_from_dir(outdir)})')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'accuracy_vs_shots.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_accuracy_em_dual(b_rows, kv, outdir):
    shots = [r['shots'] for r in b_rows]
    accs = [r.get('acc') for r in b_rows]
    ems = [r.get('em_string') for r in b_rows]
    fig, ax = plt.subplots(figsize=(11,6))
    ax.plot(shots, accs, marker='o', label='Baseline Accuracy', color='#9b59b6')
    ax.plot(shots, ems, marker='s', label='Baseline EM', color='#e67e22')
    if kv.get('acc') is not None:
        ax.axhline(y=kv['acc'], color='#2ecc71', linestyle='--', label='KV Accuracy')
    if kv.get('em_string') is not None:
        ax.axhline(y=kv['em_string'], color='#e74c3c', linestyle='--', label='KV EM')
    ymax = max([a for a in accs if a is not None] + [e for e in ems if e is not None] + [kv.get('acc') or 0, kv.get('em_string') or 0])
    ax.set_ylim(0, ymax*1.15 if ymax else 1)
    ax.set_xlabel('Shots')
    ax.set_ylabel('Score')
    ax.set_title(f'Accuracy & EM vs Shots ({_dataset_label_from_dir(outdir)})')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'accuracy_em_vs_shots.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_latency_vs_shots(b_rows, kv, outdir):
    shots = [r['shots'] for r in b_rows]
    lats = [r.get('avg_latency_s') for r in b_rows]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(shots, lats, marker='o', label='Baseline Latency')
    if kv.get('avg_latency_s') is not None:
        ax.axhline(y=kv['avg_latency_s'], color='r', linestyle='--', label='KV Latency')
    ymax = max([l for l in lats if l is not None] + [kv.get('avg_latency_s') or 0])
    ax.set_ylim(0, ymax*1.15 if ymax else None)
    ax.set_xlabel('Shots')
    ax.set_ylabel('Average Latency (s)')
    ax.set_title(f'Latency vs Shots ({_dataset_label_from_dir(outdir)})')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'latency_vs_shots.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_tokens_vs_shots(b_rows, kv, outdir):
    shots = [r['shots'] for r in b_rows]
    toks = [r.get('avg_in_tokens') for r in b_rows]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(shots, toks, marker='o', label='Baseline Avg Input Tokens')
    if kv.get('avg_in_tokens') is not None:
        ax.axhline(y=kv['avg_in_tokens'], color='r', linestyle='--', label='KV Avg Tokens')
    ymax = max([t for t in toks if t is not None] + [kv.get('avg_in_tokens') or 0])
    ax.set_ylim(0, ymax*1.15 if ymax else None)
    ax.set_xlabel('Shots')
    ax.set_ylabel('Avg Input Tokens')
    ax.set_title(f'Tokens vs Shots ({_dataset_label_from_dir(outdir)})')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'tokens_vs_shots.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_tokens_stacked(b_rows, kv, outdir):
    shots = [r['shots'] for r in b_rows]
    in_t = [r.get('avg_in_tokens') or 0 for r in b_rows]
    out_t = [r.get('avg_out_tokens') or 0 for r in b_rows]
    x = np.arange(len(shots))
    fig, ax = plt.subplots(figsize=(11,6))
    ax.bar(x, in_t, label='Input Tokens', color='#34495e')
    ax.bar(x, out_t, bottom=in_t, label='Output Tokens', color='#95a5a6')
    if kv.get('avg_in_tokens') is not None:
        ax.axhline(y=kv['avg_in_tokens'], color='#2ecc71', linestyle='--', label='KV Avg Tokens')
    ymax = max([i+o for i,o in zip(in_t,out_t)] + [kv.get('avg_in_tokens') or 0])
    ax.set_ylim(0, ymax*1.15 if ymax else None)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in shots])
    ax.set_xlabel('Shots')
    ax.set_ylabel('Tokens')
    ax.set_title(f'Token Usage (Stacked) ({_dataset_label_from_dir(outdir)})')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'tokens_stacked.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_bubble_tradeoff(b_rows, kv, outdir):
    shots = [r['shots'] for r in b_rows]
    accs = [r.get('acc') for r in b_rows]
    lats = [r.get('avg_latency_s') for r in b_rows]
    sizes = np.array(shots) * 25
    fig, ax = plt.subplots(figsize=(11,7))
    ax.scatter(lats, accs, s=sizes, c='#3498db', alpha=0.7, edgecolors='black', linewidth=1.5, label='Baseline')
    if kv.get('acc') is not None and kv.get('avg_latency_s') is not None:
        ax.scatter([kv['avg_latency_s']], [kv['acc']], s=300, c='#e74c3c', marker='*', edgecolors='black', linewidth=1.5, label='KV')
        ax.axhline(y=kv['acc'], color='#e74c3c', linestyle='--', alpha=0.5)
        ax.axvline(x=kv['avg_latency_s'], color='#e74c3c', linestyle='--', alpha=0.5)
    ax.set_xlabel('Average Latency (s)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Efficiency–Accuracy Trade-off ({_dataset_label_from_dir(outdir)})')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.0f}%'))
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'efficiency_accuracy_tradeoff.png'), dpi=150, bbox_inches='tight')
    plt.close()

def select_shots_for_radar(b_rows):
    shots = [r['shots'] for r in b_rows]
    picks = []
    for s in [8,16,32,256,512]:
        if s in shots:
            picks.append(s)
    if not picks:
        picks = shots[:min(5, len(shots))]
    return picks

def plot_radar(b_rows, kv, outdir):
    picks = select_shots_for_radar(b_rows)
    cats = ['Accuracy','1/Latency','TokenEff','ShotEff']
    angles = [n/float(len(cats))*2*np.pi for n in range(len(cats))]
    angles += angles[:1]
    accs = []
    inv_lats = []
    tok_eff = []
    shot_eff = []
    for s in picks:
        r = next((x for x in b_rows if x['shots']==s), None)
        accs.append(r.get('acc') or 0)
        inv_lats.append(1/(r.get('avg_latency_s') or 1))
        tok_eff.append(1/(r.get('avg_in_tokens') or 1))
        shot_eff.append(1/max(s,1))
    accs = np.array(accs); inv_lats = np.array(inv_lats); tok_eff = np.array(tok_eff); shot_eff = np.array(shot_eff)
    def norm(v):
        m = v.max() if v.size else 1
        return v/(m if m!=0 else 1)
    accs = norm(accs); inv_lats = norm(inv_lats); tok_eff = norm(tok_eff); shot_eff = norm(shot_eff)
    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    colors = ['#3498db','#2ecc71','#e74c3c','#9b59b6','#f1c40f']
    for i, s in enumerate(picks):
        vals = [accs[i],inv_lats[i],tok_eff[i],shot_eff[i]]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, label=f'Baseline {s}-shot', color=colors[i%len(colors)])
        ax.fill(angles, vals, alpha=0.15, color=colors[i%len(colors)])
    kv_vals = [kv.get('acc') or 0, 1/(kv.get('avg_latency_s') or 1), 1/(kv.get('avg_in_tokens') or 1), 1/(kv.get('avg_shots') or 1)]
    kv_vals = np.array(kv_vals)
    kv_vals = norm(kv_vals)
    kv_vals = list(kv_vals)+[kv_vals[0]]
    ax.plot(angles, kv_vals, linewidth=3, label='KV', color='#e67e22')
    ax.fill(angles, kv_vals, alpha=0.15, color='#e67e22')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats)
    ax.set_title(f'Radar Comparison ({_dataset_label_from_dir(outdir)})')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35,1.05))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'radar_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

def write_summary_csv(b_rows, kv, outdir):
    rows = []
    for r in b_rows:
        bt = r.get('total_time_s')
        bl = r.get('avg_latency_s')
        st = bt / kv['total_time_s'] if bt and kv['total_time_s'] else None
        sl = bl / kv['avg_latency_s'] if bl and kv['avg_latency_s'] else None
        rows.append({
            'shots': r['shots'],
            'baseline_acc': r.get('acc'),
            'baseline_em_string': r.get('em_string'),
            'baseline_total_time_s': bt,
            'baseline_avg_latency_s': bl,
            'baseline_avg_in_tokens': r.get('avg_in_tokens'),
            'baseline_avg_out_tokens': r.get('avg_out_tokens'),
            'kv_acc': kv.get('acc'),
            'kv_em_string': kv.get('em_string'),
            'kv_total_time_s': kv.get('total_time_s'),
            'kv_avg_latency_s': kv.get('avg_latency_s'),
            'kv_avg_shots': kv.get('avg_shots'),
            'kv_avg_in_tokens': kv.get('avg_in_tokens'),
            'kv_avg_out_tokens': kv.get('avg_out_tokens'),
            'speedup_total': st,
            'speedup_latency': sl,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, 'summary_comparison.csv'), index=False)

def _run_for_dataset(dataset, args):
    dataset_map = {
        'openai_gsm8k_main': {
            'baseline_dir': 'shot_openai_gsm8k_main_cot',
            'kv_dir': 'manyshot_kv_openai_gsm8k_main',
            'out': 'gsm8k',
            'model_slug': 'qwen2.5_7B_gsm8k',
        },
        'AI-ModelScope_CoT-Collection_default': {
            'baseline_dir': 'shot_AI-ModelScope_CoT-Collection_default_cot',
            'kv_dir': 'manyshot_kv_AI-ModelScope_CoT-Collection_default',
            'out': 'iirc',
            'model_slug': 'qwen2.5_7B_iirc',
        },
        'gsm8k': {
            'baseline_dir': 'shot_openai_gsm8k_main_cot',
            'kv_dir': 'manyshot_kv_openai_gsm8k_main',
            'out': 'gsm8k',
            'model_slug': 'qwen2.5_7B_gsm8k',
        },
        'iirc': {
            'baseline_dir': 'shot_AI-ModelScope_CoT-Collection_default_cot',
            'kv_dir': 'manyshot_kv_AI-ModelScope_CoT-Collection_default',
            'out': 'iirc',
            'model_slug': 'qwen2.5_7B_iirc',
        },
    }
    cfg = dataset_map.get(dataset) or dataset_map['openai_gsm8k_main']

    base_baseline_root = f"/data/oujie/oujie-data/shareShot/AdaCache/shot_baseline/Qwen2.5_7B/outputs/{cfg['baseline_dir']}"
    base_kv_root = f"/data/oujie/oujie-data/shareShot/AdaCache/Qwen2.5-7B/{cfg['kv_dir']}/{cfg['model_slug']}"
    out_dir_default = f"/data/oujie/oujie-data/shareShot/AdaCache/Qwen2.5-7B/pics/{cfg['out']}"

    baseline_root = args.baseline_root or base_baseline_root
    kv_root = args.kv_root or base_kv_root
    output_dir = args.output_dir or out_dir_default

    ensure_dir(output_dir)

    b_rows = collect_baseline(baseline_root)
    if not b_rows:
        raise RuntimeError(f'未找到 baseline 指标文件: {baseline_root}')
    kv = collect_kv(kv_root)
    if not kv:
        raise RuntimeError(f'未找到 KV 指标文件: {kv_root}')

    plot_speedup_vs_shots(b_rows, kv, output_dir)
    plot_speedup_bars(b_rows, kv, output_dir)
    plot_accuracy_vs_shots(b_rows, kv, output_dir)
    plot_accuracy_em_dual(b_rows, kv, output_dir)
    plot_latency_vs_shots(b_rows, kv, output_dir)
    plot_tokens_vs_shots(b_rows, kv, output_dir)
    plot_tokens_stacked(b_rows, kv, output_dir)
    plot_bubble_tradeoff(b_rows, kv, output_dir)
    plot_radar(b_rows, kv, output_dir)
    write_summary_csv(b_rows, kv, output_dir)

    lines = []
    lines.append('Method,Shots,Accuracy,Em_String,Total_Time_s,Avg_Latency_s,Speedup_Total,Speedup_Latency,Avg_Input_Tokens,Avg_Output_Tokens')
    for r in b_rows:
        st = (r.get('total_time_s') or 0) / (kv.get('total_time_s') or 1)
        sl = (r.get('avg_latency_s') or 0) / (kv.get('avg_latency_s') or 1)
        lines.append(
            f"Baseline {r['shots']}-shot,{r['shots']},{r.get('acc')},{r.get('em_string')},{r.get('total_time_s')},{r.get('avg_latency_s')},{st:.2f},{sl:.2f},{r.get('avg_in_tokens')},{r.get('avg_out_tokens')}"
        )
    lines.append(
        f"KV Cache (Ours),{kv.get('avg_shots')},{kv.get('acc')},{kv.get('em_string')},{kv.get('total_time_s')},{kv.get('avg_latency_s')},1.00,1.00,{kv.get('avg_in_tokens')},{kv.get('avg_out_tokens')}"
    )
    with open(os.path.join(output_dir, 'comparison_table.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('输出目录:', output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='all')
    parser.add_argument('--model', default='Qwen2.5-7B')
    parser.add_argument('--baseline_root')
    parser.add_argument('--kv_root')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset != 'all' else ['gsm8k', 'iirc']
    for ds in datasets:
        _run_for_dataset(ds, args)

if __name__ == '__main__':
    main()
