# figures_v2 数据溯源与变更说明

生成时间：2026-10-08 ｜ 脚本：`make_figures_v2.py`（配套 `probe_fused_flops.py`）
生成命令：`python make_figures_v2.py --out figures_v2`

本目录是 `visdrone-yolo26/results/figures/` 的**重画版**：图的主题、面板划分与旧版一致，
数据更新到最新权威口径。逐张说明如下。

---

## 0. 出图脚本怎么用（避免两处数据分叉）

`make_figures_v2.py` 现在是**唯一的数据源与出图入口**，支持 `--only {all,figs,tradeoff}`：

| 命令 | 产物 |
|---|---|
| `python scripts/make_figures_v2.py --out <dir>` | 全部 7 张 |
| `python scripts/make_figures.py` | 四联图 + fig1..4（**已改为转发到 v2**） |
| `python scripts/make_tradeoff_figure.py` | fig_tradeoff / fig_tradeoff_latency（**已改为转发到 v2**） |

> 这两个旧脚本里原有的硬编码效率数据（未 fuse 口径）已删除，改成 `runpy` 转发到
> `make_figures_v2.py`，并且**用脚本自身位置定位仓库**（不依赖 `YOLO_ROOT`，
> 否则会误写到工作目录）。因此 `python scripts/make_figures.py` 这条 README 里的命令
> 现在产出的是最新口径的图，不会再退回旧数字。

---

## 1. 哪些图真的过时了（审计结论）

| 图 | 数据源 | 是否过时 | 原因 |
|---|---|---|---|
| `fig1_size_missrate` | `results/oracle_logs/*.log`（2026-09-27） | **否** | 数据未变，重画后与旧图逐值一致 |
| `fig2_oracle` | 同上 | **否** | 同上 |
| `fig4_gt_size` | `datasets/visdrone/val.txt` | **否** | 数据集未变 |
| `fig3_pareto` | 硬编码效率数据 | **是** | 用的是**未 fuse** 口径 |
| `fig_all` | 上述四者 | **部分** | 仅 (c) 面板过时 |
| `fig_tradeoff` | 硬编码效率数据 | **是** | 全部 GFLOPs / 延迟都偏高 |
| `fig_tradeoff_latency` | 同上 | **是** | 同上 |
| `viz_heads_baseline_100ep`、`viz_models_compare` | 需 GPU + 数据集重跑 | 未处理 | 定性可视化，非数据对比图；本轮 GPU 不可用 |

> 根因：2026-10-02 的效率口径更正。旧图在**未 fuse** 的模型上测 GFLOPs/延迟，
> 会把推理时不用的 one2one 分支也算进去（`ultralytics/nn/modules/head.py:183-190`），
> 因此系统性偏高。`Detect.fuse()`（`head.py:277`）才删除该分支。

---

## 2. 效率数据表（图 3 / tradeoff 系列的唯一数据来源）

| 配置 | GFLOPs | 前向延迟 (ms) | mAP50-95 (md300) | (md1000) | 数据来源 |
|---|---|---|---|---|---|
| baseline@640 | **5.32** | **2.86** | 0.182 | 0.186 | README / RESULTS_FINAL.md（权威） |
| P2@640 | **6.57** | **3.47** | 0.198 | 0.203 | 同上 |
| baseline@960（640 训 / 960 推） | **12.31** | ~~3.73~~ * | 0.224 | — | GFLOPs 本机 CPU 实测；延迟推算 |
| P2-pruned（P2 neck + 三层头） | **6.06** | ~~3.19~~ * | 0.1888 | — | GFLOPs 本机 CPU 实测；延迟与 mAP 沿用旧值 |
| P2@960 | **15.13** | **6.44** | 0.263 | 0.269 | README / RESULTS_FINAL.md（权威） |

* 标注 `*` 的延迟为**推算值**，图内已加脚注。GPU 恢复后应重测。

### 2.1 GFLOPs 的实测与自检（`probe_fused_flops.py`）

在 CPU 上先 `.fuse()` 再 `get_flops()`，与文档已有值**完全吻合**，因此用于补算另两个点：

| 配置 | 本次实测 | 文档值 | 自检 |
|---|---|---|---|
| baseline@640 | 5.32 | 5.32 | ✅ |
| P2@640 | 6.57 | 6.57 | ✅ |
| P2@960 | 15.13 | 15.13 | ✅ |
| baseline@960（640 训/960 推） | **12.31** | — | 新增 |
| P2-pruned | **6.06** | — | 新增 |

> 顺带修正：旧图给 baseline@960 推理填的是 **13.6** GFLOPs（未 fuse），实际 **12.31**；
> P2-pruned 旧图填 **6.6**（未 fuse），实际 **6.06**（注意 6.6 恰好是 P2@640 fuse 后的值，两者易混）。

### 2.2 两个推算延迟的算法

按**同架构在 640 上实测的 fuse 修正系数**折算，透明可复核：

| 点 | 旧（未 fuse）延迟 | 参照架构 | fuse 系数 | 折算结果 |
|---|---|---|---|---|
| baseline@960 推理 | 5.96 ms | baseline@640：4.57 → 2.86 | 2.86/4.57 = 0.6258 | 5.96 × 0.6258 = **3.73 ms** |
| P2-pruned | 5.40 ms | P2@640：5.88 → 3.47 | 3.47/5.88 = 0.5901 | 5.40 × 0.5901 = **3.19 ms** |

---

## 3. 未过时的三张图（数据逐值核对）

### fig1 分尺寸漏检率（读取 `head=o2m` 段，conf=0.001）

| 尺寸 | baseline@640 | P2@640 | P2@960 |
|---|---|---|---|
| 0–8 px | 0.835 | 0.718 | 0.587 |
| 8–16 px | 0.593 | 0.495 | 0.380 |
| 16–32 px | 0.364 | 0.321 | 0.237 |
| 32–64 px | 0.193 | 0.178 | 0.131 |
| ≥64 px | 0.097 | 0.095 | 0.086 |

与 README「关键发现」表逐值一致。

### fig2 oracle 四象限（COCO 协议）

| 理想化 | baseline@640 | P2@640 | P2@960 |
|---|---|---|---|
| 真实 | 0.1731 | 0.1876 | 0.2474 |
| +定位完美 | +0.132 | +0.137 | +0.165 |
| +分类完美 | +0.101 | +0.105 | +0.113 |
| **+不漏检** | **+0.582** | **+0.551** | **+0.456** |
| +无误检 | +0.058 | +0.061 | +0.068 |

与 `docs/RESULTS_FINAL.md` §3 一致。

### fig4 GT 尺寸分布
`<8px` 占比在 letterbox 后由 @640 的 31% 降到 @960 的 13%，与旧图一致。

---

## 4. 已知待办（不阻塞出图）

1. **两个推算延迟待重测**：GPU 恢复后跑
   `python scripts/bench_models.py --weights <w> --imgsz 960`（需先修正该脚本的 fuse 问题）。
2. **`P2-pruned` 的 mAP 0.1888 无权威出处**：该值只在旧图 `make_tradeoff_figure.py` 中出现，
   `runs/visdrone/val_p2_model_P345_only/` 为空目录，训练日志与 JSON 里都查不到。
   本轮**沿用旧值**，但引用前建议补一次 `eval_p345_only.py` 复核。
3. **`viz_*` 两张定性图**未重新生成（需 GPU + 数据集）。
4. 仓库里 `results/FINAL_COMPARISON.md` 仍写着「17.3 vs 7.7 GFLOPs」（未 fuse 旧值），
   属于文字表格的同类过时问题，本轮未改动。
