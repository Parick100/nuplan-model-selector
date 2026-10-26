# 第一次 pre 规划与落地方案

> 项目：基于扩散模型与流匹配的车辆轨迹生成式预测（nuPlan）
> 目标：下周第一次 pre（4–7 分钟）通过；并在 5 周内产出一份可辩护的对照实验
> 本文档基于**实测**（不是 proposal 里的假设）写成

---

## 0. 三个必须先拍板的决定（TL;DR）

| # | 决定 | 为什么现在就要定 |
|---|---|---|
| **D1** | **立刻补数据**：现在手上只有 1 个 log（198 秒），proposal 里"5 万~20 万条轨迹对"是按**完整 mini 划分**算的，我们只有 1/180 | 这是唯一的关键路径。训练集只有 ~44 条车轨迹的话，一切对比都不可信 |
| **D2** | **课程项目不依赖 nuplan-devkit**：直接用 `sqlite3` 读 DB（已验证可行） | devkit 要 hydra + Python 3.9 + 地图包 + 大量依赖；而模型需要的量全在 `lidar_box` 表里。绕开它 = 砍掉最大的一类工程风险 |
| **D3** | **对比设计升级**：加"多模态回归基线(WTA)"，并且**在等 NFE 下比较**扩散 vs 流匹配 | 否则"生成模型比 MSE 回归准"和"流匹配步数更少"这两条结论**在方法上都是必然成立的**，老师一问就塌 |

---

## 1. 实测盘点（截至今天）

### 1.1 数据

| 项 | 实测值 | 备注 |
|---|---|---|
| 数据文件 | `2021.10.11.08.31.07_veh-50_01750_01948.db`，22.2 MB | 位于 `C:\Users\lenovo\Downloads\nuplan-v1.1_mini\` |
| 地点 / 日期 | `sg-one-north`（新加坡），2021-10-11 | `log.map_version = sg-one-north` |
| 时长 | **198 秒**（`lidar_pc` 3960 帧 @ **20 Hz**） | 不是 10 Hz，见 §3.1 |
| `ego_pose` | 20000 行 / 200 秒 → 100 Hz | |
| 相机 | 8 路 × 10 Hz | 本次项目不用 |
| `lidar_box`（智能体状态） | 56971 行 | ← **本项目唯一需要的表** |
| 轨迹（track） | 395 条：**vehicle 44**、pedestrian 110、traffic_cone 135、barrier 43、generic_object 39、czone_sign 17、bicycle 7 | 车辆样本极少 |
| 连续 10 秒窗口（0.5 s 步长） | 共 2241 个，其中 **vehicle 仅 482 个** | 且高度重叠，非独立样本 |
| `scenario_tag` | 4745 行、27 种类型 | **免费拿到场景真值标签** |
| 地图 | **缺失** | 需 `nuplan-maps-v1.0/sg-one-north/9.17.1964/map.gpkg` |

### 1.2 关键结论

1. **proposal 的数据量估算本身没错，错的是我们手上的份额。** 198 s/log，mini 划分约 10 小时 → 约 180 个 log → 约 8.8 万个车辆窗口，正好落在 proposal 说的"5 万~20 万"区间。问题是**我们只有 180 个里的 1 个**。
2. **482 个 vehicle 窗口来自 44 条 track，平均每条只有约 4.5 秒可用**。若按 track 划分 train/val/test，测试集只有 ~7 条轨迹 —— 无法支撑 proposal 里"多 seed 平均"的实验强度。
3. **`scenario_tag` 是意外之喜**：`on_intersection`(697)、`traversing_intersection`(654)、`on_stopline_traffic_light`(452)、`stationary`(370)、`starting_right_turn`(20)、`high_lateral_acceleration`(27)…… 意味着 proposal 里 4.2(4)"按直行/弯道/路口/跟车分类统计"**不需要靠启发式猜，有官方场景标签**。这条要写进 pre。
4. **地图包不在本机**（`map.zip` 是 Java 最短路作业，无关）。地图路径已被官方文档确认：`dataset/maps/sg-one-north/9.17.1964/map.gpkg`。
5. **`lidar_box` 直接给出 `x, y, yaw, vx, vy, width, length, height, confidence`** —— 抽取管线用纯 SQL 就够了。

### 1.3 代码与算力

| 项 | 状态 |
|---|---|
| `C:\Users\lenovo\Desktop\项目\diffusion_planner\` | **只是模型模块的抽取版**（encoder / dit / decoder / planner / diffusion_planner 五个文件），缺 `model/module/mixer.py`、`data_process/`、`utils/config`，且依赖未安装的 `timm` → **不能直接运行**，仅作结构参考 |
| `D:\Diffusion ES\diffusion-es-master\` | 完整 repo，内含 vendored `nuplan-devkit` + `tuplan_garage`，官方 `docs/dataset_setup.md` 确认了数据与地图目录结构 |
| **用哪个环境** | ✅ **`conda activate myenv`** → `G:\Anaconda\envs\myenv\python.exe`（Python 3.12.6） |
| **myenv 已装** | **torch 2.8.0+cu129（CUDA 可用）**、numpy 2.2 / scipy / pandas / sklearn / matplotlib / h5py / einops / pyarrow |
| myenv 缺 | timm、msgpack、sqlalchemy、hydra、shapely、geopandas、fiona、zarr |
| ❌ 别用 | `ml-env`（Python 3.8，**无 torch**）；base 环境（无 torch） |
| GPU | **RTX 4060 Laptop，8.6 GB，CUDA 已实测可用**；32 核 CPU |
| 结论 | **算力完全不是风险**，环境开箱即用。缺的包里 **我们一个都不需要**（见下） |

**为什么缺的包不用装：**

- `msgpack` / `sqlalchemy` / `hydra` —— 只有 nuplan-devkit 才需要。我们用 stdlib `sqlite3` 直读 DB，**零依赖**。这条实测结果直接确认了 §0 的 D2 决策。
- `timm` —— `diffusion_planner` 的代码用 `timm.models.layers.Mlp` 和 `timm.layers.DropPath`。我们自己写 10 行 `Mlp` + `DropPath` 即可，代码更自包含、更适合课程项目阅读与答辩。**不建议装。**
- `geopandas` / `fiona` —— 带 GDAL，Windows 上装起来很痛。地图阶段**只装 `shapely`**（纯 wheel、无 GDAL）即可解 WKB；GeoPackage 本身就是 SQLite，可以用 stdlib `sqlite3` 读。**不要装 geopandas。**

---

## 2. 数据方案

### 2.1 优先级最高的两件事（今天就做）

```
① 到 https://www.nuscenes.org/nuplan#download 登录（需同意 Terms of Use）
   下载 nuplan-v1.1 mini 全部 .db（我们手上是其中最后一个）
   目标：≥ 30 个 log，至少覆盖 2 个城市 → 才能做"按城市留出"的划分
② 下载 nuplan-maps-v1.0（至少 sg-one-north 的 map.gpkg）
   否则车道线条件、路口分类、off-road 失败分析都做不了
```

**备选（若下载受限）**：向正式科研项目组索要 Diffusion-ES 已缓存的 nuPlan cache（`scripts/cache.sh` 生成物），或直接用已抽好的 npz。

**降级方案（若两条都不通）**：改用"**全智能体预测**"（每个 track 都是目标），窗口数 2241 → 用 `scenario_tag` 分层；同时把预测时域缩短为 2 s 历史 + 4 s 未来来提高产量；并在报告中**明确写清 n 与划分方式**，把项目定位为"方法学对照研究"，而不是"刷指标"。这条路能交付，但说服力弱一档。

### 2.2 抽取管线：纯 SQLite，不装 devkit（已验证可读）

```python
# 目标智能体：vehicle；20 Hz 抽成 10 Hz 后，2 s 历史 = 20 点，8 s 未来 = 80 点
con = sqlite3.connect(db_path)
frames = [t for (t,) in con.execute("SELECT token FROM lidar_pc ORDER BY timestamp")]
fidx = {t: i for i, t in enumerate(frames)}          # 帧号
boxes = con.execute("""SELECT lidar_pc_token, track_token, x, y, yaw, vx, vy,
                              width, length, height
                       FROM lidar_box""")            # 全量 56971 行，秒级
```

流程：
1. 建 `track → {frame_idx: (x, y, yaw, vx, vy, w, l)}`；
2. 目标智能体筛选：`category.name == 'vehicle'`，窗口 `[t0-2s, t0+8s]` 内**逐帧连续**（不连续就丢）；
3. 同步取同帧内其它智能体作邻居（**只用历史，绝不能用邻居的未来** —— 这是数据泄漏点）；
4. 坐标变换：目标车 t0 时刻位置为原点、heading 对齐 +x，全场景旋转平移过去；
5. 地图（Phase 2）：GeoPackage **本身就是 SQLite**，可用 `sqlite3` 或 `fiona` 读，取 t0 前 50 m 内的 lane / lane_connector 折线；
6. 缓存为**每个 log 一个 `.npz`**（不要 20 万个小文件，Windows 上 I/O 会拖死）；
7. 划分：**按 log / 城市划分，绝不按帧随机划分**（否则同一条轨迹的相邻帧会同时出现在 train 和 test，指标虚高）。

### 2.3 冻结的特征契约（A/B/C/D 四人并行开发的接口）

```python
sample = {
    "hist":       (20, 4)      float32,  # x, y, vx, vy（目标车自身坐标系）
    "hist_mask":  (20,)        bool,
    "neigh":      (10, 20, 6)  float32,  # 最近 10 个邻居的历史 x,y,vx,vy,w,l
    "neigh_mask": (10,)        bool,
    "lanes":      (20, 20, 6)  float32,  # Phase 2：20 条车道 × 20 点
    "lane_mask":  (20,)        bool,
    "future":     (80, 2)      float32,  # 真值
    "scene_tags": (T,)         uint8,    # 来自 scenario_tag
    "log_id":     int,
}
```

**这张表在第一周末冻结**。冻结后 C 和 D 可以拿随机数造假的 batch 立刻开始写损失函数和采样器，不必等 A 把数据抽完。这是避免"四人串行等待"的关键。

---

## 3. 方法方案

### 3.1 任务定义（相对 proposal 的两处澄清）

- 时域：**2 s 历史 → 8 s 未来**（10 Hz：20 点 → 80 点），与 proposal 一致。DB 是 20 Hz，抽取时**每 2 帧取 1 帧**降到 10 Hz。
- 坐标系：目标车 t0 自身坐标系。**注意首点恒为 (0,0)**，建议改为预测**位移增量**，否则模型会把容量浪费在学一个常数上。

### 3.2 模型谱系：从 3 个变 4 个

| 模型 | 角色 | 训练目标 | 推理 |
|---|---|---|---|
| **A. MLP 回归** | 下界 baseline | MSE | 单次前向 |
| **A′. K 头回归 + WTA** ★新增 | **强 baseline** | `min_k ‖τ̂_k − τ‖²`（赢家通吃） | 单次前向出 K 条 |
| **B. 扩散 (DiT)** | 生成范式一 | ε-预测 | Euler / DPM-Solver |
| **C. 流匹配 (DiT)** | 生成范式二 | 速度场回归 | Euler ODE |

**为什么必须加 A′**：minADE@K 是"K 条里取最好的"。回归只出 1 条，而生成模型出 K 条 —— 生成模型赢是**构造上必然的**，不构成发现。加了 K 头 WTA 回归后，问题才变成真正的科学问题：**在同容量、同 K 条候选的条件下，迭代式生成建模是否比"锚点/回归式多模态"更好？** A′ 只多一个训练脚本，性价比极高。

三个生成/回归头共享同一个 encoder + 同一个 DiT backbone（结构照 `dit.py` 的 adaLN-Zero 精简版复刻），**参数总量与训练步数对齐并写进报告**。

### 3.3 数学约定（修正 proposal 里的两处笔误）

proposal（中文版）里两处公式有误，pre 上被问到会很尴尬，必须改：

| 原文 | 正确写法 |
|---|---|
| `x_t = √_t · x + √(1_t) · ε` | `x_t = √ᾱ_t · x₀ + √(1−ᾱ_t) · ε`（VP 余弦调度） |
| 流匹配 `v = ε x` | `v = ε − x₀`（路径 `x_t = (1−t)x₀ + tε`，对 t 求导即得） |

约定写死：流匹配取 `x₀ = 数据`、`x₁ = ε ~ N(0,I)`，**从 t=1 积分到 t=0** 得到样本，与"从 ε 出发少步 Euler"一致。

### 3.4 训练配置

batch 256、AdamW、余弦退火、20–50 epoch、EMA；`train.py --objective {reg, wta, ddpm, fm}` 统一入口 + 单一配置文件。参数量实测后如实上报（不要照抄 proposal 的"约 200 万"）。

---

## 4. 实验设计（这一段是决定 pre 分数的地方）

### 4.1 指标

| 指标 | 定义 |
|---|---|
| minADE@K / minFDE@K | K 条候选取最优 |
| MissRate@K | `minFDE@K > 2.0 m` 的比例 |
| **Diversity** | K 条候选两两 FDE 均值（刻画多模态覆盖） |
| **NFE / 延迟** | 前向次数与单样本/batch 墙钟时间 |
| **NLL** | 真值在 K 条样本 KDE 下的对数似然（**检验"覆盖"而非"最好那条"**） |

### 4.2 四条必须做的实验

1. **K 扫描** `K ∈ {1, 4, 6, 16}` → minADE 曲线。**同时画 diversity–accuracy Pareto 前沿**：单向刷 minADE 不代表更好，能同时保持多样性才是生成模型的真正价值。
2. **等 NFE 对比（核心）**：`minADE vs NFE` 曲线。proposal 说"扩散 15 步 vs 流匹配 4/8 步"，但**用了不同求解器**（DPM-Solver vs Euler），这混淆了"路径差异"和"求解器差异"。
   - 主实验：**两者都用一阶 Euler**（连续时间 VP 扩散 vs 线性插值流匹配）→ 差异纯粹来自生成范式；
   - 次实验：扩散换 DPM-Solver-2M，验证"扩散在低 NFE 下需要高阶求解器才能追上流匹配"。
3. **确定性检验（诚实定位）**：同一 ε 重复解码，流匹配 L2 ≈ 0 —— 这**是数学上必然的，属于实现自检，不是科学发现**。它的真正价值是**支撑未来工作**：确定性 ODE 解码使"在噪声空间做无梯度搜索"成为良定义问题（Diffusion-ES 的 CEM + TOAD 的信任域）。pre 上要这么讲，老师才会觉得你想清楚了。
4. **场景分层**：直接用 `scenario_tag` 的 `on_intersection` / `traversing_intersection` / `stationary` 等标签分组统计 —— **免费拿到官方场景真值**，不用启发式。这条要作为亮点讲。

### 4.3 额外加一个消融（几乎零成本、含金量高）

**地图条件 ablation（有/无 lane 输入）**：既回答"多模态收益是否集中在路口"，又顺手说明"地图信息对生成式模型 vs 回归模型哪个更关键"。

---

## 5. 第一次 pre：逐页脚本（目标 6 分 10 秒，卡在 4–7 分钟内）

| # | 内容 | 时长 | 要点 |
|---|---|---|---|
| 1 | 标题 + 一句话卖点 + 组员 | 0:15 | "同一个编码器，三种生成范式，一次控制变量的正面对比" |
| 2 | **痛点图**：路口三条真实未来（左转/直行/右转）及其**平均值指向路边** | 0:45 | 全程最有杀伤力的一页。MSE 的失败是几何可见的 |
| 3 | 研究问题 RQ1 / RQ2 + **我们控制了哪些变量** | 0:35 | 强调"同数据、同条件、同编码器、同参数量" |
| 4 | **任务与数据：实测数字**（198 s / 1 log / 44 条车轨迹 / 482 窗口 / 27 种场景标签） | 1:00 | ★ 别人念 proposal，我们念**measurement**。这一页就是差异化 |
| 5 | 方法：一个 encoder + 四个头；**扩散弯曲路径 vs 流匹配直线路径**同图对比 | 1:00 | 用图讲，不用推公式 |
| 6 | **评测协议**：六项指标、K 扫描、**等 NFE**、按 log/城市划分、A′ 强基线 | 0:50 | 直接回应"你怎么保证公平" |
| 7 | 实验清单 + 预期结果 + **什么结果会证伪我们的假设** | 0:40 | 主动说可证伪 = 研究素养 |
| 8 | 5 周排期甘特图 + 四人分工 | 0:50 | 硬性要求，必须有 |
| 9 | 风险预案 + 未来工作（噪声空间 CEM 搜索） | 0:30 | 收尾，留钩子 |

**加分项（强烈建议做）**：第 4 页放一张**从真实 DB 里抽出来的 BEV 轨迹图**（带真实车辆历史+未来），第 5 页放一个 20 秒的采样动画。第一周就能拿出"我们已经跑通数据"的实物，比多讲三页理论有效得多。—— 这一步我可以现在就帮你做出来（环境齐全：numpy + matplotlib + sqlite3，无需 torch）。

**现场纪律**：自备笔记本（硬性要求）；PPT 字体 ≥ 20pt；提前 3 分钟到场试投影；把 §8 的问答预演一遍。

---

## 6. 分工与 5 周排期

### 6.1 分工（对 proposal 的微调）

| 成员 | 负责 | 交付物 |
|---|---|---|
| **A** | 数据管线（SQLite → npz）、地图解析、划分、数据校验、BEV 可视化 | `data/` 全套 + 10 张随机样本人工校验图 |
| **B** | **`models/` 包唯一 owner**：共享 encoder + DiT backbone + 回归基线 + **WTA 多模态基线** + 指标库 | `models/` + `metrics.py` |
| **C** | 扩散目标函数 + 调度器 + 求解器（Euler / DPM-Solver）+ 训练循环 | 扩散训练与采样脚本 |
| **D** | 流匹配目标 + ODE 积分器 + 全部消融脚本 + 报告整合 | FM 脚本 + 消融表 |

**改动理由**：proposal 原文让 B 管编码器、C/D 各自管骨干 —— 但 B/C/D 都要改同一份 DiT 代码，会打架。**backbone 必须有唯一 owner（B）**，C 和 D 只写 loss + sampler。这是工程上很实际的一条，讲出来老师会觉得你们真的想过。

### 6.2 排期（风险前置）

| 周 | 任务 | 里程碑 |
|---|---|---|
| **W0**（本周） | 数据盘点（✅已完成）、**冻结特征契约**、**启动 mini + maps 下载** | 第一次 pre 通过 |
| **W1** | A: 全量抽取 + 缓存 + 划分 + 校验；B: encoder + 回归基线 + 指标库 | **回归基线出第一个数** |
| **W2** | B: WTA 基线；C: 扩散（DDPM + Euler）跑通 | 扩散能采样出合理轨迹 |
| **W3** | D: 流匹配 + ODE 采样器；等 NFE 曲线；K 扫描 | 核心对比表成型 |
| **W4** | 地图 ablation、ε-pred vs v-pred、场景分层、失败案例、BEV 图 | 全部消融完成 |
| **W5** | 报告 + 最终 pre | 交付 |

**基线优先（baseline-first）**：W1 就产出一个完整可报的结果。这样即使后面流匹配出问题，项目依然完整 —— 这就是 proposal 里"preserve a complete core result"的落地方式。

---

## 7. 风险与预案

| 风险 | 概率 | 预案 |
|---|---|---|
| **数据量不足**（当前最大风险） | 高 | 下载完整 mini（首选）；或索要科研项目缓存；或降级为全智能体 + 短时域 + 明确定位为小样本方法学研究 |
| 地图包拿不到 | 中 | Phase 1 先做无地图版本（agent-only），把地图当 ablation 而非前提；避免"地图不到位就全卡住" |
| Windows 下的 DataLoader | 中 | ✅ 环境已就绪（`myenv` + CUDA）。但 `DataLoader(num_workers>0)` **必须**加 `if __name__ == '__main__'` 保护，否则 Windows spawn 会死循环 |
| 扩散训练不稳 | 中 | 先跑通 DDPM(1000 步) 再上少步采样；EMA + 梯度裁剪；小数据先过拟合单 batch 验证实现正确 |
| 对比不公平被质疑 | 中 | §4.2 的 A′ 基线 + 等 NFE + 同 backbone + 参数量表格，四条一起上 |
| 下载要账号/审批耗时 | 中 | **今天就提交**，这是唯一会阻塞全局的事 |

---

## 8. 预判提问与准备答案（"Think deeper" 页的弹药）

**Q1：生成模型比 MSE 回归准，这不是显然的吗？**
> 对，如果只比 minADE@K 就是显然的——因为回归只能出 1 条而生成模型出 K 条。所以我们加了 K 头 WTA 多模态回归基线，并且看 diversity–accuracy 的 Pareto 前沿和 NLL，而不是单一 minADE。真正的开放问题是：**迭代式生成是否比锚点式多模态在同等容量下更好**，这个没有定论。

**Q2：流匹配"步数少"会不会只是求解器造成的假象？**
> 有可能，这正是我们把主实验设计成"两者都用一阶 Euler"的原因。DPM-Solver 只放在次实验里，用来验证"扩散需要高阶求解器才能追上流匹配"。

**Q3：ODE 确定性这个"发现"是不是废话？**
> 是数学必然，所以我们把它定位为**实现自检**（验证采样器没有引入额外随机性），而不是科学结论。它的价值在于支撑未来工作：解码确定 + 输出天然在数据流形上 ⟹ "在噪声空间做 CEM/进化搜索"是良定义且廉价的。这是接 Diffusion-ES + TOAD 的桥。

**Q4：为什么不用 devkit？**
> 实测发现模型需要的量（位置、朝向、速度、尺寸、类别、场景标签）全在 `lidar_box`/`scenario_tag` 里，纯 SQL 就能取；devkit 会引入 hydra、Python 3.9、地图 API 等一长串依赖。我们选择绕开以把时间花在建模上，并且这条路已被验证可跑。

**Q5：数据泄露怎么防？**
> 按 log 和城市划分，不按帧随机划分；邻居特征只用历史，绝不使用邻居的未来。

**Q6：为什么是 8 秒？nuPlan 规划只用 4 秒。**
> 预测任务的时域通常比规划更长，8 秒是通行设定（Argoverse/nuPlan 预测模块）。我们额外报 4 秒截断的指标以便和规划侧对齐。

**Q7：数据这么少，结论可信吗？**
> 诚实的回答：当前单 log 数据不足以支撑泛化性结论。所以第一优先级是补齐 mini 划分（约 180 个 log，可支撑按城市留出与多 seed）。若补齐受限，我们会明确标注 n、按 track 分层、并只声称"方法学对照"而非"性能结论"。

**Q8：这个和你们正式科研项目什么关系？**
> 课程项目是那个项目的**控制变量子集**：把"扩散 vs 流匹配"在同架构同数据下的差异先测清楚，为后续测试时轨迹搜索（确定性解码 ⟹ 噪声空间搜索）提供方法学依据。

---

## 9. proposal 需要修正的地方（pre 前改掉）

1. **§3.1 数据公式笔误**：`v = ε x` → `v = ε − x₀`；`√_t · x + √(1_t) · ε` → `√ᾱ_t x₀ + √(1−ᾱ_t) ε`。
2. **数据量表述**：中文版"抽取约 5 万~20 万条"应加一句"基于 mini 完整划分；当前已验证子集为 1 个 log / 198 s / 44 条车轨迹"。
3. **模型对比增加 A′（K 头 WTA 回归）**，否则 RQ1 的结论力度不足。
4. **采样步数对比改为等 NFE 且同求解器**（主实验统一 Euler）。
5. **"确定性验证"重新定位**为实现自检 + 未来工作动机，而非预期发现。
6. **分工**：明确 `models/` 唯一 owner。
7. **风险项**新增"数据规模"为最高风险（原 proposal 只写了"预处理复杂/算力/训练不稳/对比不公平"，漏了量级问题）。
8. 中文版 §3.3 说"单卡消费级 GPU 数小时内可完成" —— 实测 RTX 4060 8 GB，这句可以保留，但应写成实测配置。

---

## 10. 本周立刻要做（按顺序）

1. ✅ 数据可行性盘点（已完成，本文档 §1）
2. 🔴 **提交 nuPlan 账号申请并开始下载 mini 全量 + nuplan-maps**（唯一会阻塞全局的事）
3. 🟡 **产出第一张 BEV 图**：从真实 DB 抽 3–4 个车辆样本（历史 + 未来）画出来，直接进 pre 第 4 页
4. 🟡 组会敲定并冻结 §2.3 特征契约，明确 A/B/C/D 接口
5. 🟢 建 conda 环境装 torch + timm，跑通一个随机 batch 的前向
6. 🟢 按 §5 出 PPT 初稿，§8 问答预演一遍
