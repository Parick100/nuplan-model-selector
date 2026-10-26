# 第一次 pre 讲稿（中文对照）· 11 页版

**PPT 文件**：`Which-Model-When-Generative-Vehicle-Trajectory-Prediction\pre1_which_model_when.pptx`
**目标时长**：**5 分 45 秒**（要求 4–7 分钟）
**逐页预览图**：`pre1_ppt\preview\幻灯片N.PNG`
**版式**：淡米色底 + Times New Roman，橙色仅作点缀

> 建议逐字背 **P2**、**P3**、**P8**、**P9**。

---

## 时长分配

| 页 | 内容 | 时长 | 累计 |
|---|---|---|---|
| 1 | 封面 | 0:15 | 0:15 |
| 2 | 痛点 | 0:40 | 0:55 |
| 3 | 两种方法 | 0:50 | 1:45 |
| 4 | 模型怎么在 nuPlan 上跑 | 0:45 | 2:30 |
| 5 | 数据 | 0:35 | 3:05 |
| 6 | 我们的问题 | 0:30 | 3:35 |
| 7 | 计划与指标 | 0:40 | 4:15 |
| 8 | **要交付的：选择器** | 0:45 | 5:00 |
| 9 | 预期效果 + 怎么验证 | 0:35 | 5:35 |
| 10 | 分工与排期 | 0:35 | 6:10 |
| 11 | 收尾 | 0:15 | 6:25 |

> 略超 6 分。超时就压 P7 的指标细节，别砍 P2 / P3 / P8。

---

## P1 · 封面 「predict the future」（0:15）

> Good morning. Every self-driving car has to guess what the cars around it will do next. Our project asks a very practical question: **when should you use which method to make that guess?**

> 早上好。每辆自动驾驶车都必须猜周围的车接下来会做什么。我们问一个很实际的问题：**什么时候该用哪种方法来猜？**

---

## P2 · 痛点 「one past, three futures」（0:40）★

> One car, one piece of history — and it could go left, straight, or right. Three completely different futures. The simplest way to train a model is to ask it to be as close as possible **on average**. But look at what the average is: it is not any of the three. It drives straight into the kerb.
> This is not a tuning problem. **The average of three different futures is not a road.**

> 一辆车、一段历史，可能左转、直行或右转——三种完全不同的未来。训练模型最简单的方式是让它"平均来看"尽量接近真值。但你看这个平均值：它三条都不是，它径直开上了人行道。
> 这不是调参问题。**三种不同未来的平均值，根本不是一条路。**

**提示**：说完 "into the kerb" 停一秒。

---

## P3 · 两种方法 「two ways to learn a future」（0:50）★

> So instead of averaging, we use models that can **imagine**. Two popular ways — and they differ in *what they learn*, not in how many steps they take.
> **Diffusion** takes a real trajectory and gradually adds noise until it is pure noise — that is the top arrow. Then it learns to **undo** that, step by step, to get a real future back.
> **Flow matching** does not add noise step by step. It learns a **flow** — a velocity field — that carries a simple blob of noise across to the whole distribution of real futures. That is the red set of curves on the right.
> One learns to undo noise; the other learns a current. **How many sampling steps each needs is a consequence of that — not the definition.**

> 所以我们不用"平均"，而是用会**想象**的模型。两种主流做法，区别在于**学到的是什么**，不在于走多少步。
> **扩散模型**把真实轨迹逐步加噪到纯噪声（上面那条箭头），然后学会**反过来去噪**，一步步把真实未来还原出来。
> **流匹配**不一步步加噪。它学一个**流**——速度场，把一团噪声整体推到真实未来的分布上，就是右边那组红色曲线。
> 一个学"去噪"，一个学"水流"。**各自需要多少采样步数，是这个区别的结果，不是定义。**

---

## P4 · 模型怎么在 nuPlan 上跑 「from a scene to a future」（0:45）

> This is the recipe from Diffusion-Planner, redrawn. Everything starts from a **scene** — the bird's-eye view on the left. From it we build the inputs: **neighbours**, **lanes**, **static objects**, **navigation**.
> All of it goes into one shared **encoder** — MLP-Mixers plus self-attention — which produces a single summary of the situation.
> Then the **decoder** generates the future; in Diffusion-Planner that is a diffusion transformer that denoises over N steps.
> And this is also what the data looks like: nuPlan — real drives, bird's-eye view, ten hertz.

> 这是 Diffusion-Planner 的做法，我重画了一下。一切从**场景**开始——左边那个俯视图。我们从中构造输入：**邻居车辆**、**车道**、**静态物体**、**导航**。
> 全部进一个共享的**编码器**（MLP-Mixer 加自注意力），把局面压成一个表示。
> 然后**解码器**生成未来；在 Diffusion-Planner 里是扩散 Transformer，要去噪 N 步。
> 这也顺便说明了数据长什么样：nuPlan，真实驾驶、俯视视角、10 Hz。

**提示**：页脚已注明 "after Diffusion-Planner (ICLR 2025), redrawn"，讲时说一句"我们沿用这个配方做预测"即可。

---

## P5 · 数据 「real driving, seen from above」（0:35）

> Real nuPlan scenes: an intersection, a pedestrian crossing, an unprotected left turn. For each we give the model **two seconds of history** and everything around it, and ask for the **next eight seconds** — twenty points in, eighty points out.
> nuPlan already labels every intersection, crossing and turn, so we do not have to guess which scene is which.

> 真实 nuPlan 场景：路口、行人过街、无保护左转。每个场景给模型**过去两秒**的历史和周围环境，预测**未来八秒**——输入 20 点，输出 80 点。
> nuPlan 已经帮我们标好了每个路口、人行横道和转弯，场景分类不用猜。

---

## P6 · 我们的问题 「no model wins everywhere」（0:30）

> Look at accuracy against compute, and the lines **cross**. Flow matching gets good fast. Diffusion is slower but can end up better if you can afford many calls. Plain regression is flat — more compute never helps it.
> **No model wins everywhere.** So the honest deliverable is not "which one is better".

> 把精度和算力画在一起，两条线会**交叉**。流匹配很快变好；扩散更慢但最终可能更好；普通回归是一条平线，给再多算力也不进步。
> **没有哪个模型在所有情况下都赢。** 所以诚实的交付物不是"谁更好"。

**提示**：图是**预期形状**，页脚已注明。

---

## P7 · 计划与指标 「four models, one brain」（0:40）

> Five steps: extract the data, train the models, sample futures, score and search, and finally **pick the model** — that last box is the new part.
> The key design choice: **all four models share one encoder**. Only the decoder that imagines is swapped. That is what makes the comparison fair.
> We measure four things plainly: how close the best guess gets, whether it **covers** the futures that really happen, how often nothing lands near the truth, and what it costs.

> 五步：抽取数据、训练模型、生成未来、打分与搜索，最后**选择模型**——最后一格是新增的。
> 关键设计：**四个模型共用一个编码器**，只换负责"想象"的解码器。这才是公平对比。
> 我们量四件事：最好的猜测有多接近真值、有没有**覆盖**真正发生的未来、多久出现"一条都不靠谱"、代价多大。

---

## P8 · 要交付的：选择器 「one cheap selector, not one model」（0:45）★ 逐字背

> Here is what we actually deliver — and it is **not** a table of rules. It is a **selector**.
> We give it four trained models with very different costs: regression runs in one unit of time, flow matching in about four, diffusion in about fifteen. Running all of them, every frame, defeats the purpose.
> So the selector looks at the scene, and picks **exactly one** model to run — the cheapest one that is still good enough for *this* scene.
> The key point is the cost. **Total decision time = the selector plus exactly one model.** The selector is a tiny module next to a diffusion decoder, so its own cost is essentially noise. That is the whole trick: we can afford to be smart about the choice, because the choice is nearly free.

> 这才是我们真正要交付的——而且它**不是**一张规则表，而是一个**选择器**。
> 我们给它四个训练好的模型，成本差别很大：回归是 1 个单位时间，流匹配约 4，扩散约 15。每帧都把它们全跑一遍，那就本末倒置了。
> 所以选择器看一眼场景，只挑**一个**模型来跑——对**这个**场景来说够好的里面最便宜的那个。
> 关键是成本。**总决策时间 = 选择器 + 恰好一个模型。** 相比扩散解码器，选择器是一个极小的模块，它自己的耗时基本可以忽略。**诀窍就在这里：我们有能力去聪明地选，因为"选"这件事几乎不要钱。**

**提示**：这是本场最重要的一页。老师若问"选择器本身有多大开销"，答："它只是一个读场景特征的小模块，比起跑一次扩散解码器可以忽略。"

---

## P9 · 预期效果 + 怎么验证 「near-best quality, far less time」（0:35）★ 逐字背

> This chart is what we expect to show. The grey dashed line is what **any single fixed model** can achieve — that is your trade-off if you just pick one model and live with it.
> If our selector works, it should sit **below** that line: near-diffusion quality at roughly a third of the decision time. That gap is the value of the selector.
> We check it on our own four models first. Then we take it to **nuPlan's official scorer** and ask a blunt question: **does a routed policy actually beat every single model?** It might not — and if it does not, that is a result too, and we will report it.

> 这张图是我们预期的结果。灰色虚线是**任何单一固定模型**能达到的水平——也就是你只挑一个模型用到底的取舍。
> 如果选择器有效，它应该落在**这条线下方**：接近扩散的质量，但只花大约三分之一的决策时间。这段差距就是选择器的价值。
> 我们先在自己的四个模型上验证，然后拿到 **nuPlan 官方打分器**上问一个直白的问题：**按场景路由的策略，是不是真的比任何单一模型都好？** 有可能不是——如果真不是，那也是结果，我们会如实报告。

**提示**：页脚写明"预期形状，非实测结果"。主动说"可能是负结果"是加分项。

---

## P10 · 分工与排期 「four people, five weeks」（0:35）

> Four people, five weeks. **A** owns the data — extraction, splits, validation, and the timing harness. **B** owns the shared encoder, the baselines, the metrics, and **the selector** itself. **C** owns diffusion — the objective, training, and reduced-step solvers. **D** owns flow matching, the ablation suite, and the nuPlan scorer integration.
> The gate is **end of week one**: a complete baseline already exists by then, so the project stays intact even if a later part slips.

> 四个人、五周。**A** 管数据——抽取、划分、校验和计时工具。**B** 管共享编码器、基线、指标，以及**选择器本身**。**C** 管扩散——目标函数、训练、少步求解器。**D** 管流匹配、消融实验，以及 nuPlan 打分器的接入。
> 关键节点在第一周结束：那时已有完整可跑的基线，所以即使后面某块出问题，项目依然完整。

**提示**：分工是硬性展示要求，按上面说清每人一句即可，不用展开。

---

## P11 · 收尾 「no model wins everywhere」（0:15）

> So — no model wins everywhere. What we deliver is a **cheap selector** that picks the right one for each scene, at almost no extra cost. Thank you.

> 所以——没有哪个模型在所有情况下都赢。我们要交付的是一个**几乎不花额外代价的廉价选择器**，为每个场景挑对模型。谢谢。

---

## 讲前必做

1. **替换占位符**：封面/P11 页脚是 `TEAM A / B / C / D`，P10 用 `A/B/C/D`。填真实姓名（直接改，文字全可编辑）。
2. **自备笔记本**（硬性要求），提前 3 分钟到场试投影。
3. **计时练 2 遍**，练到 5:45。

---

## ⚠ 维护提醒（请转告组员）

**① 字体是后处理补丁。** 这个 PPTD 构建会忽略 `fontFace` 并硬编码成 Arial，所以 `_extract\patch_font.py` 把 PPTX 里的 `typeface="Arial"` 换成了 `Times New Roman`。

> **重新 `pptd_render` 一次，字体就退回 Arial —— 必须再跑一次：**
> `G:\Anaconda\envs\myenv\python.exe _extract\patch_font.py`（在项目根目录运行）

**② 页面底色与所有色块/细线/甘特条都是烘焙进 `bg/p0N.png` 的图片**（该构建不渲染 `shape` 元素）。改版式要同时改 `_extract\make_bg3.py`。插图源码在 `_extract\make_figs2.py` 与 `make_figs3.py`。

---

## 可能被问到的问题

**Q：选择器凭什么比"只用最好的那个模型"好？**
> 因为不同场景需要的精度不同：直行场景用便宜的模型就够，路口才需要贵的。而"永远用最好的"意味着每一帧都付最贵的价格。选择器把这笔钱按场景分配。

**Q：选择器本身会不会成为新的开销？**
> 它是一个读场景特征的小模块，参数量和计算量比扩散解码器小几个数量级，所以总时间 ≈ 选择器 + 一个模型 ≈ 一个模型。

**Q：怎么保证选择器不是过拟合到测试集？**
> 按 log / 城市划分训练与测试；选择器只使用推理时本来就拿得到的信息（场景特征、历史、地图），不使用未来。

**Q：如果混合策略并不更好呢？**
> 那也是结论，而且是我们预先声明要报告的结论之一。页面上写了。

**Q：扩散是不是就是步数多？**
> 不是。扩散学的是"去掉一个固定的加噪过程"，流匹配学的是"把噪声推到数据上的流"。步数是路径形状的**结果**。

**Q：官方打分和我们自己的指标什么关系？**
> 自己的指标做细致消融；官方打分器作为**外部检验**，回答"这在 benchmark 尺度上算不算数"。

**答不上来时**：
> That is on our ablation list — we have not tested it yet.
