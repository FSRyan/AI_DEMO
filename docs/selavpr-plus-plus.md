# SelaVPR++：面向高效地点识别的基础模型无缝适配

## 论文信息

| 项目 | 内容 |
|------|------|
| **标题** | SelaVPR++: Towards Seamless Adaptation of Foundation Models for Efficient Place Recognition |
| **作者** | Feng Lu, Tong Jin, Xiangyuan Lan, Lijun Zhang, Yunpeng Liu, Yaowei Wang, Chun Yuan (Senior Member, IEEE) |
| **机构** | 清华大学深圳国际研究生院、鹏城实验室、中科院沈阳自动化所、中科院重庆绿色智能技术研究院等 |
| **发表** | IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), 2026, Vol. 48, No. 3, pp. 2731–2748 |
| **预印本** | [arXiv:2502.16601v2](https://arxiv.org/abs/2502.16601)（2025-11-07） |
| **本地 PDF** | `pdf/SelaVPR++: Towards Seamless Adaptation of Foundation Models for Efficient Place Recognition.pdf` |
| **前身工作** | SelaVPR (ICLR 2024)，本地 PDF 见 `pdf/Towards Seamless Adaptation of Pre-trained Models for Visual Place Recognition.pdf` |
| **代码与模型** | [GitHub: Lu-Feng/SelaVPRplusplus](https://github.com/lu-feng/selavprplusplus)（与 SelaVPR 仓库合并发布） |
| **关键词** | Visual place recognition, foundation models, parameter-efficient transfer learning, deep hashing |

> 本文档基于仓库本地 PDF 全文阅读整理。`pdf/` 目录下文件名含换行符，路径引用时请注意。

---

## 一、研究背景与问题定义

### 1.1 视觉地点识别（VPR）

视觉地点识别（Visual Place Recognition, VPR）旨在根据查询图像，从地理标记图像数据库中检索最相似的地点，从而粗粒度估计查询位置。该任务在移动机器人定位、增强现实等场景中有广泛应用。

VPR 面临的主要挑战包括：

- **外观变化**：光照、天气、季节差异
- **视角变化**：不同拍摄角度与朝向
- **感知混淆（Perceptual Aliasing）**：不同地点但视觉外观极为相似

### 1.2 典型技术路线

VPR 通常采用图像检索与匹配框架：

- **一阶段方法**：仅用全局描述子做最近邻检索，速度快，但易忽略空间信息，对感知混淆敏感
- **两阶段方法**：先用紧凑全局特征召回 Top-K 候选，再用局部特征重排序，精度更高，但计算与存储开销大

传统 VPR 模型多采用 ImageNet 预训练再全量微调。随着模型与数据集规模扩大，训练成本（计算量、显存）急剧上升。

### 1.3 基础模型时代的机遇与鸿沟

DINOv2 等视觉基础模型具备强泛化表征能力，但直接用于 VPR 存在明显问题：

- 注意力常落在**动态前景**（行人、车辆）上
- 对**静态判别性地标**（建筑、植被、道路结构）关注不足
- 预训练任务与 VPR 任务之间存在**任务鸿沟**

全量微调基础模型还可能导致**灾难性遗忘**，损害预训练模型的迁移能力。

---

## 二、SelaVPR 的局限与 SelaVPR++ 的动机

### 2.1 SelaVPR（ICLR 2024）回顾

SelaVPR 是作者前序工作，核心思路为：

- 冻结 DINOv2 backbone，在 Transformer block 内插入轻量 **adapter** 做全局适配
- 增加上采样卷积模块输出**密集局部特征**
- 提出**互最近邻局部特征损失（MNN loss）**
- 两阶段流程：全局 GeM 描述子召回 + 局部特征直接匹配重排序（无需 RANSAC 几何验证）

![SelaVPR 全局适配结构（论文 Figure 2）](./images/selavpr/fig2-global-adaptation.png)

![SelaVPR 两阶段 VPR 流程（论文 Figure 3）](./images/selavpr/fig3-two-stage-pipeline.png)

SelaVPR 在多个数据集上取得竞争力结果，且检索速度远快于传统"局部匹配 + RANSAC"管线。

### 2.2 SelaVPR 仍存在的三大瓶颈

| 瓶颈 | 具体表现 |
|------|----------|
| **训练效率不足** | Adapter 插入 backbone 内部，梯度仍需反向传播穿过整个冻结 backbone，显存与时间开销大 |
| **重排序成本高** | 密集局部特征存储量大，重排序仍有显著延迟与内存压力 |
| **范式经济性下降** | 一阶段全局方法（SALAD、BoQ 等）在大规模全监督数据上已很强，局部重排序带来的性能增益与资源消耗不成比例 |

### 2.3 SelaVPR++ 的核心思路

SelaVPR++ 在 SelaVPR 基础上做三方面升级：

1. **内存高效 MultiConv 并行适配**：避免梯度穿过 backbone，同时引入多尺度空间先验
2. **全局特征两阶段检索新范式**：二进制哈希码快速召回 + 高维浮点全局特征重排序，彻底放弃局部特征
3. **统一训练协议**：合并 GSV-Cities、SF-XL、Pitts30k、MSLS 等数据集，全监督训练

---

## 三、方法详解

> 以下结构图摘自论文 PDF（见仓库 `pdf/` 目录），用于辅助理解整体架构。

### 3.0 论文关键结构图

#### SelaVPR（ICLR 2024）

![SelaVPR Figure 2：Global Adaptation——ViT block 内串联/并联 adapter](./images/selavpr/fig2-global-adaptation.png)

*Figure 2：在每个 Transformer block 的 MHA 后插入串联 adapter，在 MLP 旁并联 adapter；MHA/MLP 冻结，adapter 可训练。*

![SelaVPR Figure 3：局部适配与两阶段 VPR 流程](./images/selavpr/fig3-two-stage-pipeline.png)

*Figure 3：全局分支经 GeM 池化做 Top-K 召回；局部分支经上采样卷积得到密集特征，用于互最近邻匹配重排序。*

#### SelaVPR++（T-PAMI 2026）

![SelaVPR++ Fig. 2：Full Tuning / Vanilla adapter / Memory-efficient 并行适配对比](./images/selavprpp/fig2-transfer-learning-comparison.png)

*Fig. 2：(a) 全量微调；(b) SelaVPR 内置 adapter（反向传播仍穿过 backbone）；(c) SelaVPR++ 并行侧路适配（梯度仅在侧路流动）。*

![SelaVPR++ Fig. 3：SelaVPR 全局适配 vs MultiConv 并行适配](./images/selavprpp/fig3-multiconv-adaptation.png)

*Fig. 3：(a) 标准 ViT block；(b) SelaVPR 全局适配；(c) 侧路 MCA 模块逐层精炼 backbone 中间特征。*

![SelaVPR++ Fig. 4：二进制 + 浮点双分支两阶段 VPR 流程](./images/selavprpp/fig4-two-stage-pipeline.png)

*Fig. 4：冻结 foundation model + 两个独立 Side Adapter Network；上分支输出 512-dim 二进制特征做 Hamming 召回，下分支输出高维浮点特征做 L2 重排序。*

### 3.1 整体架构

```text
输入图像
    │
    ▼
冻结 DINOv2 backbone（ViT）
    │ 输出各层中间特征 x₀, x₁, …, xₗ
    ▼
并行侧路 MultiConv Adapter 网络（可训练）
    │ 逐层精炼特征
    ├──────────────────┬──────────────────┐
    ▼                  ▼                  │
线性投影 + GeM 聚合    线性投影 + GeM 聚合   │
    │                  │                  │
    ▼                  ▼                  │
FC 降维 + 哈希量化     FC 升维              │
    │                  │                  │
    ▼                  ▼                  │
512-dim 二进制特征     2048/4096-dim 浮点特征 │
（Hamming 距离召回）   （L2 距离重排序）      │
```

两个侧路 adapter 网络**相互独立**，可分别训练或联合训练，也可按需裁剪某一分支。

### 3.2 内存高效 MultiConv 适配（Memory-Efficient MultiConv Adaptation）

#### 3.2.1 与 SelaVPR 全局适配的对比

| 维度 | SelaVPR 全局适配 | SelaVPR++ 并行适配 |
|------|------------------|-------------------|
| Adapter 位置 | 插入 Transformer block 内部（MHA 后串联 + MLP 并联） | 与 backbone **并行**，侧路精炼中间特征 |
| 梯度传播 | 需穿过整个冻结 backbone | **不穿过** backbone |
| 空间建模 | 依赖后续 MHA 层做 token 交互 | Adapter 自身需建模空间交互 |

SelaVPR 中，即便只训练 adapter，计算 ∂ℒ/∂αᵢ 仍须沿后续 block 反向传播，导致巨大显存开销（参见论文式 5）。

SelaVPR++ 借鉴 Ladder Side-Tuning [23] 范式：冻结 backbone 前向提取中间特征，侧路 adapter 逐层精炼，训练时梯度仅在侧路网络中流动。

#### 3.2.2 并行适配的计算公式

对第 l 层：

- l = 1：`y₁ = Adapter(x₀ + x₁) + x₀`
- l > 1：`yₗ = Adapter(yₗ₋₁ + xₗ) + yₗ₋₁`

每层均有残差连接，融合 backbone 原始特征与侧路精炼特征。

#### 3.2.3 MultiConv Adapter 设计

SelaVPR 的 vanilla adapter 仅在**通道维度**操作，无法建模 patch token 间的空间交互。在并行适配架构中，adapter 输出不再进入后续 Transformer block，这一缺陷尤为突出。

MultiConv Adapter 在 bottleneck 结构的激活层与上投影层之间插入 **MultiConv 模块**（含 skip connection），灵感来自 GoogLeNet 的 Inception 模块：

- 三条并行卷积路径：**1×1、3×3、5×5**
- 3×3 和 5×5 路径前各有 1×1 卷积做通道压缩
- 三路输出拼接后融合

Patch token 先 reshape 为 W×H×D 特征图，再送入卷积，使空间相邻 token 可通过 3×3/5×5 卷积交互，并引入**多尺度局部先验**。

**重要变化**：由于 MultiConv adapter 已能捕获局部信息，SelaVPR++ **不再需要** SelaVPR 中专门的上采样局部适配模块。

#### 3.2.4 可扩展的 Adapter 配置

方法具有 plug-and-play 特性，可灵活控制 adapter 数量以权衡精度与效率：

- **每层一个 adapter**：DINOv2-L/14（24 层）配 24 个 adapter
- **每 m 层一个 adapter**：如每 3 层一个，共 8 个
- **仅最后若干层**：DINOv2-Large 推荐仅对最后 16 层加 adapter（性能接近饱和，显存更优）

### 3.3 高效两阶段 VPR 新范式

#### 3.3.1 设计动机

传统两阶段方法：全局特征召回 → **局部特征匹配**重排序，延迟、显存、存储开销均大。

SelaVPR++ 提出：**两阶段均使用全局特征**，彻底放弃局部特征。

| 阶段 | 特征类型 | 维度 | 距离度量 | 作用 |
|------|----------|------|----------|------|
| 第一阶段（召回） | 二进制哈希码 | 512-dim | Hamming 距离 | 快速获取 Top-K 候选 |
| 第二阶段（重排序） | 浮点全局描述子 | 2048/4096-dim | L2 距离 | 精确重排候选 |

#### 3.3.2 效率优势

- 二进制特征存储仅为同维浮点的 **1/32**
- Hamming 距离计算远快于 L2 距离（512-dim 二进制仅 0.26ms vs 512-dim 浮点 3.89ms，Pitts30k 数据库）
- 相比 TransVPR 检索延迟加速 **6000×**，相比 4096-dim 一阶段全局检索加速 **60×**

#### 3.3.3 灵活性

- 去掉二进制分支 → 退化为普通一阶段 VPR
- 去掉浮点分支 → 仅用二进制哈希做极速检索（资源受限场景）
- 可先训练一个分支，再用其参数初始化另一分支

### 3.4 相似度约束深度哈希（Similarity-Constrained Deep Hashing）

#### 3.4.1 问题

深度哈希用 sign 函数将浮点特征量化为 {-1, +1} 二进制码，但 sign 函数梯度处处为零或无穷，无法端到端训练。

#### 3.4.2 常见方案及其缺陷

- **量化前计算 metric loss + L1/L2 量化损失**：量化损失与 metric loss 目标冲突
- **Locality-Constrained Hashing [74]**：约束量化前后特征对的相似度，但使用 sigmoid 变换

#### 3.4.3 SelaVPR++ 的相似度约束损失

对浮点特征对 (fᵢ, fⱼ) 与量化后二进制码对 (bᵢ, bⱼ)，最小化：

```
ℒ_Q = Σᵢⱼ (s(fᵢ, fⱼ) - s(bᵢ, bⱼ))² / K
```

其中 s(·,·) 为余弦相似度，K 为特征对数量（每 batch 仅用约 1/5 正负样本对以节省显存）。

相比 LC hashing，该方法**不使用 sigmoid**，形式更简洁，直接保持量化前后成对特征的余弦相似度一致。

#### 3.4.4 Straight-Through Estimation (STE)

前向传播使用 sign 函数；反向传播将 sign 的导数替换为恒等映射 g(x)=x 的导数，使 metric loss 可直接在二进制码上计算。

#### 3.4.5 最终损失

```
ℒ = ℒ_M(b) + λ · ℒ_Q(f, b)
```

- ℒ_M：在二进制码上计算的 Multi-Similarity (MS) 损失
- λ = 0.1

### 3.5 训练策略与统一数据集

#### 3.5.1 从弱监督到全监督

| 项目 | SelaVPR | SelaVPR++ |
|------|---------|-----------|
| 损失函数 | Triplet loss | Multi-Similarity (MS) loss |
| 监督方式 | 弱监督（GPS 标签） | 全监督（地点类别） |
| 训练数据 | Pitts30k + MSLS 分别训练 | 多数据集统一合并训练 |

#### 3.5.2 统一数据集构建

合并四个常用训练集：**GSV-Cities、SF-XL、Pitts30k、MSLS**

核心挑战：各数据集标注方式不同，需统一为相同训练协议。

**地点类别划分方法**（借鉴 CosPlace）：

1. 合并 query 与 database 图像
2. 按 UTM 坐标 {east, north} 划分地理网格
3. 按朝向 {heading} 进一步细分类别

类别定义：

```
C_{eᵢ, nⱼ, hₖ} = {x : [east/M]=eᵢ, [north/M]=nⱼ, [heading/α]=hₖ}
```

超参数设置：

| 数据集 | 网格大小 M | 角度间隔 α | 分组参数 N×L |
|--------|-----------|-----------|-------------|
| SF-XL | 10 m | 60° | 5×2（用 1 组） |
| Pitts30k / MSLS | 15 m | 60° | 3×2（用全部组） |

- Pitts30k：利用 12 个 yaw 角（0°, 30°, …, 330°）构造伪角度标签
- MSLS：直接使用罗盘角度（compass angles）

训练时按组顺序加载，避免地理相邻但标签不同的图像进入同一 batch。

#### 3.5.3 与 SuperPlace 等方法的区别

SuperPlace 也合并多数据集，但：

- Pittsburgh 数据集需借助局部特征匹配辅助类别划分
- MSLS 假设所有图像朝向相同，未利用罗盘角数据

SelaVPR++ 更一致地利用位置与朝向信息，实现更统一的训练框架。

---

## 四、实验设置

### 4.1 实现细节

| 配置项 | 设置 |
|--------|------|
| Backbone | DINOv2-Base / DINOv2-Large |
| 训练分辨率 | 224×224 |
| 推理分辨率 | 322×322 |
| 优化器 | Adam，初始 lr=0.0004，每 3 epoch 减半 |
| Batch | 120 个地点 × 每地点 4 张图 = 480 张 |
| 早停 | MSLS-val 连续 12 epoch 无提升，最大 25 epoch |
| Adapter 配置 | DINOv2-B：12 层各 1 个；DINOv2-L：最后 16 层各 1 个 |
| 二进制特征维度 | 512-dim |
| 浮点特征维度 | 2048-dim（resource）/ 4096-dim（performance） |
| 重排序候选数 | Top-100（默认） |

### 4.2 评估数据集

| 数据集 | 场景特点 | Database | Queries |
|--------|----------|----------|---------|
| Pitts30k-test | 城市街景，大视角变化 | 10,000 | 6,816 |
| MSLS-val | 城市/郊区 | 18,871 | 740 |
| MSLS-challenge | 长期变化，多城市 | 38,770 | 27,092 |
| Tokyo24/7 | 城市，昼夜变化 | 75,984 | 315 |
| Nordland | 自然场景，季节变化 | 27,592 | 27,592 |

评估指标：**Recall@N (R@N)**，即前 N 个检索结果中至少有一个正确的查询比例。

### 4.3 两种模型配置

| 配置 | Backbone | 浮点维度 | 定位 |
|------|----------|----------|------|
| **SelaVPR++ (resource)** | DINOv2-Base | 2048-dim | 资源友好 |
| **SelaVPR++ (performance)** | DINOv2-Large | 4096-dim | 性能优先 |

两者均使用 512-dim 二进制特征做 Top-100 候选召回。

---

## 五、主要实验结果

### 5.1 与 SOTA 方法对比（Table II 核心数据）

**Performance 配置（DINOv2-L, 4096-dim, Unified 训练）**：

| 数据集 | R@1 | R@5 | R@10 |
|--------|-----|-----|------|
| Pitts30k-test | **94.4** | **97.5** | **98.1** |
| Tokyo24/7 | **98.1** | **98.7** | **99.4** |
| MSLS-val | **94.5** | **98.0** | **98.2** |
| MSLS-challenge | **84.0** | **93.7** | **94.4** |
| Nordland | **97.2** | **99.0** | **99.4** |
| **五数据集平均 R@5** | — | **97.4** | — |

关键结论：

- 在**所有评估数据集**上取得最优 R@1/R@5/R@10
- MSLS-challenge R@5 达 **93.7%**，位列官方排行榜**第一**（超越 SALAD-CM 的 91.2%）
- 相比 SelaVPR，五数据集平均 R@5 提升 **3.1%**
- 全局描述子维度不到 SALAD/BoQ（8448/12288-dim）的一半

**Resource 配置（DINOv2-B, 2048-dim）** 平均 R@5 达 96.4%，仍超越绝大多数方法。

### 5.2 训练显存对比（Table IV）

| 方法 | Backbone | 微调策略 | 显存 (GB) |
|------|----------|----------|-----------|
| **SelaVPR++** | DINOv2-L | 并行 MultiConv 适配 | **7.97** |
| BoQ | DINOv2-B | 部分微调（最后 2 层） | 8.20 |
| SALAD | DINOv2-B | 部分微调（最后 4 层） | 14.91 |

SelaVPR++ 使用更大的 DINOv2-Large，显存仍低于 SALAD 和 BoQ。

### 5.3 检索延迟对比（Pitts30k, Table XII）

| 方法 | 初始检索 (ms) | 重排序 (ms) | 总延迟 (ms) |
|------|--------------|------------|------------|
| TransVPR | 1.94 | 3096.66 | **3098.60** |
| SelaVPR | 8.02 | 67.51 | 75.53 |
| Float 4096D（一阶段） | 32.22 | — | 32.22 |
| Binary 512D（一阶段） | 0.26 | — | 0.26 |
| **SelaVPR++ (performance)** | 0.26 | 0.25 | **0.51** |
| **SelaVPR++ (resource)** | 0.26 | 0.14 | **0.40** |

加速比：

- 相比 TransVPR：**6000×**
- 相比 SelaVPR：两个数量级以上
- 相比 4096-dim 一阶段全局检索：**60×**

### 5.4 公平对比实验（Table V 摘要）

在相同 backbone（DINOv2-B）、相同维度（2048-dim）、相同训练集（Unified）下：

- SelaVPR++ 全面超越降维版 SALAD 和 BoQ
- 替换 GeM 为 SALAD/BoQ 聚合器后（SelaVPR++(SALAD/BoQ)），性能进一步提升
- 统一数据集也使 SALAD、BoQ 等方法受益，证明统一训练协议的通用价值

### 5.5 补充数据集（Table III）

在 SPED、Eynsham、SVOX（Night/Rain/Overcast）等数据集上，SelaVPR++ 同样取得最优或并列最优结果。

---

## 六、消融实验要点

### 6.1 微调策略对比（Table VI/VII）

在 GSV-Cities 上对比六种微调方式：

| 方法 | 特点 | 结论 |
|------|------|------|
| Freezing | 仅训练聚合器 | Nordland 等自然场景表现差 |
| Full-tuning | 全量微调 | DINOv2-L 上灾难性遗忘，Tokyo24/7 下降明显 |
| Partial-tuning | 微调最后 4 层 | 性能尚可，但参数量与显存较大 |
| SelaVPR-global | 内置 adapter | 参数少但显存接近 Full-tuning |
| MemEfficient-vanilla | 并行 vanilla adapter | 优于 SelaVPR-global |
| **MemEfficient-MCA** | 并行 MultiConv adapter | **全面最优** |

关键发现：

- PEFT 方法更适合大模型（DINOv2-L） than 全量/部分微调
- MemEfficient-MCA 在 DINOv2-L 上显存仅为 SelaVPR-global 的 **14.6%**，Full-tuning 的 **11.5%**
- 训练时间：DINOv2-B 上 MemEfficient-MCA 仅 6.85 min/epoch，接近 Freezing（6.75）

### 6.2 Adapter 类型对比（Table VIII）

| 方法 | Nordland R@1 |
|------|-------------|
| MemEfficient-vanilla | 69.2 |
| MemEfficient-alternating（通道-Token 交替） | 73.1 |
| **MemEfficient-MCA** | **75.2** |

MultiConv adapter 同时建模空间交互与多尺度先验，且支持推理时提升分辨率（322×322），而 alternating adapter 受限于训练分辨率。

### 6.3 Adapter 数量（Table X）

DINOv2-L + Unified 数据集：

- 所有方法均显著优于 Freezing 基线
- 最后 16 层加 adapter 性能接近全部 24 层（已达饱和）
- 仅 4 个 adapter 也能取得有竞争力的结果（平均 R@1 = 94.5%）

### 6.4 两阶段范式效果（Table XI）

| 特征 | Pitts30k R@1 | Nordland R@1 |
|------|-------------|-------------|
| Binary 512D 直接检索 | 89.0 | 78.4 |
| Float 2048D 直接检索 | 93.3 | 94.7 |
| **Binary 召回 + Float 重排序** | **93.3** | **94.6** |

重排序后性能与直接用高维浮点特征几乎一致，但检索速度快 60× 以上。Binary 与 Float 的 R@100 差距极小，说明二进制召回能有效覆盖正确候选。

### 6.5 重排序候选数（Table XIII）

| 候选数 | Nordland R@1 |
|--------|-------------|
| Top-20 | 96.5 |
| Top-50 | 97.1 |
| **Top-100** | **97.2** |
| Top-200 | 97.2（无提升） |

推荐 Top-50 至 Top-100 作为候选数。

### 6.6 深度哈希方法对比（Table XIV）

| 方法 | 平均 R@1 |
|------|---------|
| Direct hashing | 78.3 |
| LC hashing | 79.6 |
| SC hashing（本文） | 80.9 |
| STE | 81.2 |
| **SC hashing + STE（完整方法）** | **82.0** |

相似度约束损失与 STE 互补：SC 在城市场景更优，STE 在郊区/自然场景更优，组合后全面最优。

### 6.7 统一数据集效果（Table XV）

以 DINOv2-B 为例，Unified vs GSV-Cities 的 R@1 提升：

| 数据集 | GSV-Cities | Unified | 提升 |
|--------|-----------|---------|------|
| Pitts30k | 92.6 | 93.3 | +0.7% |
| Tokyo24/7 | 96.2 | 96.8 | +0.6% |
| MSLS-val | 92.6 | 94.3 | +1.7% |
| Nordland | 81.3 | 94.7 | **+13.4%** |

自然场景（Nordland）受益最大，因统一数据集覆盖了更多样的场景类型。

---

## 七、定性分析与讨论

### 7.1 注意力可视化（Fig. 6）

- **DINOv2 预训练模型**：关注动态前景（行人、车辆），忽略建筑等静态地标
- **SelaVPR**：有所改善，但仍有车辆干扰，部分建筑区域关注不足
- **SelaVPR++**：更好过滤动态干扰，关注更多判别性地标（如烟囱），细节表现更优

### 7.2 困难样例（Fig. 5, Fig. 8）

SelaVPR++ 在以下场景表现突出：

- 剧烈视角变化（如 180° 旋转）
- 夜间弱光查询
- 自然场景中缺乏明显建筑地标

仍存在的**失败案例**：

- 植被占主导且缺乏判别性特征的自然场景
- 地理距离略超阈值但视觉几乎相同的近邻地点（所有方法共同难题）

### 7.3 效率总结

| 维度 | 相比 Full-tuning (DINOv2-L) |
|------|---------------------------|
| 可训练参数 | 8.7% |
| 训练显存 | 11.5% |
| 训练时间 | < 50% |
| 检索延迟 | 6000× 快于 TransVPR |

---

## 八、与相关工作的关系定位

```text
                    参数效率
                       ↑
          SelaVPR ──── SelaVPR++（MemEfficient-MCA）
                       │
    AnyLoc（零样本）────┼──── SALAD / BoQ（部分微调）
                       │
          NetVLAD ─────┼──── MixVPR / CosPlace
                       │
                       ↓
                    全量微调

    检索效率 ←── SelaVPR++（Binary+Global 重排序）──→ 精度优先
                       │
          TransVPR ────┼──── SelaVPR（局部重排序）
                       │
                       ↓
              局部匹配 + RANSAC
```

| 对比维度 | SelaVPR++ 相对优势 |
|----------|-------------------|
| vs SelaVPR | 训练更快、显存更低、检索更快、无需局部特征存储 |
| vs SALAD/BoQ | 更低维度达到更高精度，训练显存更少 |
| vs 一阶段全局方法 | 两阶段重排序精度更高，二进制召回速度更快 |
| vs 传统两阶段（TransVPR） | 无需局部匹配与几何验证，延迟低 4 个数量级 |

---

## 九、主要贡献总结

1. **内存高效 MultiConv 适配**：并行侧路 adapter + MultiConv 模块，实现参数、时间、显存三方面高效微调，并引入多尺度空间先验
2. **全局特征两阶段新范式**：512-dim 二进制哈希快速召回 + 高维浮点全局特征重排序，彻底替代局部特征重排序
3. **相似度约束深度哈希**：解决量化损失与 metric loss 冲突，结合 STE 实现端到端训练
4. **统一训练协议**：合并 GSV-Cities、SF-XL、Pitts30k、MSLS 四数据集，全监督 MS loss 训练
5. **SOTA 性能与效率**：MSLS challenge 排行榜第一，检索延迟比 TransVPR 快 6000×

---

## 十、局限与未来方向

论文讨论的局限：

- 自然场景中植被占主导、缺乏判别性地标时仍可能失败
- 地理距离略超阈值的近邻误匹配是所有 VPR 方法的共性问题
- 提升描述子维度可改善细节表达，但需与效率权衡

潜在改进方向：

- 增大数据库采集的地理密度
- 结合更强聚合器（BoQ/SALAD aggregator）进一步提升性能
- 针对极端自然场景的专用策略

---

## 十一、核心公式速查

| 名称 | 公式 |
|------|------|
| 并行适配 | yₗ = Adapter(yₗ₋₁ + xₗ) + yₗ₋₁ |
| 二进制量化 | b = sgn(f) |
| 相似度约束损失 | ℒ_Q = Σ(s(fᵢ,fⱼ) - s(bᵢ,bⱼ))² / K |
| 总损失 | ℒ = ℒ_M(b) + 0.1 · ℒ_Q(f, b) |
| 全局特征聚合 | GeM pooling on reshaped patch tokens |

---

## 参考文献（核心）

- Lu et al., "Towards Seamless Adaptation of Pre-trained Models for Visual Place Recognition," ICLR 2024.（SelaVPR）
- Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," TMLR 2023.
- Sung et al., "LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning," NeurIPS 2022.
- Lu et al., "CricaVPR: Cross-Image Correlation-Aware Representation Learning for Visual Place Recognition," CVPR 2024.（MultiConv adapter 首次提出）
- Izquierdo & Civera, "Optimal Transport Aggregation for Visual Place Recognition," CVPR 2024.（SALAD）
- Ali-bey et al., "BoQ: A Place is Worth a Bag of Learnable Queries," CVPR 2024.
- Berton et al., "Rethinking Visual Geo-Localization for Large-Scale Applications," CVPR 2022.（CosPlace / SF-XL）
