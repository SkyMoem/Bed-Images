# 多模态大模型MLLM

这份课件涵盖了从视觉基础模型（ViT）到经典的图文对齐模型（CLIP），再到近期主流的多模态大模型（Flamingo, BLIP-2, LLaVA, MiniGPT-4）的演进路线，内容非常核心。

 **MLLM 发展概况**

- **训练范式**：通常分为“预训练（对齐视觉与文本）”和“微调（指令微调/对话能力）”两个阶段 。

- **预训练阶段（PreTraining）**：通常是为了实现视觉特征与文本特征的对齐。有些模型的这一阶

  段也会分为两个子阶段，比如针对弱标签数据训练和人工标注训练，或者在训练中增加图片分辨

  率等

- **微调阶段（Finetune）**：此阶段通常是使用指令或特定任务数据进行微调，以增强模型遵循指令

  的能力和对话能力等，有些也会分位两个子阶段

  ![image-20260118203930902](/Users/moem/Library/Application Support/typora-user-images/image-20260118203930902.png)

------

## **第一部分：视觉基础模型 —— Vision Transformer (ViT)**

这部分介绍了如何将Transformer架构从NLP领域迁移到计算机视觉领域，是理解后续多模态模型中“视觉编码器（Vision Encoder）”的基础。

### **1. 核心挑战与思路**

- **难点**：在NLP中输入是1D序列，而视觉是2D图片。直接将像素点作为序列输入会导致复杂度过高（例如 $224 \times 224$ 的图片会产生 50176 长度的序列，远超BERT常用的512）。
- **早期尝试**：曾尝试使用特征图、局部窗口（Local Window）或轴自注意力（Axial Attention）来降低复杂度，但硬件支持不足或不够直观 。
- **ViT的解决方案**：**Patch Embedding**。将图片切分为固定大小的块（Patch），将其视为“单词” 。

### **2. ViT 模型架构流程**

#### 2.1、第一阶段：数据预处理与 Embedding（输入层）

这个过程主要分为四个步骤：**分块 (Patch Partitioning)** -> **线性投射 (Linear Projection)** -> **添加分类 Token ([Class] Token)** -> **位置编码 (Position Embedding)**。

![image-20260117210433453](/Users/moem/Library/Application Support/typora-user-images/image-20260117210433453.png)

以下是详细的流程解析：

##### 2.1.1. 图片分块 (Patch Partitioning)

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260117211320594.png" alt="image-20260117211320594" style="zoom:50%;" />

Transformer 最初是为处理 NLP 任务（一维文本序列）设计的，而图片是二维的。如果直接把图片像素点输入，序列会过长（$224 \times 224 = 50176$），计算复杂度太高 。

- **操作**：ViT 将输入图片（假设尺寸为 $224 \times 224$）切分成固定大小的小块（Patch）。
- **尺寸**：通常设定 Patch 大小为 $16 \times 16$ 。
- **计算**：一张图片会生成 $224 \times 224 / (16 \times 16) = 196$ 个 Patch 。
- **结果**：此时，我们将图片视为了一个包含 196 个“视觉单词”的序列。

##### 2.1.2. 线性投射 (Linear Projection of Flattened Patches)

也就是图中粉色长条框 **"Linear Projection of Flattened Patches"** 的部分。

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260117211301012.png" alt="image-20260117211301012" style="zoom:50%;" />

- **展平 (Flatten)**：每个 $16 \times 16$ 的彩色 Patch（3通道）被展平为一个向量。向量维度为 $16 \times 16 \times 3 = 768$ 。
- **映射 (Projection)**：这 196 个维度为 768 的向量，会经过一个全连接层（线性投射层），映射到固定的 Embedding 维度（通常也是 768）。
- **状态**：此时我们得到了一个维度为 $196 \times 768$ 的序列，这步操作成功将视觉问题转化为了类似 NLP 的 Seq2Seq 问题 。

##### 2.1.3. 添加特殊分类 Token ([Class] Embedding)

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260117211338115.png" alt="image-20260117211338115" style="zoom:50%;" />

对应图中左侧带有星号 `*` 的 **0号 Token**，以及文字说明 **"Extra learnable [class] embedding"**。

- **目的**：为了让模型能够汇总整张图片的信息进行分类，ViT 借鉴了 BERT 的做法，在序列最前面插入一个可学习的特殊 Token，称为 `[CLS]` 。
- **输出**：在经过 Transformer 编码器后，这个 Token 对应的输出向量将被用于类别预测 。
- **维度变化**：序列长度从 196 变为 **197** ($196 + 1$)。此时数据整体维度变为 $197 \times 768$ 。

##### 2.1.4. 加上位置编码 (Position Embedding)

对应图中紫色的圆角矩形 **0, 1, 2...**，以及箭头指向的加号操作。

- **原因**：Transformer 的自注意力机制（Self-Attention）是“位置无关”的（即打乱序列顺序不影响计算结果），但图片中 Patch 的位置关系（比如头在身体上面）非常重要，因此必须手动注入位置信息 。

- **实现**：使用标准的、可学习的 1D 位置编码向量 。位置编码也是一张表，每一行代表一个位置向量。

  > 我们可以把它拆解为三个关键词来详细解释：**“一张表（矩阵）”**、**“可学习（Learnable）”**、**“1D（一维）”**。
  >
  > ##### 1. 它具体是什么？—— 一张参数表（Matrix）
  >
  > 你在课件中看到的“位置编码可以理解为一张表”，在计算机实现中实际上就是一个 **Matrix (矩阵)** 或者 **Tensor (张量)** 1。
  >
  > - **表的形状 (Shape)**：$(N+1) \times D$
  >   - **行数 ($N+1$)**：对应输入序列的长度。
  >     - 对于 $224 \times 224$ 的图片，切分为 $16 \times 16$ 的 patch，会产生 196 个 patch。
  >     - 加上 1 个特殊的 `[CLS]` token。
  >     - 所以共有 **197** 行，每一行对应序列中的一个位置（第0个位置是CLS，第1个位置是第一个Patch，以此类推）。
  >   - **列数 ($D$)**：对应 Embedding 的维度。
  >     - 为了能和 Patch Embedding 直接相加，位置编码的维度必须和 Patch Embedding 完全一致，通常是 **768** 。
  > - **具体数值**：
  >   - 这张表里存的全部是**浮点数**。
  >   - 例如：`Position_Embedding[1]` 取出的就是第 2 行的一个 768 维的向量，它代表了“我在序列中排第 2 个”这个位置信息。
  >
  > ##### 2. 为什么叫“可学习 (Learnable)”？
  >
  > 这与最早的 Transformer (Attention Is All You Need) 中使用的“正弦/余弦固定编码”不同。
  >
  > - **初始化**：这张 $197 \times 768$ 的表，最初是**随机初始化**的（比如服从正态分布的随机数）。此时它没有任何意义，只是一堆噪点。
  > - **训练过程**：在模型训练过程中（反向传播），这张表里的每一个数字都会被视为模型的**参数**（Parameter）。
  > - **结果**：随着梯度下降的进行，模型会自己“学会”什么样的向量能最好地表示“左上角”、“右下角”或者“中间”这些位置关系。最终，这张表里的数值会固定下来，成为能够表达空间位置信息的特征向量。
  >
  > ##### 3. 为什么是“1D (一维)”？
  >
  > 这是 ViT 论文中一个非常有意思的设计选择。
  >
  > - **视觉的直觉 (2D)**：图片是二维的，我们直观地认为应该用 2D 编码（比如一个向量表示行号 $X$，一个向量表示列号 $Y$）。
  > - **ViT 的做法 (1D)**：ViT 简单粗暴地把 $14 \times 14$ 的 Patch 栅格拉平（Flatten）成了一个长度为 196 的长条。
  >   - 它不告诉模型“Patch 1 和 Patch 15 在垂直方向是相邻的”。
  >   - 它只告诉模型：“这是第 1 个 Patch”，“这是第 15 个 Patch”。
  >   - 论文中提到的“1D”指的就是这种**线性索引**（0, 1, 2, ..., 196），而不是网格索引 ((0,0), (0,1)...)。
  > - **论文结论**：ViT 原作者在论文附录中对比了 1D 编码、2D 编码和相对位置编码，发现**并没有显著差异**。这说明 Transformer 的自注意力机制非常强大，能够通过大量数据自己从 1D 序列中“悟”出 2D 的空间拓扑关系。
  >
  > ##### 4. 最终操作流程
  >
  > 1. **准备输入**：你的图像经过处理变成了 $197 \times 768$ 的矩阵 $E_{patch}$（包含图像特征）。
  >
  > 2. **准备位置表**：模型持有一个 $197 \times 768$ 的参数矩阵 $E_{pos}$（包含位置信息）。
  >
  > 3. 相加 (Sum)：直接将两个矩阵对应元素相加 5。
  >
  >    $$Input = E_{patch} + E_{pos}$$
  >
  >    注意是相加，不是拼接（Concat）。这意味着位置信息是通过“叠加”的方式融入到图像特征中的，并没有增加向量的维度。
  >
  > 总结来说，这个“标准的可学习1D位置编码向量”就是一个**形状为 $197 \times 768$ 的可训练参数矩阵**，它让 Transformer 知道哪个 Patch 是头，哪个 Patch 是脚，尽管它看到的只是排成一排的数据。

- **操作细节**：位置编码向量与原本的 Patch Embedding 向量是进行 **Sum (相加)** 操作，而不是拼接 (Concat) 。

- **最终输入**：加上位置编码后，输入到 Transformer Encoder 的最终维度依然是 **$197 \times 768$** 。

#### 2.2、第二阶段：Transformer Encoder 编码（核心层）

![image-20260117214406038](/Users/moem/Library/Application Support/typora-user-images/image-20260117214406038.png)

这一阶段是特征提取的核心。输入是 $197 \times 768$ 的矩阵，输出依然是 $197 \times 768$ 的矩阵（但在最后一层只取第0个向量用于分类）。

对应图中右侧的灰色大框 **Transformer Encoder**，其内部由 $L$ 个层堆叠而成。根据原论文，ViT 的 Encoder 与标准 Transformer 有一个关键区别：**Layer Norm 的位置**。

##### **2.2.1. 输入归一化 (Layer Norm before Blocks)**

- **原论文实现**：ViT 采用了 **Pre-Norm** 结构。即 Layer Norm (LN) 施加在 Multi-Head Attention (MSA) 和 MLP **之前** 。
  - 公式：$\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}$
- **对应图中**：你可以看到数据流是先进入黄色的 **Norm** 框，再进入绿色的 Attention 框。

##### 2.2.2. 多头自注意力 (Multi-Head Self-Attention, MSA)

- **流程**：
  - 输入向量被映射为 $q, k, v$ (Query, Key, Value)。
  - 如果由 12 个头 (Heads)，则 $768$ 维被拆分为 $12 \times 64$。
  - 12组 $q, k, v$ 并行计算注意力，最后将结果拼接（Concat）并映射回 768 维 。
- **对应图中**：绿色的 **Multi-Head Attention** 模块。

##### **2.2.3. 残差连接 (Residual Connections)**

- **实现**：MSA 的输出会与 MSA 的输入直接相加。
- **对应图中**：模块右侧的弯曲箭头，指向一个 $\oplus$ 符号。

##### **2.2.4. MLP Block (多层感知机)**

- **结构**：包含两个线性层，中间夹一个 GELU 激活函数。
  - 第一层通常将维度放大（例如 768 -> 3072）。
  - 第二层将维度缩回（3072 -> 768）。
- **归一化**：同样，进入 MLP 之前也会先经过一层 **Norm**。
- **对应图中**：蓝色的 **MLP** 模块及其下方的 Norm。

------

#### 2.3、第三阶段：分类与输出 (MLP Head)

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260117215326877.png" alt="image-20260117215326877" style="zoom:50%;" />

这一阶段的输入是 Transformer Encoder 输出的 $197 \times 768$ 的特征矩阵，但我们不需要使用所有的向量，只关注其中的一部分。

##### **2.3.1. 提取分类特征 ([Class] Token Extraction)**

- **操作**：在 Transformer 的最后一层输出中，模型**只提取第 0 个位置的向量**（即 $\mathbf{z}_L^0$），忽略掉后面 196 个对应具体 Patch 的向量。
- **原理**：
  - 在 Transformer 的自注意力机制（Self-Attention）作用下，这个特殊的 `[class]` token 在经过多层编码器后，已经与所有的 Patch token 进行了充分的信息交互。
  - 因此，原论文认为 $\mathbf{z}_L^0$ 已经聚合了整张图像的全局语义信息，足以代表整张图片 ($\mathbf{y}$)。
- **对应图中**：你可以看到 Transformer Encoder 上方的箭头指向 MLP Head，虽然画的是一个整体，但在逻辑上这里只取了对应输入端 `0*` 位置的那一个向量输出。

##### **2.3.2. 最终层归一化 (Final Layer Normalization)**

- 论文细节：虽然架构图中没有详细画出这一个小步骤，但在原论文的公式 (4) 中明确提到：

  $$\mathbf{y} = \text{LN}(\mathbf{z}_L^0)$$

- **说明**：在将特征送入分类头之前，通常会先经过层归一化 (Layer Norm) 处理，以确保特征分布的稳定性。

##### 2.3.**3. MLP Head 分类 (Classification Head)**

- **对应图中**：橘黄色的 **MLP Head** 模块。
- **结构区分（论文关键点）**：原论文指出，这个 MLP Head 的结构在“预训练”和“微调”阶段是不同的：
  - **预训练阶段 (Pre-training)**：MLP Head 是一个**带有隐藏层的多层感知机**（MLP with one hidden layer），中间包含 `tanh` 非线性激活函数。目的是让模型有足够的容量去学习复杂的预训练任务（如 ImageNet-21k 或 JFT-300M）。
  - **微调阶段 (Fine-tuning)**：MLP Head 通常被替换为一个**单层线性层** (Single Linear Layer)，直接将 768 维的特征映射到目标类别的数量（例如 ImageNet-1k 的 1000 类，或 CIFAR-10 的 10 类）。初始时通常初始化为零。

##### **4. 输出类别概率 (Output Class Probabilities)**

- **操作**：MLP Head 的输出（Logits）经过 **Softmax** 函数处理。
- **结果**：得到属于各个类别的概率分布。
- **对应图中**：最左上角的 **Class** 椭圆框，列出了 "Bird", "Ball", "Car" 等类别标签。模型会选择概率最大的那个作为最终预测结果。

****

### **3. 核心结论**

- 当拥有足够多的数据进行预训练时，ViT的表现会超过CNN。它突破了Transformer缺少归纳偏置（Inductive Bias）的限制，具有很好的迁移效果 。
- **缺点**：对于未见过的类别无法输出正确结果；对数据分布偏移（Distribution Shift）敏感 。

------







## **第二部分：图文对齐基石 —— CLIP**

CLIP (Contrastive Language-Image Pre-training) 解决了传统分类模型只能识别固定类别的问题，通过大规模图文对比学习实现了Zero-Shot能力。

![image-20260118101647820](/Users/moem/Library/Application Support/typora-user-images/image-20260118101647820.png)

### 1、第一部分：对比预训练 (Contrastive Pre-training)

![image-20260118091646733](/Users/moem/Library/Application Support/typora-user-images/image-20260118091646733.png)

这张幻灯片主要讲解了 CLIP 的 **训练阶段（Pre-training Phase）**。作为算法专家，我将结合这张图上的文字细节和原论文的深层逻辑，为你进行深度拆解。

------

#### 1.1、双流与对比（The Two-Stream & Contrastive）

架构图，这是 CLIP 的灵魂。

##### 1.1.1. 双流架构 (Two-Stream Architecture)

CLIP 并不是像 BERT 那样一开始就把图文拼在一起（Early Fusion），而是**各自为政，最后汇合**（Late Fusion）。

> 这其实是一些**多模态 BERT 变体**（如 **ViLBERT、LXMERT、VisualBERT、UniT** 等）所采用的思路。这些模型在 BERT 的基础上进行了扩展，以支持图文等多模态输入。

- **Text Encoder（上方紫色模块）：** 输入文本（如 "Pepper the aussie pup"），输出文本特征向量 $T_n$。

  <img src="/Users/moem/Library/Application Support/typora-user-images/image-20260118092527593.png" alt="image-20260118092527593" style="zoom:50%;" />

- **Image Encoder（下方绿色模块）：** 输入图片（小狗的照片），输出图片特征向量 $I_n$。

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260118092618634.png" alt="image-20260118092618634" style="zoom:50%;" />

##### 1.1.2. 对比矩阵（The $N \times N$ Matrix）

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260118092345086.png" alt="image-20260118092345086" style="zoom:50%;" />

图中中间那个网格矩阵是核心。假设一个 Batch 有 $N$ 对数据（论文中 $N=32768$）。

- **矩阵内容：** 每一个格子代表一个“图片向量”和一个“文本向量”的**点积（Dot Product）**，即相似度。
- **对角线（蓝色格子 $I_i \cdot T_i$）：** 这是**正样本（Positive Pairs）**。即“图1”配“文1”，“图2”配“文2”。训练目标是**最大化**这些蓝色的值。
- **非对角线（白色格子）：** 这是**负样本（Negative Pairs）**。即“图1”配“文2”（错配）。训练目标是**最小化**这些值。

> **专家视点：** 这种让正样本靠近、负样本推开的过程，就是幻灯片标题所说的 **Contrastive Learning（对比学习）**。它不需要人工打标签（Class Label），只需要图文是否匹配的自然信号。

------

#### 1.2、 深度解析：Text Encoder 的细节

幻灯片右侧文字提到了 Text Encoder 的具体实现，这里有几个关键点需要结合论文理解：

- **架构选择 (GPT-2)：**

  - **幻灯片原文：** “借鉴的是 GPT-2 架构”。
  - **论文细节：** 实际上这是一个简化版的 GPT-2（Decoder-only Transformer），使用了 Masked Self-Attention。它不是为了生成文本，而是为了理解文本。

  > 结合你提供的 CLIP 原论文（*Learning Transferable Visual Models From Natural Language Supervision*），我为你深入解读这一设计选择：
  >
  > ##### 1. 机制原理：为什么 Decoder 也能做特征提取？
  >
  > 你说得对，GPT-2 本质上是 Decoder，使用的是 **Masked Self-Attention（掩码自注意力）**，即“因果注意力”——当前的词只能看见前面的词，看不见后面的词。
  >
  > 那它怎么像 BERT 那样获取整句的语义表示呢？答案在于 **[EOS] Token**。
  >
  > - **BERT (Encoder):** 使用双向注意力。句首的 `[CLS]` token 能直接“看见”句子里所有的词，所以我们用 `[CLS]` 的向量表示全句。
  > - **GPT-2 (Decoder):** 使用单向注意力。
  >   - 句首的 `[SOS]` 什么都看不见。
  >   - 中间的词只能看见它之前的词。
  >   - **句尾的 `[EOS]` (End Of Sequence):** 它位于序列的最后一位。根据因果注意力的规则，**它能看见前面所有的词**。
  >
  > **论文原话验证：** 论文在 **2.4 Model Architecture - Text Encoder** 章节中明确写道：
  >
  > > "The text sequence is bracketed with [SOS] and [EOS] tokens and the activation of the highest layer of the transformer at the [EOS] token is treated as the feature representation of the text."  （文本序列被括在 [SOS] 和 [EOS] 标记之间，Transformer 最高层在 **[EOS] 标记处的激活值** 被视为文本的特征表示。）
  >
  > 因此，对于 Decoder 架构，取最后一个 token（[EOS]）的输出，在数学上就等价于聚合了全句的信息，可以作为一个全局的 Sentence Embedding。
  >
  > ##### 2. 设计动机：为什么不仅是为了“能用”，而是“特意选择”？
  >
  > OpenAI 并没有随便选一个模型，论文中给出了具体的理由。
  >
  > ###### (1) 为了保留“生成能力”与“预训练兼容性”
  >
  > 这是最核心的原因。BERT 这种纯 Encoder 架构很难做文本生成任务。而 GPT 架构天生就是生成模型。
  >
  > **论文原话验证：**
  >
  > > "Masked self-attention was used in the text encoder to preserve the ability to initialize with a pre-trained language model or add language modeling as an auxiliary objective, though exploration of this is left as future work."  （在文本编码器中使用掩码自注意力是为了**保留使用预训练语言模型进行初始化的能力**，或者**将语言建模作为辅助目标**的能力，尽管这留作未来的工作探索。）
  >
  > **解读：** 虽然 CLIP 本身是做对比学习（判别式），但 OpenAI 的研究人员希望架构具有通用性。如果未来想在训练中加入“预测下一个词”的任务（辅助目标），或者直接加载一个训练好的 GPT-2 权重来加速，使用 Decoder 架构就非常方便。
  >
  > ###### (2) 计算效率与架构细节
  >
  > CLIP 的 Text Encoder 对原始 GPT-2 做了一些轻量化修改以适应多模态任务：
  >
  > - **规模：** 基础版本使用了一个 63M 参数、12 层、512 宽度的模型（比标准 GPT-2 Base 还要小一点）。
  > - **序列长度：** 为了计算效率，最大序列长度被限制在 **76** 个 token 。这意味着它不需要处理超长文本，Decoder 的计算开销完全可控。
  > - **激活函数：** 使用了 **QuickGELU** 替代标准的 GELU，这也是为了计算效率 。
  >
  > ##### 3. 专家总结
  >
  > CLIP 使用 GPT-2 架构并非“误用”，而是一个深思熟虑的 **Trade-off（权衡）**：
  >
  > 1. **功能上：** 利用 `[EOS]` 的全局感受野，Decoder 能够完美胜任特征提取的任务。
  > 2. **扩展性上：** 相比 BERT，Decoder 架构保留了未来扩展为生成式模型（如 Image Captioning）的潜力，且符合 OpenAI 一贯的 GPT 技术栈路线。
  >
  > 这也是为什么后来的许多多模态工作（如 BLIP 等）开始探索 Encoder-Decoder 混合架构，而 CLIP 作为开山之作，证明了 Decoder-only 在理解任务上也是完全可行的。

- **特殊 Token ([SOS] & [EOS])：**

  - **幻灯片原文：** “添加表示开始和结束的符号 [SOS] 与 [EOS]... 将 [EOS] 位置的向量作为该 prompt 的特征”。
  - **专家解读：** 这是一个非常经典的 NLP 操作。
    - `[SOS]` (Start Of Sequence)：告诉模型“开始读句子了”。
    - `[EOS]` (End Of Sequence)：告诉模型“读完了”。
    - **为什么用 [EOS] 做特征？** 在 Transformer 机制中，由于注意力机制的存在，[EOS] 能够“看到”前面所有的词。因此，[EOS] 位置输出的向量，被视为**整个句子的语义摘要（Global Sentence Representation）**。这与 BERT 使用 `[CLS]` 标记的作用是异曲同工的。

------

#### 1.3、 深度解析：Image Encoder （VIT模型）

这是最具“专家味”的部分，因为它提到了具体的模型参数 `ViT-L/14@336px`。这在原论文的实验部分是非常重要的结论。

##### 1.3.1. 架构选型 (ResNet vs ViT)

- **幻灯片原文：** “尝试过 5 种 ResNet... 和 3 种 ViT... 最终选用的是 ViT-L/14@336px”。
- **论文背景：** OpenAI 在训练时发现，**ViT (Vision Transformer)** 比 ResNet 训练效率更高，且在大规模数据下上限更高。

##### 1.3.2. 解码神秘代码：`ViT-L/14@336px`

这个字符串代表了 CLIP 性能最强版本的配置，我们可以拆解为：

- **ViT (Vision Transformer)：** 说明它把图片切成小方块（Patches），像处理单词一样处理图片像素块。
- **L (Large)：** 代表模型规模。
  - ViT-B (Base) < **ViT-L (Large)** < ViT-H (Huge)。L 代表层数更深，参数量更大。
- **14 (Patch Size)：**
  - 这是指将图片切成 $14 \times 14$ 像素的小块。
  - **专家视点：** 常见的 ViT 是 /32 或 /16。**/14 意味着切得更细**。切得越细，序列越长，计算量呈平方级增长，但捕捉的细节也越丰富。这是高性能的关键。
- **@336px (Resolution)：**
  - **幻灯片原文：** “用更高分辨率 (336*336) 的图片做了一个 epoch 的 fine-tune”。
  - **专家视点 (FixRes 策略)：** 通常预训练图片是 $224 \times 224$。但在论文中，OpenAI 发现如果先用 224 训练，最后哪怕只用 336 分辨率微调短短一个 Epoch，性能就能大幅提升。这被证明是一个性价比极高的 Trick，能让模型“看”得更清楚。

------

#### 1.4、这张图告诉你 CLIP 到底强在哪？

结合幻灯片和论文，我们可以总结出 CLIP 成功的公式：

$$\text{CLIP} = \text{海量图文对} + \text{对比学习(Contrastive)} + \text{强大的编码器(ViT-L + GPT-2)}$$

这张幻灯片展示的是 CLIP 如何**将图像和文本强行拉入同一个特征空间**。一旦训练完成（即这张图的过程结束），Text Encoder 和 Image Encoder 就达成了“共识”。

### 2.第二部分：构建分类器 (Create Dataset Classifier)

训练完成后，如何使用模型进行分类？这里展示了 CLIP 独特的“Prompt Engineering”（提示工程）机制。

![image-20260118102438258](/Users/moem/Library/Application Support/typora-user-images/image-20260118102438258.png)

1. **标签文本化:**
   - 假设我们要识别一张图是不是“飞机”、“汽车”、“狗”或“鸟”。CLIP 不会把它们看作数字 ID（0, 1, 2...），而是看作文本。
   - 我们列出所有可能的标签（object classes）。
2. **Prompt Template (提示模版):**
   - 注意图中的 **"A photo of a {object}"**。
   - **论文洞察：** 论文 *Section 2.4* 指出，直接由单词（如 "dog"）本身可能会由歧义（是动词还是名词？）。通过加上 "A photo of a..." 这样的前缀，可以给模型提供上下文，告诉它这是一个关于图像描述的任务。这种做法通常能显著提升准确率（论文中提到约提升 1.3%）。
3. **生成文本特征:**
   - 这些构造好的句子（"A photo of a plane", "A photo of a dog"...）再次通过训练好的 **Text Encoder**，生成一组代表各个类别的文本特征向量（Linear weights）。

------

### 3.第三部分：零样本预测 (Zero-shot Prediction)

![image-20260118102810729](/Users/moem/Library/Application Support/typora-user-images/image-20260118102810729.png)

这是推理（Inference）阶段。

1. **图像编码:**
   - 拿到一张新的、没见过的图片（图中的狗），通过训练好的 **Image Encoder** 提取图像特征向量。
2. **相似度计算:**
   - 将这个**图像向量**与步骤 2 中生成的**所有文本向量**进行比较（计算余弦相似度）。
3. **预测结果:**
   - 图中绿色的柱状条最高，对应的文本是 "A photo of a dog"。
   - 因为该文本向量与图像向量的距离最近（相似度最高），模型就判定这张图是“狗”。



## **第三部分：Flamingoa ：Visual Language Model for Few-Shot Learning**

Flamingo 是多模态领域的一个里程碑式工作，它的核心设计哲学是：**“冻结”强大的单模态模型（视觉 encoder 和 LLM），只通过轻量级的适配器（Adapter）将它们连接起来。**

以下是 Flamingo 结构的详细拆解：![image-20260118210047834](/Users/moem/Library/Application Support/typora-user-images/image-20260118210047834.png)

### 1. 整体架构概览 (The Big Picture)

Flamingo 的架构可以分为三个主要部分：

1. **Vision Encoder（视觉编码器）：** 负责“看”图。
2. **Perceiver Resampler（感知重采样器）：** 负责“压缩”和“翻译”视觉信息。
3. **Language Model with Gated XATTN（带门控交叉注意力的语言模型）：** 负责“理解”并生成文本。

------

### 2. 详细组件解析

#### A. Vision Encoder (冻结的视觉之眼)

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260118210732686.png" alt="image-20260118210732686" style="zoom:50%;" />

- **作用：** 将输入的图像或视频帧转换为高维特征。
- **模型：** 论文中使用的是 **NFNet (Normalizer-Free ResNet)**，这是一个在大量图像文本对上预训练好的对比学习模型（类似于 CLIP 的视觉端），输入图片，输出对应的特征，维度为 [S, d]，其中 S 表示有多少块，d 表示每一个块的特征维度。类似于其他 VLM 常用的 ViT模型。
- **特点：** 在训练 Flamingo 时，这个部分是**冻结的 (Frozen)**。
- **输入/输出：** 输入像素，输出一个 2D 的特征网格（feature map）。对于视频，它会逐帧处理，形成一个时间序列的特征。

#### B. Perceiver Resampler (核心组件：视觉-语言的桥梁)

![image-20260118214427026](/Users/moem/Library/Application Support/typora-user-images/image-20260118214427026.png)

这张图片展示的是 DeepMind 的 Flamingo 模型中至关重要的组件——**Perceiver Resampler**（感知器重采样器）的工作原理。

根据 Flamingo 的原论文，这个模块的核心作用是解决“视觉信息量巨大且长度可变”与“语言模型需要固定且有限的输入上下文”之间的矛盾 。

以下我结合论文中的细节（特别是 Section 2.1 和 Appendix A.1.1），为你逐步解析图中文字和代码的含义：

##### 1. 视觉特征提取 (Vision Encoder Output)

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260118220819823.png" alt="image-20260118220819823" style="zoom:50%;" />

- **图解内容**：每个图像经 Vision Encoder 会生成一个 [S, d] 的视觉特征，T 个图像对应 $x_f$  的维度为 [T, S, d]。

   ![image-20260118221133782](/Users/moem/Library/Application Support/typora-user-images/image-20260118221133782.png)

- **论文细节**：

  - Flamingo 使用预训练且冻结的 **NFNet-F6** 作为视觉编码器 。

  - **$T$** 代表时间步（Time），即视频的帧数（如果是单张图片，则 $T=1$）。

  - **$S$** 代表空间位置（Space），即卷积神经网络输出的特征图网格被拉平后的大小（Flattened spatial grid）。

    > 要理解 $S$ 是如何得到的，我们需要回到 CNN 的工作原理，并结合 Flamingo 论文的具体设置。
    >
    > 简单来说，$S$ **不是**像 Vision Transformer (ViT) 那样把图片“切”成小方块得到的，**而是 CNN 经过多次卷积和降采样后输出的“特征图网格（Feature Grid）”被拉平的结果。**
    >
    > 以下是详细的步骤解析：
    >
    > ###### 1. 输入图像 (Input)
    >
    > - Flamingo 在训练时，会将输入的图像（或视频帧）调整大小（Resize）到 **$320 \times 320$** 的分辨率 。
    >
    > ###### 2. CNN 卷积处理 (Feature Extraction)
    >
    > - 图像进入 **NFNet-F6**（预训练的 CNN）。
    > - CNN 的作用是逐层提取特征，同时降低空间分辨率（Downsampling）。
    > - 通常，深度 CNN（如 ResNet 或 NFNet）的最后一层输出的特征图（Feature Map），其长宽会缩小为原图的 1/32（这就叫 Stride 32）。
    >   - **计算示例**：如果输入是 $320 \times 320$，经过 stride=32 的网络后，输出的特征图大小约为 **$10 \times 10$**。
    > - 此时，我们得到了一个形状为 $[H', W', d]$ 的 3D 张量。在这里，$H'=10, W'=10$，即一个 $10 \times 10$ 的网格。
    >
    > ###### 3. 定义 S (Spatial Grid)
    >
    > - 论文中明确提到：“We use the output of the final stage, a 2D spatial grid of features” 。
    >
    > - 这个 $10 \times 10$ 的网格中的每一个点（Grid Cell），不仅仅代表一个像素，而是代表原图中对应的一块区域（感受野）的高级语义特征。
    >
    > - 因此，这里的 **$S$ 就是特征图的高度乘以宽度**。
    >
    >   - $$S = H' \times W'$$
    >
    >      （例如 $10 \times 10 = 100$）。
    >
    > ###### 4. 拉平 (Flattening)
    >
    > - 为了输入到后续的 Transformer 结构（Perceiver Resampler）中，必须把这个 2D 网格拉直。
    > - 论文描述：“...that is flattened to a 1D sequence” 。
    > - **操作**：将 $10 \times 10$ 的网格拉成一条长度为 100 的序列。
    > - **结果**：对于单张图像，我们得到了一个 $[S, d]$ 的张量。对于 $T$ 帧视频，就是 $[T, S, d]$。
    >
    > ------
    >
    > ###### 总结：与 ViT 切块的区别
    >
    > 为了防止混淆，我们可以对比一下两种“S”的来源：
    >
    > | **方式**     | **对应模型**          | **S 是如何得到的？**                                         | **物理含义**                                                 |
    > | ------------ | --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
    > | **CNN 方式** | **Flamingo (NFNet)**  | **卷积降采样**。$320 \times 320$ 的图 $\rightarrow$ $10 \times 10$ 的特征图 $\rightarrow$ 拉平得到 $S=100$。 | $S$ 中的每个点对应图像上的一个**感受野区域**，包含重叠信息。 |
    > | **ViT 方式** | **LlaVA / GPT-4V 等** | **硬切块 (Patching)**。把图切成 $16 \times 16$ 的小方块，铺成一排。 | $S$ 中的每个点对应图像上一个**互不重叠的切片**。             |
    >
    > 所以，Flamingo 的 $S$ 是**卷积神经网络最后一层输出特征图的空间分辨率乘积**。

  - **$d$** 是特征的隐藏层维度。

  - 对于视频输入，每一帧都是独立编码的，形成了一个 3D 的时空网格特征 。

##### 2. 注入时间信息 (Time Embeddings)

- **图解内容**：$x_f$ 加上维度为 $[T, 1, d]$ 的 `time_embeddings`。

  ![image-20260118221341406](/Users/moem/Library/Application Support/typora-user-images/image-20260118221341406.png)

- **论文细节**：

  - 由于视觉编码器是独立处理每一帧的，它本身不知道帧的顺序。因此，必须加上可学习的**时间位置嵌入（temporal position embeddings）**
  - 值得注意的是，论文中提到他们**只使用了时间嵌入**，并没有使用空间位置嵌入，因为 CNN 本身已经隐式包含了一定的空间信息 。

  > 这是一个非常棒的观察！你的疑惑很合理，因为现在的多模态大模型（如 LLaVA、BLIP-2 等）确实普遍使用 Vision Transformer (ViT) 作为视觉编码器。
  >
  > 但在 **Flamingo** 这篇论文中，**它的 Vision Encoder 确实是 CNN，而不是 ViT。**
  >
  > 根据论文原文，具体情况如下：
  >
  > ###### 1. Flamingo 使用的是 NFNet (CNN)
  >
  > 论文明确指出，他们的视觉编码器是一个预训练并冻结的 **Normalizer-Free ResNet (NFNet)**，具体型号是 **F6**。
  >
  > > *"Our vision encoder is a pretrained and frozen Normalizer-Free ResNet (NFNet) [10] - we use the F6 model."* 
  >
  > NFNet 是一种高性能的卷积神经网络（CNN），它是 ResNet 的变体，去掉了 Batch Normalization 层。
  >
  > ###### 2. 为什么论文说“隐式包含空间信息”？
  >
  > 这正是因为它是 CNN。
  >
  > - **CNN 的特性**：卷积操作是在局部滑动的，且层与层之间保持着二维的空间结构（Height $\times$ Width）。即使到了最后一层卷积层，输出的特征图（Feature Map）仍然是一个 $H \times W$ 的网格，特征在网格中的位置直接对应原图的空间位置。
  >
  > - **论文的解释**：在 Appendix A.1.1 中，作者解释了为什么在 Perceiver Resampler 中只加了时间嵌入（Time Embeddings）而**没有加空间位置嵌入**：
  >
  >   > *"...we did not observe improvements from the latter [spatial grid position encodings]. This rationale behind is likely that CNNs, such as our NFNet encoder, are known to implicitly include spatial information channel-wise"* 
  >
  >   也就是说，因为 CNN 输出的特征本身就“记得”自己在哪儿（左上角的特征对应左上角的像素），所以不需要像 ViT 那样，因为把图片切成了 Patch 且打散了顺序，必须强行加上 Positional Embedding 告诉模型“我是第一块砖”。
  >
  > ###### 3. 作者尝试过 ViT 吗？
  >
  > 是的，作者做过对比实验。
  >
  > 在论文的 Ablation Studies（消融实验） 部分（Table 3），作者对比了他们使用的 NFNet-F6 和当时很火的 CLIP ViT-L/14：
  >
  > - **NFNet-F6 (CNN)**：Overall score 64.9% 
  > - **CLIP ViT-L/14 (ViT)**：Overall score 62.7% 
  >
  > 在 Flamingo 的实验设置下，这个 CNN (NFNet-F6) 的表现比那个版本的 ViT (CLIP ViT-L/14) 要好 。

##### 3. 特征展平 (Flattening)

- **图解内容**：将时间和空间维度拉平，$x_f \rightarrow [T \times S, d]$。对应的代码是 `x_f = flatten(x_f)`。

- **论文细节**：

  - 这一步将所有帧的所有局部特征拼接成一个超长的 1D 序列 。

    > 这是一个非常敏锐的问题！严格来说，你的直觉是对的：**从数学张量（Tensor）的角度看，它其实是一个 2D 矩阵，而不是 1D 向量。**
    >
    > 但是，在 **Transformer 和序列建模**的语境下，论文称其为“1D 序列”是有特定含义的。让我们来拆解一下这里“1D”的真正所指：
    >
    > ###### 1. 为什么说是“1D 序列”？（语义视角）
    >
    > 论文中提到 "flattened to a 1D sequence" 111，这里的“1D”指的是**令牌（Token）的排列方式**，而不是张量的阶数。
    >
    > - **输入状态（3D 结构）：** 原始特征包含 **时间** ($T$) 和 **空间** ($H, W$) 的结构信息。比如 $[T, H, W, d]$。这时候，像素点之间有“上下左右”和“前后”的空间/时间邻居关系。
    > - **展平之后（1D 结构）：** 操作 `x_f -> [T*S, d]` 将时间和空间维度强行拉直。此时，模型不再把它们看作一个“网格”或“视频块”，而是看作**一长串排成一队的点**。
    >   - 就像把原本拼成“魔方”的小方块，拆开后排成了一条长长的线。
    >   - **“1D”指的是序列长度（Sequence Length）这一维，即 $N = T \times S$。**
    >
    > ###### 2. 实际上是什么样子的？（数学视角）
    >
    > 在代码和内存中，这个 x_f 实际上是一个 Rank-2 张量（二维矩阵），形状为：$$[N, d]$$
    >
    > - **$N = T \times S$**：序列的长度（行数）。这是被“展平”的部分。
    > - **$d$**：特征维度（列数，hidden dimension）。**这一维从来没有被展平**，它是每个像素点携带的信息量（embedding size）。
    >
    > ###### 3. 图解对比
    >
    > 可以把它想象成一篇文章：
    >
    > - **$d$ (Feature Dim)**：就像每个“单词”的词向量（例如 512 维）。
    > - **$T \times S$ (Seq Len)**：就像文章中单词的总数量。
    >
    > 虽然每个单词本身是一个 512 维的向量（导致整体是二维矩阵），但我们通常说“这是一句话”（1D 文本序列），而不是说“这是一个二维单词矩阵”。
    >
    > ###### 总结
    >
    > - **问**：这真的是 1D 吗？
    > - **答**：
    >   - **结构上**：它是 **1D 序列**（因为不仅失去了空间网格结构，也失去了时间帧结构，变成了一个列表）2。
    >   - **数据上**：它是 **2D 张量**（形状为 $[Sequence\_Length, Feature\_Dim]$）。
    >
    > 正是因为这个序列长度 $T \times S$ 可能非常大（成千上万个点），直接扔进 Transformer 计算 Attention（复杂度 $O(N^2)$）会算不过来，所以才需要 Perceiver Resampler 把它压缩成固定数量（例如 64 个）的 Token 3。

  - 如果视频很长，这个序列会非常长（$T \times S$ 很大），直接输入给 LLM 会导致计算量爆炸。这正是引入 Perceiver Resampler 的原因。

##### 4. 核心机制：Perceiver Resampler (感知器重采样)

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260118222641905.png" alt="image-20260118222641905" style="zoom:50%;" />

这是图中紫色方框和代码核心逻辑所在。

- **定义 Query (Learned Latent Queries)**：

  - **图解内容**：自定义 $R$ 个可学习的 Query Token，维度为 $[R, d]$。

    ![image-20260118223229562](/Users/moem/Library/Application Support/typora-user-images/image-20260118223229562.png)

  - **论文细节**：这些是模型预定义并随机初始化的潜在向量（Latent Queries） 。即使输入视频长度不同，这个 $R$ 的数量是固定的（论文中设定为 64）。

- **注意力机制 (Attention)**：

  - **图解内容**：将 $x_f$ 作为 Key 和 Value，Learned Latent Queries 作为 Query。

    <img src="/Users/moem/Library/Application Support/typora-user-images/image-20260118222859561.png" alt="image-20260118222859561" style="zoom:50%;" />

  - **代码细节补充**：仔细看右侧代码 `kv=concat([x_f, x])`，这对应论文 Appendix A.1.1 的一个细节：为了获得更好的性能，Keys 和 Values 实际上是**视觉特征 $x_f$ 与 潜在查询向量 $x$ 的拼接** 。

  - **运算**：通过 Cross-Attention（交叉注意力），这 $R$ 个 Query 主动去扫描那一大堆视觉特征 $[T \times S, d]$，提取出最重要的信息。

##### 5. 输出：固定长度的视觉特征

- **图解内容**：经过 `num_layers` 层 Transformer block，**得到新的特征 $x$，维度为 $[R, d]$。**
- **论文细节**：
  - 无论输入是 5 帧还是 30 帧视频，经过这个模块后，输出永远是固定的 **$R$ 个视觉 Token**（例如 64 个）。
  - 这大大降低了后续语言模型处理视觉信息的计算复杂度，实现了从“变长输入”到“定长输出”的转换。

#### C. Gated Cross-Attention Dense Block (嵌入 LLM 的适配器)

结合 Flamingo 论文，这张图解释了**如何将视觉信息“注入”到一个已经被预训练好且冻结（Frozen）的大语言模型（LLM）中**，而不破坏 LLM 原有的语言能力。

以下是结合论文对图片内容的详细解读：

![image-20260119004604300](/Users/moem/Library/Application Support/typora-user-images/image-20260119004604300.png)

------

##### 1. 核心概念：冻结与插入 (Frozen & Insertion)

- **背景**：Flamingo 的核心设计理念是**不重新训练**底层的 LLM（图中的蓝色 `LM layer`，带雪花图标 ❄️ 表示参数冻结），以保留其强大的文本生成能力。
- **动作**：为了让 LLM 能“看懂”图片，Flamingo 在现有的 LLM 层之间**插入**了新的可训练层（图中的粉色 `GATED XATTN-DENSE`）。
- **对应图中**：中间的方框示意图展示了这一点，原本连续的 LLM 层被拉开，中间塞入了粉色的新模块。![image-20260119005335783](/Users/moem/Library/Application Support/typora-user-images/image-20260119005335783.png)

##### 2. 输入数据的角色分配 (Query, Key, Value)

这是 Cross Attention（交叉注意力）机制的核心，对应图中左侧文字说明：

- **Query (查询)**：来自 **Language input (文本输入)**。
  - **LLM 正在处理的文本 token 作为 Query。它在问：“为了理解这个词，我需要看图像的哪一部分？**”
- **Key / Value (键/值)**：来自 **Vision input (视觉输入)**。
  - 图片左侧文字提到 **Perceiver Resampler**。在进入这个模块前，原始图像已经通过 Perceiver Resampler 被压缩成了固定数量的视觉 token（Visual Tokens）。这些视觉特征作为 Key 和 Value 供文本查询。

##### 3. Gated Cross-Attention 的工作流程 (右侧详细结构)

右侧放大的结构图和代码解释了数据是如何流动的：

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260119005846429.png" alt="image-20260119005846429" style="zoom:50%;" />

###### 第一步：Gated Cross Attention (门控交叉注意力)

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260119010000786.png" alt="image-20260119010000786" style="zoom:50%;" />

- **机制**：文本特征 ($Y$) 去“关注”视觉特征 ($X$)。

- 公式（对应代码）：

  <img src="/Users/moem/Library/Application Support/typora-user-images/image-20260119010228369.png" alt="image-20260119010228369" style="zoom:50%;" />

  $$y = y + \tanh(\alpha_{xattn}) \times \text{attention}(q=y, kv=x)$$

- **关键点**：注意那个 `tanh gating`。这是 Flamingo 的一个重要 trick。

###### 第二步：Gated FFW (门控前馈网络)

- **机制**：标准的 MLP 层，用于进一步处理融合了视觉信息的特征。

- 公式（对应代码）：

  <img src="/Users/moem/Library/Application Support/typora-user-images/image-20260119010353246.png" alt="image-20260119010353246" style="zoom:50%;" />

  $$y = y + \tanh(\alpha_{dense}) \times \text{ffw}(y)$$

###### 第三步：进入原始 LLM 层

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260119010720430.png" alt="image-20260119010720430" style="zoom:50%;" />

- 处理后的数据 ($Y$) 接着进入原本冻结的 `self attention` 和 `FFW` 层（图中的蓝色部分），继续进行标准的语言模型推理。

##### 4. 为什么要用 "Gated" (门控)？——初始化技巧

这是论文中最精妙的设计之一，体现在代码中的 `init at 0`：

- **问题**：如果在训练初期直接把视觉噪音加到训练好的 LLM 中，会瞬间破坏 LLM 原有的语言分布，导致模型“变傻”或训练崩溃。

  <img src="/Users/moem/Library/Application Support/typora-user-images/image-20260119011016038.png" alt="image-20260119011016038" style="zoom:50%;" />

- **解决方案 (`tanh gating`)**：

  - 引入一个可学习的参数 `alpha` (即代码中的 `alpha_xattn` 和 `alpha_dense`)。
  - **初始化为 0**。
  - 因为 $\tanh(0) = 0$，所以在训练**刚开始时**，新加入的模块输出为 0。
  - 这意味着：**初始状态下，整个网络等同于那个原始的、纯文本的 LLM。**
  - **随着训练进行**，`alpha` 逐渐更新，门慢慢打开，视觉信息才一点点地融入进来。

------

### 3. Flamingo 模型如何处理交错排列（Interleaved）的多模态（图像+文本）输入数据

![image-20260119011841189](/Users/moem/Library/Application Support/typora-user-images/image-20260119011841189.png)

#### 1. 视觉信号的处理 (Visual Encoder + Perceiver Resampler)

图片的第一点提到：“视觉图像全部需要经过 **Vision Encoder + Perceiver Resampler** 生成的 Vision input 作为 Key, Value 输入。”

- **对应原文机制：** Flamingo 首先使用一个预训练且冻结的 **Vision Encoder**（通常是 NFNet）将图像像素转换为特征 。
- **Perceiver Resampler 的作用：** 这是一个关键的架构创新。Vision Encoder 输出的特征数量很大且可变（取决于图像或视频的分辨率/长度），**Perceiver Resampler** 的作用是将这些变长的特征转换为**固定数量**（例如 64 个）的视觉 Token 。
- **作为 Key/Value：** 在注意力机制中，这些固定数量的视觉 Token 被用作 **Keys (键)** 和 **Values (值)**，供语言模型查询 。

#### 2. 文本信号与特殊 Token (Tokenization)

![image-20260119013346637](/Users/moem/Library/Application Support/typora-user-images/image-20260119013346637.png)

图片的第二点提到：“文本全部经 Tokenization 后输入... 插入 `<BOS>`, `<EOC>` 等起止 Token，也会插入 `<image>` Token 作为图像的位置标识。”

- **交错数据的构建：** Flamingo 的核心能力是处理“交错”的图文数据（例如网页截图）。论文指出，模型通过在文本中图像出现的位置插入 `<image>` 标签来构建输入序列 。
- **特殊 Token 的含义：**
  - **`<image>`**：指示模型此处有一个图像，模型需要在此处关注视觉信息 。
  - **`<BOS>` (Beginning of Sequence)**：序列的开始。
  - **`<EOC>` (End of Chunk)**：用于标记一段内容的结束 。
- **作为 Query：** 处理后的文本 Token 在 Cross Attention 层中作为 **Queries (查询)**，**去“查询”前面提到的视觉 Key/Value 。**

#### 3. 掩码交叉注意力机制 (Masked Cross Attention)

图片的第三点提到：“其中的 **Cross Attention Mask** 也经过特**殊设计，让文本只和相关图像进行交互**。”

- **设计目的：** 这是一个非常关键的设计，称为“per-image/video attention masking”（逐图像/视频注意力掩码）。

- **工作原理：** 在一个包含多个图像和文本的长序列中，**模型限制了文本 Token 的视野**。具体来说，**特定的文本 Token 只能与“刚刚出现”在它前面的那张图像（或视频）的视觉 Token 进行交互**，而**不能看到序列中更前面的其他图像** 。

- **图解含义：**

  ![image-20260119013708790](/Users/moem/Library/Application Support/typora-user-images/image-20260119013708790.png)

  - 图片底部展示了 `<image>` 标签如何插入到文本流中。
  - 图片中部的**蓝色条状图（Masked cross attention）**直观地展示了这一点：深蓝色块表示允许注意的部分。你可以看到，第一段关于 "puppy" 的文本只对应第一张 "puppy" 的图像；第二段关于 "cat" 的文本只对应第二张 "cat" 的图像 。

- **优势：** 这种单图像交叉注意力机制允许模型无缝地泛化到任意数量的视觉输入，无论训练时使用了多少张图像 。

> 结合 Flamingo 论文原文，这个机制被称为 **"Per-image/video attention masking"（逐图像/视频注意力掩码）**。它的设计非常巧妙，主要解决了多模态模型在处理任意长度序列时的泛化问题和计算效率问题。
>
> 以下结合论文 **2.3 节 (Multi-visual input support)**、**附录 A.1.3** 以及 **消融实验 3.3 节** 的内容，为您详细解释其背后的原理和逻辑：
>
> ###### 1. 核心机制：由掩码控制的单图像交互
>
> 在 Flamingo 的架构中，文本生成的每一步都需要“看”图像。但是，如果文本序列很长，包含了几十张图像，应该看哪一张呢？Flamingo 的选择是：**只看最近的那一张**。
>
> - **定义函数 $\phi$：** 论文在 **附录 A.1.3** 中定义了一个函数 $\phi(l)$。对于文本序列中的第 $l$ 个 Token，$l$ 位置之前出现的最后一张图像（或视频）的索引就是 $\phi(l)$。如果前面没有图像，则为 0。
> - **掩码的设计：** 在计算 Cross Attention（交叉注意力）矩阵时，模型会应用一个掩码（Mask）。这个掩码强制规定：**当前的文本 Token $y_l$ 只能与图像 $x_{\phi(l)}$ 的视觉 Token 进行交互** 。
> - **图解（Figure 7）：** 正如您上传图片中的深蓝色块所示，特定的一段文本（例如描述 "my puppy" 的文字）在 Cross Attention 层中，其“视野”被物理限制，只能看到对应的 "puppy" 图片，而完全看不到后面的 "cat" 图片，也看不到前面的其他无关图片 。
>
> ###### 2. 为什么要这样设计？（设计原理与优势）
>
> 您可能会问：“如果只能看到最近的一张图，模型怎么理解整个故事的上下文呢？” 论文在 **2.3 节** 和 **3.3 节** 给出了三个关键理由：
>
> ###### A. 隐式全局上下文 (Implicit Global Context via Self-Attention)
>
> 虽然 **Cross Attention**（交叉注意力）被限制为只能看“当前图像”，但 Flamingo 的 **Self-Attention**（自注意力，即语言模型本身的部分）是全局的 。
>
> - **原理：** 之前的文本 Token 已经与其对应的图像交互过了，并且将这些视觉信息“编码”进了文本的隐层状态（Hidden States）中。
> - **结果：** 当前的文本 Token 可以通过**文本自注意力机制**关注到之前的文本 Token，从而间接地获取之前所有图像的信息。论文指出：“这种依赖关系通过 LM 中的自注意力得以保留” 。
>
> ###### B. 泛化能力 (Seamless Generalization)
>
> 这是该设计最大的工程优势。
>
> - **问题：** 如果允许模型同时关注所有前面的图像，训练时如果只用了 5 张图（为了节省显存），测试时突然给它 32 张图，模型可能会因为处理不了这么多视觉 Token 而崩溃或性能下降。
> - **解决：** 通过这种“一次只看一张”的限制，模型在每一步 Cross Attention 中处理的视觉 Token 数量是**固定**的（即一张图的 Token 数）。
> - **效果：** 论文提到，虽然训练时每个序列最多只用了 5 张图像（$N=5$），但模型在推理时可以无缝处理包含 32 张甚至更多图像的序列 。
>
> ###### C. 避免歧义 (Disambiguation)
>
> 论文在 **3.3 节** 的消融实验中对比了“只看最近一张图”和“看前面所有图”的效果。
>
> - **发现：** “只看最近一张图”的效果要好得多（性能提升 7.2%）。
> - **原因：** 论文推测，如果让模型直接关注前面所有图像，模型很难区分哪些视觉特征属于哪张图片（除非加入复杂的图像位置编码，但实验证明这并不鲁棒）。限制只看当前图像，消除了这种特征归属的歧义 。
>
> 简单来说，Flamingo 的原理是：**“在视觉上只关注当下，在文本上回顾历史。”**
>
> - **视觉上 (Cross Attention)**：通过 Mask 强制文本只与最近的一张图交互，确保了模型能处理任意长度的图文流，且不会混淆不同图片的内容。
> - **逻辑上 (Self Attention)**：通过语言模型强大的记忆能力，将之前图片的信息通过文本上下文传递下来，从而实现对长序列的整体理解。

------

### 4.《Flamingo: a Visual Language Model for Few-Shot Learning》）的**数据构建**、**预训练流程**以及**微调策略**。

结合Flamingo的原始论文，我将为你深入解析这两张图片中的关键信息。

------

#### 1、数据及处理 (Data Pipeline)

Flamingo之所以强大，核心原因之一在于它构建了极其多样化且高质量的多模态数据。论文强调，为了让模型具备“Few-Shot（少样本）”能力，必须让模型在训练时就见过“图文交错”的复杂序列，而不仅仅是简单的“图-文”对。

##### 1. M3W (MultiModal Massive Web) 数据集

这是Flamingo最重要的创新数据集，目的是模拟人类浏览网页时的真实体验。

- **来源与规模**：从4300万（43M）个网页中提取，原始数据非常庞大。
- **结构特点**：它不是简单的图片配标题，而是**“交替的图像文本”**（Interleaved image-text）。
- **采样策略**：对于每个文档，随机选择一段长为 $L=256$ tokens 的文本序列，并选取该序列前的 $N=5$ 张图片。总共包含了1.85亿（185M）张图片。
- **论文背景**：这种数据格式让Flamingo学会了在上下文中进行推理（In-context learning），例如：“图1是猫，图2是狗，图3是...”模型能学会补全“兔子”。

##### 2. 图像-文本对 (Image-Text Pairs)

为了增强视觉和文本的强对齐能力，使用了传统的成对数据：

- **ALIGN 数据集**：包含18亿（1.8B）个图文对。这是基于噪声文本监督的大规模数据集，用于提供广泛的视觉概念覆盖。
- **LTIP (Long Text & Image Pairs)**：包含3.12亿（312M）个对。
  - **关键点**：标准的图文对（如COCO或ALIGN）通常文本很短（如“一只猫”）。LTIP旨在提供**更长的描述**和**更高的质量**，这对于训练模型生成详细的图像描述（Captioning）至关重要。

##### 3. 视频-文本对 (Video-Text Pairs - VTP)

- 包含2700万（27M）个短视频（平均22秒），配有句子描述。
- **论文背景**：引入视频数据是为了让Flamingo不仅能处理静态图像，还能理解时间序列上的视觉信息，使其成为真正的多模态模型。

##### 4. 数据清洗和去重 (Cleaning & Deduplication)

为了保证Scaling Law（缩放定律）有效，数据质量至关重要。DeepMind采取了严格的清洗策略：

- **过滤内容**：删除非英文、低质量、重复文档。
- **过滤图像**：
  - 尺寸：删除长或宽小于64像素的图（太糊）。
  - 比例：删除长宽比大于3的图（太扁或太长，通常是Banner或侧边栏广告）。
  - 质量：删除单一颜色的图（如纯色背景图）。

------

#### 2、预训练和微调 (Pre-training & Fine-tuning)

Flamingo复杂的三阶段训练流水线。Flamingo并非从头训练，而是“嫁接”了两个强大的预训练模型：**视觉编码器（Vision Encoder）和语言模型（Langauge Model, Chinchilla）**。

##### 第一阶段：Vision Encoder 预训练

在训练Flamingo整体架构之前，先要把“眼睛”练好。

- **模型架构**：论文中提到使用的是**NFNet (Normalizer-Free ResNet)** 系列（F6型号）。
- **训练数据**：基于ALIGN和LTIP数据集。
- **硬件与规模**：
  - 使用512个TPUv4芯片。
  - Batch size高达16,384，利用对比学习（Contrastive Learning）拉近图文距离。
  - **分辨率**：$288 \times 288$。
  - 训练了120万（1.2M）步。

##### 第二阶段：Flamingo 预训练 (核心阶段)

这是将视觉信息注入语言模型的阶段。

- **模型规格**：训练了三个版本，分别对应不同的Chinchilla语言模型底座大小：
  - **Flamingo-3B** (基于1.4B LM)
  - **Flamingo-9B** (基于7B LM)
  - **Flamingo-80B** (基于70B LM)
- **关键策略**：
  - **冻结 LM**：在这个阶段，**冻结**了预训练好的Chinchilla语言模型层，只训练新加入的**Gated Cross-Attention（门控交叉注意力）**层。这既节省了算力，又保留了语言模型原本强大的文本能力。
  - **分辨率提升**：输入图像分辨率从288提升到 $320 \times 320$。
  - **大规模并行**：Flamingo-80B在1536个TPUv4上训练了15天。使用了模型并行（Model Parallelism），将层切分为16个shard。
  - **精度混合**：参数和优化器用FP32（保证稳定），激活和梯度用BF16（提升速度）。

##### 第三阶段：微调 (Fine-tuning)

这是为了进一步提升性能，特别是针对特定任务或视觉细节。

- **策略变化**：
  - **解冻 Vision Encoder**：这是与预训练阶段最大的不同。在微调时，**Vision Encoder也参与训练**。这允许视觉特征适应具体的下游任务。
  - **分辨率再次提升**：从320提升到 $480 \times 480$。更高的分辨率意味着模型能看清图片中的更多细节（如OCR文字识别）。
- **效果**：论文指出，这种解冻视觉编码器并提高分辨率的策略，通常能显著改善模型在VQA（视觉问答）等任务上的效果。





## **第四部分：BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**

BLIP-2的核心贡献在于提出了一种轻量级的对齐机制，能够低成本地利用现成的冻结视觉模型和语言模型。

### **1.  **BLIP-2 (Bootstrapping Language-Image Pre-training) 的核心架构

BLIP-2 的核心思想是**利用现有的、强大的预训练单模态模型（视觉模型和语言模型）**，通过一个轻量级的**中间组件（Q-Former）**将它们连接起来，从而实现高效的视觉-语言预训练 。

![image-20260119094427912](/Users/moem/Library/Application Support/typora-user-images/image-20260119094427912.png)

#### 1. Image Encoder (图像编码器)

**图中文字**：和 Flamingo 模型的 Vision Encoder 作用一样，用于提取视觉特征，采用的是 CLIP ViT-L/14 和 EVA-CLIP ViT-g/14。

**论文原理解读**：

- **冻结参数 (Frozen):** BLIP-2 的一个关键策略是**冻结**图像编码器的参数 。这意味着在训练过程中，图像编码器的权重不会更新。这样做既节省了计算成本，又利用了预训练模型高质量的视觉表达能力 。
- **模型选择:** 论文中确实尝试了两种最先进的视觉 Transformer 模型：**来自 CLIP 的 $ViT-L/14$ 和来自 EVA-CLIP 的 $ViT-g/14$** 。
- **作用:** 它的任务是**将输入的图像转换为高维的视觉特征向量**，供后续的 Q-Former 提取使用 。

#### 2. Q-Former (Querying Transformer)

**图中文字**：**Query Transformer**，**用来弥补 image 模态和 text 模态的差距，实现特征对齐**。

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260119095704586.png" alt="image-20260119095704586" style="zoom:50%;" />

**论文原理解读**：

- **核心组件:** **Q-Former 是 BLIP-2 中最关键的可训练模块** 。它是一个轻量级的 Transformer，**充当了冻结的图像编码器和冻结的 LLM 之间的“信息瓶颈” (Information Bottleneck) 。**
- **工作机制:**
  - 它使用一组可学习的查询向量 (**Learned Queries**，图中 Q-Former下方的小方块) 作为输入 。
  - 通过**交叉注意力机制 (Cross-Attention)**，**这些 Queries 与图像编码器输出的特征进行交互，提取出与文本最相关的视觉特征** 。



##### 第一阶段：视觉-语言表征学习 (Vision-and-Language Representation Learning)

**对应图中左侧虚线框部分：Bootstrapping Pre-trained Image Models**

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260119114906150.png" alt="image-20260119114906150" style="zoom:50%;" />

这一阶段的目标是训练 Q-Former，让它学会**理解图像**，并将图像特征对齐到文本空间。

1. **输入处理：**
   - **图像：** 输入到冻结的 **Image Encoder**（如 ViT-L/14），提取出通用的视觉特征。
   - **Queries：** 一组可学习的向量输入到 Q-Former。
   - **文本：** 对应的图片描述文本也输入到 Q-Former。
2. **Q-Former 的内部工作：**
   - Q-Former 内部包含两个 Transformer 子模块：一个用于处理图像特征，一个用于处理文本。
   - Queries 在这里起到了“瓶颈”作用，它们被迫从图像编码器输出的海量特征中，只提取与文本最相关的视觉信息。
3. **三个预训练目标（论文核心细节）：** 虽然图中简化了，但为了理解“表征学习”，你需要知道 Q-Former 在这一阶段通过三个任务联合训练：
   - **ITC (Image-Text Contrastive Learning)：** 对比学习。拉近匹配的图文对（Queries 输出 vs 文本特征），推开不匹配的。这让 Queries 学会捕捉图像的高层语义。
   - **ITG (Image-grounded Text Generation)：** 基于图像的文本生成。给 Q-Former 图像特征，让它生成对应的文本。这训练了 Q-Former 的多模态理解能力。
   - **ITM (Image-Text Matching)：** 图文匹配。判断输入的图像和文本是否是一对。

**阶段一总结：** 这个阶段不涉及 LLM。它的成果是训练好了一个能够提取高质量、包含语义信息的视觉 Query Embedding 的 Q-Former。

------

##### 第二阶段：视觉到语言生成学习 (Vision-to-Language Generative Learning)

**对应图中右侧虚线框部分：Bootstrapping Pre-trained LLMs**

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260119115047323.png" alt="image-20260119115047323" style="zoom:50%;" />

这一阶段的目标是将上一阶段学到的视觉信息，转换成 LLM 能够理解的“语言”，从而激活 LLM 的生成能力。

1. **连接 LLM：**

   - 现在，我们引入了冻结的 **Large Language Model (LLM)**（如 OPT 或 Flan-T5）。

   <img src="/Users/moem/Library/Application Support/typora-user-images/image-20260119120110457.png" alt="image-20260119120110457" style="zoom:25%;" />

   - **关键连接点：** Q-Former 输出的 Queries（也就是上一阶段训练好的视觉表征），**经过一个全连接层（Projection Layer，图中箭头所示）**，其维度被调整得和 LLM 的文本 Embedding 维度一致。

2. **Soft Prompts (软提示)：**

   - 这些经过转换的视觉 Queries，被当作**软提示 (Soft Prompts)** 直接输入到 LLM 中。
   - 这就好比我们把图片“翻译”成了几个特殊的单词向量，告诉 LLM：“这就是那张图片的内容”。

3. **生成过程（如图中示例）：**

   - **输入：** [Q-Former 输出的视觉特征] + [用户文本提示 "Write a romantic message..."]。
   - **处理：** 冻结的 LLM 接收到这些信息，结合它本身强大的语言知识。
   - **输出：** 生成高质量的回复："Love is like a sunset..."。

**阶段二总结：** 这个阶段主要训练的是 Q-Former 到 LLM 的线性映射层（以及微调 Q-Former），让视觉特征真正变成 LLM 可读的输入。

------

##### 为什么要分两个阶段？(原理总结)

1. **解决模态对齐问题（阶段一）：** 直接把图像特征扔给 LLM 效果通常不好，因为图像特征空间和语言特征空间差异巨大。第一阶段迫使 Q-Former 提取出“像语言一样思考”的视觉特征。
2. **降低计算成本：** 两个阶段都冻结了最大的两个模型（Image Encoder 和 LLM），只训练中间轻量级的 Q-Former，这使得在消费级显卡上训练超大模型成为可能。
3. **避免遗忘：** 因为 LLM 是冻结的，它保留了原本预训练通过海量文本学到的所有知识（常识、推理能力等），不会因为学习看图而变笨。

图中的 **"Sunset"** 例子非常直观：

- **Image Encoder** 看到了像素。
- **Q-Former (阶段一能力)** 提取了“日落”、“海面”、“美丽”等视觉语义。
- **LLM (阶段二能力)** 接收这些语义，并结合用户的“写一句浪漫的话”的指令，利用其内部文学知识，写出了那句关于爱的诗句。

#### 3. Large Language Model (大型语言模型)

**图中文字**：和 Flamingo 模型的 LLM 作用相同，用于生成文本，没有对 LLM 的结构进行修改。

**论文原理解读**：

- **冻结参数:** 与图像编码器一样，LLM 在整个预训练过程中也是**完全冻结**的 。这与 Flamingo 不同，Flamingo 会在 LLM 中插入新的交叉注意力层并进行训练 13，**而 BLIP-2 不改变 LLM 的架构**。
- **模型类型:** 论文实验了两种类型的 LLM：基于 Decoder 的模型 (如 OPT) 和 基于 Encoder-Decoder 的模型 (如 FlanT5) 。
- **生成过程:** Q-Former 提取的视觉特征**经过一个全连接层 (Fully Connected Layer) 投影后**，**直接作为 LLM 的输入前缀 (Soft visual prompts)** 。LLM 根据这些视觉信息生成相应的文本描述（例如图中右侧生成的 "Love is like a sunset..."）。





### 2.**Q-Former (Querying Transformer)** 的架构和工作原理

 **BLIP-2** 论文中最核心的组件——**Q-Former (Querying Transformer)** 的架构和工作原理。

BLIP-2 的核心思想是利用**冻结（Frozen）的图像编码器和冻结**的大语言模型（LLM），通过一个轻量级的中间件来桥接两者。这个中间件就是 **Q-Former**。

结合论文原文，我为你详细拆解图中提到的关键点：

![image-20260119121040563](/Users/moem/Library/Application Support/typora-user-images/image-20260119121040563.png)

#### 1. Q-Former 的核心作用

 Q-Former 的主要目标：

> **“从 Image Encoder 中提取固定数量的输出特征，与输入图像分辨率无关。”**

- **瓶颈（Bottleneck）设计：** **无论输入图片的分辨率多大，或者 Image Encoder 输出的特征图有多大**，**Q-Former 都会将其压缩成一组固定数量的向量**（论文中通常是 **32 个 Query Embeddings**）。
- **为什么要这样做？** 这样可以大大减轻后续 LLM 的计算负担，并且强迫模型提取最精华的视觉信息。

#### 2. 内部架构：两个 Transformer

图片中提到 Q-Former 由两个共享 **Self-Attention** 的 Transformer 子模块组成：

1. **左侧 Image Transformer：** 负责与冻结的 Image Encoder 交互，提取视觉特征。
2. **右侧 Text Transformer：** 既可以作为文本编码器（Encoder），也可以作为文本解码器（Decoder）。

**关键点：** 这两部分是**权重共享（Share Parameters）**的，特别是 Self-Attention 层，这意味着视觉查询向量（Queries）和文本向量（Text）在同一个特征空间内交互。

#### 3. Query Embedding（可学习的查询向量）

这是 Q-Former 最精妙的设计，对应图中中间下方的 `Learned Queries`：

- **输入：** 这一组 $K$ 个（例如 32 个）可学习的 Query Embeddings 是 Image Transformer 的输入。

- **交互机制（图中中间部分）：**

  - **Self Attention：** 这些 Queries 之间相互作用，同时（根据任务不同）与 Text Embeddings 相互作用。

  > 下一部分会具体讲解

  - **Cross Attention：** 专门用于引入视觉信息。这些 Queries 通过 Cross Attention 层，与冻结的 Image Encoder 输出的特征进行交互。
  - **注意细节：** 图片文字特别注明 `每隔一个 transformer block 有一个 Cross attention`。这是一种节省计算量的设计，与原始 Transformer 每一层都有 Cross Attention 不同。

#### 4. 预训练权重与初始化

- **初始化：** Q-Former 使用 **BERT-base** 的预训练权重进行初始化。这利用了 **BERT** 强大的语言理解能力。
- **随机初始化：** 因为 BERT 原本没有处理图像的能力，所以其中的 **Cross Attention 层是随机初始化**的，**需要从头学习如何“看”图片。**
- **参数量：** 整个 Q-Former 包含 **188M** 个参数，相比于巨大的 LLM 和 ViT，它是非常轻量级的。

------

#### 5. 三种预训练任务流程原理

为了训练这个 Q-Former，论文设计了三种特定的“注意力掩码（Attention Masking）”策略（对应图中右侧的三个方框），让同一组参数同时学习三种任务：

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260119124226294.png" alt="image-20260119124226294" style="zoom:50%;" />

##### 1. Q-Former 的核心设计理念

Q-Former 的目的是作为一个“桥梁”，连接冻结的图像编码器（Frozen Image Encoder）和冻结的大语言模型（Frozen LLM）。它通过一组可学习的 **Query Embeddings** 从图像编码器中提取与文本最相关的视觉特征 。

图片中的三种模式对应了 **Q-Former 在第一阶段预训练时的三个训练目标（Objectives）**，它们**共享同样的模型参数**，但通过**不同的 Mask 策略**来实现不同的功能。

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260119131156573.png" alt="image-20260119131156573" style="zoom:50%;" />

##### 2. 三种 Mask 策略详解

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260119125539254.png" alt="image-20260119125539254" style="zoom:50%;" />

###### (1) Image-Text Contrastive Learning (ITC)

- **对应图片标签：** `Uni-modal Self-Attention Mask`

- **掩码策略：** **Query 和 Text 彼此不可见**（Mask 掉彼此）。

  - Query 只能看到其他的 Query；Text 只能看到其他的 Text 。

- **功能解释：**  此时 Text Transformer 充当 **单模态编码器（Unimodal Encoder）**。

  - 目标是对齐图像表示（Query 输出的 Z）和文本表示（Text 的 [CLS] token）。

    ![image-20260119133156791](/Users/moem/Library/Application Support/typora-user-images/image-20260119133156791.png)

  - 这迫使 Query 提取出的视觉特征在语义上与对应的文本特征尽可能接近，用于区分正负样本对 。

  > 基于 BLIP-2 论文原文，以下是关于 **Image-Text Contrastive Learning (ITC)** 如何实现图像表示 $Z$ 与文本表示 $t$ 对齐的详细解释：
  >
  > ### 核心机制：多对一的相似度计算 (Many-to-One Alignment)
  >
  > 在 ITC 任务中，Q-Former 的目标是最大化对应的图像和文本表示之间的互信息 1。由于 Q-Former 的架构设计，图像端和文本端的输出形式并不对称，因此对齐过程包含一个关键的“**最大值筛选**”步骤。
  >
  > ###### 1. 定义表示 (Representations)
  >
  > - 图像表示 ($Z$)：
  >
  >   Q-Former 输出的查询表示（Output Query Representation）。注意，$Z$ 并不是单个向量，而是包含多个输出嵌入（Output Embeddings），对应每一个 Learned Query。在 BLIP-2 的实验设置中，共有 32 个 Query，因此 $Z$ 的大小为 $32 \times 768$ 。
  >
  > - 文本表示 ($t$)：
  >
  >   文本 Transformer 输出的 [CLS] token 的嵌入向量（embedding），它代表了整个输入文本的语义特征 3。
  >
  > ###### 2. 对齐计算步骤 (Alignment Step)
  >
  > 由于图像表示 $Z$ 包含 32 个向量，而文本表示 $t$ 只有 1 个向量，模型无法直接进行简单的一对一点积。论文中明确描述了对齐的计算逻辑：
  >
  > 1. **计算成对相似度**：首先计算文本表示 $t$ 与 $Z$ 中 **每一个** Query 输出向量之间的成对相似度（Pairwise Similarity） 。
  >
  > 2. **选择最大值**：从这 32 个相似度数值中，**选择最高的一个**作为该图像-文本对最终的相似度分数 。
  >
  >    $$S(I, T) = \max_{i=1}^{32} (Sim(z_i, t))$$
  >
  > ###### 3. 目的与意义
  >
  > 这种策略允许 Q-Former 的 32 个 Query 分别关注图像的不同方面（例如不同的物体或区域）。只要其中 **某一个** Query 提取到了与当前文本描述高度匹配的视觉特征，模型就会判定该图像与文本是匹配的。这实际上是在训练 Query 去“抢占”最能解释文本的视觉信息。
  >
  > ###### 训练目标 (Training Objective)
  >
  > 在得到相似度分数后，ITC 采用了标准的对比学习目标：
  >
  > - **正样本对**：最大化对应的图像-文本对的相似度。
  > - **负样本对**：最小化不匹配对的相似度。
  > - **In-batch Negatives**：由于使用了冻结的图像编码器，显存占用较小，BLIP-2 可以使用很大的 Batch Size，因此它直接利用 Batch 内的其他样本作为负样本（In-batch negatives），而不需要像 BLIP-1 那样维护额外的动量队列（Momentum Queue） 。
  >
  > ###### 掩码策略的作用 (Role of the Mask)
  >
  > 正如你所提到的，这里使用了 **Uni-modal Self-Attention Mask** ：
  >
  > - **Query 和 Text 彼此不可见**：这是为了防止信息泄漏（Information Leak）。如果允许它们在 Attention 层交互，模型可能会直接通过“看答案”来完成匹配任务，而不是学习将视觉特征映射到与文本特征相同的语义空间中。必须强迫 $Z$ 和 $t$ 独立提取特征，通过最后的对比损失函数来拉近它们的距离。

###### (2) Image-Grounded Text Generation (ITG)

- **对应图片标签：** `Multi-modal Causal Self-Attention Mask`

  ![image-20260119134201399](/Users/moem/Library/Application Support/typora-user-images/image-20260119134201399.png)

- **掩码策略：** **Query 能看到所有 Query，但看不到 Text**（防止作弊，因为这是生成任务）。

  - **Text 能看到所有的 Query**（获取视觉上下文）**以及当前 Token 之前的 Text**（因果推断），但看不到未来的 Text 。

- **功能解释：**

  - 此时 Text Transformer 充当 **解码器（Decoder）**。
  - **模型被要求给定视觉特征（Query）作为条件来生成文本 。**
  - 这强迫 Query 提取出的特征必须包含足够丰富的信息，以便能够重建出文本内容 。

###### (3) Image-Text Matching (ITM)

- **对应图片标签：** `Bi-directional Self-Attention Mask`

  ![image-20260119135606369](/Users/moem/Library/Application Support/typora-user-images/image-20260119135606369.png)

- **掩码策略：**  **全员可见：** Query 和 Text 中的每个 Token 都能看到彼此所有的 Token 。

  > 基于 BLIP-2 原论文，以下是对 **Image-Text Matching (ITM)** 任务及其数据流动的详细解析。这个阶段的核心在于利用“全员可见”的注意力机制，让图像和文本信息深度融合，从而实现细粒度的对齐判断。
  >
  > ###### 1. 核心目标：细粒度对齐 (Fine-grained Alignment)
  >
  > ITM 是一个二分类任务（Binary Classification）1。它的目标是预测给定的图像-文本对是“匹配（Positive）”还是“不匹配（Negative）”。
  >
  > 与 ITC（对比学习）不同，ITC 只是拉近了整体特征的距离，而 ITM 允许模型深入比较图像细节和文本词汇之间的逻辑关系，因此被称为“细粒度对齐” 。
  >
  > ###### 2. 掩码策略：双向自注意力 (Bi-directional Self-Attention Mask)
  >
  > 正如你所指出的，这里使用的是 **Bi-directional Self-Attention Mask**。
  >
  > - **机制**：所有的 Query Token 和 Text Token 都可以相互“看见”（attend to each other）。
  > - **意义**：这种掩码策略打破了模态间的隔阂。Q-Former 在这里不再是两个独立的编码器，而是变成了一个多模态融合编码器。Query 可以读取文本信息，Text 也可以读取由 Query 提取的视觉信息。
  >
  > ###### 3. 详细数据流动 (Data Flow)
  >
  > 根据论文描述，数据在 ITM 任务中的流动过程如下：
  >
  > ###### Step 1: 视觉特征提取 (Visual Feature Extraction)
  >
  > 图像经过冻结的 Image Encoder，生成视觉特征。Q-Former 中的 **Learned Queries** 通过 **Cross-Attention** 层与这些视觉特征进行交互 。这是 Query 获取图像信息的唯一途径。
  >
  > ###### Step 2: 多模态融合 (Multimodal Fusion via Self-Attention)
  >
  > 在 Q-Former 的 Self-Attention 层中，应用了上述的“全员可见”掩码。
  >
  > - **Queries** 既能看到彼此，也能看到所有的 **Text Tokens**。
  > - **Text Tokens** 既能看到彼此，也能看到所有的 **Queries**。
  > - 结果：输出的 Query 表征 $Z$（Output Query Embeddings）因此捕获了 **多模态信息 (Multimodal Information)**，即它既包含图像特征，也融合了文本语境 。
  >
  > ###### Step 3: 分类与打分 (Classification & Scoring)
  >
  > 这是 ITM 与通常做法不太一样的地方。BLIP-2 并没有简单地把所有 Query 聚合为一个向量，而是对 **每一个** Query 输出向量分别进行判断：
  >
  > 1. **独立分类**：将输出的每一个 Query Embedding $Z$（共有 32 个）分别输入到一个**二分类线性分类器 (Two-class Linear Classifier)** 中。
  > 2. **获取 Logits**：针对每个 Query，分类器都会输出一个 Logit（预测分数）。
  > 3. **平均聚合**：将所有 Query 输出的 Logits 取 **平均值 (Average)**，作为最终的图像-文本匹配分数 。
  >
  > ###### Step 4: 损失计算 (Loss Calculation)
  >
  > 模型根据最终的平均分数计算二分类交叉熵损失。为了提高训练效率和难度，BLIP-2 同样使用了 **Hard Negative Mining**（困难负样本挖掘）策略，即特意挑选那些虽然不匹配但在特征空间上很相似的负样本进行训练 。
  >
  > ###### 总结图示
  >
  > 可以这样想象数据流：
  >
  > 1. **输入**：Image $\rightarrow$ [Image Encoder] & Text $\rightarrow$ [Q-Former Input]
  > 2. **交互**：[Queries] $\leftrightarrow$ [Text Tokens] （在 Q-Former 内部疯狂交换信息）
  > 3. **输出**：32 个融合了图文信息的 Query 向量 $Z$。
  > 4. **判决**：32 个向量 $\rightarrow$ [分类器] $\rightarrow$ 32 个分数 $\rightarrow$ **求平均** $\rightarrow$ 最终结果 (匹配/不匹配)。
  >
  > 这种设计迫使 Query 必须充分吸收文本信息，并将其与视觉信息比对，从而使得 $Z$ 成为非常强的多模态对齐特征。

------

##### 3. 信息瓶颈（Information Bottleneck）的设计

图片右下角的蓝色文字解释了 Q-Former 作为一个“信息瓶颈”的关键作用，这与论文中的描述完全一致：

- **Query 的设置：** 使用了 **32 个可学习的 Query**，每个维数是 **768** 。
- **维度压缩：**
  - 冻结的图像编码器（如 ViT-L/14）输出的特征维度非常大（例如 $257 \times 1024$）。
  - Q-Former 将这些繁杂的视觉特征压缩为固定的 **$32 \times 768$** 的输出 $Z$ 。
- **核心作用：**
  - 正如蓝框中所述，这种架构迫使 Query 必须**提取与文本最相关的视觉信息** 。
  - 它过滤掉了与文本无关的视觉冗余，从而减轻了后续 LLM 学习视觉-语言对齐的负担，这也是 BLIP-2 能够高效利用冻结 LLM 的关键原因 。

这张图解释了 BLIP-2 如何用一个轻量级的 **Q-Former** 作为“转换器”。它就像一个精通双语的翻译官：

1. **听（Cross Attention）：** 它通过 Learned Queries 从 Image Encoder 那里“听”取视觉信息。
2. **思考（Self Attention）：** 它结合文本信息进行内部处理。
3. **说（Output）：** 它输出高度压缩的视觉特征，这些特征是可以被 LLM 理解的。

### 3.模型预训练和微调

BLIP-2 的核心思想是：利用现有的、强大的**冻结（Frozen）\**预训练图像编码器和\**冻结**的大语言模型（LLM），通过训练一个轻量级的中间件 —— **Q-Former**，来实现视觉与语言的对齐。

------

#### 1. 基础组件：Vision Encoder 预训练 

这一部分描述了 BLIP-2 视觉感知的“眼睛”。

- **冻结参数 (Frozen Image Encoder):** 论文强调在训练过程中，图像编码器的参数是完全冻结的。这意味着 BLIP-2 不会改变视觉模型已有的能力，而是“借用”它。
- **模型选择:**
  - **CLIP ViT-L/14:** OpenAI 的 CLIP 模型，用于较小版本的 BLIP-2。
  - **EVA-CLIP ViT-g/14:** 这是一个更大、更强的视觉模型 (ViT-giant)，用于构建 SOTA (State-of-the-Art) 级别的 BLIP-2 模型。
- **关键技巧 (论文细节):** 图片中提到“**删除了其最后的一个 transformer block**”。
  - **原文解释:** 论文作者发现，对于 CLIP 这种对比学习训练出来的模型，最后一层通常聚集了过于抽象或特定于原任务的特征。**移除最后一层，直接使用倒数第二层的输出特征，**能让视觉特征更通用，从而在后续的多模态任务中**稍微提升性能**。

------

#### 2. 第一阶段：视觉-语言表征学习 (中间部分)

这是 Q-Former 的“预热”阶段。此时**没有 LLM 参与**，只有**图像编码器和 Q-Former**。

- **目标:** 让 Q-Former 学会从图像编码器中提取与文本最相关的视觉特征。由于图像编码器是冻结的，Q-Former 必须学会充当一个“过滤器”或“瓶颈 (Bottleneck)”。
- **架构:**
  - Q-Former 被连接到冻结的 Image Encoder 上。
  - Q-Former 内部有一组可学习的 **Query Vectors (查询向量)**。
- **训练任务 (结合原文):** 虽然图片只写了“使用图像-文本对进行预训练”，但原文指出了三个具体的训练目标（这也是 Q-Former 强大的原因）：
  1. **ITC (Image-Text Contrastive Learning):** 图像文本对比学习，对齐视觉和文本特征空间。
  2. **ITG (Image-Text Generation):** 给定图像生成文本，训练 Q-Former 的文本生成能力。
  3. **ITM (Image-Text Matching):** 细粒度的图像文本匹配，判断图文是否对应。
- **训练细节:**
  - **分辨率:** $224 \times 224$ (标准 ImageNet 分辨率)。
  - **高效性:** 图片强调了“单台 16 x A100-40G 机器，最大的 ViT-g 需要不到 6 天”。这在多模态大模型领域是非常高效的，证明了冻结视觉主干带来的计算成本优势。

------

#### 3. 第二阶段：视觉到语言的生成预训练 (最右侧)

这是将视觉能力“注入”到大语言模型 (LLM) 的阶段。

- **目标:** 利用 LLM 强大的语言生成能力来理解图像。Q-Former 输出的特征（Queries）在这里充当了 LLM 的 **Soft Prompts (软提示)**。
- **架构:**
  - **连接方式:** Q-Former (带冻结 Image Encoder) -> **全连接层 (FC)** -> **冻结的 LLM**。
  - **冻结的 LLM:** 这里的 LLM 参数也是不更新的，这避免了“灾难性遗忘 (Catastrophic Forgetting)”，即避免 LLM 忘记原本的语言知识。
- **LLM 的类型 (原文对比):**
  - **Decoder-Only (OPT 系列):** 类似于 GPT 的架构。在这种情况下，Q-Former 的输出作为输入的“前缀 (Prefix)”，LLM 负责接续生成文本。这里使用的是无监督预训练的模型。
  - **Encoder-Decoder (FlanT5 系列):** 类似于 T5 的架构。Q-Former 的输出作为 Encoder 的输入，Decoder 负责生成文本。这里使用的是经过指令微调 (Instruction Tuned) 的模型，指令跟随能力更强。
- **训练细节:**
  - **步数:** 80K step (比第一阶段短，因为只需要训练 Q-Former 适应 LLM 的嵌入空间)。
  - **精度:** 为了节省显存和加速，使用了半精度 (FP16/BF16)。
  - **极高的效率:** 图片特别指出，训练最大的 FlanT5-XXL (11B 参数) 版本，在 16 张 A100 上**不到 3 天**就完成了。这是 BLIP-2 最令人印象深刻的成果之一——让训练多模态大模型的门槛大幅降低。

BLIP-2 的成功秘诀：**通过两阶段训练，利用轻量级的 Q-Former 作为桥梁，低成本地连接了现成的“最强之眼”（EVA-CLIP）和“最强之脑”（LLM，如 FlanT5），从而实现了强大的图文理解与生成能力。**

------





## **第五部分：LLaVA-v1**

LLaVA (Large Language and Vision Assistant) 证明了简单的结构配合高质量的指令数据也能取得极好的效果。

### 1. **LLaVA (Large Language-and-Vision Assistant)** 模型的核心架构

LLaVA 的核心设计理念是**“简单即是美”**。不同于之前复杂的模型（如 DeepMind 的 Flamingo 或 Salesforce 的 BLIP-2），LLaVA 证明了一个简单的线性层就足以让大语言模型“看懂”图片。

![image-20260119215405947](/Users/moem/Library/Application Support/typora-user-images/image-20260119215405947.png)

------

#### 1. Vision Encoder (视觉编码器)

- **图片内容**：这也采用了 **CLIP ViT-L/14**，作用与 Flamingo 的 Vision Encoder 一样，用于提取视觉特征。
- **论文深度解读**：
  - LLaVA 使用了预训练好的 **CLIP (Contrastive Language-Image Pre-training)** 模型的视觉部分（ViT-L/14）。
  - **关键点**：它不仅仅提取整张图片的全局特征（CLS token），而是提取图片经过 Patch 后的**网格特征（Grid Features）**。
  - 在 LLaVA 的第一阶段训练中，这个 Vision Encoder 的参数通常是**冻结（Frozen）**的，即不参与更新，以保留预训练好的视觉识别能力。

#### 2. Projection W (投影层 / 连接层) [核心创新]

- **图片内容**：这是 LLaVA 与 Flamingo（使用 Perceiver Resampler）和 BLIP-2（使用 Q-Former）最大的不同。LLaVA 这里非常简单，只用了一层 **Linear（线性层）**，将图像特征 ($Z_v$) 映射到 LLM 的词嵌入（Word Embedding）空间。
- **论文深度解读**：
  - **数学表达**：$H_v = W \cdot Z_v + b$。
  - **作用**：视觉编码器出来的特征向量与 LLM 能理解的文本向量在维度和语义空间上是不对齐的。这个投影层的作用就是充当“翻译官”，把视觉特征转换成 LLM 能够理解的“视觉 Token”。
  - **对比优势**：BLIP-2 的 Q-Former 结构非常复杂，包含多个 Transformer 层；而 LLaVA 证明了**只需要一个简单的矩阵乘法（线性层）**，就能取得极好的效果。这大大降低了模型的训练难度和推理开销。
  - *注：图片中提到是线性层（Linear），这是 LLaVA v1 的设计。在后续的 LLaVA v1.5 中，这一层被升级为了两层 MLP 以增强表达能力，但基本逻辑不变。*

#### 3. Large Language Model (大语言模型)

- **图片内容**：作用是生成文本。作者没有修改 LLM 的内部结构，直接使用了 **Vicuna-v1.5 13B** 模型。
- **论文深度解读**：
  - LLaVA 选用了 **Vicuna**（小羊驼），这是一个基于 LLaMA 进行指令微调（Instruction Tuning）后的模型，以对话能力强著称。
  - **处理流程**：LLM 接收到的输入是一个**拼接序列**。它将**“视觉 Token” ($H_v$) 和“语言指令 Token” ($H_q$) 串联**在一起，就像看一段“图片描述 + 用户问题”的文本一样，然后基于概率预测生成回答 ($X_a$)。

------

#### 4.图解流程数据流向，

这是多模态模型工作的标准范式：

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260119215405947.png" alt="image-20260119215405947" style="zoom:50%;" />

1. **输入图像 ($X_v$)**：通过 `Vision Encoder` 变成视觉特征 $Z_v$。
2. **特征对齐**：$Z_v$ 经过 `Projection W` 变换为 $H_v$。此时，$H_v$ 在数学上已经等同于文本的 Embedding 向量。
3. **输入指令 ($X_q$)**：用户的文本指令（例如：“描述这张图片”）被转化为文本 Embedding $H_q$。
4. **拼接与生成**：
   - $H_v$（视觉部分）和 $H_q$（语言部分）按顺序输入到 `Language Model` ($f_\phi$) 中。
   - 模型最终输出语言回答 **Language Response $X_a$**。



### **2.LLaVA 的**两个训练阶段

#### 第一阶段：特征对齐预训练 (Feature Alignment Pre-training)

这一阶段的目标非常单纯：让大语言模型（LLM）能“看懂”视觉编码器（Vision Encoder）传过来的信号。

- **数据来源与处理（论文中的 CC3M-595K 数据集）：**
  - **原始数据**：使用了 CC3M（Conceptual Captions 3M）数据集。
  - **过滤策略（图片中的详细文本）**：论文作者认为原始数据太杂，为了提高训练效率和覆盖面，他们用 Spacy 工具提取图片描述中的“名词短语”（Noun Phrases）。
    - **剔除太偏门的**：频率小于 3 的名词短语被丢弃（太罕见，没代表性）。
    - **平衡数据分布**：为了防止某些常见概念（如“人”、“猫”）出现太多次导致模型过拟合，作者从频率**最低**的概念开始选。如果某个概念出现频率超过 100 次，就只随机选 100 张。
  - **最终产出**：通过这种精心筛选，最终得到了 **595K（59.5万）** 个高质量的“图像-文本对”。
- **训练设置（核心）：**
  - **冻结部分**：Vision Encoder（视觉眼睛）和 LLM（大脑）**全部冻结**，参数不更新。
  - **训练部分**：**只训练 Projection Layer（投影层/连接层）**。
  - **目的**：就像给 LLM 配一副眼镜，调节这副眼镜的焦距（Projection Matrix），让图片特征能准确地映射到 LLM 熟悉的词向量空间中。

------

#### 第二阶段：端到端微调 (End-to-End Fine-tuning)

这一阶段的目标是让模型学会像 ChatGPT 一样与人进行多模态对话，不仅仅是“认出物体”，还能“描述细节”和“逻辑推理”。

- **数据来源（论文中的 LLaVA-Instruct-158K 数据集）：**
  - 这一阶段使用的数据不是简单的图片描述，而是**指令跟随（Instruction Following）**数据。
  - **数据构成（图片中列出的 158K 样本）：**
    1. **58K 对话 (Conversation)**：日常的多轮对话，比如“这张图里有什么好玩的？”。
    2. **23K 详细描述 (Detailed Description)**：要求模型极其详尽地描述图片细节。
    3. **77K 复杂推理 (Complex Reasoning)**：不仅要看图，还要推理。比如“图里的人为什么在笑？”或者“根据这张图表，下一年的趋势是什么？”。
  - *注：图片还提到了 Science QA，这是 LLaVA 用于测试科学问答能力的另一个微调场景。*
- **训练设置（核心的变化）：**
  - **冻结部分**：Vision Encoder 依然**冻结**（因为视觉特征提取能力已经够强了，不需要动）。
  - **训练部分**：**Projection Layer 和 LLM 都会更新**。
  - **目的**：此时不仅仅是调眼镜（Projection）了，还在训练大脑（LLM）。通过微调 LLM，让它学会根据用户的指令，结合看到的图片信息，输出符合人类偏好的回答。

------





## 第六部分：MiniGPT-4

MiniGPT-4 的目标是让视觉理解能力与先进的大语言模型（如Vicuna）对齐。

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260119224027146.png" alt="image-20260119224027146" style="zoom:50%;" />

### **1. **MiniGPT 的核心模型架构

#### 1. 视觉编码器 (Vision Encoder) - 图片底部

- **对应图片文字**：1. Vision Encoder: 直接使用了 BLIP-2 的方案，作者用的是 EVA-CLIP ViT-G/14。
- **论文原理解析**：
  - MiniGPT-4 为了获取视觉感知能力，直接采用了 **BLIP-2** 模型中相同的预训练视觉组件 。
  - 具体来说，它包含一个 **ViT (Vision Transformer)** 主干网络（使用的是 EVA-CLIP 的 ViT-G/14 版本）和一个 **Q-Former** 网络 。
  - **关键点**：在 MiniGPT-4 的训练过程中，这两个视觉组件是**冻结的 (Frozen)**，即它们的参数保持不变，不参与更新 。这意味着模型直接利用了 BLIP-2 已经学到的高质量视觉特征。

#### 2. 线性投影层 (Linear Projection Layer) - 图片中间

- **对应图片文字**：2. Projection: BLIP-2 的 Q-Former 也完整保留，同样后面增加了一层可训练的 Linear 层。
- **论文原理解析**：
  - 这是 MiniGPT-4 架构中最重要的连接桥梁。由于视觉编码器的输出特征无法直接被大语言模型（LLM）理解，作者引入了一个**单一的线性投影层 (Linear Layer)** 。
  - **功能**：这个层的作用是将编码后的视觉特征对齐到 Vicuna 语言模型的空间中 。
  - **训练策略**：在第一阶段的预训练中，**只有这一个线性层的参数是可训练的**，其他的视觉和语言模型参数都被冻结 。这使得训练非常高效（大约只需10小时）。

#### 3. 大语言模型 (Large Language Model) - 图片顶部

- **对应图片文字**：3. Large Language Model: 使用 Vicuna-v0 模型作为 LLM。
- **论文原理解析**：
  - MiniGPT-4 能够生成长篇、详细且逻辑清晰的文本（如图片上方显示的 Logo 设计理念描述），主要归功于它使用了一个非常先进的大语言模型作为解码器 。
  - **模型选择**：作者选择了 **Vicuna** 。Vicuna 是基于 LLaMA 构建的，据报道其对话质量能达到 ChatGPT 的 90% 。
  - **状态**：与视觉部分一样，这个 LLM 在训练过程中也是**冻结的 (Frozen)** 。它接收来自线性层的**视觉特征（作为一种 Soft Prompt）和用户的文本指令**，然后生成回答 。

#### 4.数据流向

<img src="/Users/moem/Library/Application Support/typora-user-images/image-20260119224027146.png" alt="image-20260119224027146" style="zoom:50%;" />

结合图片中的箭头和示例，整个流程是这样的：

1. **输入**：一张火烈鸟的图片。
2. **处理**：图片经过 Q-Former 和 ViT 提取特征，通过 Linear Layer 转换。
3. **交互**：**这些视觉特征与人类的提问（Prompt: "What do you think of this logo design?"）拼接在一起** 。
4. **输出**：Vicuna 接收到这些信息后，发挥其强大的语言能力，生成了一段关于 Logo 设计风格、象征意义及适用场景的详细评价 。

### 2.MINIGPT的预训练和微调

#### 1. 左侧：特征对齐预训练 (First Pretraining Stage)

这一阶段对应论文的 **Section 3.1**。它的核心目的是让大语言模型（LLM）学会“看图”，即理解视觉特征。

- **冻结策略 (Frozen Strategy)**：
  - 图片提到“Vision Encoder + Q-Former 和 LLM 都保持冻结状态，只训练线性投影层”。
  - 论文证实了这一点：为了保持效率，作者冻结了预训练的视觉编码器和 Vicuna LLM，**只预训练那一层线性投影层 (Linear Projection Layer)** 。
- **训练数据 (Datasets)**：
  - 图片列出了 Conceptual Caption, SBU 和 LAION 数据集。
  - 论文中提到使用了这三个数据集的组合，目的是让模型学习广泛的视觉-语言知识 。
- **训练规模与硬件 (Specs)**：
  - 图片数据：Batch-size 256, 20,000 steps, 覆盖 5M 图像-文本对, 4x A100 耗时 10 小时。
  - 论文原文完全一致：模型在大约 **500万** 个图像-文本对上训练了 **20,000** 步，Batch size 为 **256**，使用了 **4张 A100 (80GB)** 显卡，耗时约 **10小时** 。

#### 2. 右侧：端到端微调 (Second-Stage Finetuning)

这一阶段对应论文的 **Section 3.2 和 3.3**。

**背景**：论文指出，第一阶段训练后的模型虽然能看懂图，但生成的语言经常不连贯、有重复或碎片化，无法像人类一样自然交流 。因此需要第二个阶段来“教”模型如何自然地说话。

- **数据构建 (Data Curation)**：
  - 图片提到“从 Conceptual Caption 中随机选择 5,000 个图像...最终挑选出 3,500 左右”。
  - 论文详细描述了这个过程：作者随机选取了 **5,000** 张图片 ，先让第一阶段的模型生成非常详细的描述 ，然后利用 ChatGPT 修正语法错误和不自然的表达 ，最后经过人工验证，筛选出约 **3,500** 个高质量的图像-文本对 。
- **训练效率 (Efficiency)**：
  - 图片提到“只训练 400 个 step, batch size 为 12, 单个 A100 上 7 分钟”。
  - 这是 MiniGPT-4 最惊人的地方之一。论文证实，基于这 3,500 条高质量数据，微调过程极其高效，只需 **400步**，Batch size 为 **12**，单张 A100 显卡仅需 **7分钟** 即可完成 。

------

## **专家总结与建议**

- **演进逻辑**：从ViT解决视觉Token化，到CLIP解决图文特征对齐，再到Flamingo/BLIP-2探索如何“无损”利用预训练大模型，最后LLaVA和MiniGPT证明了“Adapter/Projection + 高质量指令数据”是目前最高效的范式。
- **学习重点**：
  1. **ViT** 的 Patch Embedding 维度计算。
  2. **CLIP** 的对比学习矩阵理解。
  3. **BLIP-2** Q-Former 的三种Mask机制。
  4. **LLaVA** 的数据构造方式（利用纯文本LLM生成多模态指令数据）。