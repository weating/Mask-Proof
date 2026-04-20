# MaskProof-Bench

`MaskProof-Bench` 是一个面向研究级数学证明的 masked-step benchmark。它对应论文 **Mask-Proof: An LLM-based Automated Data Curation and Evaluation Pipeline on Research-Level Mathematical Proofs**，目标是把真实论文中的证明步骤转成可自动评测的 `[MASK]` 恢复任务，用来测量模型在长链条数学证明里的中间推理能力。

## 仓库结构

```text
.
├── MaskProof-Bench.jsonl
├── Mask_Proof__An_LLM_based_Automated_Data_Curation_and_Evaluation_Pipeline_on_Research_Level_Mathematical_Proofs.pdf
├── code/
│   ├── infer-extract-judge.py
│   └── run_infer_models.sh
├── prompt/
│   ├── judge.md
│   ├── mask.md
│   └── selfcontained.md
└── codex skills/
    ├── apply-proof-mask/
    └── extract-proof-formulas/
```

## 数据格式

每条样本包含 6 个核心字段：

- `index`
- `problem`
- `reference_solution`
- `mask_text`
- `mask_content`
- `arxiv_id`

其中：

- `problem` 提供题目与必要的自包含补充；
- `reference_solution` 是原始完整证明；
- `mask_text` 是插入 `[MASK]` 后给模型作答的证明；
- `mask_content` 是被遮住的标准答案。

## 方法概览

`Mask-Proof` 的核心流程可以概括为三步：

1. **Proof extraction**
   - 从 arXiv 论文 LaTeX 中抽取 theorem-proof 对。

2. **Automated curation**
   - 先做 self-contained recovery，补回证明依赖的 lemma、definition、符号说明与宏展开；
   - 再做 agentic masking，挑出必须依赖全局证明思路才能恢复的关键公式步骤；
   - 把该步骤替换成 `[MASK]`，并保存 `mask_content`。

3. **Evaluation**
   - 模型对 `[MASK]` 作答；
   - judge 判断模型答案与标准答案是否在数学上等价，而不是做字符串匹配；
   - 论文用 `Avg@4` 统计多次采样后的表现。

## 三张图

### Figure 1: 任务形式

![Figure 1](<fig/截屏2026-04-20 20.46.35.png>)

这张图展示了三种形态：

- 原始 `IMO-ProofBench` 题目与解答；
- 转成 masked proof 后的竞赛证明样本；
- 扩展到研究级证明后的 masked-step 任务形式。

### Figure 2: Mask-Proof 流水线

![Figure 2](<fig/截屏2026-04-20 20.46.47.png>)

这张图对应完整数据构造流程：

- 从论文源文件中提取证明；
- 把问题补成 self-contained；
- 选择关键步骤并插入 `[MASK]`；
- 形成最终 benchmark，并交给 judge 自动评测。

### Figure 3: 主实验排行榜

![Figure 3](<fig/截屏2026-04-20 20.47.00.png>)

这张图展示论文主实验的 `Avg@4` 排名。论文的核心观察是：reasoning-enhanced 模型和 standard 模型之间存在明显差距，而 `Mask-ProofBench` 能把这种差距比较稳定地拉开。

