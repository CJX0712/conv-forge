# ✦ Conv Forge · 卷积神经网络锻造炉

单文件 · 零依赖 · 全自验证。手写 2D 卷积 / 最大池化 / ReLU / Softmax+交叉熵 / 解析反向传播，
数值梯度检验对拍到机器精度，端到端训练真实收敛。属于 **forge** 单文件实验室家族，与 `ai-forge` 总枢纽对齐。

**作者：晨星 (CJX0712)**

---

## 为什么是它（零重写 · 集大成）

晨星已有的 forge 家族已覆盖经典 ML 全谱系、Transformer / GPT、MoE、Diffusion、AlphaZero、PPO、
状态空间与最优传输等前沿（见 `ai-forge` / `apex-forge` / `forge-ai` / `forge-1024`）。
唯一明显缺口是 **卷积神经网络（CNN / ConvNet）从零手写** —— 视觉与现代架构的基石，故锻造本炉。

每一行都站在世界顶级成果之上：
- 高效卷积走 **im2col + GEMM**（Chetlur et al. 2014，cuDNN/各框架同款），与朴素七重循环**交叉验证**。
- 反向传播用 **解析梯度 ≡ 数值中心差分** 对拍（`∂f/∂x ≈ [f(x+εeᵢ)−f(x−εeᵢ)]/2ε`）。
- Softmax 用 **log-sum-exp** 稳定化，数值 1000 级也不溢出。
- 交叉熵与 softmax 联合梯度有闭式 `softmax − onehot`。

---

## 运行

浏览器直接打开 `conv-forge.html`：一键跑 10 项不变量自检 + 实时训练演示（可视化卷积核与 loss 曲线）。

无头复现（Node）——自检逻辑内嵌于脚本，直接运行即可：

```bash
# 提取内联脚本并运行（单文件 HTML 内嵌全部 JS）
node -e "const fs=require('fs');const m=fs.readFileSync('conv-forge.html','utf8').match(/<script>([\s\S]*?)<\/script>/);fs.writeFileSync('_run.js',m[1]);"
node _run.js
```

期望输出：

```
Conv Forge · 不变量自检 · author 晨星
✅ #1 softmax log-sum-exp 稳定性  [Σp=1.000000000000000 mono=true]
✅ #2 im2col+GEMM ≡ 朴素卷积  [maxDiff=0.00e+0]
✅ #3 卷积反向 ≡ 数值梯度 (W,b,X)  [maxRelErr=5.28e-10]
✅ #4 最大池化反向 ≡ 数值梯度  [maxRelErr=4.29e-12]
✅ #5 线性层反向 ≡ 数值梯度  [maxRelErr=1.03e-9]
✅ #6 ReLU 前向/反向  [y=[0,0,0,0.001,3]]
✅ #7 端到端 CNN 反向 ≡ 数值梯度  [maxRelErr=6.59e-10]
✅ #8 端到端训练收敛 (acc≥0.9)  [acc=1.000 loss 45.537→0.000]
✅ #9 确定性可复现 (同种子同结果)  [lossA=… lossB=…]
✅ #10 卷积输出 shape 公式  [Ysh=[1,3,5,5]]
结果: 10/10 绿
```

---

## 架构

```
输入 [N,1,8,8]
  → Conv2D(1→4, k=3, s=1, p=0)      → [N,4,6,6]
  → ReLU
  → MaxPool(2,2)                      → [N,4,4,4]
  → Flatten                           → [N,64]
  → Linear(64→2)
  → Softmax + 交叉熵(求和约定)
  → loss
优化器：全批量 SGD（解析梯度）
```

前向提供**朴素循环**与 **im2col+GEMM** 双实现互校；反向传播全链路（conv / pool / linear）解析推导，
与中心差分对拍到机器精度。

---

## 10 条可验证不变量（硬金标准）

| # | 不变量 | 度量 |
|---|--------|------|
| 1 | softmax 经 log-sum-exp 稳定，概率和恒为 1，且保持序 | Σp=1, 1000 级不溢出 |
| 2 | im2col+GEMM ≡ 朴素卷积 | maxDiff = 0 |
| 3 | 卷积反向 ≡ 数值梯度（W, b, X） | maxRelErr ≈ 5e-10 |
| 4 | 最大池化反向 ≡ 数值梯度 | maxRelErr ≈ 4e-12 |
| 5 | 线性层反向 ≡ 数值梯度（W, b, A） | maxRelErr ≈ 1e-9 |
| 6 | ReLU 前向/反向逐点正确 | — |
| 7 | 端到端 CNN 反向 ≡ 数值梯度（全参数） | maxRelErr ≈ 7e-10 |
| 8 | 端到端训练真实收敛（acc ≥ 0.9，loss 显著下降） | acc = 1.000 |
| 9 | 确定性：同种子 → 同结果 | loss 逐位相等 |
| 10 | 卷积输出 shape 公式 `⌊(H+2p−k)/s⌋+1` 正确 | Ysh=[1,3,5,5] |

---

## 许可

MIT · 作者 **晨星** (CJX0712) · 可运行 · 可复现 · 可验证
