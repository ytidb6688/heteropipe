# arXiv 提交指南（heteropipe makespan 论文）

对应论文：`Exact Makespan Bounds and the GPipe Homogeneity Bias for Heterogeneous LLM Inference Pipelines`
英文源：`paper/arxiv/main.tex`（纯 ASCII，pdfTeX/XeTeX 双兼容，已本地编译通过）

## 一、提交包内容

| 文件 | 作用 |
|---|---|
| `main.tex` | LaTeX 主文件（含内嵌参考文献 thebibliography，无需 .bib/.bbl） |
| `fig1_makespan_hetero.png` | Figure 1（定理 1 对比图） |
| `fig2_comm_additive.png` | Figure 2（异步通信平行线） |
| `fig3_blocking_phasetransition.png` | Figure 3（阻塞相变） |
| `fig4_async_block_sandwich.png` | Figure 4（夹逼区间） |
| `fig_real_vs_formula.png` | Figure 5（生产调度器实测对比） |

上传时把 `main.tex` + 5 张 PNG（共 6 个文件）打成 zip 即可（arXiv 支持 zip 上传，自动解包；也可以逐个上传）。

**与 md 版的差异（有意为之）**：
1. 未包含"中文摘要"块——arXiv 元数据与 PDF 走英文规范，中文版完整保留在 `paper/预印本_*.md`；
2. 正文"算力一张网"改为英文 "compute-as-one-network"（纯 ASCII 源码要求，pdflatex 无 CJK 宏包依赖）；
3. 定理/推论/命题改用标准 `amsthm` 环境（自动编号，与 md 中编号一致）；式 (1)(2) 用 `equation` 自动编号并 `\eqref` 交叉引用。

## 二、提交前置条件（硬门槛）

1. **arXiv 账号**：https://arxiv.org/account/register 注册（邮箱验证；建议用机构邮箱 yvcct.edu.cn，能减少审核疑虑）。
2. **Endorser 背书（最关键的一步）**：cs 分区对新作者要求背书（endorsement）。
   - 在提交系统里填入 `cs.DC` 后，arXiv 会提示你需要 endorser，并给出一封可转发的说明邮件；
   - 把该邮件转发给一位**近年在 cs.DC 或 cs.PF 发表过 ≥5 篇论文**的研究者（高校系统/并行计算方向的老师都可以），请对方点击邮件里的背书链接；
   - 背书只解锁"提交资格"，不评审内容，一般几分钟内生效。
   - 找不到背书人时的兜底：arXiv 官方帮助 https://arxiv.org/help/endorsement 有"请求人工审核"入口。
3. **类别选择**：
   - Primary：`cs.DC`（Distributed, Parallel, and Cluster Computing）
   - Cross-list：`cs.PF`（Performance）+ `cs.LG`（可选）

## 三、网页提交流程（约 20 分钟）

1. 登录后点 **Start New Submission**；
2. **License**：选默认 `arXiv non-exclusive license` 即可（后续仍可投会议/期刊）；
3. **Articles**：选 "New submission"，填写标题/作者/摘要/类别（摘要直接复制 main.tex 中 abstract 的英文段）；
4. **Files**：上传 zip（或逐个上传 6 个文件）→ arXiv 自动用 TeX Live 编译；
5. **预览 PDF**：逐页核对公式编号 (1)(2)、定理 1–3、图 1–5、表 1–2、32 条参考文献；
6. **同意条款 → Submit**。

## 四、提交后

- 新提交先进入 on hold（人工 moderation），工作日一般 1–2 天；周末与美股节假日顺延（今天是美国周日，预计周一至周二出结果）；
- 状态从 on hold → announced 后即获得 arXiv ID（如 arXiv:2610.xxxxx），此时把正式链接回填到：
  - `paper/arXiv_heteropipe_makespan_bounds.md` 与中文版 md 的数据可用性段；
  - GitHub README；
- 若编译失败被退回（常见原因：图片缺失/宏包过冷门），本包只用 `amsmath/amssymb/amsthm/graphicx/booktabs/geometry/hyperref` 七个标准包，全部在 arXiv 默认 TeX Live 内，风险极低。

## 五、快速复验本地编译

```bash
# Windows（tectonic 0.15.0，首次联网取宏包）
tectonic main.tex     # 产出 main.pdf（约 15–17 页）
```

---

*指南生成：2026-10-04。修复记录：md 版的 GitHub 渲染问题（\tag 竖排、$j^*$ 被强调抢占、CJK 后 $ 不渲染、斜体内公式失效）已在 commit 001ca7e 修复并推送；LaTeX 版不受影响（LaTeX 原生支持这些写法）。*
