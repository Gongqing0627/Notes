# Gongqing 的学习笔记

这里记录我在大模型应用、Agent 和相关工程技术上的学习过程，包括概念梳理、代码示例、实践记录和参考资料。

**我会持续更新这个仓库，补充新的学习笔记与实践内容，也会随着理解的深入回头修正和完善已有笔记。**

## 阅读导航

### LangChain 与 LangGraph

| 笔记 | 内容 |
| --- | --- |
| [LangChain](Exploring%20the%20Lang%20Ecosystem/LangChain.md) | Agent 的基本组成、工具调用与结构化输出示例 |
| [LangGraph](Exploring%20the%20Lang%20Ecosystem/LangGraph.md) | 工作流与 Agent 的阅读入口，具体示例见下方 Building Effective Agents |

### Agent 推理与规划

| 笔记 | 内容 |
| --- | --- |
| [Building Effective Agents](Modern%20Agent/Building%20Effective%20Agents.md) | 提示词链、并行化、路由、编排器与执行者、生成与评估循环，以及工具调用示例 |
| [Planning Agent](Modern%20Agent/Planning%20Agent.md) | Plan-and-Execute 的规划、执行与重新规划流程，以及共享状态的组织方式 |
| [Agent Reasoning & Planning](Modern%20Agent/Agent%20Reasoning%20%26%20Planning.md) | 推理与规划的概念梳理，以及 ReAct、Plan-and-Solve、ToT、LLM+P 等方法的学习记录 |

### 其他技术笔记

| 笔记 | 内容 |
| --- | --- |
| [FastAPI](Pieces/FastAPI.md) | FastAPI 学习笔记 |
| [Vision Transformer](Pieces/Vision%20Transformer.md) | ViT 基本概念与论文阅读记录，持续补充中 |

## 阅读建议

如果你关注 Agent 工作流，可以先阅读 **Building Effective Agents**，再结合 **Planning Agent** 理解多步骤任务的规划与执行。

各篇笔记中的参考资料和图片放在对应主题目录中，方便结合原文与示例阅读。后续新增内容也会补充到这份导航中。
