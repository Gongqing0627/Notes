# Reasoning

> **Reasoning（推理）是分析问题、得出判断；Planning（规划）是确定为了达成目标，要按什么步骤行动。** 规划通常需要推理，是推理的一种应用。
> **Reasoning体现着真正的intelligence。**

没有reasoning就没有planning，也不会有reflection。

## Reasoning Model

- **早期通用大语言模型**：已经能做一定的数学、逻辑判断和代码分析，但复杂多步问题容易出错。

- **2022 年：思维链（CoT）研究**：研究者发现，在提示里给模型展示“逐步推导”的例子，就能提高它解决一些复杂问题的能力。**无需更换模型，也能改善推理表现。** [Google 研究](https://research.google/blog/language-models-perform-reasoning-via-chain-of-thought/)

  ![](Pasted%20image%2020260909204546.png)

- **2024 年 9 月：OpenAI o1-preview 发布**：一个重要转折点：通过大规模强化学习，专门训练模型分解问题、检查错误、尝试不同方法，并在回答前投入更多推理计算。它是这条路线的代表性产品，不能说是所有 AI 推理研究的起点。[发布说明](https://openai.com/index/learning-to-reason-with-llms/)

- **后来的可调思考模型**：将这种能力做成不同思考档位，也就是 `low / medium / high` 等。[官方说明](https://developers.openai.com/api/docs/guides/reasoning)


# Planning

>**What**
>- Decomposition
>- Think ahead, think multiple-step

>**Why**
>- Limits of llm to solve complex queries or tasks
	- Long horizon（长程任务）：需要持续完成很多步骤，并始终保持目标、不遗漏要求。例如开发一个功能，从理解需求到修改代码、测试和修复。
	- Multi-hop（多跳推理）：需要串联多条信息才能得到答案。例如先查公司创始人，再查其毕业学校，最后确定学校所在城市
	- Multi-interactions（多次交互）：需要反复与用户、工具或环境互动，根据反馈继续行动。例如查询库存、发现缺货、查找替代品，再根据用户选择继续处理。
>- Leverage models performance on hard tasks

>**How ????????**
![523](Pasted%20image%2020260910033515.png)
## Decomposition

### ReAct
- **CoT-SC(Cot-Self-Consistency)**：让模型生成多份推理和答案，投票决定最终答案
- **ReAct**：Thought（判断下一步）→ Action（执行操作）→ Observation（接收结果）→ 继续判断，直到完成
- **ReAct → CoT-SC**：先执行ReAct，如果多步没有找到答案，采用CoT-SC。
- **CoT-SC→ ReAct**：先生成n份推理和答案，如果没有大于n/2的统一答案，采用ReAct策略。
![](Pasted%20image%2020260910203810.png)

## Selection




Reference:
1. [Understanding the planning of LLM agents: A survey](https://arxiv.org/pdf/2402.02716)
2. [ReAct](https://arxiv.org/pdf/2210.03629)

