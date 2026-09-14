# Reasoning

> **Reasoning（推理）是分析问题、得出判断；Planning（规划）是确定为了达成目标，要按什么步骤行动。** 规划通常需要推理，是推理的一种应用。
> **Reasoning体现着真正的intelligence。**

没有reasoning就没有planning，也不会有reflection。

## Reasoning Model

- **早期通用大语言模型**：已经能做一定的数学、逻辑判断和代码分析，但复杂多步问题容易出错。

- **2022 年：思维链（CoT）研究**：研究者发现，在提示里给模型展示“逐步推导”的例子，就能提高它解决一些复杂问题的能力。**无需更换模型，也能改善推理表现。** [Google 研究](https://research.google/blog/language-models-perform-reasoning-via-chain-of-thought/)

  ![](assets/Pasted%20image%2020260909204546.png)

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

![523](assets/Pasted%20image%2020260910033515.png)
## Decomposition

### ReAct
- **CoT-SC(Cot-Self-Consistency)**：让模型生成多份推理和答案，投票决定最终答案
- **ReAct**：Thought（判断下一步）→ Action（执行操作）→ Observation（接收结果）→ 继续判断，直到完成
- **ReAct → CoT-SC**：先执行ReAct，如果多步没有找到答案，采用CoT-SC。
- **CoT-SC→ ReAct**：先生成n份推理和答案，如果没有大于n/2的统一答案，采用ReAct策略。
![](assets/Pasted%20image%2020260910203810.png)

### Plan and Solve
普通的“请一步步思考”可能仍然漏步骤。PS 先让模型明确需要做哪些事，主要针对这种**遗漏中间步骤**的问题。论文还提出 PS+，加入提取变量、注意计算等更详细的指令，减少计算错误。
![](assets/Pasted%20image%2020260914014717.png)
## Selection
### ToT
ToT 将问题的求解过程组织成一棵思维树：每一步生成多个候选思路，由大模型评估各候选状态的潜力，再由搜索算法选择保留和继续探索的分支，必要时剪枝或回退，直到得到答案。
![688](assets/Pasted%20image%2020260912132152.png)

## External Planner
### LLM + P

![](assets/Pasted%20image%2020260913135609.png)
- **提供规则和示例**：人工定义可用动作、执行条件和效果，并提供一组“自然语言问题—PDDL”的示例。
- **转换任务**：大模型将用户需求转成 PDDL 问题文件，明确对象、初始状态和目标。
- **求解计划**：外部规划器结合规则和问题文件，搜索满足目标的动作序列。
- **输出步骤**：大模型将动作序列翻译成易懂的自然语言。


Reference:
1. [Understanding the planning of LLM agents: A survey](https://arxiv.org/pdf/2402.02716)
2. [ReAct](https://arxiv.org/pdf/2210.03629)
3. [Plan and Solve](https://arxiv.org/pdf/2305.04091)
4. [ToT: Tree of Thought](http://arxiv.org/pdf/2305.10601)
5. [LLM+P](https://arxiv.org/pdf/2304.11477)

