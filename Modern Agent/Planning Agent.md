# Plan-And-Execute

>The core idea is to first come up with a multi-step plan, and then go through that plan one item at a time. After accomplishing a particular task, you can then revisit the plan and modify as appropriate.

![](assets/Pasted%20image%2020260922204121.png)
**Plan**: top-down(subgoal) decomposition
- output: list of steps
- introducing generator-critic mode
**Execute**: consume a step of plan
- outcome/feedbacks
**Replan**: bottom-up
- input: raw input/raw plan/past steps

This compares to a typical [ReAct](https://arxiv.org/abs/2210.03629) style agent where you think one step at a time. The advantages of this "plan-and-execute" style agent are:

1. Explicit long term planning (which even really strong LLMs can struggle with)
2. Ability to use smaller/weaker models for the execution step, only using larger/better models for the planning step

