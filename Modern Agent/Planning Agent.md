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



# Define Execution Agent

```python
from langchain.agents import create_agent

tools = [SearchTool]

agent_executor = create_agent(
    model="chat-openai",
    tools=tools,
    system_prompt="You are a helpful assistant.",
)

agent_executor.invoke(
    {
        "messages": [
            {"role": "user", "content": "Who is the winnner of the us open"}
        ]
    }
)
```

## Agent 执行记录

```python
{
    "messages": [
        # 1. 用户提问
        HumanMessage(
            content="who is the winnner of the us open"
        ),

        # 2. 模型请求调用搜索工具
        AIMessage(
            content="",
            tool_calls=[{
                "name": "tavily_search_results_json",
                "args": {"query": "US Open 2023 winner"},
                "id": "call_1",
                "type": "tool_call",
            }],
        ),

        # 3. 工具返回搜索结果（内容摘要）
        ToolMessage(
            content="搜索返回 3 条结果，涉及高芙和德约科维奇赢得 2023 年美网冠军。",
            name="tavily_search_results_json",
            tool_call_id="call_1",
        ),

        # 4. 模型根据搜索结果回答（内容摘要）
        AIMessage(
            content="2023 年美网女单冠军是高芙，男单冠军是德约科维奇。"
        ),
    ]
}
```

执行流程：**用户提问 → 模型请求搜索 → 工具返回结果 → 模型生成回答**。

# Define the State

这里的 **state（状态）就是 Agent 执行任务时，各个步骤共享的一份数据记录**。它保存用户的问题、当前计划、已完成的步骤和最终答案。每一步读取这份数据，再返回需要更新的内容。

```python
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
```

- **`input`：用户的原始问题**
  - 作用：保存用户提出的任务，让后续步骤明确要解决什么问题。
  - 类型：`str`（字符串）。
  - 例子：`"2023 年美网男单和女单冠军分别是谁？"`

- **`plan`：当前的执行计划**
  - 作用：把任务拆成具体步骤，供执行节点依次处理；后续可以根据执行结果调整计划。
  - 类型：`List[str]`（字符串列表）。
  - 例子：`["搜索男单冠军", "搜索女单冠军", "整理最终答案"]`

- **`past_steps`：已执行的步骤及其结果**
  - 作用：记录已经完成的工作，供后续判断是否需要继续执行或调整计划。
  - 类型：`Annotated[List[Tuple], operator.add]`（元组列表，并指定追加合并规则）。
  - 例子：`[("搜索男单冠军", "2023 年美网男单冠军是德约科维奇")]`
  
- **`response`：最终回复**
  - 作用：保存任务完成后，准备返回给用户的答案。
  - 类型：`str`（字符串）。
  - 例子：`"2023 年美网男单冠军是德约科维奇，女单冠军是高芙。"`

# Planning Step

```python
class Plan(BaseModel):
	""" Plan to follow in future """
	
	steps: List[str] = Feild(
		description="different steps to follow, should be in sorted order"
	)
```
定义模型输出的格式，并校验数据类型：
- 输出包含一个 **`steps` 字段**。
- `steps` 是一个**字符串列表**。
- 描述告诉模型：步骤要按执行顺序排列，但代码本身不会自动排序。
## Create a Planner
```python

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "For the given objective, come up with a simple step by step plan. "
                "This plan should involve individual tasks, that if executed correctly "
                "will yield the correct answer. Do not add any superfluous steps. "
                "The result of the final step should be the final answer. "
                "Make sure that each step has all the information needed "
                "- do not skip steps."
            ),
        ),
        ("placeholder", "{messages}"),
    ]
)

planner = planner_prompt | ChatOpenAI(
    model="gpt-4o",
    temperature=0,
).with_structured_output(Plan)

planner.invoke(
    {
        "messages": [
            ("user", "what is the hometown of the current Australia open winner?")
        ]
    }
)
```
```python
Plan(steps=['Identify the current winner of the Australia Open.', 'Find the hometown of the identified winner.'])
```

# Re-Plan Step

## Create a Replanner
```python
class Response(BaseModel):
    """Response to user."""

    response: str

class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed c·rrectly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)

replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0
).with_structured_output(Act)

```

# Create the Graph
Graph 是用来控制整个任务流程的：先规划，再执行，执行后判断下一步做什么，直到完成。
Graph中的Nodes之间通过 **`state` 共享数据**，例如用户任务、当前计划、已经完成的步骤。节点读取这些数据，再返回需要更新的字段；Graph 的**连线和条件分支**决定接下来运行哪个节点。
## Nodes
```python
from langgraph.graph import END

async def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = await agent_executor.ainvoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }

async def plan_step(state: PlanExecute):
    plan = await planner.ainvoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}

async def replan_step(state: PlanExecute):
    output = await replanner.ainvoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}

def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"

```

# Graph
```python
from langgraph.graph import StateGraph, START

workflow = StateGraph(PlanExecute)

# Add the plan node
workflow.add_node("planner", plan_step)

# Add the execution step
workflow.add_node("agent", execute_step)

# Add a replan node
workflow.add_node("replan", replan_step)

workflow.add_edge(START, "planner")

# From plan we go to agent
workflow.add_edge("planner", "agent")

# From agent, we replan
workflow.add_edge("agent", "replan")

workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["agent", END],
)

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()
```
![](assets/Pasted%20image%2020260926212922.png)
# Run and Stream
```python
config = {"recursion_limit": 50}
inputs = {"input": "what is the hometown of the mens 2024 Australia open winner?"}
async for event in app.astream(inputs, config=config):
    for k, v in event.items():
        if k != "__end__":
            print(v)
```
```python
{'plan': ["Identify the winner of the men's 2024 Australian Open.", 'Research the hometown of the identified winner.']}
{'past_steps': [("Identify the winner of the men's 2024 Australian Open.", "The winner of the men's singles tennis title at the 2024 Australian Open was Jannik Sinner. He defeated Daniil Medvedev in the final with scores of 3-6, 3-6, 6-4, 6-4, 6-3 to win his first major singles title.")]}
{'plan': ['Research the hometown of Jannik Sinner.']}
{'past_steps': [('Research the hometown of Jannik Sinner.', "Jannik Sinner's hometown is Sexten, which is located in northern Italy.")]}
{'response': "The hometown of the men's 2024 Australian Open winner, Jannik Sinner, is Sexten, located in northern Italy."}
```
# References:
1. [Plan-and-Execute](https://github.com/langchain-ai/langgraph/blob/23961cff61a42b52525f3b20b4094d8d2fba1744/docs/docs/tutorials/plan-and-execute/plan-and-execute.ipynb)