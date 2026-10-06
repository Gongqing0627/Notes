# Workflows and agents

![](assets/Pasted%20image%2020260927145334.png)
- Workflows have predetermined code paths and are designed to operate in a certain order.
- Agents are dynamic and define their own processes and tool usage.

# LLMs and augmentations
Workflows and agentic systems are based on LLMs and the various augmentations you add to them. **Tool calling, structured outputs,** and **shot term memory** are a few options for tailoring LLMs to your needs.
![](assets/Pasted%20image%2020260927150953.png)
## Structured outputs
```python
# Schema for structured output
class SearchQuery(BaseModel):
    search_query: str | None = Field(
        default=None,
        description="The search query to execute.",
    )
    justification: str | None = Field(
        default=None,
        description="The justification for the search query.",
    )


structured_agent = create_agent(
    model="deepseek-chat",
    tools=[],
    system_prompt="You are a helpful assistant.",
    # Define the response format
    response_format=SearchQuery
)

# Invoke the augmented LLM
output = agent.invoke({"messages": [{"role": "user", "content": "How does Calcium CT score relate to high cholesterol?"}]})


result = output["structured_response"]

print(result.search_query)
print(result.justification)
```
```python
Calcium CT score relationship to high cholesterol coronary artery calcification
To find medical evidence on how coronary calcium CT score relates to high cholesterol.
```
## Tools
```python
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


agent_with_tool = create_agent(
    model="deepseek-chat",
    tools=[multiply],
    system_prompt="You are a helpful assistant.",
)

output = agent_with_tool.invoke({"messages": [{"role": "user", "content": "Multiply 2 and 3"}]})
```

# Prompt Chaining
将一个可以拆解的大任务分成多个步骤，每次调用 LLM 时，都把上一步的输出作为下一步的输入。
- **翻译文档**：先翻译，再润色或检查。
- **检查生成内容的一致性**：先生成内容，再让另一次调用检查前后是否矛盾，例如人物名字、日期、结论是否一致。
![](assets/Pasted%20image%2020260928015357.png)
## Define LLM and Graph State
[Planning Agent](Planning%20Agent.md)该文档包含对 State 的相关说明。
```python
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)


class State(TypedDict):
    topic: str
    joke: str
    improved_joke: str
    final_joke: str
```
## Define Nodes
```python
def generate_joke(state: State):
    """First LLM to generate a joke about a topic."""
    msg = llm.invoke(f"Generate a joke about {state['topic']}")
    return {"joke": msg.content}

"""注意⚠️：该函数在后续流程中仅用于条件判断，不作为节点注册。"""
def check_punchline(state: State):
    """Gate function to check the punchline of a joke."""
    if "!" in state["joke"] or "?" in state["joke"]:
        return "Pass"
    return "Fail"

def improve_joke(state: State):
    """Second LLM to improve a joke."""
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}


def polish_joke(state: State):
    """Third LLM to polish a joke."""
    msg = llm.invoke(f"Add a surprising twist to this joke: {state['improved_joke']}")
    return {"final_joke": msg.content}
```
## Build Graph
```python
workflow = StateGraph(State)

workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_node("polish_joke", polish_joke)

workflow.add_edge(START, "generate_joke")
"""注意⚠️：check_punchline只作为一个判断函数"""
workflow.add_conditional_edges(
    "generate_joke", check_punchline, {"Pass": END, "Fail": "improve_joke"}
)
workflow.add_edge("improve_joke", "polish_joke")
workflow.add_edge("polish_joke", END)

chain = workflow.compile()

# runrunrun！！！！！！
state = chain.invoke({"topic": "cats"})
```
# Parallelization
关于并行化，LLMs可以一起同时工作。具体做法包括：同时执行多个相互独立的子任务，或多次执行同一项任务，以比较不同的输出结果。并行处理通常用于：
- 拆分子任务并同时执行，提高处理速度。
- 多次执行同一项任务并比较输出结果，提高结果的可信度。
![](assets/Pasted%20image%2020261001180951.png)
```python
class State(TypedDict):
    topic: str
    joke: str
    story: str
    poem: str
    combined_output: str


def call_llm1(state: State) -> State:
    response = llm.invoke("Write a joke about " + state["topic"])
    return {"joke": response.content}


def call_llm2(state: State) -> State:
    response = llm.invoke("Write a story about " + state["topic"])
    return {"story": response.content}


def call_llm3(state: State) -> State:
    response = llm.invoke("Write a poem about " + state["topic"])
    return {"poem": response.content}


def aggregator(state: State) -> State:
    combined = f"Here's a story, joke, and poem about {state['topic']}!\n\n"
    combined += f"STORY:\n{state['story']}\n\n"
    combined += f"JOKE:\n{state['joke']}\n\n"
    combined += f"POEM:\n{state['poem']}"
    return {"combined_output": combined}


workflow = StateGraph(State)

workflow.add_node("call_llm1", call_llm1)
workflow.add_node("call_llm2", call_llm2)
workflow.add_node("call_llm3", call_llm3)
workflow.add_node("aggregator", aggregator)

workflow.add_edge(START, "call_llm1")
workflow.add_edge(START, "call_llm2")
workflow.add_edge(START, "call_llm3")

workflow.add_edge("call_llm1", "aggregator")
workflow.add_edge("call_llm2", "aggregator")
workflow.add_edge("call_llm3", "aggregator")

workflow.add_edge("aggregator", END)

workflow = workflow.compile()

result = workflow.invoke({"topic": "cats"})
print(result["combined_output"])
```
# Routing
路由工作流会先识别输入的类型或意图，再将其分配到相应的处理节点。
![](assets/Pasted%20image%2020261001203151.png)
根据不同的用户输入路由到对应的节点：
```python
class State(TypedDict):
    input: str
    decision: str
    output: str


router = llm.with_structured_output(Route)


def router_node(state: State) -> State:
    decision = router.invoke([
        SystemMessage(
            content="Route the input to story, joke, or poem based on the user's request."
        ),
        HumanMessage(content=state["input"]),
    ])
    return {"decision": decision.step}

"""注意⚠️：此函数作为判断函数，不会作为节点注册"""
def router_decision(state: State) -> str:
    if state["decision"] == "joke":
        return "call_llm1"
    elif state["decision"] == "story":
        return "call_llm2"
    else:
        return "call_llm3"


def call_llm1(state: State) -> State:
    response = llm.invoke(state["input"])
    return {"output": response.content}


def call_llm2(state: State) -> State:
    response = llm.invoke(state["input"])
    return {"output": response.content}


def call_llm3(state: State) -> State:
    response = llm.invoke(state["input"])
    return {"output": response.content}


workflow = StateGraph(State)

workflow.add_node("router_node", router_node)
workflow.add_node("call_llm1", call_llm1)
workflow.add_node("call_llm2", call_llm2)
workflow.add_node("call_llm3", call_llm3)

workflow.add_edge(START, "router_node")

"""注意⚠️：条件边的写法"""
workflow.add_conditional_edges(
    "router_node",
    router_decision,
    {
        "call_llm1": "call_llm1",
        "call_llm2": "call_llm2",
        "call_llm3": "call_llm3",
    },
)

workflow.add_edge("call_llm1", END)
workflow.add_edge("call_llm2", END)
workflow.add_edge("call_llm3", END)

workflow = workflow.compile()

result = workflow.invoke({"input": "Write a joke about cats"})
print(result["output"])
```
# Orchestrator-worker
**Orchestrator** 英语原意是“编曲者”，在AI 工作流中通常译为 **“编排器”或“协调者”**。
在 **Orchestrator-worker范式**中，Orchestrator 主要负责：
- **分解任务**：将复杂任务拆解为多个子任务。
- **分配任务**：将子任务交给相应的执行者（Worker，如子代理）处理。
- **整合结果**：汇总各执行者的输出，形成完整的最终结果。
![](assets/Pasted%20image%2020261002000040.png)
这段代码实现了一个**自动拆分任务、并行撰写章节、最后汇总成报告的工作流**
```python
class Section(BaseModel):
	"""BaseModel 会在运行时校验数据"""
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )


class Sections(BaseModel):
    sections: List[Section] = Field(
        description="Sections of the report.",
    )


# Augment the LLM with schema for structured output
planner = llm.with_structured_output(Sections)
```
Orchestrator-worker（编排器—执行者）工作流很常见，LangGraph 为这种模式提供了内置支持。`Send` API 允许动态创建 Worker 节点，并向它们发送指定的输入。
每个 Worker 都有自己的状态，而所有 Worker 的输出都会写入一个共享的状态字段。
```python
# Graph state
class State(TypedDict):
    topic: str  # Report topic
    sections: list[Section]  # List of report sections
    completed_sections: Annotated[
        list, operator.add
    ]  # All workers write to this key in parallel
    final_report: str  # Final report

# Worker state
class WorkerState(TypedDict):
    section: Section

# Nodes
def orchestrator(state: State):
    """Orchestrator that generates a plan for the report"""

    # Generate queries
    report_sections = planner.invoke(
        [
            SystemMessage(content="Generate a plan for the report."),
            HumanMessage(content=f"Here is the report topic: {state['topic']}"),
        ]
    )

    return {"sections": report_sections.sections}

def llm_call(state: WorkerState):
    """Worker writes a section of the report"""

    # Generate section
    section = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Write a report section following the provided name and "
                    "description. Include no preamble for each section. "
                    "Use markdown formatting."
                )
            ),
            HumanMessage(
                content=(
                    f"Here is the section name: {state['section'].name} "
                    f"and description: {state['section'].description}"
                )
            ),
        ]
    )

    # Write the updated section to completed sections
    return {"completed_sections": [section.content]}


def synthesizer(state: State):
    """Synthesize full report from sections"""

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Format completed sections as a string
    completed_report_sections = "\n\n---\n\n".join(completed_sections)

    return {"final_report": completed_report_sections}


# Create a worker task for each section
def assign_workers(state: State):
    """Assign a worker to each section in the plan"""

    # Kick off section writing in parallel via Send() API
    return [
        Send("llm_call", {"section": s})
        for s in state["sections"]
    ]


# Build workflow
orchestrator_worker_builder = StateGraph(State)

# Add nodes
orchestrator_worker_builder.add_node("orchestrator", orchestrator)
orchestrator_worker_builder.add_node("llm_call", llm_call)
orchestrator_worker_builder.add_node("synthesizer", synthesizer)

# Add edges
orchestrator_worker_builder.add_edge(START, "orchestrator")

orchestrator_worker_builder.add_conditional_edges(
    "orchestrator",
    assign_workers,
    ["llm_call"],
)

orchestrator_worker_builder.add_edge("llm_call", "synthesizer")
orchestrator_worker_builder.add_edge("synthesizer", END)

# Compile workflow
orchestrator_worker = orchestrator_worker_builder.compile()

# Show workflow
display(Image(orchestrator_worker.get_graph().draw_mermaid_png()))

# Invoke
state = orchestrator_worker.invoke(
    {"topic": "Create a report on LLM scaling laws"}
)

from IPython.display import Markdown

Markdown(state["final_report"])
```
# Evaluator-optimizer
在 **Evaluator-optimizer范式**中，Generator先生成答案，再由 Evaluator检查是否符合要求。若通过评估，则输出答案；若未通过，则提供修改建议，由 Generator 根据反馈改进答案，并再次接受评估，直到满足要求或达到预设的迭代上限。
这种工作流通常适用于**有明确的验收标准，但需要多次迭代才能达标的任务**。

![](assets/Pasted%20image%2020261002222359.png)

这段代码围绕topic生成笑话，并根据评价反馈反复改进，直到被评为Funny。
```python
class State(TypedDict):
    topic: str
    joke: str
    feedback: str
    funny_or_not: str


def llm_call_generator(state: State) -> State:
    if state.get("feedback"):
        joke = llm.invoke([
            SystemMessage(content="Improve the joke based on the feedback."),
            HumanMessage(
                content=f"Joke: {state['joke']}\nFeedback: {state['feedback']}"
            ),
        ])
    else:
        joke = llm.invoke([
            SystemMessage(content="Generate a joke based on the topic."),
            HumanMessage(content=state["topic"]),
        ])

    return {"joke": joke.content}


class Feedback(BaseModel):
    funny_or_not: Literal["funny", "not funny"] = Field(
        description="Whether the joke is funny or not.",
    )
    feedback: None | str = Field(
        description=(
            "If the joke is not funny, please provide feedback to make "
            "the joke more funny. Otherwise, please leave it empty."
        ),
    )


evaluator = llm.with_structured_output(Feedback)


def llm_call_evaluator(state: State) -> State:
    feedback = evaluator.invoke([
        SystemMessage(content="Evaluate the joke whether it is funny or not."),
        HumanMessage(content=state["joke"]),
    ])

    return {
        "funny_or_not": feedback.funny_or_not,
        "feedback": feedback.feedback,
    }


def route_joke(state: State):
    return "funny" if state["funny_or_not"] == "funny" else "not funny"


workflow = StateGraph(State)

workflow.add_node("generator", llm_call_generator)
workflow.add_node("evaluator", llm_call_evaluator)

workflow.add_edge(START, "generator")
workflow.add_edge("generator", "evaluator")

workflow.add_conditional_edges(
    "evaluator",
    route_joke,
    {
        "funny": END,
        "not funny": "generator",
    },
)

workflow = workflow.compile()

result = workflow.invoke({"topic": "cat"})
```
# Agent
> **When to use agents:** Agents can be used for open-ended problems where it’s difficult or impossible to predict the required number of steps, and where you can’t hardcode a fixed path.

![](assets/Pasted%20image%2020261002230037.png)
Define tools
```python
# Define tools
@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Adds `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`.

    Args:
        a: First int
        b: Second int
    """
    return a / b


# Augment the LLM with tools
tools = [add, multiply, divide]
"""按名字找到工具，再传入参数"""
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)
```
Define nodes
```python
# Nodes
def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ]
    }


def tool_node(state: MessagesState):
    """Performs the tool call"""

    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


# Conditional edge function to route to the tool node or end based upon whether the LLM made a tool call
def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()

# Show the agent
display(Image(agent.get_graph(xray=True).draw_mermaid_png()))

# Invoke
messages = [HumanMessage(content="2 * 3 + 4.")]
messages = agent.invoke({"messages": messages})
for m in messages["messages"]:
    m.pretty_print()
```
```python
================================ Human Message =================================

2 * 3 + 4
================================== Ai Message ==================================

I'll solve this step by step, following order of operations (multiplication before addition).

First, let me multiply 2 * 3:
Tool Calls:
  multiply (call_00_4ie2Wovd3EwdXraWcrfA1216)
 Call ID: call_00_4ie2Wovd3EwdXraWcrfA1216
  Args:
    a: 2
    b: 3
================================= Tool Message =================================

6
================================== Ai Message ==================================

Now let me add 4 to that result:
Tool Calls:
  add (call_00_WDwahgLaYEwteuiPUkeM2102)
 Call ID: call_00_WDwahgLaYEwteuiPUkeM2102
  Args:
    a: 6
    b: 4
================================= Tool Message =================================

10
================================== Ai Message ==================================

**2 * 3 + 4 = 10**

Following order of operations: first 2 × 3 = 6, then 6 + 4 = 10.
```


1. [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
2. [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
