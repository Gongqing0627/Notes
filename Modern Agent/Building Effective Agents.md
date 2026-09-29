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



# References
1. [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
2. [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
