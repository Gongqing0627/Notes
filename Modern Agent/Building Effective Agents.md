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


# References
1. [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
2. [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)