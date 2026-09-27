# Workflows and agents

![](assets/Pasted%20image%2020260927145334.png)
- Workflows have predetermined code paths and are designed to operate in a certain order.
- Agents are dynamic and define their own processes and tool usage.

# LLMs and augmentations
Workflows and agentic systems are based on LLMs and the various augmentations you add to them. **Tool calling, structured outputs,** and **shot term memory** are a few options for tailoring LLMs to your needs.
![](assets/Pasted%20image%2020260927150953.png)

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


agent = create_agent(
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


# References
1. [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
2. [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)