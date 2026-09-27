> Agent = Model + Harness
> The harness is everything around the model loop: the prompt, the tools, and any middleware that shapes behavior.

# Create an agent

用 LangChain 将 DeepSeek 模型和一个模拟天气查询工具组合成智能体，回答用户的天气问题。
```python
from langchain.agents import create_agent


def get_weather(location: str) -> str:
    """Get the weather of a location."""
    return f"{location} is sunny"


agent = create_agent(
    model="deepseek-chat",
    tools=[get_weather],
    system_prompt="You are a helpful assistant that can get the weather of a location.",
)

result = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "What's the weather in San Francisco?"}
        ]
    }
)

print(result["messages"][-1].content_blocks)
```
