# Overview
> Agent = Model + Harness
> The harness is everything around the model loop: the prompt, the tools, and any middleware that shapes behavior.
## Create an agent
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

# Agent
Agent是指让模型循环调用工具，直到完成指定任务的一种机制。
![696](assets/Pasted%20image%2020260929183840.png)
如图所示，这是一个典型的 ReAct 架构：模型在“思考—行动—观察”的循环中推进任务。LangChain 的 `create_agent` 底层采用了类似的机制，让模型根据当前上下文决定是否调用工具，并结合工具返回的结果继续处理，直到任务完成。

Harness 是围绕 Agent 循环的一整套机制，包括提示词（Prompt）、工具（Tools）以及用于塑造模型行为的各类中间件（Middleware）。
> **Agent = Model + Harness**
> Harness 的职责是：在完成特定任务的恰当时机，为模型提供恰当的上下文。

[`create_agent`](https://reference.langchain.com/python/langchain/agents/factory/create_agent?_gl=1*l1t5zd*_gcl_au*MTc5MzkwNDE1OC4xNzg5NDYzMDQ4*_ga*MTExMTExNzkwNS4xNzg5NDYzMDQ4*_ga_47WX3HKKY2*czE3OTA2NzM2NTQkbzkkZzAkdDE3OTA2NzM2NTQkajYwJGwwJGgw) is a highly configurable harness:
```python
class Answer(BaseModel): 
	summary: str 
	confidence: float 

agent = create_agent(
	model="openai:gpt-5.5",  # model
	tools=tools, # tools
	system_prompt="You are a helpful assistant. Be concise and accurate.",
	response_format=Answer # structured output
	)
```



# References
1. [Langchain](![](assets/Pasted%20image%2020260929183942.png))