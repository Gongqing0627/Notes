# HTTP

>HTTP 定义了调用方与被调用方之间请求和响应的通信规范。

![672](assets/Pasted%20image%2020260914193833.png)
- **起始行**
	- 请求报文：请求方法、请求目标、HTTP 版本。
    - 响应报文：HTTP 版本、状态码、原因短语。
- **头部（Header）**
    - 描述请求或响应的附加信息，如内容类型、内容长度等。
- **空行**
    - 分隔头部和正文。
- **正文（Body，可选）**
    - 承载实际发送或返回的数据，如 JSON、HTML、图片等。
    - 可以是文本，也可以是二进制数据。


## HTTP请求拆解

### 从 curl 命令看 HTTP 请求的组成
```
curl https://api.deepseek.com/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${DEEPSEEK_API_KEY}" \
  -d '{
        "model": "deepseek-flash",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hello!"}
        ],
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
        "stream": false
      }'
```
- **请求方法：`POST`**
    - 使用 `-d` 发送数据时，`curl` 默认使用 `POST`。
- **请求地址：`https://api.deepseek.com/chat/completions`**
    - 主机是 `api.deepseek.com`，请求路径是 `/chat/completions`。
- **请求头：通过 `-H` 指定**
    - `Content-Type: application/json`：说明请求体采用 JSON 格式。
    - `Authorization: Bearer ...`：携带 API Key，用于身份验证。
- **请求体：通过 `-d` 指定**
    - 后面的整段 JSON 就是请求体，包含 `model`、`messages`、`thinking`、`reasoning_effort`、`stream` 等字段。
- **空行及其他必要信息：由 `curl` 自动处理**
    - 你不需要手动写出头部与正文之间的空行。

### 请求结果
```
> POST /chat/completions HTTP/2
> Host: api.deepseek.com
> User-Agent: curl/8.7.1
> Accept: */*
> Content-Type: application/json
> Authorization: Bearer apikey
> Content-Length: 299
> 
* upload completely sent off: 299 bytes

< HTTP/2 200 
< server: openresty
< content-type: application/json
< vary: origin, access-control-request-method, access-control-request-headers
< access-control-allow-credentials: true
< x-ds-trace-id: 7fde6eeb372cece2a25fa752da6d90d9
< strict-transport-security: max-age=31536000; includeSubDomains; preload
< x-content-type-options: nosniff
< date: Mon, 14 Sep 2026 12:08:43 GMT
< eo-log-uuid: 12238438622816626444
< eo-cache-status: MISS
< 
* Connection #0 to host api.deepseek.com left intact
{"id":"41ac167c-4c17-4e5d-b0a9-559c86601806","object":"chat.completion","created":1789387723,"model":"deepseek-flash","choices":[{"index":0,"message":{"role":"assistant","content":"Hello! How can I help you today?","reasoning_content":"The user just said \"Hello!\" — a simple greeting. I should respond warmly and briefly, and invite them to share what they need. No need for tools or lengthy output."},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":38,"completion_tokens":46,"total_tokens":84,"prompt_tokens_details":{"cached_tokens":0},"completion_tokens_details":{"reasoning_tokens":36},"prompt_cache_hit_tokens":0,"prompt_cache_miss_tokens":38},"system_fingerprint":"aeb56401ca74e127821c4f9126dcb669"}CQ9RQR9CXL:~ bytedance$
```
- **`>` 开头：发送的请求**
    - `POST /chat/completions HTTP/2`：使用 HTTP/2，向 `/chat/completions` 发送 POST 请求。
    - `Host`：目标服务器。
    - `User-Agent`：调用工具是 curl，版本为 8.7.1。
    - `Accept: */*`：接受任意类型的响应。
    - `Content-Type: application/json`：请求体是 JSON。
    - `Authorization: Bearer apikey`：身份验证信息。
    - `Content-Length: 299`：请求体大小为 **299 字节**。
- **`*` 开头：curl 的运行提示**
    - `upload completely sent off: 299 bytes`：请求体已全部发送；这段日志没有显示请求体的具体内容。
    - `Connection #0 ... left intact`：连接保留，可供复用，不是报错。
- **`<` 开头：收到的响应**
    - `HTTP/2 200`：服务器成功处理了请求。
    - `content-type: application/json`：返回的数据是 JSON。
    - `server: openresty`：服务器声明使用的软件。
    - 其余响应头主要涉及跨域、安全策略、时间、缓存和请求追踪。
- **最后的 JSON：响应体**
    - `id`：本次回复的标识。
    - `model`：本次使用的模型是 `deepseek-flash`。
    - `choices[0].message.content`：模型的正式回答——**“Hello! How can I help you today?”**
    - `reasoning_content`：接口返回的推理说明字段。
    - `finish_reason: "stop"`：模型正常结束输出。
    - `usage`：输入 **38 tokens**，输出 **46 tokens**，总计 **84 tokens**；输出中包含 **36 个推理 tokens**。

## 简单的 HTTP API 服务端实现（GET）

```python
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

# 返回给客户端的示例数据
profile = {
    "name": "bytedance"
}


class Handler(BaseHTTPRequestHandler):
    # 处理客户端发起的 GET 请求
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(profile).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write("404 Not Found".encode("utf-8"))


# 创建 HTTP 服务并持续监听 8000 端口
server = HTTPServer(("localhost", 8000), Handler)
print("Server is running on port 8000")
server.serve_forever()
```
```
curl -v http://localhost:8000/
* Uses proxy env variable no_proxy == '.tiktokd.net,tiktokd.net,.byted.org,byted.org,.bytedance.net,bytedance.net,.tiktokd.net,tiktokd.net,.byted.org,byted.org,.bytedance.net,bytedance.net'
*   Trying 127.0.0.1:8000...
* Connected to localhost (127.0.0.1) port 8000 (#0)
> GET / HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.88.1
> Accept: */*
> 
* HTTP 1.0, assume close after body
< HTTP/1.0 200 OK
< Server: BaseHTTP/0.6 Python/3.11.2
< Date: Mon, 14 Sep 2026 13:28:36 GMT
< Content-type: application/json
< 
* Closing connection 0
{"name": "bytedance"}
```


# FastAPI
![](assets/Pasted%20image%2020260914222301.png)
## HTTP API  FastAPI重实现（GET）
```python
from fastapi import FastAPI
from pydantic import BaseModel

class AnalyzeRequest(BaseModel):
	text: str

app = FastAPI()

profile = {
	"name": "bytedance"
}

@app.get("/profile")
def read_profile():
	return profile

@app.post("/analyze")
def analyze(request: AnalyzeRequest):
	return {
		"text": request.text,
		"message": "analyze success"
		}
```

```bash
uvicorn main:app --reload --port 8000
```
