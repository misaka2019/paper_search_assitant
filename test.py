from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11435/v1",
    api_key="your_api_key"
)

# 普通调用
response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "介绍AI模型的安全威胁"}]
)

# 流式调用
for chunk in client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[{"role": "user", "content": "介绍AI模型的安全威胁"}],
    stream=True
):
    print(chunk.choices[0].delta.content or "", end="")