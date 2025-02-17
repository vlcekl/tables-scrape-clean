import os

from smolagents import CodeAgent, HfApiModel, LiteLLMModel, ToolCallingAgent

model = LiteLLMModel(
    model_id="ollama/llama3.1",
    api_key="ollama"
)
agent = CodeAgent(tools=[], model=model, add_base_tools=True, additional_authorized_imports=['math'])

#hf_token = os.environ['HUGGINGFACEHUB_API_TOKEN']
#model = HfApiModel(model_id="meta-llama/Llama-3.3-70B-Instruct", token=hf_token)
#model = HfApiModel(model_id="ollama/llama3.2", token=hf_token)
#agent = CodeAgent(tools=[], model=model, additional_authorized_imports=['math'])


result = agent.run("What is square root of 144?")

print(result)
