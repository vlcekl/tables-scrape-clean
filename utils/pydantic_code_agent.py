import asyncio
from pydantic_ai import Agent, Tool, RunContext
from typing import Union
import subprocess

# Define the function to execute Python code
async def execute_code(ctx: RunContext, code: str) -> str:
    """
    Safely executes Python code and returns the result or any errors.
    """
    try:
        # Use subprocess to safely execute the code in an isolated environment
        result = subprocess.run(
            ['python3', '-c', code],
            capture_output=True,
            text=True,
            timeout=5  # Prevent runaway code execution
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr.strip()}"
    except Exception as e:
        return f"Exception during execution: {str(e)}"

# Create a Tool instance for the execute_code function
execute_code_tool = Tool(
    function=execute_code,
    name="execute_code",
    description="Executes Python code and returns the result."
)

# Initialize the Agent with the tool
agent = Agent(
    model="ollama:llama3.2",  # Replace with your desired model identifier
    tools=[execute_code_tool]
)

# Function to interact with the agent
async def ask_code_agent(prompt: str) -> str:
    """
    Process user input, generate Python code, execute it, and return the result.
    """
    # Run the agent with the provided prompt
    result = await agent.run(prompt)
    return result

# Example usage
if __name__ == "__main__":
    user_prompt = "Act as a code agent and use python to solve square root of 111. Only return the actual result with no additional words."
    result = asyncio.run(ask_code_agent(user_prompt))
    print(f"Execution Result:\n{result}")
