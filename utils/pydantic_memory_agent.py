from pydantic import BaseModel
from typing import List
import asyncio
from pydantic_ai import Agent

# Define the memory structure
class Interaction(BaseModel):
    user_input: str
    agent_response: str

class AgentMemory(BaseModel):
    interactions: List[Interaction] = []

# Extend Pydantic AI Agent with integrated memory
class AIAgentWithMemory(Agent):
    def __init__(self, model: str, system_prompt: str):
        super().__init__(model=model, system_prompt=system_prompt)
        self.memory = AgentMemory()  # Initialize internal memory

    async def run(self, user_input: str) -> str:
        """
        Override the run method to include memory context automatically.
        """
        # Prepare memory context
        memory_context = "\n".join(
            [f"User: {i.user_input}\nAgent: {i.agent_response}" for i in self.memory.interactions]
        )

        # Call the parent Agent's run method with memory context
        complete_prompt = f"{memory_context}\n\n{user_input}"
        print('PROMPT:', complete_prompt, "\nPROMPT")
        result = await super().run(complete_prompt)

        # Get the agent's response
        agent_response = result.data

        # Update memory
        self.memory.interactions.append(
            Interaction(user_input=user_input, agent_response=agent_response)
        )

        return agent_response

    def get_memory(self) -> AgentMemory:
        """Expose memory for external inspection if needed."""
        return self.memory

    def save_memory_to_json(self, file_path: str):
        """Save memory to a JSON file."""
        with open(file_path, "w") as file:
            file.write(self.memory.json())

    def load_memory_from_json(self, file_path: str):
        """Load memory from a JSON file."""
        from pathlib import Path
        if Path(file_path).is_file():
            with open(file_path, "r") as file:
                self.memory = AgentMemory.parse_raw(file.read())

# Main loop
async def main():
    # Initialize the AI agent
    agent = AIAgentWithMemory(
        model="ollama:llama3.2", # Local model
        system_prompt="You are an AI assistant with memory."
    )

    # Optional: Load memory from a file
    agent.load_memory_from_json("agent_memory.json")

    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                break

            # Get agent response
            agent_response = await agent.run(user_input)
            print(f"Agent: {agent_response}")
    finally:
        # Save memory to a file before exiting
        agent.save_memory_to_json("agent_memory.json")

# Run the main loop
if __name__ == '__main__':
    asyncio.run(main())
