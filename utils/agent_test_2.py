from pydantic_ai import Agent

if __name__ == "__main__":
    # Define a very simple agent including the model to use, you can also set the model when running the agent.
    agent = Agent(
        'ollama:llama3.2',
            # Register a static system prompt using a keyword argument to the agent.
        # For more complex dynamically-generated system prompts, see the example below.
        system_prompt='Be concise, reply with one sentence.',
    )

    # Run the agent synchronously, conducting a conversation with the LLM.
    # Here the exchange should be very short: PydanticAI will send the system prompt and the user query to the LLM,
    # the model will return a text response. See below for a more complex run.
    result = agent.run_sync('Where does "hello world" come from?')
    print(result.data)
    """
    The first known use of "hello, world" was in a 1974 textbook about the C programming language.
    """