# LLM Agent

The LLM Agent is a reinforcement learning agent that uses a local Ollama language model to make policy decisions. Instead of learning through traditional Q-learning or neural networks, it leverages the reasoning capabilities of large language models to select actions based on state descriptions.

## Features

- **LLM-powered decision making**: Uses Ollama models for intelligent action selection
- **Robust fallback system**: Falls back to random action selection if LLM fails
- **Configurable models**: Support for any Ollama model (llama2, mistral, codellama, etc.)
- **Temperature control**: Adjustable randomness in LLM responses
- **Custom system prompts**: Customize the agent's behavior through system prompts
- **Save/Load functionality**: Persist and restore agent configurations
- **Logging**: Comprehensive logging of decisions and errors

## Prerequisites

1. **Ollama installation**: Install Ollama from [https://ollama.ai/](https://ollama.ai/)
2. **Model download**: Pull a model, e.g., `ollama pull llama3.2`
3. **Python dependencies**: The `ollama` package is included in requirements.txt

## Usage

```python
from elsciRL.agents.LLM_Agent import LLMAgent

# Initialize the agent
agent = LLMAgent(
    parameter=0.1,                    # General parameter for future use
    model_name="llama2",              # Ollama model name
    temperature=0.7,                  # Controls randomness (0.0-1.0)
    system_prompt="Custom prompt..."  # Optional custom system prompt
)

# Use the agent to make decisions
state = "You are in a maze with walls to the north and west. There are open paths east and south."
legal_actions = ["move_east", "move_south", "stay"]

chosen_action = agent.policy(state, legal_actions)
print(f"Agent chose: {chosen_action}")


```

## Configuration Options

### Model Selection
- `llama3.2`: General-purpose model, good balance of performance and speed
- `mistral`: Fast and efficient, good for quick decisions
- `codellama`: Optimized for code and logic-based reasoning
- `llama3.2:13b`: Larger model for better reasoning (requires more resources)

### Temperature Settings
- `0.0`: Deterministic, always chooses the most likely action
- `0.3-0.5`: Slightly creative but mostly consistent
- `0.7`: Good balance of creativity and consistency (default)
- `1.0+`: Very creative and unpredictable

### System Prompt Examples

**Strategic Agent:**
```python
system_prompt = """You are a strategic decision maker. Analyze the situation carefully, 
consider long-term consequences, and choose the action that maximizes expected future reward."""
```

**Risk-Averse Agent:**
```python
system_prompt = """You are a cautious agent that prioritizes safety and risk minimization. 
Always choose the safest action that still makes progress toward the goal."""
```

**Exploratory Agent:**
```python
system_prompt = """You are an curious explorer. When faced with unknown situations, 
prefer actions that provide new information or lead to unexplored areas."""
```

## Error Handling

The agent includes robust error handling:

1. **Invalid action selection**: If the LLM suggests an action not in `legal_actions`, falls back to random selection
2. **JSON parsing errors**: If LLM response isn't valid JSON, uses first available action
3. **Ollama connection issues**: Falls back to random selection and logs the error
4. **Model not found**: Provides helpful error messages about installing models

## Save/Load Functionality

```python
# Save agent configuration
saved_config = agent.save()

# Load configuration into a new agent
new_agent = LLMAgent(parameter=0.1)
new_agent.load(saved_config)
```

## Integration with Environment

The LLM Agent follows the same interface as other elsciRL agents:

- `policy(state, legal_actions)`: Returns chosen action
- `learn(state, next_state, reward, action)`: Updates from experience  
- `save()`/`load()`: Persistence functionality

## Limitations

- **Latency**: LLM inference can be slower than traditional RL agents
- **Consistency**: May make different decisions for identical states
- **Resource usage**: Requires local GPU/CPU resources for model inference
- **Learning**: Currently doesn't update the underlying model from rewards (future enhancement)

## Future Enhancements

- **Fine-tuning**: Use reward signals to fine-tune the model
- **Experience replay**: Store successful action patterns
- **Multi-turn reasoning**: Allow the agent to "think" through complex decisions
- **Ensemble methods**: Combine multiple models for better decisions 