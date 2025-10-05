"""
Helper utilities for Claude Agent SDK with improved defaults.
"""

from claude_agent_sdk import ClaudeAgentOptions
from typing import Any


def create_claude_options(**kwargs: Any) -> ClaudeAgentOptions:
    """
    Create ClaudeAgentOptions with claude-sonnet-4-5 as the default model.
    
    This helper function provides a convenient way to create ClaudeAgentOptions
    with sensible defaults, particularly setting the model to claude-sonnet-4-5
    (Claude Sonnet 4.5) which is the latest and most capable model for most tasks.
    
    Args:
        **kwargs: Any arguments that ClaudeAgentOptions accepts
        
    Returns:
        ClaudeAgentOptions: Configured options with claude-sonnet-4-5 as default model
        
    Example:
        # Use default model (claude-sonnet-4-5)
        options = create_claude_options(
            allowed_tools=["Read", "Write"],
            system_prompt="You are a helpful assistant"
        )
        
        # Override with different model
        options = create_claude_options(
            model="claude-opus-4-1",
            allowed_tools=["Read", "Write"]
        )
    """
    # Set default model if not provided
    if 'model' not in kwargs:
        kwargs['model'] = 'claude-sonnet-4-5'
    
    return ClaudeAgentOptions(**kwargs)


# Model name constants for convenience
class ClaudeModels:
    """Constants for Claude model names based on the official documentation."""
    
    # Claude 4.x models (latest generation)
    SONNET_4_5 = "claude-sonnet-4-5"  # Alias for claude-sonnet-4-5-20250929
    SONNET_4_5_FULL = "claude-sonnet-4-5-20250929"  # Full model name
    SONNET_4 = "claude-sonnet-4-0"  # Alias for claude-sonnet-4-20250514
    SONNET_4_FULL = "claude-sonnet-4-20250514"  # Full model name
    OPUS_4_1 = "claude-opus-4-1"  # Alias for claude-opus-4-1-20250805
    OPUS_4_1_FULL = "claude-opus-4-1-20250805"  # Full model name
    OPUS_4 = "claude-opus-4-0"  # Alias for claude-opus-4-20250514
    OPUS_4_FULL = "claude-opus-4-20250514"  # Full model name
    
    # Claude 3.x models
    SONNET_3_7 = "claude-3-7-sonnet-latest"  # Alias for claude-3-7-sonnet-20250219
    SONNET_3_7_FULL = "claude-3-7-sonnet-20250219"  # Full model name
    HAIKU_3_5 = "claude-3-5-haiku-latest"  # Alias for claude-3-5-haiku-20241022
    HAIKU_3_5_FULL = "claude-3-5-haiku-20241022"  # Full model name
    HAIKU_3 = "claude-3-haiku-20240307"  # Full model name
    
    # Recommended defaults
    DEFAULT = SONNET_4_5  # Best overall model for most tasks
    FASTEST = HAIKU_3_5   # Fastest model
    MOST_CAPABLE = OPUS_4_1  # Most capable for complex tasks
