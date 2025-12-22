"""
Custom neural network architectures for RL agents.

Implements custom actor-critic networks with flexible architectures.
"""

import logging
from typing import Dict, List, Optional, Tuple, Type

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

logger = logging.getLogger(__name__)


class CustomFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for state processing."""

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        hidden_dims: Optional[List[int]] = None,
    ):
        """
        Initialize feature extractor.

        Args:
            observation_space: Observation space
            features_dim: Output feature dimension
            hidden_dims: Hidden layer dimensions
        """
        super().__init__(observation_space, features_dim)

        if hidden_dims is None:
            hidden_dims = [256, 256]

        # Build network
        layers = []
        input_dim = observation_space.shape[0]

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, features_dim))
        layers.append(nn.ReLU())

        self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(observations)


class CustomActorCriticPolicy(ActorCriticPolicy):
    """Custom actor-critic policy with flexible architecture."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule,
        net_arch: Optional[List[Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        """
        Initialize custom policy.

        Args:
            observation_space: Observation space
            action_space: Action space
            lr_schedule: Learning rate schedule
            net_arch: Network architecture specification
            activation_fn: Activation function
        """
        if net_arch is None:
            net_arch = [dict(pi=[256, 256], vf=[256, 256])]

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs,
        )


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """LSTM-based feature extractor for sequential data."""

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 2,
    ):
        """
        Initialize LSTM feature extractor.

        Args:
            observation_space: Observation space
            features_dim: Output feature dimension
            lstm_hidden_size: LSTM hidden size
            num_lstm_layers: Number of LSTM layers
        """
        super().__init__(observation_space, features_dim)

        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=observation_space.shape[0],
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        # Output layer
        self.linear = nn.Linear(lstm_hidden_size, features_dim)
        self.activation = nn.ReLU()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM."""
        # Add sequence dimension if needed
        if len(observations.shape) == 2:
            observations = observations.unsqueeze(1)

        # LSTM forward
        lstm_out, _ = self.lstm(observations)

        # Take last timestep
        last_hidden = lstm_out[:, -1, :]

        # Linear projection
        features = self.activation(self.linear(last_hidden))

        return features


class AttentionFeatureExtractor(BaseFeaturesExtractor):
    """Attention-based feature extractor."""

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        num_heads: int = 4,
        hidden_dim: int = 256,
    ):
        """
        Initialize attention feature extractor.

        Args:
            observation_space: Observation space
            features_dim: Output feature dimension
            num_heads: Number of attention heads
            hidden_dim: Hidden dimension
        """
        super().__init__(observation_space, features_dim)

        # Input projection
        self.input_proj = nn.Linear(observation_space.shape[0], hidden_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention."""
        # Project input
        x = self.input_proj(observations)

        # Add sequence dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Self-attention
        attn_output, _ = self.attention(x, x, x)

        # Take last timestep
        output = attn_output[:, -1, :]

        # Output projection
        features = self.output_proj(output)

        return features
