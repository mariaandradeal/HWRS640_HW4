"""
Model definitions for streamflow prediction.
"""

from typing import Tuple

import torch
import torch.nn as nn


class LSTMStreamflowModel(nn.Module):
    """
    LSTM-based streamflow prediction model with static attribute fusion.

    Inputs
    ------
    x_seq : torch.Tensor
        Shape (batch, seq_len, num_dynamic_features)
    x_static : torch.Tensor
        Shape (batch, num_static_features)

    Output
    ------
    y_pred : torch.Tensor
        Shape (batch, 1)
    """

    def __init__(
        self,
        num_dynamic_features: int,
        num_static_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        static_hidden_size: int = 32,
        fusion_hidden_size: int = 32,
    ) -> None:
        super().__init__()

        # LSTM encoder for dynamic meteorological sequence
        self.lstm = nn.LSTM(
            input_size=num_dynamic_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Small MLP for static basin attributes
        self.static_encoder = nn.Sequential(
            nn.Linear(num_static_features, static_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion head combines LSTM representation + static representation
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size + static_hidden_size, fusion_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_size, 1),
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x_seq : torch.Tensor
            Shape (batch, seq_len, num_dynamic_features)
        x_static : torch.Tensor
            Shape (batch, num_static_features)

        Returns
        -------
        torch.Tensor
            Predicted streamflow, shape (batch, 1)
        """
        # LSTM output:
        # lstm_out shape = (batch, seq_len, hidden_size)
        # h_n shape      = (num_layers, batch, hidden_size)
        _, (h_n, _) = self.lstm(x_seq)

        # Use final hidden state from the last LSTM layer
        seq_repr = h_n[-1]  # shape: (batch, hidden_size)

        # Encode static attributes
        static_repr = self.static_encoder(x_static)  # shape: (batch, static_hidden_size)

        # Concatenate sequence + static representations
        fused = torch.cat([seq_repr, static_repr], dim=1)

        # Predict streamflow
        y_pred = self.regressor(fused)  # shape: (batch, 1)

        return y_pred


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(
    num_dynamic_features: int,
    num_static_features: int,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.1,
    static_hidden_size: int = 32,
    fusion_hidden_size: int = 32,
) -> LSTMStreamflowModel:
    """
    Convenience factory function to create the model.
    """
    return LSTMStreamflowModel(
        num_dynamic_features=num_dynamic_features,
        num_static_features=num_static_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        static_hidden_size=static_hidden_size,
        fusion_hidden_size=fusion_hidden_size,
    )


def run_model_sanity_check() -> Tuple[torch.Tensor, int]:
    """
    Simple local test for model shape correctness.

    Returns
    -------
    y : torch.Tensor
        Output tensor from a dummy forward pass.
    n_params : int
        Number of trainable parameters.
    """
    batch_size = 8
    seq_len = 60
    num_dynamic_features = 5
    num_static_features = 16

    model = create_model(
        num_dynamic_features=num_dynamic_features,
        num_static_features=num_static_features,
        hidden_size=64,
        num_layers=1,
        dropout=0.1,
        static_hidden_size=32,
        fusion_hidden_size=32,
    )

    x_seq = torch.randn(batch_size, seq_len, num_dynamic_features)
    x_static = torch.randn(batch_size, num_static_features)

    y = model(x_seq, x_static)
    n_params = count_trainable_parameters(model)

    return y, n_params
