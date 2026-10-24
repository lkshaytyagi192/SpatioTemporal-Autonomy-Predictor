"""
Deep Learning Architecture: Spatio-Temporal Seq2Seq LSTM with Temporal Attention.
Designed for non-linear trajectory forecasting in autonomous robotics and multi-agent systems.

Architecture Overview:
1. Encoder: Recurrent LSTM extracting temporal motion dynamics from observed timesteps (t=1..8).
2. Temporal Attention Mechanism: Dynamic alignment scoring that highlights critical kinematic phases
   (e.g., turning points, braking, acceleration).
3. Decoder: Autoregressive LSTM with context projection generating future coordinates (t=9..20).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TemporalAttention(nn.Module):
    """
    Bahdanau-style Additive Attention to compute alignment between decoder hidden state
    and all encoder temporal representations.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_enc = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: Current decoder hidden state (Batch_Size, Hidden_Dim)
            encoder_outputs: All encoder outputs (Batch_Size, Obs_Len, Hidden_Dim)
        Returns:
            context_vector: (Batch_Size, Hidden_Dim)
            attn_weights: (Batch_Size, Obs_Len)
        """
        # Expand decoder_hidden across temporal dimension: (Batch_Size, 1, Hidden_Dim)
        dec_proj = self.W_dec(decoder_hidden).unsqueeze(1)
        enc_proj = self.W_enc(encoder_outputs)  # (Batch_Size, Obs_Len, Hidden_Dim)

        # Energy scores: tanh(W_enc * h_enc + W_dec * h_dec)
        energy = torch.tanh(enc_proj + dec_proj)  # (Batch_Size, Obs_Len, Hidden_Dim)
        scores = self.v(energy).squeeze(-1)       # (Batch_Size, Obs_Len)

        # Attention weights distribution
        attn_weights = F.softmax(scores, dim=-1)   # (Batch_Size, Obs_Len)

        # Context vector as weighted sum: (Batch_Size, Hidden_Dim)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context, attn_weights


class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, embedding_dim: int = 32, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, obs_traj: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            obs_traj: (Batch_Size, Obs_Len, 2)
        Returns:
            outputs: (Batch_Size, Obs_Len, Hidden_Dim)
            (hidden, cell): Last recurrent state tuples
        """
        embedded = self.embedding(obs_traj)
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)


class TrajectoryDecoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 2,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        pred_len: int = 12
    ):
        super().__init__()
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Sequential(
            nn.Linear(output_dim, embedding_dim),
            nn.ReLU()
        )
        self.attention = TemporalAttention(hidden_dim)
        # Input to LSTM is concatenated [embedded_coord, context_vector]
        self.lstm_cell = nn.LSTMCell(embedding_dim + hidden_dim, hidden_dim)
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        init_state: Tuple[torch.Tensor, torch.Tensor],
        last_obs_coord: torch.Tensor,
        target_traj: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive multi-step prediction.
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        h_t = init_state[0][-1]  # Top layer hidden state: (Batch, Hidden_Dim)
        c_t = init_state[1][-1]  # Top layer cell state: (Batch, Hidden_Dim)

        current_input = last_obs_coord  # (Batch, 2)
        predictions = []
        all_attentions = []

        for t in range(self.pred_len):
            embedded = self.embedding(current_input)  # (Batch, embedding_dim)
            context, attn_w = self.attention(h_t, encoder_outputs)
            all_attentions.append(attn_w)

            lstm_in = torch.cat([embedded, context], dim=-1)  # (Batch, embedding_dim + hidden_dim)
            h_t, c_t = self.lstm_cell(lstm_in, (h_t, c_t))

            step_pred = self.fc_out(h_t)  # (Batch, 2)
            predictions.append(step_pred)

            # Scheduled sampling / Teacher forcing during training
            if target_traj is not None and torch.rand(1).item() < teacher_forcing_ratio:
                current_input = target_traj[:, t, :]
            else:
                current_input = step_pred

        pred_traj = torch.stack(predictions, dim=1)        # (Batch, Pred_Len, 2)
        attn_history = torch.stack(all_attentions, dim=1)  # (Batch, Pred_Len, Obs_Len)

        return pred_traj, attn_history


class Seq2SeqAttentionTrajectory(nn.Module):
    """
    Complete end-to-end trajectory forecasting model.
    """
    def __init__(
        self,
        coord_dim: int = 2,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        obs_len: int = 8,
        pred_len: int = 12
    ):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.encoder = TrajectoryEncoder(input_dim=coord_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.decoder = TrajectoryDecoder(output_dim=coord_dim, embedding_dim=embedding_dim, hidden_dim=hidden_dim, pred_len=pred_len)

    def forward(
        self,
        obs_traj: torch.Tensor,
        target_traj: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_outputs, (h_n, c_n) = self.encoder(obs_traj)
        last_obs_coord = obs_traj[:, -1, :]
        pred_traj, attn_history = self.decoder(
            encoder_outputs=encoder_outputs,
            init_state=(h_n, c_n),
            last_obs_coord=last_obs_coord,
            target_traj=target_traj,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        return pred_traj, attn_history
