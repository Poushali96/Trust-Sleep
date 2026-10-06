
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

EVENT_CLASSES = ("no_event", "apnea", "hypopnea", "artifact_or_uncertain")
SUBTYPE_CLASSES = ("obstructive", "central", "mixed", "indeterminate")


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None


def reverse_gradient(x, strength: float):
    return ReverseGradient.apply(x, strength)


class EventDescriptorLayer(nn.Module):
    bases = (
        "mean", "std", "min", "max", "range", "slope_mean", "slope_std",
        "negative_drop", "positive_rise", "instability", "low_tail",
        "high_tail", "event_density",
    )

    def forward(self, x):
        mean = x.mean(1)
        std = x.std(1, unbiased=False)
        xmin, xmax = x.amin(1), x.amax(1)
        dx = x[:, 1:] - x[:, :-1]
        slope_std = dx.std(1, unbiased=False)
        return torch.cat([
            mean, std, xmin, xmax, xmax-xmin, dx.mean(1), slope_std,
            F.relu(-dx).mean(1), F.relu(dx).mean(1),
            slope_std / (std + 1e-6),
            F.relu(mean[:, None] - x).mean(1),
            F.relu(x - mean[:, None]).mean(1),
            (dx.abs() > slope_std[:, None]).float().mean(1),
        ], 1)


class TemporalEncoder(nn.Module):
    def __init__(self, channels=4, hidden=96, dropout=0.30):
        super().__init__()
        self.input = nn.Conv1d(channels, hidden, 7, padding=3, bias=False)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, 5, padding=2, bias=False),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden, hidden, 5, padding=2, bias=False),
                nn.BatchNorm1d(hidden),
            ) for _ in range(3)
        ])

    def forward(self, x):
        z = self.input(x.transpose(1, 2))
        for block in self.blocks:
            z = F.gelu(z + block(z))
        return z


@dataclass
class Output:
    binary_logit: torch.Tensor
    event_logits: torch.Tensor
    signal_subtype_logits: torch.Tensor
    context_prior_logits: torch.Tensor
    subtype_logits: torch.Tensor
    domain_logits: torch.Tensor
    localization_logits: torch.Tensor
    descriptors: torch.Tensor
    gates: torch.Tensor
    contributions: torch.Tensor
    signal_embedding: torch.Tensor
    context_embedding: torch.Tensor
    modality_embedding: torch.Tensor


class HierarchicalTrustSleepV4(nn.Module):
    """
    Safety-oriented hierarchy.

    * Binary event head is signal-only.
    * Signal subtype head is signal-only.
    * Context prior is bounded and exposed separately.
    * Localization head yields frame-level candidate-event probabilities.
    * Modality-presence mask prevents missing channels being interpreted as true zero.
    """
    def __init__(
        self,
        channels=4,
        metadata_dim=24,
        domains=2,
        dropout=0.30,
        max_context_weight=0.20,
    ):
        super().__init__()
        self.channels = channels
        self.max_context_weight = float(max_context_weight)
        descriptor_dim = channels * len(EventDescriptorLayer.bases)

        self.temporal = TemporalEncoder(channels, hidden=96, dropout=dropout)
        self.localization_head = nn.Conv1d(96, 1, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.descriptors = EventDescriptorLayer()
        self.gate = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, descriptor_dim),
            nn.Sigmoid(),
        )
        self.event_encoder = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 80),
            nn.GELU(),
        )

        self.modality_encoder = nn.Sequential(
            nn.Linear(channels, 16), nn.GELU(), nn.Linear(16, 16), nn.GELU()
        )
        self.signal_projection = nn.Sequential(
            nn.Linear(96 + 80 + 16, 192),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.context = nn.Sequential(
            nn.LayerNorm(metadata_dim),
            nn.Linear(metadata_dim, 96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, 48),
            nn.GELU(),
        )

        self.binary_head = nn.Linear(192, 1)
        self.event_head = nn.Linear(192, len(EVENT_CLASSES))
        self.signal_subtype_head = nn.Linear(192, len(SUBTYPE_CLASSES))
        self.context_prior_head = nn.Linear(48, len(SUBTYPE_CLASSES))
        self.domain_head = nn.Sequential(
            nn.Linear(192, 64), nn.GELU(), nn.Linear(64, domains)
        )

    def forward(
        self,
        signal,
        metadata,
        modality_present: Optional[torch.Tensor] = None,
        domain_strength: float = 0.0,
    ):
        if modality_present is None:
            modality_present = torch.ones(
                signal.shape[0], signal.shape[2], device=signal.device
            )
        masked_signal = signal * modality_present[:, None, :]

        temporal = self.temporal(masked_signal)
        localization_logits = self.localization_head(temporal).squeeze(1)
        temporal_global = self.pool(temporal).squeeze(-1)

        descriptors = self.descriptors(masked_signal)
        gates = self.gate(descriptors)
        contributions = descriptors * gates
        event_embedding = self.event_encoder(contributions)
        modality_embedding = self.modality_encoder(modality_present)
        signal_embedding = self.signal_projection(
            torch.cat([temporal_global, event_embedding, modality_embedding], 1)
        )
        context_embedding = self.context(metadata)

        signal_subtype_logits = self.signal_subtype_head(signal_embedding)
        context_prior_logits = torch.tanh(self.context_prior_head(context_embedding))
        subtype_logits = (
            signal_subtype_logits
            + self.max_context_weight * context_prior_logits
        )

        return Output(
            binary_logit=self.binary_head(signal_embedding).squeeze(-1),
            event_logits=self.event_head(signal_embedding),
            signal_subtype_logits=signal_subtype_logits,
            context_prior_logits=context_prior_logits,
            subtype_logits=subtype_logits,
            domain_logits=self.domain_head(
                reverse_gradient(signal_embedding, domain_strength)
            ),
            localization_logits=localization_logits,
            descriptors=descriptors,
            gates=gates,
            contributions=contributions,
            signal_embedding=signal_embedding,
            context_embedding=context_embedding,
            modality_embedding=modality_embedding,
        )
