"""
Modular Expert Implementation for nanoMoE Integration.

Refactors nanoMoE's batched MLPExperts into nn.ModuleList for surgical
lifecycle operations. Trades batched efficiency for modularity.

This is the dev target version - optimize later after lifecycle proven.
"""

import torch
import torch.nn as nn


class ModularMLP(nn.Module):
    """
    Single MLP expert module (matches nanoMoE MLP architecture).

    Used as independent expert in ModuleList, enabling clean add/remove.
    """

    def __init__(self, n_embd: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class ModularMLPExperts(nn.Module):
    """
    Modular expert implementation using nn.ModuleList.

    Drop-in replacement for nanoMoE's batched MLPExperts, but with
    independent expert modules for lifecycle surgery.

    Trades efficiency for modularity - correct choice for dev target.
    """

    def __init__(self, n_exp: int, n_embd: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.n_exp = n_exp
        self.n_embd = n_embd

        # ModuleList of independent experts (surgery-friendly)
        self.experts = nn.ModuleList([
            ModularMLP(n_embd, bias=bias, dropout=dropout)
            for _ in range(n_exp)
        ])

    def forward(self, x):
        """
        Forward pass matching nanoMoE's batched interface.

        Args:
            x: [n_exp, exp_capacity, n_embd] - batched inputs

        Returns:
            [n_exp, exp_capacity, n_embd] - batched outputs
        """
        # Process each expert independently (loop instead of bmm)
        outputs = []
        for i, expert in enumerate(self.experts):
            expert_input = x[i]  # [exp_capacity, n_embd]
            expert_output = expert(expert_input)  # [exp_capacity, n_embd]
            outputs.append(expert_output)

        # Stack back into batch format
        return torch.stack(outputs, dim=0)  # [n_exp, exp_capacity, n_embd]

    def add_expert(self, expert: ModularMLP):
        """Add a new expert to the ModuleList."""
        self.experts.append(expert)
        self.n_exp += 1

    def remove_expert(self, expert_id: int):
        """Remove an expert from the ModuleList."""
        del self.experts[expert_id]
        self.n_exp -= 1

    def get_expert(self, expert_id: int) -> ModularMLP:
        """Get a specific expert module."""
        return self.experts[expert_id]


class ModularMOELayer(nn.Module):
    """
    MOELayer using modular experts.

    Drop-in replacement for nanoMoE's MOELayer with surgery-friendly
    expert management.
    """

    def __init__(self, router, n_exp: int, n_embd: int, bias: bool = True, dropout: float = 0.0):
        super().__init__()
        self.router = router
        self.experts = ModularMLPExperts(n_exp, n_embd, bias=bias, dropout=dropout)

    def forward(self, x: torch.Tensor):
        """Forward pass matching nanoMoE's MOELayer interface."""
        B, T, n_embd = x.size()
        num_tokens = (B * T)

        # Pass tokens through router
        used_capacity, exp_weight, exp_mask = self.router(x)

        # Flatten input
        x = x.view(num_tokens, n_embd)

        # Reshape tokens into batches for each expert
        # [n_exp, exp_capacity, B*T] * [B*T, n_embd] -> [n_exp, exp_capacity, n_embd]
        exp_batches = exp_mask.permute(1, 2, 0).type_as(x) @ x

        # Compute expert outputs (modular instead of batched)
        exp_out = self.experts(exp_batches)  # [n_exp, exp_capacity, n_embd]

        # Aggregate expert outputs based on router weights
        exp_weight = exp_weight.view(num_tokens, -1)  # [B*T, n_exp * exp_capacity]
        exp_out = exp_out.view(-1, n_embd)  # [n_exp * exp_capacity, n_embd]
        output = exp_weight @ exp_out  # [B*T, n_embd]

        # Resize output
        return output.view(B, T, n_embd)


def convert_batched_to_modular(batched_experts, config):
    """
    Convert nanoMoE's batched MLPExperts to modular version.

    Extracts parameters from batched tensors and creates independent modules.
    Useful for loading pretrained nanoMoE checkpoints.

    Args:
        batched_experts: nanoMoE's MLPExperts instance
        config: nanoMoE config

    Returns:
        ModularMLPExperts with same parameters
    """
    modular = ModularMLPExperts(
        n_exp=config.n_exp,
        n_embd=config.n_embd,
        bias=config.bias,
        dropout=config.dropout,
    )

    # Copy parameters from batched to modular
    with torch.no_grad():
        for i in range(config.n_exp):
            # Copy c_fc weights and bias
            modular.experts[i].c_fc.weight.copy_(
                batched_experts.c_fc[i].permute(1, 0)  # [4*n_embd, n_embd]
            )
            if config.bias:
                modular.experts[i].c_fc.bias.copy_(
                    batched_experts.fc_bias[i].squeeze(0)
                )

            # Copy c_proj weights and bias
            modular.experts[i].c_proj.weight.copy_(
                batched_experts.c_proj[i].permute(1, 0)  # [n_embd, 4*n_embd]
            )
            if config.bias:
                modular.experts[i].c_proj.bias.copy_(
                    batched_experts.proj_bias[i].squeeze(0)
                )

    return modular


def create_blank_expert(n_embd: int, bias: bool = True, dropout: float = 0.0) -> ModularMLP:
    """
    Create a blank expert with random initialization.

    For blank_spawn_with_probation strategy - expert learns from scratch.
    """
    return ModularMLP(n_embd, bias=bias, dropout=dropout)


def create_clone_expert(parent: ModularMLP, perturbation_scale: float = 0.01) -> ModularMLP:
    """
    Create a cloned expert with small perturbation.

    For clone_seeded strategy - expert inherits parent knowledge.
    """
    clone = ModularMLP(
        n_embd=parent.c_fc.in_features,
        bias=parent.c_fc.bias is not None,
        dropout=parent.dropout.p,
    )

    # Clone parameters with perturbation
    with torch.no_grad():
        for clone_param, parent_param in zip(clone.parameters(), parent.parameters()):
            noise = torch.randn_like(parent_param) * perturbation_scale
            clone_param.copy_(parent_param + noise)

    return clone
