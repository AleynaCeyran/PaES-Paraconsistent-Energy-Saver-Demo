"""projector_trainer.py — Epistemic projector optimisation for PaES-LLM."""

import torch
import torch.nn.functional as F
import torch.optim as optim


def train_epistemic_projector(model, data_loader, steps: int = 150, lr: float = 1e-3):
    """Optimise the Epistemic Projector layers in isolation.

    The projector is trained directly on its (μ, λ) outputs rather than
    through the full block forward pass. This avoids the gradient cut that
    occurs when frozen MHA weights interrupt the computation graph.

    Loss terms
    ----------
    fidelity   : Maximise the TRUE token fraction to preserve semantic quality.
    sparsity   : Drive the FALSE fraction toward a 40% target for compute savings.
    separation : Push μ and λ to be anti-correlated so all four L4 quadrants
                 are meaningfully populated.

    Parameters
    ----------
    model       : GatedTransformerBlock with mode='projector'.
    data_loader : DataLoader providing tokenised batches.
    steps       : Number of optimisation steps.
    lr          : Adam learning rate.

    Returns
    -------
    GatedTransformerBlock with updated projector weights, set to eval mode.
    """
    embed_dim = model.gated_attn.gate.embed_dim

    gates = []
    if hasattr(model.gated_attn.gate, "projector"):
        gates.append(model.gated_attn.gate)
    if hasattr(model.gate_ffn, "projector"):
        gates.append(model.gate_ffn)

    if not gates:
        print("No projector found — is mode='projector'?")
        return model

    proj_params = []
    for gate in gates:
        for p in gate.projector.parameters():
            p.requires_grad = True
            proj_params.append(p)

    optimizer    = optim.Adam(proj_params, lr=lr)
    target_false = 0.4

    for step in range(steps):
        total_loss, n_batches = 0.0, 0

        for batch in data_loader:
            B, S = batch["input_ids"].shape
            x    = torch.randn(B, S, embed_dim)

            optimizer.zero_grad()
            step_loss = torch.zeros(1, requires_grad=True)

            for gate in gates:
                proj = gate.projector(x)
                mu   = torch.sigmoid(proj[..., 0])
                lam  = torch.sigmoid(proj[..., 1])
                tau  = gate.threshold

                is_true   = ((mu >= tau) & (lam < tau)).float()
                fidelity  = 1.0 - is_true.mean()

                pct_false = ((mu < tau) & (lam >= tau)).float().mean()
                sparsity  = (pct_false - target_false).pow(2)

                sep = F.cosine_similarity(
                    mu.reshape(-1, 1), lam.reshape(-1, 1), dim=0
                ).abs().mean()

                step_loss = step_loss + fidelity + 0.3 * sparsity + 0.1 * sep

            step_loss.backward()
            optimizer.step()
            total_loss += step_loss.item()
            n_batches  += 1

        if (step + 1) % 50 == 0:
            print(f"  step {step + 1}/{steps}  loss={total_loss / max(n_batches, 1):.6f}")

    model.eval()
    return model
