import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Basic Conv Blocks ----------

class ConvBlock1D(nn.Module):
    """Simple Conv1D -> Norm -> Activation block.

    Optionally downsamples via stride > 1 and supports multiple
    normalization types (batch, instance, group, or none).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int | None = None,
        norm: str = "bn",  # "bn", "in", "gn", or "none"
        activation: str | None = "gelu",
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2  # keep length (if stride=1)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Normalization
        norm = (norm or "none").lower()
        if norm == "bn":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == "in":
            self.norm = nn.InstanceNorm1d(out_channels, affine=True)
        elif norm == "gn":
            # Choose a valid number of groups (<=32) that divides out_channels
            num_groups = min(32, out_channels)
            while out_channels % num_groups != 0 and num_groups > 1:
                num_groups //= 2
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        else:
            self.norm = nn.Identity()

        # Activation
        if activation is None:
            self.act = nn.Identity()
        else:
            act = activation.lower()
            if act == "relu":
                self.act = nn.ReLU()
            elif act == "gelu":
                self.act = nn.GELU()
            elif act == "elu":
                self.act = nn.ELU()
            else:
                self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DeconvBlock1D(nn.Module):
    """Mirror of ConvBlock1D for the decoder using ConvTranspose1d.

    Supports the same normalization and activation options.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int | None = None,
        output_padding: int = 0,
        norm: str = "bn",  # "bn", "in", "gn", or "none"
        activation: str | None = "gelu",
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.deconv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )

        # Normalization
        norm = (norm or "none").lower()
        if norm == "bn":
            self.norm = nn.BatchNorm1d(out_channels)
        elif norm == "in":
            self.norm = nn.InstanceNorm1d(out_channels, affine=True)
        elif norm == "gn":
            num_groups = min(32, out_channels)
            while out_channels % num_groups != 0 and num_groups > 1:
                num_groups //= 2
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        else:
            self.norm = nn.Identity()

        # Activation
        if activation is None:
            self.act = nn.Identity()
        else:
            act = activation.lower()
            if act == "relu":
                self.act = nn.ReLU()
            elif act == "gelu":
                self.act = nn.GELU()
            elif act == "elu":
                self.act = nn.ELU()
            else:
                self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# ---------- Encoder ----------

class Conv1DEncoder(nn.Module):
    """Conv1D encoder that maps (B, 1, L) -> (B, d_model, L_latent) + (B, d_global).

    The (B, d_model, L_latent) sequence is where you can later insert
    Mamba / Transformer / attention blocks.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_downsamples: int = 3,
        d_model: int = 256,
        d_global: int = 256,
        norm: str = "bn",
    ):
        super().__init__()

        layers = []
        c_in = in_channels
        c = base_channels

        # Progressive downsampling via stride=2
        for i in range(num_downsamples):
            stride = 2
            layers.append(
                ConvBlock1D(
                    in_channels=c_in,
                    out_channels=c,
                    kernel_size=7 if i == 0 else 5,
                    stride=stride,
                    norm=norm,
                    activation="gelu",
                )
            )
            c_in = c
            c *= 2

        # Final conv to project to d_model channels (no further downsampling)
        layers.append(
            ConvBlock1D(
                in_channels=c_in,
                out_channels=d_model,
                kernel_size=3,
                stride=1,
                norm=norm,
                activation="gelu",
            )
        )

        self.conv_stack = nn.Sequential(*layers)

        # Global pooling + linear to get global latent (B, d_global)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_proj = nn.Linear(d_model, d_global)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input spectra.

        Args:
            x: (B, 1, L)

        Returns:
            z_seq:    (B, d_model, L_latent)  sequence latent
            z_global: (B, d_global)           global latent
        """
        # Convolutional feature extraction + downsampling
        z_seq = self.conv_stack(x)  # (B, d_model, L_latent)

        # Global latent
        pooled = self.global_pool(z_seq)  # (B, d_model, 1)
        pooled = pooled.squeeze(-1)       # (B, d_model)
        z_global = self.global_proj(pooled)  # (B, d_global)

        return z_seq, z_global


# ---------- Decoder ----------

class Conv1DDecoder(nn.Module):
    """Conv1D decoder that reconstructs the spectrum from:

      - z_seq:    (B, d_model, L_latent)
      - z_global: (B, d_global)

    We broadcast the global vector along the latent length
    and concatenate it with z_seq as extra channels.
    """

    def __init__(
        self,
        out_channels: int = 1,
        d_model: int = 256,
        d_global: int = 256,
        base_channels: int = 64,
        num_upsamples: int = 3,
        norm: str = "bn",
    ):
        super().__init__()

        # Combine global + local latent
        self.merge_proj = nn.Conv1d(
            in_channels=d_model + d_global,
            out_channels=d_model,
            kernel_size=1,
        )

        # Progressive upsampling back to original length
        layers = []
        c_in = d_model
        c = d_model // 2

        # We mirror num_upsamples with stride=2
        for _ in range(num_upsamples):
            stride = 2
            # output_padding=1 exactly doubles the length for
            # kernel=5, padding=2, stride=2.
            layers.append(
                DeconvBlock1D(
                    in_channels=c_in,
                    out_channels=c,
                    kernel_size=5,
                    stride=stride,
                    output_padding=1,
                    norm=norm,
                    activation="gelu",
                )
            )
            c_in = c
            c = max(base_channels, c // 2)

        # Final conv to map to 1 channel spectrum, no norm or activation
        layers.append(
            ConvBlock1D(
                in_channels=c_in,
                out_channels=out_channels,
                kernel_size=5,
                stride=1,
                norm="none",
                activation=None,  # no nonlinearity; let loss decide
            )
        )

        self.deconv_stack = nn.Sequential(*layers)

    def forward(
        self,
        z_seq: torch.Tensor,
        z_global: torch.Tensor,
        target_length: int | None = None,
    ) -> torch.Tensor:
        """Decode latent representation back to spectrum.

        Args:
            z_seq: (B, d_model, L_latent)
            z_global: (B, d_global)
            target_length: if provided, crop or interpolate to match this length.

        Returns:
            x_recon: (B, out_channels, L_out)
        """
        B, d_model, L_latent = z_seq.shape

        # Broadcast z_global across positions and concat
        z_global_expanded = z_global.unsqueeze(-1).expand(B, z_global.shape[1], L_latent)
        z = torch.cat([z_seq, z_global_expanded], dim=1)  # (B, d_model + d_global, L_latent)

        # Merge channels back to d_model
        z = self.merge_proj(z)  # (B, d_model, L_latent)

        # Upsample using deconv stack
        x_recon = self.deconv_stack(z)  # (B, out_channels, L_out_approx)

        # Optional: force exact length match (e.g., for training)
        if target_length is not None:
            L_out = x_recon.shape[-1]
            if L_out > target_length:
                # center crop
                start = (L_out - target_length) // 2
                x_recon = x_recon[..., start : start + target_length]
            elif L_out < target_length:
                # interpolate to match
                x_recon = F.interpolate(
                    x_recon,
                    size=target_length,
                    mode="linear",
                    align_corners=False,
                )

        return x_recon


# ---------- Latent blocks ----------

class SeqLatentWrapper(nn.Module):
    """Wrapper to apply a (B, L, d_model) sequence model on (B, d_model, L).

    This lets you plug in Mamba / Transformer blocks that expect
    a (batch, length, dim) layout without changing the rest of the code.
    """

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, z_seq: torch.Tensor) -> torch.Tensor:
        # z_seq: (B, d_model, L)
        z_seq = z_seq.permute(0, 2, 1)  # (B, L, d_model)
        z_seq = self.core(z_seq)        # user-defined core expects (B, L, d_model)
        z_seq = z_seq.permute(0, 2, 1)  # back to (B, d_model, L)
        return z_seq


class SimpleSeqMLP(nn.Module):
    """Small residual MLP block over the sequence latent.

    Expects input as (B, L, d_model) and returns the same shape.
    This is a lightweight stand-in for a more complex sequence model
    like Mamba or a Transformer encoder.
    """

    def __init__(self, d_model: int, hidden_factor: float = 2.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(d_model * hidden_factor)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        return x + residual


# ---------- Full Autoencoder Wrapper ----------

class Conv1DAutoencoder(nn.Module):
    """Full Conv1D spectrum autoencoder with explicit latent space.

    Later, you can:
      - Replace the SimpleSeqMLP with Mamba/Transformer blocks via `seq_latent_block`.
      - Use z_global as a condition or "style"/context vector.
      - Use `baseline` output to model background / smooth components.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_downsamples: int = 3,
        d_model: int = 256,
        d_global: int = 256,
        norm: str = "bn",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model

        self.encoder = Conv1DEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_downsamples=num_downsamples,
            d_model=d_model,
            d_global=d_global,
            norm=norm,
        )
        self.decoder = Conv1DDecoder(
            out_channels=in_channels,
            d_model=d_model,
            d_global=d_global,
            base_channels=base_channels,
            num_upsamples=num_downsamples,
            norm=norm,
        )

        # Baseline / background head on the sequence latent
        # (operates at latent length, then upsampled to input length).
        self.baseline_head = nn.Conv1d(
            in_channels=d_model,
            out_channels=in_channels,
            kernel_size=9,
            padding=4,
        )

        # Sequence latent block: small MLP over (B, L, d_model)
        self.seq_latent_block = SeqLatentWrapper(SimpleSeqMLP(d_model=d_model))

        # Global latent block: simple MLP over z_global
        self.global_latent_block = nn.Sequential(
            nn.LayerNorm(d_global),
            nn.Linear(d_global, d_global),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Args:
            x: (B, in_channels, L_in)

        Returns:
            x_recon:  (B, in_channels, L_in)  reconstructed spectrum
            baseline: (B, in_channels, L_in)  smooth baseline/background estimate
            z_seq:    (B, d_model, L_latent)  sequence latent
            z_global: (B, d_global)           global latent
        """
        B, C, L_in = x.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}."

        # Encode
        z_seq, z_global = self.encoder(x)

        # Latent processing (sequence + global)
        z_seq = self.seq_latent_block(z_seq)        # apply sequence MLP
        z_global = self.global_latent_block(z_global)

        # Baseline / background prediction at latent resolution
        baseline_latent = self.baseline_head(z_seq)  # (B, in_channels, L_latent)
        baseline = F.interpolate(
            baseline_latent,
            size=L_in,
            mode="linear",
            align_corners=False,
        )  # (B, in_channels, L_in)

        # Decode (ensure reconstruction has same length as input)
        x_recon = self.decoder(z_seq, z_global, target_length=L_in)

        return x_recon, baseline, z_seq, z_global


# ---------- Simple training loop for smoke testing ----------

def smoothness_loss(signal: torch.Tensor) -> torch.Tensor:
    """Encourage smooth baselines via a finite-difference penalty.

    Args:
        signal: (B, C, L)
    Returns:
        scalar loss
    """
    diff = signal[..., 1:] - signal[..., :-1]
    return (diff ** 2).mean()


def main_train_example():
    """Tiny training loop on random data for sanity-checking the model.

    This is not meant for real NMR training, just to verify that the
    forward/backward passes work and losses go down.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = 4
    L = 32768  # ~32k points for NMR spectrum

    model = Conv1DAutoencoder(
        in_channels=1,
        base_channels=64,
        num_downsamples=3,
        d_model=256,
        d_global=256,
        norm="bn",  # try "in" or "gn" if batch sizes are tiny
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_steps = 10  # keep this small for a quick smoke test
    model.train()

    for step in range(1, num_steps + 1):
        # Synthetic input: you can replace this with real spectra.
        x = torch.randn(B, 1, L, device=device)

        optimizer.zero_grad()

        x_recon, baseline, z_seq, z_global = model(x)

        # Reconstruction loss
        loss_recon = F.mse_loss(x_recon, x)

        # Smoothness penalty on baseline (encourage slowly varying background)
        loss_baseline_smooth = smoothness_loss(baseline)

        # Total loss (weights are arbitrary for this demo)
        loss = loss_recon + 0.1 * loss_baseline_smooth

        loss.backward()
        optimizer.step()

        if step == 1 or step % 2 == 0:
            print(
                f"Step {step:02d} | "
                f"loss={loss.item():.6f} "
                f"(recon={loss_recon.item():.6f}, baseline_smooth={loss_baseline_smooth.item():.6f})"
            )


if __name__ == "__main__":
    # Quick forward-shape sanity check
    B = 4
    L = 32768
    x = torch.randn(B, 1, L)

    model = Conv1DAutoencoder(
        in_channels=1,
        base_channels=64,
        num_downsamples=3,
        d_model=256,
        d_global=256,
        norm="bn",
    )

    with torch.no_grad():
        x_recon, baseline, z_seq, z_global = model(x)
    print("[Sanity check]")
    print("Input shape:    ", x.shape)
    print("Reconstructed:  ", x_recon.shape)
    print("Baseline:       ", baseline.shape)
    print("Seq latent:     ", z_seq.shape)
    print("Global latent:  ", z_global.shape)

    # Run a tiny training loop on random data
    print("\n[Running tiny training loop on random data]\n")
    main_train_example()
