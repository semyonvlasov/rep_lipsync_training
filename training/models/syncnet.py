"""
SyncNet: Audio-visual sync discriminator.

Learns a joint embedding where lip region frames and mel spectrogram
are close when in sync and far apart when out of sync.
Trained with contrastive loss on positive (aligned) and negative (shifted) pairs.

Reference: "Out of time: automated lip sync in the wild" (Chung & Zisserman, 2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, kernel, stride, padding, residual=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, kernel, stride, padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, x):
        out = self.conv(x)
        if self.residual:
            out = out + x
        return self.act(out)


class SyncNet(nn.Module):
    """
    Two-stream network:
      - Visual stream: T consecutive lower-face crops → 512-d embedding
      - Audio stream: T×mel spectrogram → 512-d embedding
    Trained with cosine contrastive loss.
    """

    def __init__(self, T=5, audio_temporal_kernels=None):
        super().__init__()
        self.T = T
        if audio_temporal_kernels is None:
            audio_temporal_kernels = [3] * 8
        if len(audio_temporal_kernels) != 8:
            raise ValueError(
                f"audio_temporal_kernels must have 8 entries, got {len(audio_temporal_kernels)}"
            )
        for kernel_t in audio_temporal_kernels:
            if int(kernel_t) < 1 or int(kernel_t) % 2 == 0:
                raise ValueError(
                    f"audio_temporal_kernels entries must be positive odd ints, got {audio_temporal_kernels}"
                )

        def audio_kernel(kernel_t):
            return (3, int(kernel_t))

        def audio_padding(kernel_t):
            return (1, int(kernel_t) // 2)

        # Visual encoder: input (B, T*3, 48, 96) — lower half of face, T frames stacked
        # 48 = img_size/2 height (lower half), 96 = width
        # We keep it resolution-agnostic by adaptive pooling at the end
        self.visual_encoder = nn.Sequential(
            ConvBlock(T * 3, 64, 7, 1, 3),

            ConvBlock(64, 128, 5, 2, 2),
            ConvBlock(128, 128, 3, 1, 1, residual=True),

            ConvBlock(128, 256, 3, 2, 1),
            ConvBlock(256, 256, 3, 1, 1, residual=True),

            ConvBlock(256, 512, 3, 2, 1),
            ConvBlock(512, 512, 3, 1, 1, residual=True),

            ConvBlock(512, 512, 3, 2, 1),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
        )

        # Audio encoder: input (B, 1, 80, T*mel_step)
        self.audio_encoder = nn.Sequential(
            ConvBlock(1, 64, audio_kernel(audio_temporal_kernels[0]), 1, audio_padding(audio_temporal_kernels[0])),

            ConvBlock(64, 128, audio_kernel(audio_temporal_kernels[1]), 2, audio_padding(audio_temporal_kernels[1])),
            ConvBlock(128, 128, audio_kernel(audio_temporal_kernels[2]), 1, audio_padding(audio_temporal_kernels[2]), residual=True),

            ConvBlock(128, 256, audio_kernel(audio_temporal_kernels[3]), 2, audio_padding(audio_temporal_kernels[3])),
            ConvBlock(256, 256, audio_kernel(audio_temporal_kernels[4]), 1, audio_padding(audio_temporal_kernels[4]), residual=True),

            ConvBlock(256, 512, audio_kernel(audio_temporal_kernels[5]), 2, audio_padding(audio_temporal_kernels[5])),
            ConvBlock(512, 512, audio_kernel(audio_temporal_kernels[6]), 1, audio_padding(audio_temporal_kernels[6]), residual=True),

            ConvBlock(512, 512, audio_kernel(audio_temporal_kernels[7]), 2, audio_padding(audio_temporal_kernels[7])),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
        )

    def forward_visual(self, x):
        """x: (B, T*3, H/2, W) lower-face frames stacked along channels."""
        return F.normalize(self.visual_encoder(x), dim=-1)

    def forward_audio(self, x):
        """x: (B, 1, 80, T*mel_step) mel spectrogram."""
        return F.normalize(self.audio_encoder(x), dim=-1)

    def forward(self, visual, audio):
        v_emb = self.forward_visual(visual)
        a_emb = self.forward_audio(audio)
        return v_emb, a_emb

    @staticmethod
    def cosine_loss(v_emb, a_emb, labels):
        """
        Contrastive cosine loss.
        labels: 1 for in-sync, 0 for out-of-sync.
        """
        cos_sim = F.cosine_similarity(v_emb, a_emb, dim=-1)
        # BCE loss: sync pairs should have cos_sim ≈ 1, out-of-sync ≈ 0
        return F.binary_cross_entropy_with_logits(cos_sim * 5.0, labels.float())
