"""
Official-style Wav2Lip HQ visual quality discriminator.

This is a local port of the open-source Wav2Lip_disc_qual model so that the
training pipeline can stay self-contained inside rep_lipsync_training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonNormConv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size, stride, padding)
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class OfficialQualityDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(NonNormConv2d(3, 32, kernel_size=7, stride=1, padding=3)),  # 48x96

            nn.Sequential(
                NonNormConv2d(32, 64, kernel_size=5, stride=(1, 2), padding=2),  # 48x48
                NonNormConv2d(64, 64, kernel_size=5, stride=1, padding=2),
            ),

            nn.Sequential(
                NonNormConv2d(64, 128, kernel_size=5, stride=2, padding=2),  # 24x24
                NonNormConv2d(128, 128, kernel_size=5, stride=1, padding=2),
            ),

            nn.Sequential(
                NonNormConv2d(128, 256, kernel_size=5, stride=2, padding=2),  # 12x12
                NonNormConv2d(256, 256, kernel_size=5, stride=1, padding=2),
            ),

            nn.Sequential(
                NonNormConv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 6x6
                NonNormConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            ),

            nn.Sequential(
                NonNormConv2d(512, 512, kernel_size=3, stride=2, padding=1),  # 3x3
                NonNormConv2d(512, 512, kernel_size=3, stride=1, padding=1),
            ),

            nn.Sequential(
                NonNormConv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1x1
                NonNormConv2d(512, 512, kernel_size=1, stride=1, padding=0),
            ),
        ])

        self.binary_pred = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    @staticmethod
    def get_lower_half(face_sequences):
        if face_sequences.dim() == 5:
            return face_sequences[:, :, :, face_sequences.size(3) // 2:, :]
        return face_sequences[:, :, face_sequences.size(2) // 2:, :]

    @staticmethod
    def to_2d(face_sequences):
        if face_sequences.dim() == 5:
            batch_size = face_sequences.size(0)
            return torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0), batch_size
        return face_sequences, face_sequences.size(0)

    def perceptual_forward(self, false_face_sequences):
        false_face_sequences, _ = self.to_2d(false_face_sequences)
        false_face_sequences = self.get_lower_half(false_face_sequences)

        x = false_face_sequences
        for block in self.face_encoder_blocks:
            x = block(x)

        pred = self.binary_pred(x).view(len(x), -1)
        targets = torch.ones((len(pred), 1), device=pred.device, dtype=pred.dtype)
        return F.binary_cross_entropy(pred, targets)

    def forward(self, face_sequences):
        x, _ = self.to_2d(face_sequences)
        x = self.get_lower_half(x)

        for block in self.face_encoder_blocks:
            x = block(x)

        return self.binary_pred(x).view(len(x), -1)
