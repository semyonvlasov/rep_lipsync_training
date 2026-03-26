"""
LipSync Generator: Audio-conditioned face generator.

Fixed encoder-decoder with skip connections (reference-style).
Resolution controlled by number of up/down blocks.
Optional alpha mask output for seamless blending.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, kernel=3, stride=1, padding=1, residual=False):
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


class DeconvBlock(nn.Module):
    def __init__(self, cin, cout, kernel=3, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel, stride, padding, output_padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.deconv(x))


class LipSyncGenerator(nn.Module):
    """
    Encoder channels at each level (going down):
      Level 0: 6 → C0        (no downscale, first conv)
      Level 1: C0 → C1       (stride 2)
      Level 2: C1 → C2       (stride 2)
      ...
      Level N: → 512          (bottleneck, spatial=1x1)

    Decoder mirrors this with skip connections from encoder.
    After each decoder block, concat with encoder skip of the same spatial size.
    """

    def __init__(self, img_size=192, base_channels=32, predict_alpha=True):
        super().__init__()
        self.img_size = img_size
        self.predict_alpha = predict_alpha
        B = base_channels

        if img_size == 96:
            # Mirror the open-source Wav2Lip architecture exactly for the
            # 96px path. Higher resolutions still use our generalized variant.
            self.face_encoder_blocks = nn.ModuleList([
                nn.Sequential(ConvBlock(6, 16, 7, 1, 3)),

                nn.Sequential(
                    ConvBlock(16, 32, 3, 2, 1),
                    ConvBlock(32, 32, 3, 1, 1, residual=True),
                    ConvBlock(32, 32, 3, 1, 1, residual=True),
                ),

                nn.Sequential(
                    ConvBlock(32, 64, 3, 2, 1),
                    ConvBlock(64, 64, 3, 1, 1, residual=True),
                    ConvBlock(64, 64, 3, 1, 1, residual=True),
                    ConvBlock(64, 64, 3, 1, 1, residual=True),
                ),

                nn.Sequential(
                    ConvBlock(64, 128, 3, 2, 1),
                    ConvBlock(128, 128, 3, 1, 1, residual=True),
                    ConvBlock(128, 128, 3, 1, 1, residual=True),
                ),

                nn.Sequential(
                    ConvBlock(128, 256, 3, 2, 1),
                    ConvBlock(256, 256, 3, 1, 1, residual=True),
                    ConvBlock(256, 256, 3, 1, 1, residual=True),
                ),

                nn.Sequential(
                    ConvBlock(256, 512, 3, 2, 1),
                    ConvBlock(512, 512, 3, 1, 1, residual=True),
                ),

                nn.Sequential(
                    ConvBlock(512, 512, 3, 1, 0),
                    ConvBlock(512, 512, 1, 1, 0),
                ),
            ])

            self.audio_encoder = nn.Sequential(
                ConvBlock(1, 32, 3, 1, 1),
                ConvBlock(32, 32, 3, 1, 1, residual=True),
                ConvBlock(32, 32, 3, 1, 1, residual=True),

                ConvBlock(32, 64, 3, (3, 1), 1),
                ConvBlock(64, 64, 3, 1, 1, residual=True),
                ConvBlock(64, 64, 3, 1, 1, residual=True),

                ConvBlock(64, 128, 3, 3, 1),
                ConvBlock(128, 128, 3, 1, 1, residual=True),
                ConvBlock(128, 128, 3, 1, 1, residual=True),

                ConvBlock(128, 256, 3, (3, 2), 1),
                ConvBlock(256, 256, 3, 1, 1, residual=True),

                ConvBlock(256, 512, 3, 1, 0),
                ConvBlock(512, 512, 1, 1, 0),
            )

            self.face_decoder_blocks = nn.ModuleList([
                nn.Sequential(ConvBlock(512, 512, 1, 1, 0)),

                nn.Sequential(
                    DeconvBlock(1024, 512, 3, 1, 0, 0),
                    ConvBlock(512, 512, 3, 1, 1, residual=True),
                ),

                nn.Sequential(
                    DeconvBlock(1024, 512, 3, 2, 1, 1),
                    ConvBlock(512, 512, 3, 1, 1, residual=True),
                    ConvBlock(512, 512, 3, 1, 1, residual=True),
                ),

                nn.Sequential(
                    DeconvBlock(768, 384, 3, 2, 1, 1),
                    ConvBlock(384, 384, 3, 1, 1, residual=True),
                    ConvBlock(384, 384, 3, 1, 1, residual=True),
                ),

                nn.Sequential(
                    DeconvBlock(512, 256, 3, 2, 1, 1),
                    ConvBlock(256, 256, 3, 1, 1, residual=True),
                    ConvBlock(256, 256, 3, 1, 1, residual=True),
                ),

                nn.Sequential(
                    DeconvBlock(320, 128, 3, 2, 1, 1),
                    ConvBlock(128, 128, 3, 1, 1, residual=True),
                    ConvBlock(128, 128, 3, 1, 1, residual=True),
                ),

                nn.Sequential(
                    DeconvBlock(160, 64, 3, 2, 1, 1),
                    ConvBlock(64, 64, 3, 1, 1, residual=True),
                    ConvBlock(64, 64, 3, 1, 1, residual=True),
                ),
            ])

            self.output_face = nn.Sequential(
                ConvBlock(80, 32, 3, 1, 1),
                nn.Conv2d(32, 3, 1, 1, 0),
                nn.Sigmoid(),
            )
            if predict_alpha:
                self.output_alpha = nn.Sequential(
                    ConvBlock(80, 16, 3, 1, 1),
                    nn.Conv2d(16, 1, 1, 1, 0),
                    nn.Sigmoid(),
                )
            return

        # Encoder channel progression (same for all resolutions)
        # More blocks for larger resolutions
        if img_size == 128:
            enc_channels = [B, B*2, B*4, B*8, 512, 512, 512, 512]
            # spatial:      128 64   32   16    8    4    2    1
            final_conv = (2, 1, 0)  # 2→1: kernel=2, stride=1, padding=0
        elif img_size == 192:
            enc_channels = [B, B*2, B*4, B*8, 512, 512, 512, 512]
            # spatial:      192 96   48   24   12    6    3    1
            final_conv = (3, 1, 0)  # 3→1
        elif img_size == 256:
            enc_channels = [B, B*2, B*4, B*8, 512, 512, 512, 512, 512]
            # spatial:      256 128  64   32   16    8    4    2    1
            final_conv = (2, 1, 0)  # 2→1
        else:
            raise ValueError(f"Unsupported img_size={img_size}")

        # Build encoder
        enc_blocks = []
        in_ch = 6
        for i, out_ch in enumerate(enc_channels):
            if i == 0:
                # First block: no downscale
                enc_blocks.append(nn.Sequential(ConvBlock(in_ch, out_ch, 7, 1, 3)))
            elif i == len(enc_channels) - 1:
                # Last block: reduce to 1x1
                k, s, p = final_conv
                enc_blocks.append(nn.Sequential(
                    ConvBlock(in_ch, 512, k, s, p),
                    ConvBlock(512, 512, 1, 1, 0),
                ))
            else:
                enc_blocks.append(nn.Sequential(
                    ConvBlock(in_ch, out_ch, 3, 2, 1),
                    ConvBlock(out_ch, out_ch, 3, 1, 1, residual=True),
                ))
            in_ch = out_ch
        self.face_encoder_blocks = nn.ModuleList(enc_blocks)

        # Build decoder
        # Decoder block i takes: decoder_output + encoder_skip[N-1-i]
        # We need to compute the cat channels at each level
        N = len(enc_channels)
        dec_blocks = []

        # First decoder block: process 512 (audio emb) → 512
        dec_blocks.append(nn.Sequential(ConvBlock(512, 512, 1, 1, 0)))

        # Remaining decoder blocks: upsample + concat with skip
        dec_in_ch = 512 + enc_channels[-1]  # after cat with last encoder output (512+512=1024)
        for i in range(1, N):
            skip_ch = enc_channels[N - 1 - i]  # skip from mirror encoder level
            # Output channels: match the skip or use 512 for top levels
            out_ch = min(512, skip_ch * 2) if skip_ch < 512 else 512

            if i == 1:
                # 1x1 → 3x3 (or 2x2)
                k, s, p = final_conv
                dec_blocks.append(nn.Sequential(
                    DeconvBlock(dec_in_ch, out_ch, k, s, p, 0),
                    ConvBlock(out_ch, out_ch, 3, 1, 1, residual=True),
                ))
            else:
                dec_blocks.append(nn.Sequential(
                    DeconvBlock(dec_in_ch, out_ch, 3, 2, 1, 1),
                    ConvBlock(out_ch, out_ch, 3, 1, 1, residual=True),
                ))

            dec_in_ch = out_ch + skip_ch  # next input = this output + next skip

        self.face_decoder_blocks = nn.ModuleList(dec_blocks)

        # Audio encoder: (1, 1, 80, 16) → (1, 512, 1, 1)
        self.audio_encoder = nn.Sequential(
            ConvBlock(1, B, 3, 1, 1),
            ConvBlock(B, B, 3, 1, 1, residual=True),
            ConvBlock(B, B*2, 3, (3, 1), 1),
            ConvBlock(B*2, B*2, 3, 1, 1, residual=True),
            ConvBlock(B*2, B*4, 3, 3, 1),
            ConvBlock(B*4, B*4, 3, 1, 1, residual=True),
            ConvBlock(B*4, B*8, 3, (3, 2), 1),
            ConvBlock(B*8, B*8, 3, 1, 1, residual=True),
            ConvBlock(B*8, 512, 3, 1, 0),
            ConvBlock(512, 512, 1, 1, 0),
        )

        # Output: last dec_in_ch is the final concat channels
        # For 96: final = 64+32 = 96 (from last decoder out=64 + skip[0]=32)
        # Actually the last decoder block outputs to match skip[0]=B, and we DON'T
        # concat after the last block (no more skips). So output channels = last dec output.
        # Let me think... the last decoder block produces out_ch, then we cat with skip[0],
        # giving dec_in_ch for the "next" block which doesn't exist.
        # The output head should take dec_in_ch (the final catted channels).
        final_ch = dec_in_ch  # channels after last cat

        self.output_face = nn.Sequential(
            ConvBlock(final_ch, B, 3, 1, 1),
            nn.Conv2d(B, 3, 1, 1, 0),
            nn.Sigmoid(),
        )
        if predict_alpha:
            self.output_alpha = nn.Sequential(
                ConvBlock(final_ch, B // 2, 3, 1, 1),
                nn.Conv2d(B // 2, 1, 1, 1, 0),
                nn.Sigmoid(),
            )

    def forward(self, mel, face):
        input_dim_size = len(face.size())
        batch_size = face.size(0)

        if input_dim_size > 4:
            mel = torch.cat([mel[:, i] for i in range(mel.size(1))], dim=0)
            face = torch.cat([face[:, :, i] for i in range(face.size(2))], dim=0)

        audio_emb = self.audio_encoder(mel)

        feats = []
        x = face
        for block in self.face_encoder_blocks:
            x = block(x)
            feats.append(x)

        x = audio_emb
        for block in self.face_decoder_blocks:
            x = block(x)
            if feats:
                x = torch.cat((x, feats[-1]), dim=1)
                feats.pop()

        face_out = self.output_face(x)
        if self.predict_alpha:
            alpha = self.output_alpha(x)

            if input_dim_size > 4:
                face_out = torch.stack(torch.split(face_out, batch_size, dim=0), dim=2)
                alpha = torch.stack(torch.split(alpha, batch_size, dim=0), dim=2)
            return face_out, alpha

        if input_dim_size > 4:
            face_out = torch.stack(torch.split(face_out, batch_size, dim=0), dim=2)
        return face_out
