from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..modules.utils import convert_module_to_f16, convert_module_to_f32
from ..modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from ..modules.spatial import patchify, unpatchify


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseStructurePoseFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        use_point_embedder: bool = False,
        point_embedder_out_channels: int = 1024,
        use_mask_embedder: bool = False,
        mask_embedder_out_channels: int = 1024,
        use_frame_token_embedder: bool = False,
        use_asymmetric_mask: bool = False,
        pose_representation: Literal["quaternion_translation_scale", "6d_translation_scale", "9d_translation_scale"] = "quaternion_translation_scale",
        num_pose_tokens: int = 1,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.use_point_embedder = use_point_embedder
        self.point_embedder_out_channels = point_embedder_out_channels
        self.use_mask_embedder = use_mask_embedder
        self.mask_embedder_out_channels = mask_embedder_out_channels
        self.use_frame_token_embedder = use_frame_token_embedder
        self.num_pose_tokens = num_pose_tokens
        self.use_asymmetric_mask = use_asymmetric_mask
        self.num_pose_tokens = num_pose_tokens
        self.dtype = torch.float16 if use_fp16 else torch.float32
        if pose_representation == "quaternion_translation_scale":
            self.pose_channels = 8  # 4 (quaternion) + 3 (translation) + 1 (scale)
        elif pose_representation == "6d_translation_scale":
            self.pose_channels = 10  # 6 (6D rotation) + 3 (translation) + 1 (scale)
        elif pose_representation == "9d_translation_scale":
            self.pose_channels = 13  # 9 (9D rotation) + 3 (translation) + 1 (scale)

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            pos_embedder = AbsolutePositionEmbedder(model_channels, 3)
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution // patch_size] * 3], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            pos_emb = pos_embedder(coords)
            self.register_buffer("pos_emb", pos_emb)

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)
            
        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        # Add pose input/output layers
        self.input_layer_pose = nn.Linear(self.pose_channels, model_channels)
        self.out_layer_pose = nn.Linear(model_channels, self.pose_channels)

        # Add optional point embedder for pointmap conditioning
        if self.use_point_embedder:
            self.point_embedder = nn.Conv2d(3, self.point_embedder_out_channels, kernel_size=14, stride=14)

        if self.use_mask_embedder:
            self.mask_embedder = nn.Conv2d(1, self.mask_embedder_out_channels, kernel_size=14, stride=14)

        if self.use_frame_token_embedder:
            self.frame_token_embedder = nn.Parameter(torch.zeros(2, cond_channels))

        if self.num_pose_tokens > 1:
            self.pose_token_embedder = nn.Parameter(torch.zeros(num_pose_tokens, model_channels))

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

        # Zero-out pose output layer:
        nn.init.constant_(self.out_layer_pose.weight, 0)
        nn.init.constant_(self.out_layer_pose.bias, 0)

        # Initialize point embedder if present:
        if self.use_point_embedder:
            nn.init.xavier_uniform_(self.point_embedder.weight)
            if self.point_embedder.bias is not None:
                nn.init.constant_(self.point_embedder.bias, 0)

        # Initialize mask embedder if present:
        if self.use_mask_embedder:
            nn.init.xavier_uniform_(self.mask_embedder.weight)
            if self.mask_embedder.bias is not None:
                nn.init.constant_(self.mask_embedder.bias, 0)

        # Initialize frame token embedder if present:
        if self.use_frame_token_embedder:
            nn.init.normal_(self.frame_token_embedder, std=0.02)

        # Initialize pose token embedder if present:
        # Use larger scale than frame_token_embedder (0.02) to ensure pose tokens
        # are distinguishable through 24 self-attention layers and shared out_layer_pose.
        if self.num_pose_tokens > 1:
            nn.init.normal_(self.pose_token_embedder, std=1.0)

    def forward(self, x: torch.Tensor, pose: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, **kwargs) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

        h = self.input_layer(h)
        h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)

        # Add pose to x and build asymmetric attention mask
        attn_mask = None
        if pose is not None:
            h_pose = self.input_layer_pose(pose)  # (B, D) -> (B, model_channels) or (B, K, D) -> (B, K, model_channels)
            h = h.type(self.dtype)
            if h_pose.ndim == 2:
                # Single pose token: (B, model_channels) -> (B, 1, model_channels)
                h_pose = h_pose.unsqueeze(1)
            # Add per-view identity embeddings so pose tokens know which view they predict
            if self.num_pose_tokens > 1 and hasattr(self, 'pose_token_embedder'):
                h_pose = h_pose + self.pose_token_embedder[None, :h_pose.shape[1], :]
            n_pose = h_pose.shape[1]
            h = torch.cat([h, h_pose], dim=1)  # (B, N_shape + n_pose, model_channels)

            if self.use_asymmetric_mask:
                # Asymmetric mask: shape tokens cannot attend to pose tokens,
                # but pose tokens can attend to everything
                L = h.shape[1]  # N_shape + n_pose
                attn_mask = torch.zeros(L, L, device=h.device, dtype=h.dtype)
                attn_mask[:L - n_pose, L - n_pose:] = float('-inf')

        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)

        for block in self.blocks:
            h = block(h, t_emb, cond, attn_mask=attn_mask)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])

        if pose is not None:
            h, h_pose = torch.split(h, [h.shape[1] - n_pose, n_pose], dim=1)
            h_pose = self.out_layer_pose(h_pose)  # (B, n_pose, pose_channels)
            if n_pose == 1:
                h_pose = h_pose.squeeze(1)  # (B, pose_channels) for backward compat

        h = self.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()

        if pose is not None:
            return h, h_pose
        return h

    def encode_pointmap(self, pointmap: torch.Tensor) -> torch.Tensor:
        """
        Encode pointmap using the point embedder.
        
        Args:
            pointmap: Pointmap tensor of shape (B, 3, H, W)
            
        Returns:
            Encoded pointmap features of shape (B, N, C) where N = (H/14) * (W/14)
        """
        if not self.use_point_embedder:
            raise ValueError("Point embedder is not enabled for this model. Set use_point_embedder=True during initialization.")
        
        features = self.point_embedder(pointmap)  # (B, C, H', W')
        features = features.flatten(2).transpose(1, 2)  # (B, H'*W', C)
        return features

    def encode_frame_tokens(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Apply per-frame token-type embeddings to multiview conditioning.

        Args:
            cond: Conditioning tensor of shape (B, 2, N, D)

        Returns:
            Conditioning with frame embeddings added, flattened to (B, 2*N, D)
        """
        if not self.use_frame_token_embedder:
            raise ValueError("Frame token embedder is not enabled. Set use_frame_token_embedder=True during initialization.")

        # cond: (B, 2, N, D), frame_token_embedder: (2, D)
        cond = cond + self.frame_token_embedder[None, :, None, :]
        B, F, N, D = cond.shape
        return cond.reshape(B, F * N, D)

    def encode_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Encode mask using the mask embedder.

        Args:
            mask: Mask tensor of shape (B, 1, H, W)

        Returns:
            Encoded mask features of shape (B, N, C) where N = (H/14) * (W/14)
        """
        if not self.use_mask_embedder:
            raise ValueError("Mask embedder is not enabled for this model. Set use_mask_embedder=True during initialization.")

        features = self.mask_embedder(mask)  # (B, C, H', W')
        features = features.flatten(2).transpose(1, 2)  # (B, H'*W', C)
        return features
