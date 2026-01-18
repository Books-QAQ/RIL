# vision_transformer.py
# Integrated with Domain-Specific LayerNorm Bank (DSLN-Bank) + Router
# Author: zhangqunshu (based on VIL ECCV 2024 codebase)
# Date: 2025-11
# --------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_

# --------------------------------------------------------------
# 1. Domain-Specific LayerNorm Bank (DSLN-Bank)
# --------------------------------------------------------------
class DomainNormBank(nn.Module):
    def __init__(self, normalized_shape, num_domains=4, eps=1e-6, affine=True,
                 router_hidden=128, topk=1, temperature=1.0):
        super().__init__()
        self.num_domains = num_domains
        self.topk = topk
        self.temperature = temperature

        # K 个 LayerNorm 分支
        self.lns = nn.ModuleList([
            nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=affine)
            for _ in range(num_domains)
        ])

        # Router MLP
        self.router = nn.Sequential(
            nn.Linear(normalized_shape, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, num_domains)
        )

        # 分支使用频率统计
        self.register_buffer("usage", torch.zeros(num_domains), persistent=False)

        # 保存前一任务参数快照
        self.prev_gamma = [None for _ in range(num_domains)]
        self.prev_beta = [None for _ in range(num_domains)]

    def forward(self, x):
        if x.dim() == 3:
            cls = x[:, 0]
        else:
            cls = x
        logits = self.router(cls.detach())
        prob = F.softmax(logits / self.temperature, dim=-1)  # (B, K)

        # 更新使用频率
        with torch.no_grad():
            self.usage += prob.sum(dim=0).detach()

        # Top-1 硬路由
        if self.topk == 1:
            idx = prob.argmax(dim=-1)
            outs = []
            for b in range(x.size(0)):
                outs.append(self.lns[idx[b].item()](x[b]))
            out = torch.stack(outs, dim=0)
            return out, prob
        else:
            # 混合路由（Top-k 加权）
            out_sum = 0
            if x.dim() == 3:
                for k, ln in enumerate(self.lns):
                    out_sum = out_sum + ln(x) * prob[:, k].view(-1, 1, 1)
            else:
                for k, ln in enumerate(self.lns):
                    out_sum = out_sum + ln(x) * prob[:, k].view(-1, 1)
            return out_sum, prob

    def take_snapshot(self):
        """保存任务结束时的 LayerNorm 权重快照"""
        for k, ln in enumerate(self.lns):
            if ln.weight is not None:
                self.prev_gamma[k] = ln.weight.detach().clone()
            if ln.bias is not None:
                self.prev_beta[k] = ln.bias.detach().clone()

    def drift_loss(self):
        """计算漂移正则项"""
        loss = 0.0
        usage = self.usage + 1e-6
        usage = usage / usage.sum()
        for k, ln in enumerate(self.lns):
            if self.prev_gamma[k] is not None and ln.weight is not None:
                loss += usage[k] * F.mse_loss(ln.weight, self.prev_gamma[k])
            if self.prev_beta[k] is not None and ln.bias is not None:
                loss += usage[k] * F.mse_loss(ln.bias, self.prev_beta[k])
        return loss


# --------------------------------------------------------------
# 2. Vision Transformer 基础模块
# --------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 use_dnbank=False, dnbank_k=4, dnbank_topk=1):
        super().__init__()
        if use_dnbank:
            self.norm1 = DomainNormBank(dim, num_domains=dnbank_k, topk=dnbank_topk)
            self.norm2 = DomainNormBank(dim, num_domains=dnbank_k, topk=dnbank_topk)
        else:
            self.norm1 = norm_layer(dim)
            self.norm2 = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)
        self.use_dnbank = use_dnbank

    def forward(self, x):
        if self.use_dnbank:
            nx, _ = self.norm1(x)
            x = x + self.attn(nx)
            nx, _ = self.norm2(x)
            x = x + self.mlp(nx)
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------
# 3. Vision Transformer Backbone
# --------------------------------------------------------------

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 use_dnbank=False, dnbank_k=4, dnbank_topk=1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.use_dnbank = use_dnbank

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                                     stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  use_dnbank=use_dnbank, dnbank_k=dnbank_k, dnbank_topk=dnbank_topk)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    # ---- DSLN-Bank 扩展接口 ----
    def dnbank_modules(self):
        if not self.use_dnbank:
            return []
        banks = []
        for blk in self.blocks:
            if isinstance(blk.norm1, DomainNormBank):
                banks.append(blk.norm1)
            if isinstance(blk.norm2, DomainNormBank):
                banks.append(blk.norm2)
        return banks

    def dnbank_drift_loss(self):
        loss = 0.0
        for bank in self.dnbank_modules():
            loss = loss + bank.drift_loss()
        return loss

    def dnbank_take_snapshot(self):
        for bank in self.dnbank_modules():
            bank.take_snapshot()

    def dnbank_parameters(self):
        params = []
        for bank in self.dnbank_modules():
            for ln in bank.lns:
                if ln.weight is not None:
                    params.append(ln.weight)
                if ln.bias is not None:
                    params.append(ln.bias)
            for p in bank.router.parameters():
                params.append(p)
        return params
