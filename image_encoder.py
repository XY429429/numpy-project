"""Pure-NumPy implementation of the OpenCLIP ViT-B/32 image tower.

This is the student solution for the course project.  The learned parameters
are supplied by the course; only the forward arithmetic is implemented here.

Shape flow (float32 throughout)::

    images [B,3,224,224]
      -> patchify (32x32 patches, C-H-W order)       [B,49,3072]
      -> patch projection (no bias)                  [B,49,768]
      -> prepend CLS + positional embedding          [B,50,768]
      -> LayerNorm (pre) -> 12 x TransformerBlock    [B,50,768]
      -> LayerNorm (post) -> CLS row                 [B,768]
      -> visual.proj [768,512]                       [B,512]
      -> L2 unit vector                              [B,512]

Conventions:
    * Linear weights are stored as [out, in], so every linear layer uses W.T.
    * visual.proj is the exception: [in, out] = [768, 512], no transpose.
    * Every intermediate array must stay float32.  In particular, NumPy 2
      promotes ``float32_array / np.float64_scalar`` to float64 (NEP 50);
      the score scale below is therefore cast to float32 explicitly.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    """Pre-LayerNorm multi-head self-attention with fixed checkpoint weights."""

    def __init__(self, weights, prefix, heads):
        # in_proj packs Q, K and V into one [3D, D] matrix; out_proj is [D, D].
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02: [B,N,D] -> [B,H,N,D/H], preserving token order.

        The last feature axis is split as (H, D/H).  The head axis is then
        moved before the token axis so that every head can attend
        independently: [B, N, H, d] -> [B, H, N, d].
        """
        # D=768 -> (H=12, d=64), then move H in front of N.
        x = x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2 * self.width])
        v = self.split_heads(qkv[:, :, 2 * self.width:])

        # TODO-03a: scaled dot-product scores, q/k/v: [B,H,N,64].
        # Only the final two axes are transposed, so the B/H groups stay
        # intact: k^T is [B,H,64,N] and scores is [B,H,N,N].
        #
        # The explicit float32 cast is essential: np.sqrt(64) returns a
        # NumPy float64 scalar in NumPy 2, and float32 / np.float64 would
        # promote the entire attention path (and every later residual) to
        # float64.  The reference tests require float32 output.
        scale = np.float32(np.sqrt(self.head_dim))
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied softmax helper

        # TODO-03b: weighted values: [B,H,N,N] @ [B,H,N,d] -> [B,H,N,d].
        context = coefficients @ v

        # TODO-03b/c: [B,H,N,d] -> [B,N,H,d] -> [B,N,D], then out projection.
        # The head order matches the feature-axis split performed above.
        context = context.transpose(0, 2, 1, 3)
        merged = context.reshape(context.shape[0], context.shape[1], self.width)

        # TODO-03c: stored out_proj is [out,in], hence the transpose.
        return merged @ self.out_weight.T + self.out_bias


class TransformerBlock:
    """One pre-LayerNorm Transformer block with two residual paths."""

    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix + '.ln_1.weight'],
                               weights[prefix + '.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix + '.ln_2.weight'],
                               weights[prefix + '.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix + '.attn', heads)
        self.fc1_weight = weights[prefix + '.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix + '.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix + '.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix + '.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        # TODO-04a: first residual addition, shape stays [B,N,D].
        residual = x + update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        # TODO-04b: MLP = GELU(u @ W1.T + b1) @ W2.T + b2.
        # hidden is [B,N,3072]; after c_proj it is [B,N,768] again.
        hidden = u @ self.fc1_weight.T + self.fc1_bias
        activated = self.gelu(hidden)
        mlp_update = activated @ self.fc2_weight.T + self.fc2_bias

        # Second residual addition: both paths retain [B,N,D].
        return residual + mlp_update


class ImageEncoder:
    """OpenCLIP ViT-B/32 image tower: images -> one 512-d unit vector each."""

    def __init__(self, weights):
        # conv1 is a 32x32 stride-32 convolution without bias; flattening its
        # kernel to [768,3072] makes the patch projection a plain matmul.
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'],
                                  weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'],
                                   weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01: [B,3,224,224] -> [B,49,3072].

        Patch order is left-to-right, then top-to-bottom.  Within one patch,
        C-H-W is flattened so that the row matches the kernel layout of
        visual.conv1 (whose weight is [768,3,32,32]).
        """
        # Split H and W into (grid, patch) axes: 224 = 7*32.
        # x: (B, C, G_h, P_h, G_w, P_w)
        B = images.shape[0]
        x = images.reshape(B, 3, 7, 32, 7, 32)

        # Move the patch-grid axes forward and keep C,P_h,P_w inside a patch.
        # (B, G_h, G_w, C, P_h, P_w)
        x = x.transpose(0, 2, 4, 1, 3, 5)

        # Row-major flattening over (G_h,G_w) gives 49 patches; each patch
        # contributes C*P_h*P_w = 3*32*32 = 3072 values.
        return x.reshape(B, 49, 3072)

    def embed(self, images):
        """TODO-05: patch projection, CLS token and learned positions."""
        patches = self.patchify(images)  # [B,49,3072]

        # TODO-05a: project each patch to 768 dimensions.  There is no bias
        # because visual.conv1 has bias=False.  [B,49,3072] @ [3072,768].
        patches = patches @ self.patch_weight.T

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        # TODO-05b: CLS first, then the 49 patch tokens.  The [50,768]
        # positional table broadcasts over the batch axis.
        return np.concatenate([cls, patches], axis=1) + self.position

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06: return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        # TODO-06: select the CLS row, apply the post-LayerNorm, and project.
        # [B,50,768] -> [B,768] -> [B,512].  visual.proj is [in,out], so it
        # is applied directly (no transpose).
        output = self.post_norm(tokens)[:, 0] @ self.projection

        # The supplied helper L2-normalizes each 512-d row.
        return unit_vector(output)
