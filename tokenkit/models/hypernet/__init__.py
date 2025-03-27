import math
from pprint import pprint
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from praxis import base_layer, layers, pax_fiddle
from praxis.layers import activations as activations_lib
from praxis.layers import embedding_softmax

from tokenkit.models import param

LayerTpl = pax_fiddle.Config[base_layer.BaseLayer]
template_field = base_layer.template_field
WeightInit = base_layer.WeightInit
WeightHParams = base_layer.WeightHParams

EPSILON = 1e-8


class IdentityLayerNorm(base_layer.BaseLayer):
    def __call__(self, x, *args, **kwargs):
        return x


class IdentityAttention(base_layer.BaseLayer):
    def __call__(self, x, *args, **kwargs):
        return jnp.zeros_like(x), jnp.zeros(())


class EmbeddingRescaler(base_layer.BaseLayer):
    shape: tuple[int] = ()
    axes: tuple[int] = (0,)

    def setup(self):
        self.create_variable(
            "w",
            WeightHParams(
                # NOTE: axes must start from the left atm, and this fails potentially silently if not
                shape=(1,) * len(self.axes) + self.shape,
                init=WeightInit.Constant(1.0),
                dtype=self.dtype,
            ),
            trainable=False,
        )
        self.create_variable(
            "b",
            WeightHParams(
                shape=(1,) * len(self.axes) + self.shape,
                init=WeightInit.Constant(0.0),
                dtype=self.dtype,
            ),
            trainable=False,
        )

    def __call__(self, x):
        return x * self.get_var("w") + self.get_var("b")

    def scale_to(x, target=None, target_means=None, target_stds=None, axes=(0,)):
        if target_stds is None:
            target_stds = target.std(axis=axes)
        if target_means is None:
            target_means = target.mean(axis=0)

        w = (target_stds / (x.std(axis=axes) + EPSILON))[None]
        b = (target_means - (x * w).mean(axis=axes))[None]

        return w, b


class HypernetTransformer(layers.Transformer):
    use_attention: bool = False

    def create_child(self, name, params):
        if self.use_attention:
            super().create_child(name, params)
        else:
            if "layer_norm" in name:
                super().create_child(name, pax_fiddle.Config(IdentityLayerNorm))
            elif "attention" in name:
                super().create_child(name, pax_fiddle.Config(IdentityAttention))
            else:
                super().create_child(name, params)


class HypernetStackedTransformer(layers.StackedTransformer):
    transformer_layer_params_tpl: LayerTpl | Sequence[LayerTpl] = template_field(
        HypernetTransformer
    )


class Hypernet(base_layer.BaseLayer):
    hidden_size: int = 0
    max_seq_length: int = 0
    num_embeddings: int = 2
    vocab_size: int = None
    residual: bool = True
    shared: bool = True
    use_attention_mask: bool = True
    pooling: str = "first"  # "first", "mean"
    residual_pooling: str = "first"  # "first", "mean"
    architecture: str = "transformer"  # 'transformer', 'linear', 'identity'
    embedding_lora_rank: int = 0
    embedding_lora_alpha: float = 8.0
    embedding_lora_position: str = "post"  # 'pre', 'post'

    rescaler_tpl: LayerTpl = template_field(EmbeddingRescaler)
    stacked_transformer_tpl: LayerTpl = template_field(
        HypernetStackedTransformer,
    )
    position_emb_tpl: LayerTpl = template_field(
        embedding_softmax.TrainablePositionalEmbedding
    )

    residual_alpha: float = 8.0

    # redefine for ease of access
    use_attention: bool = True
    multiply_hidden_dim_by_num_embeddings: bool = True
    hidden_expansion_factor: int = 2
    num_layers: int = 3
    num_heads: int = 16

    def setup(self):
        self.rescaler_tpl.shape = (self.num_embeddings, self.hidden_size)

        in_rescaler_tpl = self.rescaler_tpl.clone()
        in_rescaler_tpl.axes = (0, 1)
        out_rescaler_tpl = self.rescaler_tpl.clone()

        self.create_child("in_rescaler", in_rescaler_tpl)
        self.create_child("out_rescaler", out_rescaler_tpl)

        if self.embedding_lora_rank > 0:
            lora_embedding_a_config = pax_fiddle.Config(layers.Embedding)
            lora_embedding_a_config.num_classes = self.vocab_size
            lora_embedding_a_config.input_dims = self.embedding_lora_rank

            lora_linear_b_config = pax_fiddle.Config(layers.Linear)
            lora_linear_b_config.input_dims = self.embedding_lora_rank
            lora_linear_b_config.output_dims = self.hidden_size
            lora_linear_b_config.weight_init = WeightInit.Constant(0.0)

            self.create_children(
                "lora_embedding_a",
                [lora_embedding_a_config.clone() for _ in range(self.num_embeddings)],
            )
            self.create_children(
                "lora_linear_b",
                [lora_linear_b_config.clone() for _ in range(self.num_embeddings)],
            )

        if self.architecture == "transformer":
            hidden_dims = self.hidden_size
            if self.multiply_hidden_dim_by_num_embeddings:
                hidden_dims *= self.num_embeddings

            self.stacked_transformer_tpl.model_dims = hidden_dims
            self.stacked_transformer_tpl.hidden_dims = (
                hidden_dims * self.hidden_expansion_factor
            )
            self.stacked_transformer_tpl.num_layers = self.num_layers
            self.stacked_transformer_tpl.num_heads = self.num_heads
            self.stacked_transformer_tpl.transformer_layer_params_tpl.use_attention = (
                self.use_attention
            )
            self.stacked_transformer_tpl.transformer_layer_params_tpl.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
                activations_lib.SiLU
            )

            if self.shared:
                if not self.multiply_hidden_dim_by_num_embeddings:
                    input_linear_config = pax_fiddle.Config(layers.Linear)
                    input_linear_config.input_dims = (
                        self.hidden_size * self.num_embeddings
                    )
                    input_linear_config.output_dims = self.hidden_size

                    self.create_child("input_linear", input_linear_config)

                output_linear_config = pax_fiddle.Config(layers.Linear)
                output_linear_config.input_dims = hidden_dims
                output_linear_config.output_dims = (
                    self.hidden_size * self.num_embeddings
                )

                self.create_child("output_linear", output_linear_config)

                self.create_child("transformer", self.stacked_transformer_tpl)

                self.position_emb_tpl.embedding_dims = hidden_dims
                self.position_emb_tpl.max_seq_length = self.max_seq_length
                self.create_child("position_emb", self.position_emb_tpl)
            else:
                raise NotImplementedError("Non-shared transformer not implemented")
        elif self.architecture == "linear":
            if not self.shared:
                raise NotImplementedError("Non-shared linear not implemented")

            linear_config = pax_fiddle.Config(layers.Linear)
            linear_config.input_dims = self.hidden_size
            linear_config.output_dims = self.hidden_size
            self.create_children(
                "linear", [linear_config.clone() for _ in range(self.num_embeddings)]
            )
        elif self.architecture == "identity":
            pass
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

    def compute_embedding_lora(self, vocab_indices):
        lora_embeddings = []
        for i in range(self.num_embeddings):
            x = self.lora_embedding_a[i](vocab_indices)
            x = self.lora_linear_b[i](x)
            lora_embeddings.append(x)
        lora_embeddings = jnp.stack(lora_embeddings, axis=1)

        return lora_embeddings

    def __call__(self, embeddings, attention_mask, vocab_indices=None):
        # input (embeddings): [vocab_size, seq_length, num_embeddings, hidden_size]
        # output: [vocab_size, num_embeddings, hidden_size]
        if self.architecture == "identity":
            return embeddings[:, 0, :, :]

        if not self.use_attention_mask:
            attention_mask = jnp.ones_like(attention_mask)

        vocab_size, seq_length, _, _ = embeddings.shape
        embeddings = self.in_rescaler(embeddings)

        if self.embedding_lora_rank > 0 and self.embedding_lora_position == "pre":
            assert vocab_indices is not None

            lora_embeddings = self.compute_embedding_lora(vocab_indices)
            scaler = self.embedding_lora_alpha / math.sqrt(self.embedding_lora_rank)
            embeddings = embeddings.at[:, 0, :, :].add(lora_embeddings * scaler)

        if self.architecture == "transformer":
            if self.shared:
                x = jnp.reshape(
                    embeddings,
                    (vocab_size, seq_length, self.hidden_size * self.num_embeddings),
                )
                if not self.multiply_hidden_dim_by_num_embeddings:
                    x = self.input_linear(x)
                x = x + self.position_emb(seq_length)

                # TODO: impl packing
                x = self.transformer(x, paddings=~attention_mask)

                # take embedding of the first token in the sequence as the pooled prediction
                if self.pooling == "first":
                    x = x[:, 0, :]
                elif self.pooling == "mean":
                    x = (x * attention_mask[:, :, None]).sum(axis=1) / (
                        attention_mask.sum(axis=1) + EPSILON
                    )[:, None]
                else:
                    raise ValueError(f"Unknown pooling method: {self.pooling}")

                x = self.output_linear(x)
                x = jnp.reshape(x, (vocab_size, self.num_embeddings, self.hidden_size))

                if self.residual:
                    residual_weight = self.residual_alpha / math.sqrt(self.hidden_size)
                    if self.residual_pooling == "first":
                        non_residual = embeddings[:, 0, :, :]
                    elif self.residual_pooling == "mean":
                        non_residual = (
                            embeddings * attention_mask[:, :, None, None]
                        ).sum(axis=1) / (attention_mask.sum(axis=1) + EPSILON)[
                            :, None, None
                        ]
                    else:
                        raise ValueError(
                            f"Unknown pooling method: {self.residual_pooling}"
                        )

                    predicted_embeddings = non_residual + residual_weight * x
                else:
                    predicted_embeddings = x
            else:
                raise NotImplementedError("Non-shared transformer not implemented")
        elif self.architecture == "linear":
            raise NotImplementedError("Linear architecture not implemented")

        if self.embedding_lora_rank > 0 and self.embedding_lora_position == "post":
            assert vocab_indices is not None

            lora_embeddings = self.compute_embedding_lora(vocab_indices)
            scaler = self.embedding_lora_alpha / math.sqrt(self.embedding_lora_rank)
            predicted_embeddings = predicted_embeddings.at[:, 0, :, :].add(
                lora_embeddings * scaler
            )

        return self.out_rescaler(predicted_embeddings)

    def init(self, rngs, embeddings, attention_mask, vocab_indices=None):
        params = super().init(
            rngs, embeddings, attention_mask, vocab_indices=vocab_indices
        )

        # somewhat arbitrary, use ~Xavier normal
        in_std = math.sqrt(2.0 / self.hidden_size)

        in_w, in_b = EmbeddingRescaler.scale_to(
            embeddings, target_means=0, target_stds=in_std, axes=(0, 1)
        )

        params = param.put(params, "non_trainable.in_rescaler.w", in_w)
        params = param.put(params, "non_trainable.in_rescaler.b", in_b)

        # TODO: move to setup / accomplish via updating init config
        if self.residual:
            if self.architecture == "linear":
                for i in range(self.num_embeddings):
                    params = param.put(params, f"params.linear_{i}.w", 0.0)
            elif self.architecture == "transformer":
                if self.shared:
                    params = param.put(params, "params.output_linear.w", 0.0)
                else:
                    for i in range(self.num_embeddings):
                        for j in range(self.num_layers):
                            params = param.put(
                                params,
                                f"params.transformer_{i}.x_layers_{j}.ff_layer.ffn_layer2.linear.w",
                                0.0,
                            )

        preds = self.apply(
            params, embeddings, attention_mask, vocab_indices=vocab_indices
        )

        out_w, out_b = EmbeddingRescaler.scale_to(
            preds, target=embeddings[:, 0], axes=(0,)
        )

        params = param.put(params, "non_trainable.out_rescaler.w", out_w)
        params = param.put(params, "non_trainable.out_rescaler.b", out_b)

        return params


if __name__ == "__main__":
    model = Hypernet(hidden_size=768, architecture="transformer", use_attention=False)
    x = np.random.randn(128, 2, 768)
    params = model.init(jax.random.PRNGKey(0), x)

    preds = model.apply(params, x)

    pprint(jax.tree.map(jnp.shape, params))
