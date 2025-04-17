from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, FlaxAutoModelForCausalLM
from tokenkit.hf.configuration_tpu_llama import TPULlamaConfig
from tokenkit.hf.modelling_tpu_llama import TPULlamaForCausalLM, TPULlamaModel
from tokenkit.hf.modelling_flax_tpu_llama import FlaxTPULlamaForCausalLM, FlaxTPULlamaModel

from tokenkit.hf.configuration_tpu_gemma2 import TPUGemma2Config
from tokenkit.hf.modelling_tpu_gemma2 import TPUGemma2ForCausalLM, TPUGemma2Model
from tokenkit.hf.modelling_flax_tpu_gemma2 import FlaxTPUGemma2ForCausalLM, FlaxTPUGemma2Model

AutoConfig.register("tpu_llama", TPULlamaConfig)
AutoModel.register(TPULlamaConfig, TPULlamaModel)
AutoModelForCausalLM.register(TPULlamaConfig, TPULlamaForCausalLM)
TPULlamaForCausalLM.register_for_auto_class("AutoModelForCausalLM")
TPULlamaModel.register_for_auto_class("AutoModel")
FlaxAutoModelForCausalLM.register(TPULlamaConfig, FlaxTPULlamaForCausalLM)
FlaxTPULlamaForCausalLM.register_for_auto_class("FlaxAutoModelForCausalLM")
FlaxTPULlamaModel.register_for_auto_class("FlaxAutoModel")

AutoConfig.register("tpu_gemma2", TPUGemma2Config)
AutoModel.register(TPUGemma2Config, TPUGemma2Model)
AutoModelForCausalLM.register(TPUGemma2Config, TPUGemma2ForCausalLM)
TPUGemma2ForCausalLM.register_for_auto_class("AutoModelForCausalLM")
TPUGemma2Model.register_for_auto_class("AutoModel")
FlaxAutoModelForCausalLM.register(TPUGemma2Config, FlaxTPUGemma2ForCausalLM)
FlaxTPUGemma2ForCausalLM.register_for_auto_class("FlaxAutoModelForCausalLM")
FlaxTPUGemma2Model.register_for_auto_class("FlaxAutoModel")

__all__ = ["TPULlamaConfig", "TPULlamaModel", "TPULlamaForCausalLM", "FlaxTPULlamaForCausalLM", "FlaxTPULlamaModel", "TPUGemma2Config", "TPUGemma2Model", "TPUGemma2ForCausalLM", "FlaxTPUGemma2ForCausalLM", "FlaxTPUGemma2Model"]


def get_config(pretrained_model_name_or_path: str, **kwargs):
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    # compatibility with outside jax checkpoints
    if config.model_type in {"llama", "tpu_llama"}:
        config = TPULlamaConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        config.model_type = "tpu_llama"
        return config
    elif config.model_type in {"gemma2", "tpu_gemma2"}:
        config = TPUGemma2Config.from_pretrained(pretrained_model_name_or_path, **kwargs)
        config.model_type = "tpu_gemma2"
        return config
    else:
        return config
