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
FlaxAutoModelForCausalLM.register(TPULlamaConfig, FlaxTPULlamaForCausalLM)

AutoConfig.register("tpu_gemma2", TPUGemma2Config)
AutoModel.register(TPUGemma2Config, TPUGemma2Model)
AutoModelForCausalLM.register(TPUGemma2Config, TPUGemma2ForCausalLM)
FlaxAutoModelForCausalLM.register(TPUGemma2Config, FlaxTPUGemma2ForCausalLM)

FlaxTPULlamaForCausalLM.register_for_auto_class("FlaxAutoModelForCausalLM")
FlaxTPULlamaModel.register_for_auto_class("FlaxAutoModel")
FlaxTPUGemma2ForCausalLM.register_for_auto_class("FlaxAutoModelForCausalLM")
FlaxTPUGemma2Model.register_for_auto_class("FlaxAutoModel")

__all__ = ["TPULlamaConfig", "TPULlamaModel", "TPULlamaForCausalLM", "FlaxTPULlamaForCausalLM", "FlaxTPULlamaModel", "TPUGemma2Config", "TPUGemma2Model", "TPUGemma2ForCausalLM", "FlaxTPUGemma2ForCausalLM", "FlaxTPUGemma2Model"]