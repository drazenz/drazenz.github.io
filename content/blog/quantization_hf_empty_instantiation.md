+++
title = 'LLM Quantization From Scratch - How are empty models instantiated for low memory usage?'
date = 2024-04-07T23:36:23+02:00
draft = false
+++

Say we want to do post-training quantization of an LLM.

For PyTorch models, we'll usually have an implementation defaulting to `bfloat16` and `torch.nn` layers, such as `torch.nn.Linear` and `torch.nn.Embedding`.

We'll also have pretrained weights. For a HuggingFace model they'll come in a bunch of `.safetensors` files, accompanied by model configs.

To get a quantized model, we can simply:
1. load the pretrained model into memory (cpu or gpu)

    Do this with the default, non-quantized dtype, usually `bfloat16`.
 
2. replace each layer with its quantized implementation.

    For an LLM, that's going to be `nn.Linear` and `nn.Embeddings` as that's where almost all the parameters of the model are. (The only other weights are in LayerNorms, 2 per each decoder block + 1 before the token prediction head).

See the problem with this approach? If we simply load the pretrained model weights in `float32` or `bfloat16`, we'll need the full amount of memory for the model. If we're doing quantization, it's likely we're already memory constrained, so this won't cut it.

What we'll do instead is load the model layer by layer, quantizing the weights on the fly. This way, we'll avoid having all weight tensors in memory in full precision.

Now our plan is:

1. Instantiate an empty model, without taking up the memory needed for full precision weights
2. For each layer, load its weights, quantize them and add the quantized version to the model. Discard the full precision weights.

Let's see how.

### PyTorch `meta` device

From [PyTorch docs](https://pytorch.org/docs/stable/meta.html):

>The “meta” device is an abstract device which denotes **a tensor which records only metadata, but no actual data.** Meta tensors have two primary use cases:
> 
>- Models can be loaded on the meta device, allowing you to load a representation of the model without actually loading the actual parameters into memory. This can be helpful if you need to make transformations on the model before you load the actual data
>
> [...]

This is exactly what we need - load the *representation* of the model, without taking up any space.

Once we've done it, our step 2 is to iterate over submodules of the model and monkey-patch the ones we want to have quantized.

### Doing it to an actual HF model

Let's say we want to quantize a HuggingFace implementation of Llama3 model.

To load the model onto `meta` device, we could simply do:

```python {indent=2 lineNos=inline}
import os
import torch    
from transformers.models.llama import LlamaConfig, LlamaForCausalLM

HF_MODEL_NAME = 'meta-llama/Llama-3.2-1B-Instruct'

with open(os.path.expanduser('~/.hf_token')) as token_file:
  token = token_file.read().strip()

model_config = LlamaConfig.from_pretrained(HF_MODEL_NAME, token=token)

with torch.no_grad(), torch.device('meta'):
  model = LlamaForCausalLM(config=model_config)
```
Note that no weight loading is happening yet.  The key here is line 12, where we use the `torch.device('meta')` context manager. This causes all tensors under this context to be created on `meta`, unless explicitly specified otherwise.

However, with HuggingFace implementation there's one big gotcha here. It makes things a lot more complicated.

Here's the thing. When we instantiate the model:

```python {indent=2 lineNos=inline}
with torch.no_grad(), torch.device('meta'):
  model = LlamaForCausalLM(config=model_config)
```

all its parameters will be created on `meta`. But so will **all the [buffers](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Buffer.html)**, too.

Buffers are considered a part of the model's state, and by default, when a model is saved they'll be saved too, alongside all the parameters. That is unless a buffer is registered with `persistent=false`.

In the HuggingFace implementation of Llama (and Phi3, probably other LLMs too), the `inv_freq` constants for RoPE embeddings are created as non-persistent buffers ([source](https://github.com/huggingface/transformers/blob/f4684a6eb2d224989d17cda62007de185acf7e01/src/transformers/models/llama/modeling_llama.py#L97-L99)).
```python {indent=2 lineNos=inline}
...
inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
self.register_buffer("inv_freq", inv_freq, persistent=False) # <-- Here
self.original_inv_freq = self.inv_freq
...
```

This means that:
- The values of these buffers (there's one per transformer block) are not persisted with weights
- When we instantiate the model using `torch.device('meta')`, they're initialized but on `meta`, so the actual values are lost
- When monkey-patching quantized layers, we'll move their parameters to cpu or gpu and then copy the values. We have no way to do this for non-persistent registered buffers, as their values are meant to be set during model's `__init__`, and not persisted to weights files

### The HuggingFace solution

Digging deeper into the [transformers codebase](https://github.com/huggingface/transformers), we find that when loading quantized models, they're using the `init_empty_weights` context manager.

It's defined in HuggingFace's `accelerate` project, and in turn refers to another context manager - [`init_on_device`](https://github.com/huggingface/accelerate/blob/6fcc8efd2e27a2f25393097a3ec19c44cc12340b/src/accelerate/big_modeling.py#L93-L166)

Inside `init_on_device`, there are two key parts:
1. `register_empty_parameter`, a method used to patch `torch.nn.Module.register_parameter`
2. `register_empty_buffer`, but optionally. That is, we can have parameters pushed to `meta`, but buffers will go to the default device

Thus, to load a model with empty weights HuggingFace will first [monkey-patch](https://github.com/huggingface/accelerate/blob/6fcc8efd2e27a2f25393097a3ec19c44cc12340b/src/accelerate/big_modeling.py#L93-L166) the `torch.nn.Module.register_parameter` method with the following: 

```python {indent=2 lineNos=inline}
...
@contextmanager
def init_on_device(device: torch.device, include_buffers: bool = None):
  ...

  old_register_parameter = nn.Module.register_parameter

  def register_empty_parameter(module, name, param):
    old_register_parameter(module, name, param)
    if param is not None:
      param_cls = type(module._parameters[name])
      kwargs = module._parameters[name].__dict__
      kwargs["requires_grad"] = param.requires_grad
      module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

  ...

  try:
    nn.Module.register_parameter = register_empty_parameter
    ...
    yield
  finally:
    nn.Module.register_parameter = old_register_parameter
    ...
```

What the new `register_parameter` does is:
- (line 9) let the `nn.Module.register_parameter` do its thing
- (line 14) then create a new parameter but send it to `device`, in this case it will be `meta`

Knowing this, we can easily create our own standalone context manager to fix our initial attempt.

```python {indent=2 lineNos=inline}s
@contextmanager
def init_params_on_meta():
    device = 'meta'
    old_register_parameter = torch.nn.Module.register_parameter

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)
    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
```

With this, our code to create an empty model turns into:


```python {indent=2 lineNos=inline}s
with torch.no_grad(), init_params_on_meta(): # <-- Here we use the new context manager
  model = LlamaForCausalLM(config=model_config)
```

Now, during model initialization all the weights will go to `meta`, but the buffers (ie. `LlamaRotaryEmbedding.inv_freq`) will remain on the torch default device, so we don't lose their initialized values.

The next step is to load each layer's weights, quantize them and replace the standard `torch.nn` modules with the quantized implementations. That we'll do in another post.






