# Migration from PyTorch to MLX

This document outlines the migration from PyTorch to MLX for the LLM training framework.

## Why MLX?

MLX is Apple's machine learning framework specifically optimized for Apple Silicon (M1/M2/M3). Benefits:

- **Native Apple Silicon optimization** - Built specifically for M-series chips
- **Unified memory** - Efficiently uses the unified memory architecture
- **Faster on M3** - Better performance than PyTorch MPS for many operations
- **Simpler API** - More Pythonic and easier to use
- **Better memory efficiency** - Optimized for limited RAM scenarios

## Key Changes

### Dependencies

**Removed:**
- `torch` / `torchvision` / `torchaudio`
- `accelerate`
- `peft` (PyTorch-specific)
- `tensorboard` (not commonly used with MLX)

**Added:**
- `mlx` - Core MLX library
- `mlx-lm` - MLX language model utilities

**Kept:**
- `transformers` - For tokenizers and model configs
- `tokenizers` - Tokenization
- Other data processing libraries

### Module Changes

#### config.py
- Removed PyTorch-specific settings (fp16, bf16, use_mps, dataloader_num_workers)
- Simplified training config (MLX handles optimization internally)
- Changed default model to MLX-compatible models
- Added MLX-specific settings (grad_checkpoint)

#### dataset.py
- Keep markdown processing logic
- Replace PyTorch Dataset with simpler Python class
- Use numpy arrays instead of torch tensors
- MLX will convert numpy arrays automatically

#### training.py
- Complete rewrite using `mlx.optimizers` and `mlx.nn`
- Use MLX's automatic differentiation
- Simplified training loop (MLX handles device placement automatically)
- Use mlx-lm utilities for model loading

#### evaluation.py
- Rewrite to use MLX models
- Use mlx-lm for generation
- Keep perplexity and loss calculations

#### finetuning.py
- Use MLX's built-in LoRA support
- Simpler API than PEFT
- Better memory efficiency

#### utils.py
- Remove PyTorch device detection
- MLX automatically uses Apple Silicon
- Update memory estimation for MLX

### Model Format

**PyTorch:**
- Models from Hugging Face Hub in PyTorch format
- Loaded with `AutoModelForCausalLM.from_pretrained()`

**MLX:**
- Models from `mlx-community` namespace on Hugging Face
- Or convert PyTorch models to MLX format
- Loaded with `mlx_lm.load()`

### Training Loop Changes

**PyTorch:**
```python
for batch in dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

**MLX:**
```python
def loss_fn(model, inputs, targets):
    logits = model(inputs)
    return mx.mean(nn.losses.cross_entropy(logits, targets))

loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

for batch in data:
    loss, grads = loss_and_grad_fn(model, batch['inputs'], batch['targets'])
    optimizer.update(model, grads)
```

## Migration Checklist

- [x] Update requirements.txt
- [x] Rewrite config.py
- [ ] Rewrite dataset.py
- [ ] Rewrite training.py
- [ ] Rewrite evaluation.py
- [ ] Rewrite finetuning.py
- [ ] Update utils.py
- [ ] Update CLI
- [ ] Update documentation
- [ ] Update tests
- [ ] Update example configs
- [ ] Test end-to-end training

## Compatibility Notes

- MLX only works on Apple Silicon (M1/M2/M3)
- For other platforms, users would need to use PyTorch or another framework
- MLX models are stored in a different format than PyTorch
- Conversion tools are available in mlx-lm

## Performance Expectations

On M3 MacBook Pro (16GB RAM):
- **Faster inference** - 2-3x faster than PyTorch MPS
- **Better memory efficiency** - Can handle larger models
- **Simpler code** - Less boilerplate
- **Native optimization** - No need for MPS/CUDA abstraction
