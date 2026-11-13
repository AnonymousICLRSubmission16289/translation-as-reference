# Reference-centered Pixel MeanFlow

Minimal model, pMF training objective, and one-step inference code.

## Install

```bash
pip install -r requirements.txt
```

## Inference

```bash
python -m reference_pmf.infer --weights weights/reference_pmf_3m.pt --input 10301.npy --output prediction_3m.npy
python -m reference_pmf.infer --weights weights/reference_pmf_6m.pt --input 10001.npy --output prediction_6m.npy
```

Input: a uint8 OCT `.npy` volume in `[depth, height, width]` order. Output: a uint8 OCTA volume of the same shape. CUDA uses BF16; add `--fp32` for FP32, or `--device cpu` for CPU inference.

Each supplied submission checkpoint contains **both** the reference and correction networks. No separate Stage-I checkpoint is needed. Check file integrity with `(cd weights && sha256sum -c SHA256SUMS)`.

- `models.py`: reference network and time-conditioned correction network.
- `meanflow.py`: original pMF forward/loss and one-step prediction.
- `optim.py`: original Muon/AdamW helper for training integration.
- `infer.py`: safe weight loading and native-volume sliding-window inference.

The two files in `weights/` are the original submission models. Upload source files to the repository and weight files as separate release assets; weight binaries are excluded from ordinary Git tracking.
