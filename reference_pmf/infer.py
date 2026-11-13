from pathlib import Path
from contextlib import nullcontext
import argparse, hashlib, time
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from .models import build_model
from .meanflow import one_step_prediction

def normalize_raw255(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float32) * (2.0 / 255.0) - 1.0


def sliding_starts(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, max(1, stride)))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def autocast_context(device: torch.device, enabled: bool, dtype: str):
    if not enabled or device.type != "cuda":
        return nullcontext()
    torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=torch_dtype)


def stable_seed(base_seed: int, text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return int((int(base_seed) + value) % (2**31 - 1))


def _extract_padded_patch(volume: Tensor, z0: int, y0: int, x0: int, patch_shape: tuple[int, int, int], pad_value: float) -> Tensor:
    pd, ph, pw = patch_shape
    d, h, w = volume.shape[-3:]
    patch = volume[..., z0:min(z0+pd,d), y0:min(y0+ph,h), x0:min(x0+pw,w)]
    missing = (pd-patch.shape[-3], ph-patch.shape[-2], pw-patch.shape[-1])
    if any(value > 0 for value in missing):
        patch = F.pad(patch, (0, missing[2], 0, missing[1], 0, missing[0]), value=pad_value)
    return patch


def predict_native_volume(
    model: torch.nn.Module,
    oct_volume: Tensor,
    *,
    task: str,
    patch_shape: tuple[int, int, int],
    stride_shape: tuple[int, int, int],
    eval_noise_seed: int,
    pid: str,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
    t_clip: float,
) -> tuple[Tensor, dict[str, float]]:
    """Sliding-window native-volume inference with one coherent noise field."""
    if oct_volume.ndim != 4 or oct_volume.shape[0] != 1:
        raise ValueError(f"Expected OCT shape [1,D,H,W], got {tuple(oct_volume.shape)}")
    oct_cpu = oct_volume.unsqueeze(0).contiguous()  # [1,1,D,H,W]
    d, h, w = oct_cpu.shape[-3:]
    pd, ph, pw = patch_shape
    sd, sh, sw = stride_shape
    z_starts = sliding_starts(d, pd, sd)
    y_starts = sliding_starts(h, ph, sh)
    x_starts = sliding_starts(w, pw, sw)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(eval_noise_seed, pid))
    full_noise = torch.randn(oct_cpu.shape, generator=generator, dtype=torch.float32)
    accum = torch.zeros_like(oct_cpu, dtype=torch.float32)
    weight = torch.zeros_like(oct_cpu, dtype=torch.float32)
    patch_times: list[float] = []
    model.eval()
    with torch.inference_mode():
        for z0 in z_starts:
            for y0 in y_starts:
                for x0 in x_starts:
                    oct_patch = _extract_padded_patch(oct_cpu, z0, y0, x0, patch_shape, -1.0).to(device, non_blocking=True)
                    noise_patch = _extract_padded_patch(full_noise, z0, y0, x0, patch_shape, 0.0).to(device, non_blocking=True)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with autocast_context(device, amp, amp_dtype):
                        if task == "anchor":
                            prediction = model(oct_patch)
                        else:
                            prediction, _ = one_step_prediction(model, oct_patch, noise=noise_patch, t_clip=t_clip)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    patch_times.append(time.perf_counter() - started)
                    prediction = prediction.float().cpu()
                    valid_d = min(pd, d-z0)
                    valid_h = min(ph, h-y0)
                    valid_w = min(pw, w-x0)
                    accum[..., z0:z0+valid_d, y0:y0+valid_h, x0:x0+valid_w] += prediction[..., :valid_d, :valid_h, :valid_w]
                    weight[..., z0:z0+valid_d, y0:y0+valid_h, x0:x0+valid_w] += 1.0
    if float(weight.min()) <= 0:
        raise RuntimeError("Sliding inference left uncovered voxels")
    output = (accum / weight).squeeze(0)
    return output, {
        "num_patches": float(len(patch_times)),
        "patch_seconds_mean": float(np.mean(patch_times)),
        "patch_seconds_total": float(np.sum(patch_times)),
    }



def load_model(weights, device="cpu"):
    payload = torch.load(weights, weights_only=True, map_location="cpu", mmap=True)
    if payload.get("format_version") != 1 or "config" not in payload:
        raise ValueError("Expected a weights-only Reference pMF checkpoint")
    model = build_model(payload["config"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload

def main():
    parser = argparse.ArgumentParser(description="One-step OCT-to-OCTA inference")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--input", required=True, help="uint8 OCT .npy, shape D,H,W")
    parser.add_argument("--output", required=True, help="output uint8 OCTA .npy")
    parser.add_argument("--subject-id", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.resolve() == Path(args.input).resolve():
        raise FileExistsError("Choose a new output path")
    torch.set_num_threads(2)
    raw = np.load(args.input, mmap_mode="r", allow_pickle=False)
    if raw.dtype != np.uint8 or raw.ndim != 3:
        raise ValueError("Input must be uint8 OCT with shape [D,H,W]")
    device = torch.device(args.device)
    model, payload = load_model(args.weights, device)
    settings = payload["inference"]
    volume = torch.from_numpy(np.ascontiguousarray(normalize_raw255(raw)[None]))
    prediction, _ = predict_native_volume(model, volume, task=payload["config"]["task"],
        patch_shape=tuple(settings["patch_shape"]), stride_shape=tuple(settings["stride_shape"]),
        eval_noise_seed=args.seed, pid=args.subject_id or Path(args.input).stem,
        device=device, amp=(device.type=="cuda" and not args.fp32), amp_dtype="bf16", t_clip=settings["t_clip"])
    result = np.clip((prediction.numpy()[0] + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        np.save(f, result, allow_pickle=False)
    print(output)

if __name__ == "__main__":
    main()
