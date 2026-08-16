"""Metrics.

Every reconstruction reports PSNR, SSIM, LPIPS and runtime per image.  LPIPS is a headline
metric of this benchmark and is not optional in a normal run.

Secondary diagnostics
    measurement consistency   RMSE(A(x_hat) - y).  Important because MPC explicitly
                              optimises measurement consistency while SDEdit does not, so a
                              PSNR difference could otherwise be misread.
    masked-region metrics     for box/random inpainting, full-image PSNR/SSIM are dominated
                              by the observed region; the missing region is reported too.

PSNR/SSIM use data_range = 2.0 because the canonical range is [-1,1].  This is numerically
identical to evaluating in [0,1] with data_range = 1.0.  Consistency check with the
MPC-Flow paper: for denoising with sigma = 0.2 the degraded PSNR is
10*log10(2^2/0.2^2) = 20.00 dB, exactly the "Degraded" entry of its Table 2 -- which is what
fixes the [-1,1] convention used throughout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .utils import PIXEL_PEAK, NUMPY_BACKEND

_LPIPS_STATE: Dict[str, Any] = {"model": None, "enabled": True, "net": "alex", "failed": False}


def configure_lpips(enabled: bool = True, net: str = "alex") -> None:
    _LPIPS_STATE["enabled"] = bool(enabled)
    _LPIPS_STATE["net"] = str(net)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))
    return float("inf") if mse <= 0 else float(10.0 * np.log10(PIXEL_PEAK ** 2 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(np.asarray(a, np.float64), np.asarray(b, np.float64),
                                       channel_axis=2, data_range=PIXEL_PEAK))


LPIPS_WEIGHT_URL = ("https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
                    "master/lpips/weights/v0.1/%s.pth")


def repair_lpips_weights(net: str = "alex", verbose: bool = True) -> bool:
    """Restore the small linear-head weights that ship inside the `lpips` package.

    LPIPS needs two things: a torchvision backbone (downloaded to TORCH_HOME) and a ~6 KB
    calibration head bundled in the package itself.  Some redistributed wheels -- the
    Alliance `lpips+computecanada` build among them -- omit the bundled file, so the model
    imports fine and then fails on a missing path.  Downloading it once repairs the install
    permanently; it requires network, so it happens during --prefetch, never in a job.
    """
    try:
        import lpips as _lp
        target = Path(_lp.__file__).parent / "weights" / "v0.1" / ("%s.pth" % net)
        if target.exists() and target.stat().st_size > 1000:
            return True
        target.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        if verbose:
            print("   LPIPS head weights missing from the installed package; fetching %s"
                  % target.name)
        urllib.request.urlretrieve(LPIPS_WEIGHT_URL % net, str(target))
        ok = target.exists() and target.stat().st_size > 1000
        if verbose:
            print("   %s %s (%d bytes)" % ("repaired" if ok else "FAILED to repair",
                                           target, target.stat().st_size if ok else 0))
        return ok
    except Exception as exc:
        if verbose:
            print("   could not repair the LPIPS weights: %s" % exc)
        return False


def lpips_per_image(recon: np.ndarray, truth: np.ndarray) -> Optional[List[float]]:
    """Per-image LPIPS (AlexNet by default).  Returns None if the package is unavailable."""
    if not _LPIPS_STATE["enabled"] or _LPIPS_STATE["failed"]:
        return None
    try:
        import torch
        if _LPIPS_STATE["model"] is None:
            import lpips as _lp
            weights = (Path(_lp.__file__).parent / "weights" / "v0.1"
                       / ("%s.pth" % _LPIPS_STATE["net"]))
            if not weights.exists():
                raise FileNotFoundError(
                    "The installed `lpips` package is missing its bundled weight file %s. "
                    "This is a packaging defect in some redistributed wheels, not a "
                    "download failure. Repair it on a machine with network access:\n"
                    "    python -c \"from src.metrics import repair_lpips_weights; "
                    "repair_lpips_weights()\"\n"
                    "or reinstall from PyPI: pip install --force-reinstall --no-deps lpips"
                    % weights)
            net = _lp.LPIPS(net=_LPIPS_STATE["net"])
            net.eval().requires_grad_(False)
            _LPIPS_STATE["model"] = net
        net = _LPIPS_STATE["model"]
        with torch.no_grad():
            a = torch.from_numpy(np.ascontiguousarray(
                np.asarray(recon, np.float32).transpose(0, 3, 1, 2))).float()
            b = torch.from_numpy(np.ascontiguousarray(
                np.asarray(truth, np.float32).transpose(0, 3, 1, 2))).float()
            # Inputs are already in [-1,1], which is what LPIPS expects.
            values = net(a, b).reshape(-1)
            return [float(v) for v in values.cpu().numpy()]
    except Exception as exc:
        print("   LPIPS unavailable (%s) -- LPIPS columns will be empty. "
              "Install it with `pip install lpips`." % exc)
        _LPIPS_STATE["failed"] = True
        return None


def measurement_consistency(recon_pixels: np.ndarray, problem) -> float:
    """RMSE of A(x_hat) - y in measurement space (backend-independent, NumPy)."""
    residual = np.asarray(problem.apply(np.asarray(recon_pixels, np.float32), NUMPY_BACKEND),
                          np.float32) - problem.measurement
    return float(np.sqrt(np.mean(residual ** 2)))


def masked_region_metrics(recon: np.ndarray, truth: np.ndarray,
                          mask: np.ndarray) -> Dict[str, Any]:
    """Errors restricted to the missing and observed regions of an inpainting problem.

    `mask` is (N,H,W,1) with 1 = observed.  Full-image PSNR/SSIM can be dominated by the
    already-observed region, so the missing region is reported separately.  Region-restricted
    PSNR uses the same data_range convention as the full-image metric; SSIM is not
    well-defined on a scattered pixel set and is therefore reported only for the box mask,
    where the missing region is a contiguous rectangle.
    """
    recon = np.asarray(recon, np.float64)
    truth = np.asarray(truth, np.float64)
    m = np.asarray(mask, np.float64)
    out: Dict[str, Any] = {}
    for label, weight in (("missing", 1.0 - m), ("observed", m)):
        w = np.broadcast_to(weight, recon.shape)
        total = float(w.sum())
        if total <= 0:
            out["%s_rmse" % label] = None
            out["%s_psnr" % label] = None
            continue
        mse = float((w * (recon - truth) ** 2).sum() / total)
        out["%s_rmse" % label] = float(np.sqrt(mse))
        out["%s_psnr" % label] = (float("inf") if mse <= 0
                                  else float(10.0 * np.log10(PIXEL_PEAK ** 2 / mse)))
    return out


def box_region_ssim(recon: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> Optional[float]:
    """SSIM inside a contiguous rectangular hole, where it is mathematically sensible."""
    hole = (np.asarray(mask[0, ..., 0]) < 0.5)
    if not hole.any():
        return None
    rows = np.where(hole.any(axis=1))[0]
    cols = np.where(hole.any(axis=0))[0]
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    if not hole[r0:r1, c0:c1].all():          # not a solid rectangle (e.g. random mask)
        return None
    if (r1 - r0) < 8 or (c1 - c0) < 8:        # SSIM's default window needs some support
        return None
    values = [ssim(np.asarray(recon[i, r0:r1, c0:c1], np.float64),
                   np.asarray(truth[i, r0:r1, c0:c1], np.float64))
              for i in range(len(recon))]
    return float(np.mean(values))


def evaluate_reconstruction(recon: np.ndarray, problem, spec=None) -> Dict[str, Any]:
    """The full metric record for one atomic job.

    Returns per-image PSNR/SSIM/LPIPS alongside their means, the measurement-consistency
    diagnostic, and -- for the inpainting tasks -- missing/observed region errors.
    """
    recon = np.asarray(recon, np.float32)
    truth = np.asarray(problem.ground_truth[:len(recon)], np.float32)
    ps = [psnr(recon[i], truth[i]) for i in range(len(recon))]
    ss = [ssim(recon[i], truth[i]) for i in range(len(recon))]
    lp = lpips_per_image(recon, truth)

    record: Dict[str, Any] = {
        "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss)),
        "lpips": (float(np.mean(lp)) if lp else None),
        "psnr_per_image": [float(v) for v in ps],
        "ssim_per_image": [float(v) for v in ss],
        "lpips_per_image": ([float(v) for v in lp] if lp else []),
        "measurement_rmse": measurement_consistency(recon, problem.subset(range(len(recon)))),
    }
    if problem.mask is not None:
        sub = problem.subset(range(len(recon)))
        record.update(masked_region_metrics(recon, truth, sub.mask))
        record["missing_ssim"] = box_region_ssim(recon, truth, sub.mask)
    return record


def degraded_baseline(problem) -> Dict[str, Any]:
    """Metrics of the displayed observation itself -- the paper's "Degraded" table row."""
    return {"psnr": psnr(problem.display_measurement, problem.ground_truth),
            "ssim": float(np.mean([ssim(problem.display_measurement[i], problem.ground_truth[i])
                                   for i in range(len(problem.ground_truth))])),
            "lpips": (lambda v: float(np.mean(v)) if v else None)(
                lpips_per_image(problem.display_measurement, problem.ground_truth)),
            "guide_psnr": psnr(problem.initialization_guide, problem.ground_truth)}
