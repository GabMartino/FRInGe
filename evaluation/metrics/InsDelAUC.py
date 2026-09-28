import torch
import torch.nn.functional as F
from typing import Dict


class CausalMetricScorer:
    def __init__(self, model, steps=50, device=None, eps: float = 1e-6):
        self.model = model.eval()
        self.steps = steps
        self.eps = eps
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.inference_mode()
    def score(
        self,
        x: torch.Tensor,
        attributions: torch.Tensor,
        baseline_mode: str = "blur",
        chunk_size: int = 8,
        normalize: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        x:            [Total_B, 3, H, W]
        attributions: [Total_B, H, W] or [Total_B, 1/3, H, W]
        Returns:
          deletion_auc, insertion_auc: [Total_B] (honoring `normalize`)
          deletion_auc_raw, insertion_auc_raw: [Total_B] (always unnormalized)

        Both curves are generated in a single pass per chunk regardless of
        `normalize`: normalization is a cheap post-hoc rescale of the same
        step-by-step insertion/deletion sweep, not a separate one, so calling
        this twice (once per normalize value) to get both variants doubles
        the model forward-pass cost for nothing. Read `*_raw` alongside the
        requested `normalize` output instead of issuing a second call.
        """
        Total_B = x.shape[0]
        all_del_auc = []
        all_ins_auc = []
        all_del_auc_raw = []
        all_ins_auc_raw = []

        for i in range(0, Total_B, chunk_size):
            batch_x = x[i: i + chunk_size].to(self.device)
            batch_attr = attributions[i: i + chunk_size].to(self.device)

            res = self._score_batch(batch_x, batch_attr, baseline_mode)
            all_del_auc.append(res["del_norm"] if normalize else res["del_raw"])
            all_ins_auc.append(res["ins_norm"] if normalize else res["ins_raw"])
            all_del_auc_raw.append(res["del_raw"])
            all_ins_auc_raw.append(res["ins_raw"])

        return {
            "deletion_auc": torch.cat(all_del_auc, dim=0),
            "insertion_auc": torch.cat(all_ins_auc, dim=0),
            "deletion_auc_raw": torch.cat(all_del_auc_raw, dim=0),
            "insertion_auc_raw": torch.cat(all_ins_auc_raw, dim=0),
        }

    def _make_baseline(self, x: torch.Tensor, baseline_mode: str) -> torch.Tensor:
        if baseline_mode == "blur":
            return self._make_blur_baseline(x)
        if baseline_mode in ["black", "zero", "zeros"]:
            return torch.zeros_like(x)
        if baseline_mode == "white":
            return torch.ones_like(x)
        return torch.zeros_like(x)

    @torch.inference_mode()
    def _score_batch(self, x: torch.Tensor, attr: torch.Tensor, baseline_mode: str):
        B, C, H, W = x.shape
        HW = H * W

        # 1) Reduce attribution to spatial [B,H,W]
        if attr.dim() == 4:
            if attr.shape[1] == 3:
                attr_spatial = attr.abs().max(dim=1)[0]
            else:
                attr_spatial = attr.squeeze(1)
        else:
            attr_spatial = attr

        attr_flat = attr_spatial.reshape(B, -1)  # [B,HW]

        # 2) Targets are the original top-1 class
        logits_x = self.model(x)
        targets = logits_x.argmax(dim=1)  # [B]

        # 3) Baseline image
        baseline = self._make_baseline(x, baseline_mode)

        # 4) Sort pixels by |attr|
        sorted_indices = torch.argsort(attr_flat.abs(), descending=True, dim=1)

        # 5) Generate raw probability curves (target prob) once
        del_curves = self._generate_curves(x, baseline, targets, sorted_indices, mode="deletion")  # [B,steps+1]
        ins_curves = self._generate_curves(x, baseline, targets, sorted_indices, mode="insertion") # [B,steps+1]

        # 6) Raw AUC
        del_auc_raw = torch.trapezoid(del_curves, dx=1.0 / self.steps, dim=1)  # [B]
        ins_auc_raw = torch.trapezoid(ins_curves, dx=1.0 / self.steps, dim=1)  # [B]

        # 7) Normalized AUC (cheap rescale of the same curves; two extra
        # forward passes for p_orig/p_base, not a second curve sweep)
        probs_x = F.softmax(logits_x, dim=1)
        p_orig = probs_x.gather(1, targets.unsqueeze(-1)).squeeze(1)  # [B]

        logits_b = self.model(baseline)
        probs_b = F.softmax(logits_b, dim=1)
        p_base = probs_b.gather(1, targets.unsqueeze(-1)).squeeze(1)  # [B]

        del_curves_norm = self._normalize_curves(del_curves, p_orig, p_base)
        ins_curves_norm = self._normalize_curves(ins_curves, p_orig, p_base)
        del_auc_norm = torch.trapezoid(del_curves_norm, dx=1.0 / self.steps, dim=1)
        ins_auc_norm = torch.trapezoid(ins_curves_norm, dx=1.0 / self.steps, dim=1)

        return {
            "del_raw": del_auc_raw,
            "ins_raw": ins_auc_raw,
            "del_norm": del_auc_norm,
            "ins_norm": ins_auc_norm,
        }

    def _normalize_curves(self, curves: torch.Tensor, p_orig: torch.Tensor, p_base: torch.Tensor) -> torch.Tensor:
        """
        curves: [B,T], p_orig/p_base: [B]
        Normalizes each sample's curve to [0,1] using baseline/original anchors.
        """
        denom = (p_orig - p_base).clamp(min=self.eps)  # [B]
        norm = (curves - p_base.unsqueeze(1)) / denom.unsqueeze(1)
        return norm.clamp(0.0, 1.0)

    @torch.inference_mode()
    def _generate_curves(self, x, baseline, targets, sorted_indices, mode):
        B, C, H, W = x.shape
        HW = H * W

        if sorted_indices.shape[1] != HW:
            raise ValueError(
                f"Sorted indices width ({sorted_indices.shape[1]}) must match HW ({HW}). "
                f"Did you forget to reduce color channels?"
            )

        pixel_steps = torch.linspace(0, HW, self.steps + 1, dtype=torch.long, device=self.device)
        curves = torch.zeros((B, self.steps + 1), device=self.device)

        canvas = x.clone() if mode == "deletion" else baseline.clone()
        source = baseline if mode == "deletion" else x

        canvas_flat = canvas.view(B, C, HW)
        source_flat = source.view(B, C, HW)

        for s in range(self.steps + 1):
            if s > 0:
                start_idx = pixel_steps[s - 1]
                end_idx = pixel_steps[s]
                if end_idx > start_idx:
                    step_indices = sorted_indices[:, start_idx:end_idx]  # [B,m]
                    step_indices_expanded = step_indices.unsqueeze(1).expand(-1, C, -1)  # [B,C,m]
                    gathered = torch.gather(source_flat, 2, step_indices_expanded)
                    canvas_flat.scatter_(2, step_indices_expanded, gathered)

            logits = self.model(canvas_flat.view(B, C, H, W))
            probs = F.softmax(logits, dim=1)
            curves[:, s] = probs.gather(1, targets.unsqueeze(-1)).squeeze(1)

        return curves

    def _make_blur_baseline(self, x):
        k = max(3, int(min(x.shape[-2:]) / 10) | 1)
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
