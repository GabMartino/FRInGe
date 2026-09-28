import torch
import torch.nn.functional as F


class BatchMASMetric:
    def __init__(
        self,
        model,
        step_size: int = 224,
        baseline_type: str = "black",
        klen: int = 15,
        nsig: float = 3.0,
        device=None,
        eps: float = 1e-6,
        enforce_monotonicity: bool = True,
    ):
        self.model = model.eval()
        self.step_size = int(step_size)
        self.baseline_type = str(baseline_type).lower()
        self.klen = int(klen)
        self.nsig = float(nsig)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = float(eps)
        self.enforce_monotonicity = bool(enforce_monotonicity)

        self.model.to(self.device)

    @torch.inference_mode()
    def score_batch(self, img_tensors, saliency_maps, inference_batch_size: int = 50):
        """
        Processes a batch of images and attributions.
        Returns: Dict[str, torch.Tensor] containing AUCs of shape [B]

        NOTE: inference_batch_size is currently unused (kept for API compatibility).
        """
        img_tensors = img_tensors.to(self.device)
        B, C, H, W = img_tensors.shape
        HW = H * W

        # 1) Preprocess saliency -> [B, HW] (importance is magnitude)
        if saliency_maps.ndim == 4:
            # Accept [B,3,H,W] or [B,1,H,W]
            if saliency_maps.shape[1] == 3:
                saliency_maps = saliency_maps.abs().max(dim=1)[0]
            else:
                saliency_maps = saliency_maps.squeeze(1)
        saliency_flat = saliency_maps.to(self.device).reshape(B, -1).abs()
        sorted_indices = torch.argsort(saliency_flat, descending=True, dim=1)  # [B,HW]

        # 2) Baseline and anchors
        baselines = self._get_baseline(img_tensors)  # [B,C,H,W]
        n_steps = (HW + self.step_size - 1) // self.step_size

        logits = self.model(img_tensors)
        targets = logits.argmax(dim=1)  # [B]

        orig_probs = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(-1)).squeeze(1)  # [B]
        base_probs = F.softmax(self.model(baselines), dim=1).gather(1, targets.unsqueeze(-1)).squeeze(1)  # [B]

        # valid_mask: denom not too small
        denom = (orig_probs - base_probs).abs()
        valid_mask = denom > (10.0 * self.eps)

        results = {}
        for mode in ["del", "ins"]:
            # 3) Density curve based on attribution mass
            d_resp = self._calculate_density_batch(saliency_flat, sorted_indices, n_steps, mode)  # [B,n_steps+1]

            # 4) Model response curve
            m_resp = self._get_model_response_batch(
                img_tensors, baselines, sorted_indices, n_steps, targets, mode
            )  # [B,n_steps+1]

            # 5) Normalize response curve to [0,1] using (orig, base) anchors
            norm_mr, valid_mask = self._normalize_batch(m_resp, orig_probs, base_probs, mode)

            # 6) MAS score curve
            mas_curve = self._compute_mas_batch(norm_mr, d_resp, mode)

            # 7) AUC
            auc = torch.trapezoid(mas_curve, dx=1.0 / n_steps, dim=1)  # [B]
            results["deletion" if mode == "del" else "insertion"] = auc

        # Optional: expose mask (handy for debugging/reporting)
        results["valid_mask"] = valid_mask
        return results

    def _get_baseline(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        imgs: [B,C,H,W]
        Returns baseline images with same shape/device/dtype.
        """
        t = self.baseline_type
        if t in ["black", "zero", "zeros"]:
            return torch.zeros_like(imgs)
        if t in ["white", "one", "ones"]:
            return torch.ones_like(imgs)
        if t == "blur":
            k = max(3, self.klen | 1)  # ensure odd >=3
            sigma = max(self.eps, self.nsig)

            # Build Gaussian kernel on correct device/dtype
            x = torch.linspace(-(k - 1) / 2, (k - 1) / 2, k, device=imgs.device, dtype=imgs.dtype)
            gauss = torch.exp(-0.5 * (x / sigma) ** 2)
            kernel2d = (gauss[:, None] * gauss[None, :])
            kernel2d = kernel2d / kernel2d.sum().clamp_min(self.eps)

            # Depthwise conv kernel: [C,1,k,k]
            kernel = kernel2d.view(1, 1, k, k).repeat(imgs.shape[1], 1, 1, 1)
            return F.conv2d(imgs, kernel, padding=k // 2, groups=imgs.shape[1])

        # fallback
        return torch.zeros_like(imgs)

    def _calculate_density_batch(self, saliency_flat, sorted_indices, n_steps, mode):
        """
        Density response based on attribution mass removed/added.
        Returns [B, n_steps+1]
        """
        B, HW = saliency_flat.shape
        total_attr = saliency_flat.sum(dim=1, keepdim=True).clamp_min(self.eps)

        density = torch.zeros((B, n_steps + 1), device=self.device, dtype=saliency_flat.dtype)
        density[:, 0] = 1.0 if mode == "del" else 0.0

        current_mass = torch.zeros((B, 1), device=self.device, dtype=saliency_flat.dtype)
        for s in range(n_steps):
            idx_s = s * self.step_size
            idx_e = min((s + 1) * self.step_size, HW)
            step_idx = sorted_indices[:, idx_s:idx_e]

            current_mass += torch.gather(saliency_flat, 1, step_idx).sum(dim=1, keepdim=True)
            frac = (current_mass / total_attr).squeeze(1)  # [B]

            density[:, s + 1] = (1.0 - frac) if mode == "del" else frac

        return density

    @torch.inference_mode()
    def _get_model_response_batch(self, imgs, baselines, sorted_indices, n_steps, targets, mode):
        """
        Returns model response curve (target prob) of shape [B, n_steps+1]
        """
        B, C, H, W = imgs.shape
        HW = H * W

        canvas = (imgs if mode == "del" else baselines).clone().view(B, C, HW)
        finish = (baselines if mode == "del" else imgs).view(B, C, HW)

        responses = torch.zeros((B, n_steps + 1), device=self.device, dtype=imgs.dtype)

        # initial
        logits0 = self.model(canvas.view(B, C, H, W))
        p0 = F.softmax(logits0, dim=1).gather(1, targets.unsqueeze(-1)).squeeze(1)
        responses[:, 0] = p0

        for s in range(n_steps):
            idx_s = s * self.step_size
            idx_e = min((s + 1) * self.step_size, HW)
            step_idx = sorted_indices[:, idx_s:idx_e]  # [B,m]
            expanded_idx = step_idx.unsqueeze(1).expand(-1, C, -1)  # [B,C,m]

            canvas.scatter_(2, expanded_idx, torch.gather(finish, 2, expanded_idx))

            logits = self.model(canvas.view(B, C, H, W))
            p = F.softmax(logits, dim=1).gather(1, targets.unsqueeze(-1)).squeeze(1)
            responses[:, s + 1] = p

        return responses

    def _normalize_batch(self, m_resp, orig_p, base_p, mode):
        """
        Normalize model response curve to [0,1] using anchors:
          - orig_p = p(target | x)
          - base_p = p(target | baseline)
        Expected: orig_p > base_p for meaningful insertion/deletion normalization.

        Returns:
          norm_mr: (B, T)
          valid_mask: (B,) bool
        """
        orig_p = orig_p.view(-1, 1)
        base_p = base_p.view(-1, 1)

        denom = orig_p - base_p
        eps = 1e-6

        # Valid only if baseline is truly "worse" than original for the target prob
        valid_mask = (denom > eps).squeeze(1)  # (B,)

        # Avoid divide-by-zero (we'll zero-out invalid samples later)
        denom_safe = torch.where(valid_mask.view(-1, 1), denom, torch.ones_like(denom))

        norm_mr = (m_resp - base_p) / denom_safe
        norm_mr = torch.clamp(norm_mr, 0.0, 1.0)

        # Force invalid curves to 0 to avoid polluting averages
        norm_mr = torch.where(valid_mask.view(-1, 1), norm_mr, torch.zeros_like(norm_mr))

        # Enforce monotonicity (vectorized)
        if mode == "del":
            # deletion should be non-increasing -> cumulative minimum
            norm_mr = torch.cummin(norm_mr, dim=1).values
        else:
            # insertion should be non-decreasing -> cumulative maximum
            norm_mr = torch.cummax(norm_mr, dim=1).values

        return norm_mr, valid_mask

    def _compute_mas_batch(self, norm_mr, d_resp, mode):
        penalty = (d_resp - norm_mr).abs()
        if mode == "del":
            return (norm_mr + penalty).clamp(0.0, 1.0)
        return (norm_mr - penalty).clamp(0.0, 1.0)
