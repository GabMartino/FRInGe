import torch
import numpy as np
from scipy.ndimage import gaussian_filter


def auc(arr):
    """Returns normalized Area Under Curve of the array (Trapezoidal rule)."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class MASMetric:
    def __init__(self, model, step_size=224, baseline_type='blur', klen=15, nsig=3):
        """
        Args:
            model (nn.Module): Black-box model.
            step_size (int): Pixels modified per iteration.
            baseline_type (str): 'blur', 'black', 'white', or 'mean'.
            klen (int): Kernel size (for 'blur' baseline).
            nsig (int): Sigma (for 'blur' baseline).
        """
        self.model = model
        self.step_size = step_size
        self.baseline_type = baseline_type
        self.klen = klen
        self.nsig = nsig

    def _get_baseline(self, img_tensor):
        """Generates the baseline image based on the selected type."""
        if self.baseline_type == 'blur':
            return self._blur_image(img_tensor)
        elif self.baseline_type == 'black':
            return torch.zeros_like(img_tensor)
        elif self.baseline_type == 'white':
            return torch.ones_like(img_tensor)
        elif self.baseline_type == 'mean':
            # Channel-wise mean
            mean_vals = img_tensor.mean(dim=(2, 3), keepdim=True)
            return mean_vals.expand_as(img_tensor)
        else:
            raise ValueError(f"Unknown baseline_type: {self.baseline_type}")

    def _blur_image(self, img_tensor):
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # Gaussian Kernel generation
        inp = np.zeros((self.klen, self.klen))
        inp[self.klen // 2, self.klen // 2] = 1
        k = gaussian_filter(inp, self.nsig)
        kern = torch.from_numpy(k[None, None, :, :].astype('float32'))

        channels = img_tensor.shape[1]
        weight = kern.expand(channels, 1, self.klen, self.klen).to(img_tensor.device)
        padding = self.klen // 2
        return torch.nn.functional.conv2d(img_tensor, weight, padding=padding, groups=channels)

    def single_run(self, img_tensor, saliency_map, device, batch_size=50):
        """Returns: dict {'deletion': float, 'insertion': float}"""
        img_tensor = img_tensor.to(device)
        _, _, H, W = img_tensor.shape
        HW = H * W

        # 1. Preprocess Saliency
        if hasattr(saliency_map, 'cpu'): saliency_map = saliency_map.cpu().numpy()
        if saliency_map.ndim == 4: saliency_map = saliency_map.squeeze(0)
        if saliency_map.ndim == 3 and saliency_map.shape[0] == 3:
            saliency_map = np.max(np.abs(saliency_map), axis=0)

        saliency_flat = np.abs(saliency_map).flatten()
        sorted_indices = np.argsort(saliency_flat)[::-1].copy()

        # 2. Setup
        n_steps = (HW + self.step_size - 1) // self.step_size
        baseline = self._get_baseline(img_tensor)

        with torch.no_grad():
            target_class = self.model(img_tensor).argmax(dim=1).item()

        results = {}
        # We calculate both by iterating through modes
        for mode in ['del', 'ins']:
            start_img, finish_img = (img_tensor, baseline) if mode == 'del' else (baseline, img_tensor)

            d_resp = self._calculate_density(saliency_flat, sorted_indices, n_steps, mode)
            m_resp = self._get_model_response(start_img, finish_img, sorted_indices,
                                              n_steps, batch_size, device, H, W, target_class)

            norm_mr = self._normalize_model_response(m_resp, mode)
            scores = self._compute_mas_score(norm_mr, d_resp, mode)
            results['deletion' if mode == 'del' else 'insertion'] = auc(scores)

        return results

    def _calculate_density(self, saliency_flat, sorted_indices, n_steps, mode):
        total_attr = saliency_flat.sum() or 1e-9
        density_response = np.zeros(n_steps + 1)
        density_response[0] = 1.0 if mode == 'del' else 0.0
        current_mass = 0.0

        for k in range(n_steps):
            idx_start, idx_end = k * self.step_size, min((k + 1) * self.step_size, len(sorted_indices))
            current_mass += saliency_flat[sorted_indices[idx_start:idx_end]].sum()
            fraction = current_mass / total_attr
            density_response[k + 1] = (1.0 - fraction) if mode == 'del' else fraction
        return density_response

    def _get_model_response(self, start_img, finish_img, sorted_indices, n_steps, batch_size, device, H, W,
                            target_class):
        with torch.no_grad():
            start_score = torch.softmax(self.model(start_img), dim=1)[0, target_class].item()

        model_response = np.zeros(n_steps + 1)
        model_response[0] = start_score
        current_flat = start_img.view(1, 3, -1).clone()
        finish_flat = finish_img.view(1, 3, -1)

        for i in range(0, n_steps, batch_size):
            batch_images = []
            this_batch = min(batch_size, n_steps - i)
            for j in range(this_batch):
                step = i + j
                idx_start, idx_end = step * self.step_size, min((step + 1) * self.step_size, len(sorted_indices))
                current_flat[0, :, sorted_indices[idx_start:idx_end]] = finish_flat[
                    0, :, sorted_indices[idx_start:idx_end]]
                batch_images.append(current_flat.view(1, 3, H, W).clone())

            with torch.no_grad():
                preds = self.model(torch.cat(batch_images, dim=0))
                scores = torch.softmax(preds, dim=1)[:, target_class].cpu().numpy()
            model_response[i + 1: i + 1 + this_batch] = scores
        return model_response

    def _normalize_model_response(self, model_response, mode):
        min_s, max_s = np.min(model_response), np.max(model_response)
        denom = max_s - min_s if (max_s - min_s) > 1e-9 else 1.0
        norm_mr = np.clip((model_response - min_s) / denom, 0, 1)
        for i in range(1, len(norm_mr)):
            norm_mr[i] = min(norm_mr[i - 1], norm_mr[i]) if mode == 'del' else max(norm_mr[i - 1], norm_mr[i])
        return norm_mr

    def _compute_mas_score(self, norm_mr, density_response, mode):
        penalty = np.abs(density_response - norm_mr)
        scores = (norm_mr + penalty) if mode == 'del' else (norm_mr - penalty)
        scores = np.clip(scores, 0, 1)
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min > 1e-9:
            return (scores - s_min) / (s_max - s_min)
        return np.linspace(1, 0, len(scores)) if mode == 'del' else np.linspace(0, 1, len(scores))