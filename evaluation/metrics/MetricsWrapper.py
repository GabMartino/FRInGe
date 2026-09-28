import torch
from typing import Dict, Optional

# Ensure you have renamed your classes to match the batched versions we wrote
from evaluation.metrics.Infidelity import InfidelityScorer  # The vectorized version
from evaluation.metrics.InsDelAUC import  CausalMetricScorer # RENAMED from CausalMetricScorer
from evaluation.metrics.MAS import BatchMASMetric  # RENAMED from MASMetric
from evaluation.metrics.MaxSensitivityBatched import MaxSensitivityScorer
from evaluation.metrics.Sparseness import SparsenessScorer


class MetricsWrapper:
    def __init__(self,
                 attr_method,
                 model: torch.nn.Module,
                 device: Optional[torch.device] = None,
                 causal_steps: int = 100,
                 causal_baseline: str = "blur",
                 mas_step_size: int = 224,
                 mas_baseline: str = "blur",
                 mas_klen: int = 15,
                 mas_nsig: int = 3):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.attr_method = attr_method

        # 1. FIXED: Using the Chunked version we built for batches
        self.causal_scorer = CausalMetricScorer(
            self.model,
            steps=causal_steps,
            device=self.device
        )
        self.causal_baseline = causal_baseline

        # 2. FIXED: Using the BatchMASMetric instead of the old single_run MASMetric
        self.mas_scorer = BatchMASMetric(
            self.model,
            step_size=mas_step_size,
            baseline_type=mas_baseline,
            klen=mas_klen,
            nsig=mas_nsig,
            device=self.device
        )

        # 3. Max Sensitivity (Ensure this class also accepts device)
        self.max_sensitivity_scorer = MaxSensitivityScorer(attr_method)

        # 4. Infidelity (The vectorized version)
        self.infidelity_scorer = InfidelityScorer(self.model, device=self.device)

        # 5. Sparseness
        self.sparseness_scorer = SparsenessScorer()

    def _get_targets(self, inputs: torch.Tensor) -> torch.Tensor:
        """Helper to get the model's top predicted class for the batch."""
        with torch.no_grad():
            logits = self.model(inputs.to(self.device))
            return logits.argmax(dim=1)

    def extract_insertion_deletion_auc(self,
                                       inputs: torch.Tensor,
                                       attributions: torch.Tensor,
                                       chunk_size: int = 8) -> Dict[str, torch.Tensor]:
        # This now correctly calls the score method of ChunkedCausalScorer
        return self.causal_scorer.score(
            inputs,
            attributions,
            baseline_mode=self.causal_baseline,
            chunk_size=chunk_size
        )

    def extract_mas_score(self,
                          inputs: torch.Tensor,
                          attributions: torch.Tensor,
                          inference_batch_size: int = 50) -> Dict[str, torch.Tensor]:
        # This now correctly calls score_batch of BatchMASMetric
        return self.mas_scorer.score_batch(
            inputs,
            attributions,
            inference_batch_size=inference_batch_size
        )

    def extract_infidelity_score(self,
                                 inputs: torch.Tensor,
                                 attributions: torch.Tensor,
                                 target_indices: Optional[torch.Tensor] = None,
                                 n_perturbations: int = 50,
                                 noise_scale: float = 0.02,
                                 seed: int = 42,
                                 chunk_size: int = 10) -> torch.Tensor:
        if target_indices is None:
            target_indices = self._get_targets(inputs)

        # Calls the vectorized infidelity score
        return self.infidelity_scorer.score(
            inputs,
            attributions,
            target_indices,
            n_perturbations=n_perturbations,
            noise_scale=noise_scale,
            seed=seed,
            max_examples_per_pass=chunk_size
        )

    def extract_sparseness_score(self, attributions: torch.Tensor) -> torch.Tensor:
        return self.sparseness_scorer.score(attributions)