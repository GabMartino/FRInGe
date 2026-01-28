import torch

from metrics.Infidelity import InfidelityScorer
from metrics.InsDelAUC import CausalMetricScorer
from metrics.MAS import MASMetric
from metrics.MaxSensitivityBatched import MaxSensitivityScorer
from metrics.Sparseness import SparsenessScorer


class MetricsWrapper:


    def __init__(self,
                 attr_method,
                 model: torch.nn.Module,
                 causal_metric_scorer_steps: int = 100,
                 causal_metric_scorer_baseline: str = "blur",
                 mas_metric_step_size: int = 224,
                 mas_metric_baseline: str = "blur",
                 mas_metric_klen: int = 15,
                 mas_metric_nsig: int = 3):
        self.attr_method = attr_method
        #########################################
        self.causal_metrics = CausalMetricScorer(model,
                                                 steps=causal_metric_scorer_steps)
        self.causal_metric_scorer_baseline = causal_metric_scorer_baseline
        ########################################

        self.mas_metric = MASMetric(model=model,
                                    step_size = mas_metric_step_size,
                                    baseline_type=mas_metric_baseline,
                                    klen = mas_metric_klen,
                                    nsig=mas_metric_nsig)

        #################################################
        self.max_sensitivity = MaxSensitivityScorer(attr_method)
        ###################################################

        self.infidelity = InfidelityScorer(model)
        ##############################################

        self.sparseness = SparsenessScorer()

    def extract_insertion_deletion_auc(self, input, attributions):
        print("Computing insertion deletion AUC")
        return self.causal_metrics.score(input, attributions, self.causal_metric_scorer_baseline)


    def extract_mas_score(self, input, attributions, batch_size = 50):
        print("Computing mas score")
        device = input.device
        return self.mas_metric.single_run(input, attributions, device, batch_size)


    def extract_max_sensitivity_score(self,  input: torch.Tensor,
                                            attributions: torch.Tensor = None,
                                            radius: float = 0.02,
                                            n_perturbations: int = 10,
                                                **attr_kwargs):
        print("Computing max sensitivity score")
        return self.max_sensitivity.score(input, attributions, radius, n_perturbations, **attr_kwargs)

    def extract_infidelity_score(self, input: torch.Tensor,
                                 attributions: torch.Tensor,
                                 target_idx: int,
                                 n_perturbations: int = 50,
                                 noise_scale: float = 0.02):
        return self.infidelity.score(input, attributions, target_idx, n_perturbations, noise_scale)


    def extract_sparseness_score(self, attributions: torch.Tensor):
        print("Computing sparseness score")
        return self.sparseness.score(attributions)