import os

import hydra
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from tqdm import tqdm

from metrics.MetricsWrapper import MetricsWrapper
from utils import load_model, load_image  # Assuming these exist in your env



@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):

    images_path = os.listdir("../examples")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    avail_models = cfg.avail_models


    cfg = cfg.IG
    for model_name in avail_models:
        print("Loading model: ", model_name)
        all_results_metrics = []
        all_attributions = []

        model, transform = load_model(model_name, device=device)
        model.eval()
        for image_path in tqdm(images_path[:1]):
            results_metrics = {}
            results_attributions = {}
            img = load_image(os.path.join("../examples", image_path), transform, device=device)

            with torch.no_grad():
                target_idx = model(img).argmax(dim=1)

            def model_forward_batch(x):
                logits = model(x)
                return logits[:, target_idx]

            ig_classic = IntegratedGradients(model_forward_batch)
            attr_ig, delta = ig_classic.attribute(inputs=img,
                                                    baselines=torch.zeros_like(img),
                                                    n_steps=cfg.n_steps,
                                                    method="riemann_trapezoid",
                                                    return_convergence_delta=True)

            metrics = MetricsWrapper(ig_classic, model)

            ############## INSERTION/DELETION AUC
            ins_del_scores = metrics.extract_insertion_deletion_auc(img, attr_ig)
            results_metrics["insertion_auc"] = ins_del_scores["insertion_auc"]
            results_metrics["deletion_auc"] = ins_del_scores["deletion_auc"]

            ############# MAS INSERTION/DELETION
            mas_scores = metrics.extract_mas_score(img, attr_ig)
            results_metrics["mas_insertion_auc"] = mas_scores["insertion"]
            results_metrics["mas_deletion_auc"] = mas_scores["deletion"]

            ############# MAX Sensitivity

            max_sensitivity_score = metrics.extract_max_sensitivity_score(img, attr_ig,
                                                                          baselines=torch.zeros_like(img),
                                                                          n_steps=cfg.n_steps,
                                                                          method="riemann_trapezoid",
                                                                          return_convergence_delta=True)
            results_metrics["max_sensitivity_score"] = max_sensitivity_score.item()
            ################### INFIDELITY

            infidelity_score = metrics.extract_infidelity_score(img, attr_ig, target_idx=target_idx,
                                                                n_perturbations=50,
                                                                noise_scale=0.02)
            results_metrics["infidelity_score"] = infidelity_score.item()

            ####################### SPARSENESS

            sparseness_score = metrics.extract_sparseness_score(attr_ig)
            results_metrics["sparseness_score"] = sparseness_score.item()

            ################################################
            results_metrics["image_path"] = image_path
            results_metrics["delta"] = delta.item()

            results_attributions["image_path"] = image_path
            results_attributions["attr"] = attr_ig.to("cpu").detach().numpy().tolist()

            all_results_metrics.append(results_metrics)
            all_attributions.append(results_attributions)
            del img
        df = pd.DataFrame(all_results_metrics)
        # If you want to keep it simple:
        os.makedirs(f"./results/IG/{model_name}", exist_ok=True)
        df.to_parquet(f"./results/IG/{model_name}/results_metrics.parquet", index=False, compression="zstd")
        del df
        df = pd.DataFrame(all_attributions)
        df.to_parquet(f"./results/IG/{model_name}/results_attributions.parquet", index=False, compression="zstd")


        del model




if __name__ == "__main__":
    main()