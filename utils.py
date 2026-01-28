import json
import os
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from torchvision.models import resnet18, resnet50, vgg19, inception_v3, VGG19_Weights, Inception_V3_Weights, \
    ResNet50_Weights, ResNet18_Weights, ResNet152_Weights, resnet152, ResNet101_Weights, resnet101
from torchvision.transforms import transforms, InterpolationMode



def load_model(model_name, device="cuda"):
    model, weights = None, None
    if model_name == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    elif model_name == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    elif model_name == 'vgg19':
        weights = VGG19_Weights.IMAGENET1K_V1
        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
    elif model_name == "resnet152":
        weights = ResNet152_Weights.IMAGENET1K_V1
        model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)

    elif model_name == "resnet101":
        weights = ResNet101_Weights.IMAGENET1K_V1
        model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)

    elif model_name == 'inception_v3':
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    else:
        raise ValueError('Invalid model name')
    #model = torch.compile(model)
    return model.to(device).eval(), weights.transforms()



def normalize_image(image):
    preprocess = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

    return preprocess(image)


def denormalize_image(image):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    import torchvision.transforms as T

    inv_normalize = T.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1 / s for s in std]
    )
    return inv_normalize(image)

def load_image(path, preprocess, device="cuda"):

    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    return x

def append_results(path, results):
    # Load existing data if file exists
    if os.path.exists(path):
        with open(path, "r") as f:
            existing_results = json.load(f)
    else:
        existing_results = {
            "Insertion AUC": [],
            "Deletion AUC": [],
            "AIC AUC": [],
            "SIC AUC": [],
            "Infidelity": []
        }

    for key, val in results.items():
        # Ensure key exists in existing_results
        if key not in existing_results:
            existing_results[key] = []

        # If val is a list/tuple/etc. -> extend
        if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
            existing_results[key].extend(val)
        else:
            # Single scalar -> append
            existing_results[key].append(val)

    # Write updated results
    with open(path, "w") as f:
        json.dump(existing_results, f, indent=4)




def preprocess_attributions(attr):
    #attr = attr.cpu().detach().numpy()
    lower_bound = np.percentile(attr, 80)
    upper_bound = np.percentile(attr, 99)

    attr[attr < lower_bound] = lower_bound
    attr[attr > upper_bound] = upper_bound

    attr = (attr - attr.max())/ (attr.max() - attr.min())

    return attr



def save_attribution(cfg, image_path, attributions, image):

    path = "results"
    path = os.path.join(path, cfg.method_name, cfg.model)
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, image_path.split("/")[-1])

    attr = attributions.sum(1, keepdim=True).cpu().squeeze().numpy()
    if attr.ndim == 3 and attr.shape[2] == 1:
        attr = attr[..., 0]  # -> (H,W)

    # Use positive attributions (or np.abs(attr) if you prefer magnitude)
    #attr = np.maximum(attr, 0)
    attr = np.abs(attr)
    # Robust normalize to [0,1] (percentile avoids outliers). Use min-max if you prefer.
    hi = np.percentile(attr, 99.0)
    attr_norm = np.clip(attr / (hi + 1e-12), 0, 1)  # (H,W)

    # --- prep image: (H,W,3) in [0,1] ---
    image = image.cpu()
    image = denormalize_image(image).squeeze().numpy().transpose(1, 2, 0)  # (H,W,3)
    image = np.clip(image, 0, 1)

    # --- highlight only salient regions ---
    mask3 = attr_norm[..., None]  # (H,W,1) -> broadcast to 3 channels
    highlight = image * mask3  # (H,W,3)

    plt.imshow(highlight)
    plt.axis('off')
    plt.savefig(path)

