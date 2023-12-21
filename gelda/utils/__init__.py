from typing import Any
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

IMAGE_TRANSFORMS = {
    "from_numpy": transforms.Compose(
        [
            torch.from_numpy,
            transforms.Lambda(lambda x: x / 255.),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ]
    ),
    "from_pil_resize_224": transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ]
    ),
    "from_pil_resize_384": transforms.Compose(
        [
            transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ]
    )
}


class EasyDict(dict):
    """
    Convenience class that behaves like a dict but allows access with the attribute syntax.
    Taken from: https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main/dnnlib
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def annotations_to_df(
        attributes_dict: dict,  # attributes_dict is the output of gelda.generate_attributes()
        annotations_dict: dict,  # annotations_dict is the output of gelda.generate_annotations()
        blip_threshold: float = 0  # threshold for BLIP attribute caption score above base caption score. Default is 0.
):
    """
    Create a pandas dataframe from the generated attributes and annotations.
    """
    # Initialize dataframe dictionary to store annotations in
    df_dict = dict(filenames=annotations_dict['filenames'].copy())

    # Loop through each attribute category
    for attr_c, attr_labels_dict in attributes_dict['attributes'].items():
        attr_list = attr_labels_dict['labels'].copy()

        # Multilabel annotations for objects and items (i.e. annotations from OWL)
        if attr_labels_dict['is_object']:
            for attr in attr_list:
                # loop through each label and identify if
                for det in annotations_dict[f"{attr_c}_{attr}"]:
                    df_dict.setdefault(f"{attr_c}_{attr}", []).append(len(det['scores']) > 0)

        # Multiclass annotations for image concepts (i.e. annotations from BLIP)
        else:
            # get all label scores
            scores_per_attribute = []
            for attr in attr_list:
                scores_per_attribute.append(annotations_dict[f"{attr_c}_{attr}"])
            scores_per_attribute = np.array(scores_per_attribute)
            # select attribute corresponding to best score
            best_attr_idxs = np.argmax(scores_per_attribute, axis=0)
            pred_attrs = np.array([attr_list[attr_idx] for attr_idx in best_attr_idxs])
            # if best score is below zero (i.e. below base prompt), attribute is undetermined
            pred_attrs[np.max(scores_per_attribute, axis=0) < - blip_threshold] = f"unknown {attr_c}"
            df_dict[attr_c] = pred_attrs
    return pd.DataFrame.from_dict(df_dict)
