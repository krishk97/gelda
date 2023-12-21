import os
from typing import Optional

import numpy as np
import PIL
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
from torchvision.utils import make_grid

from . import annotations_to_df


def plot_image_grid(imgs, ratio=0.05, ax=None, **make_grid_kwargs):
    """
    Plot grid of images
    """
    grid_img = make_grid(imgs, **make_grid_kwargs)
    fig_dims = (int(ratio * grid_img.shape[-1]), int(ratio * grid_img.shape[-2]))
    grid_img = to_pil_image(grid_img.detach())
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_dims)
    else:
        fig = ax.get_figure()
    ax.imshow(grid_img)
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    return grid_img, fig, ax


def plot_annotation_barplot(attribute_dict,
                            annotations_dict):
    # get annotations dataframe
    annotation_df = annotations_to_df(attribute_dict, annotations_dict)

    # initialize list to store sorted image proportions and attributes
    proportions_list_sorted = []
    attr_list_sorted = []
    # loop through attribute categories
    for attr_c in attribute_dict['attributes'].keys():
        # get attribute labels for category
        attr_list = attribute_dict['attributes'][attr_c]['labels'].copy()
        # save proportions in list
        proportions_list = []
        # loop through each attribute label
        for attr in attr_list:
            # owl detections -- is_object = True
            if attribute_dict['attributes'][attr_c]["is_object"]:
                _df = annotation_df[annotation_df[f"{attr_c}_{attr}"]]
            # blip multiclassification -- is_object = False
            else:
                _df = annotation_df[annotation_df[attr_c] == attr]
                # save proportions of attribute
            proportions_list.append(len(_df) / len(annotation_df))

        # sort proportions
        sorted_indxs = np.flip(np.argsort(proportions_list))
        proportions_list_sorted.append(np.array(proportions_list)[sorted_indxs])
        attr_list_sorted.extend([f"{attr_c}: {attr_list[idx]}" for idx in sorted_indxs])

    proportions_list_sorted = np.concatenate(proportions_list_sorted)

    # plot bar chart
    fig, ax = plt.subplots(figsize=(4, 12))
    ax = sns.barplot(x=proportions_list_sorted * 100.,
                     y=attr_list_sorted,
                     ax=ax)
    ax.set_xlabel("Proportion of images (%)")
    ax.set_xlim([0, 100])
    ax.grid(visible=True, axis='x')
    ax.grid(visible=True, axis='y', linestyle='--')
    ax.set_yticklabels(attr_list_sorted)

    return fig, ax


def plot_annotation_examples(
        attribute_dict: dict,  # attributes_dict is the output of gelda.generate_attributes()
        annotations_dict: dict,  # annotations_dict is the output of gelda.generate_annotations()
        attribute_category: str,  # attribute category to plot
        img_dir: Optional[str] = None,  # Optional path to image directory
        n_examples=10,  # number of examples to plot (Default: 10)
        n_rows=5  # number of images per row (Default: 5)
):

    if attribute_dict['attributes'][attribute_category]['is_object']:
        figs_dict = _plot_annotation_examples_owl(attribute_dict,
                                                  annotations_dict,
                                                  attribute_category,
                                                  img_dir=img_dir,
                                                  n_examples=n_examples,
                                                  n_rows=n_rows)

    else:
        annotation_df = annotations_to_df(attribute_dict, annotations_dict)
        figs_dict = _plot_annotation_examples_blip(attribute_dict,
                                                   annotation_df,
                                                   attribute_category,
                                                   img_dir=img_dir,
                                                   n_examples=n_examples,
                                                   n_rows=n_rows)

    # return figure dictionary containing figure, axes, and grid image for each attribute label in category
    return figs_dict


def _plot_annotation_examples_blip(
        attribute_dict,
        annotation_df,
        attribute_category,
        img_dir=None,
        n_examples=10,
        n_rows=5
):
    # Get attribute labels for category
    attr_list = attribute_dict['attributes'][attribute_category]['labels'].copy()
    # dictionary to store attribute figures and axes
    figs_dict = dict()
    # loop through each attribute label
    for attr in attr_list:
        _df = annotation_df[annotation_df[attribute_category] == attr]

        # Visualize categories
        if len(_df) > 0:
            plot_imgs = []
            for i in np.random.choice(len(_df), min(len(_df), n_examples), replace=False):
                img_path = _df.iloc[i]['filenames'] if img_dir is None else os.path.join(img_dir, _df.iloc[i]['filenames'])
                img = PIL.Image.open(img_path).resize((384, 384))
                plot_imgs.append(pil_to_tensor(img))
            grid_img, fig, ax = plot_image_grid(plot_imgs, ratio=0.01, nrow=n_rows)
            figs_dict[attr] = {'fig': fig, 'ax': ax, 'grid_img': grid_img}

    return figs_dict


def _plot_annotation_examples_owl(
        attribute_dict,
        annotations_dict,
        attribute_category,
        img_dir=None,
        n_examples=10,
        n_rows=5
):
    # get attribute labels for category
    attr_list = attribute_dict['attributes'][attribute_category]['labels'].copy()

    # dictionary to store attribute figures and axes
    figs_dict = dict()

    # loop through each attribute label
    for i, attr in enumerate(attr_list):
        key = f"{attribute_category}_{attr}"
        attr_examples_list = [{'file': fname, 'bboxes': dets['bboxes'], 'scores': dets['scores']}
                              for k, (fname, dets) in enumerate(zip(annotations_dict['filenames'], annotations_dict[key]))
                              if len(dets['scores']) > 0]

        if len(attr_examples_list) > 0:
            plot_imgs = []
            for attr_example in np.random.choice(attr_examples_list, min(len(attr_examples_list), n_examples),
                                                 replace=False):
                img_path = attr_example['file'] if img_dir is None else os.path.join(img_dir, attr_example['file'])
                img = PIL.Image.open(img_path).resize((384, 384))  # resize to shape used for OWL detections

                # draw bounding boxes
                img_np = np.array(img)
                for box in attr_example['bboxes']:
                    img_np = cv2.rectangle(img_np,
                                           (int(box[0]), int(box[1])),
                                           (int(box[2]), int(box[3])),
                                           color=(255, 0, 0),
                                           thickness=4)
                img_with_bbox = torch.from_numpy(img_np).permute(2, 0, 1)
                plot_imgs.append(img_with_bbox)
            grid_img, fig, ax = plot_image_grid(plot_imgs, ratio=0.01, nrow=n_rows)
            figs_dict[key] = {'fig': fig, 'ax': ax, 'grid_img': grid_img}

    return figs_dict
