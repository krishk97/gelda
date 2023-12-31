import os
import argparse
import json
import pickle as pkl

import torch

from gelda import generate_annotations


def get_arguments():
    parser = argparse.ArgumentParser(description="Get annotations using VLMs")
    parser.add_argument(
        "--data_path", "-p",
        type=str,
        help="path to images",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="custom",  # 'custom', 'deepfashion', 'cub', 'celeba'
        help="dataset name",
    )
    parser.add_argument(
        "--attr", "-a",
        type=str,
        default="attributes",
        help="path to attributes file",
    )
    parser.add_argument(
        "--save", "-s",
        type=str,
        default="output/vlm_labels.pkl",
        help="path to save labels generated by VLMs",
    )
    parser.add_argument(
        "--blip_model",
        type=str,
        default="Salesforce/blip-itm-large-coco",  # "Salesforce/blip-itm-base-coco", "Salesforce/blip-itm-large-coco"
        help="architecture name",
    )
    parser.add_argument(
        "--owl_model",
        type=str,
        default="google/owlv2-large-patch14-ensemble",
        help="architecture name",
    )
    parser.add_argument(
        "--base_text",
        action="store_true",
        help="use base text",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.3,
        help="threshold for owl object detection",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=1,
    )

    return parser.parse_args()


def main(args):
    # Load attributes
    with open(args.attr, "rb") as f:
        attributes_dict = json.load(f)

    # Generate annotations
    results = generate_annotations(attributes_dict,
                                   args.data_path,
                                   dataset_module=args.dataset,
                                   blip_model_name=args.blip_model,
                                   owl_model_name=args.owl_model,
                                   device="cuda" if torch.cuda.is_available() else "cpu",
                                   batch_size=args.batch_size,
                                   threshold=args.threshold,
                                   base_text=args.base_text)

    # Save results
    save_dir = os.path.dirname(args.save_path)
    save_fname = os.path.basename(args.save_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, save_fname), "wb") as f:
        print(f"Saving results to file: {args.save}")
        pkl.dump(results, f)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
