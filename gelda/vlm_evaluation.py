import copy
from tqdm import tqdm

import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection, AutoProcessor, BlipForImageTextRetrieval

from .datasets import ImageFolderDataset, DeepFashionDataset, CUBDataset, CelebADataset
from .utils import IMAGE_TRANSFORMS

def get_dataset(dataset_module: str, data_path: str):
    print(f"Loading dataset {dataset_module}")
    if dataset_module == "celeba":
        dataset = CelebADataset(data_path,
                                split="test",
                                transform=IMAGE_TRANSFORMS["from_pil_resize_384"])
    elif dataset_module == "deepfashion":
        dataset = DeepFashionDataset(data_path,
                                     split="test",
                                     resolution=384,
                                     use_labels=False,
                                     transform=IMAGE_TRANSFORMS["from_numpy"])
    elif dataset_module == "cub":
        dataset = CUBDataset(data_path,
                             resolution=384,
                             use_labels=False,
                             transform=IMAGE_TRANSFORMS["from_numpy"])
    else:
        print(f"Using general purpose dataloader for {dataset_module} dataset")
        dataset = ImageFolderDataset(data_path,
                                     resolution=384,
                                     transform=IMAGE_TRANSFORMS['from_numpy'])

    return dataset


def get_text_features_from_attrs(attrs_dict, blip_processor, owl_tokenizer, device, base_text=False):
    # initialize output dict
    text_features_dict = dict()

    # loop through attribute categories
    for attr_c in attrs_dict['attributes']:
        labels = attrs_dict['attributes'][attr_c]['labels'].copy()
        label_prompts = attrs_dict['attributes'][attr_c]['prompts'].copy()
        text_features_dict[attr_c] = dict(labels=labels, prompts=label_prompts, features=None)

        # get text features for owl attributes
        if attrs_dict['attributes'][attr_c]['is_object']:
            text_features_dict[attr_c]["features"] = owl_tokenizer(labels,
                                                                   padding="max_length",
                                                                   return_tensors='pt').to(device)
        # get text tokens for blip attributes
        else:
            text_features_dict[attr_c]["features"] = dict()
            for label, prompt in zip(labels, label_prompts):
                text_features_dict[attr_c]["features"][label] = blip_processor(text=prompt,
                                                                               padding='max_length',
                                                                               return_tensors="pt").to(device)

    # include "base" text prompt for BLIP
    if base_text:
        text_features_dict["base_prompt"] = blip_processor(text=attrs_dict['caption'],
                                                           padding='max_length',
                                                           return_tensors="pt").to(device)
    return text_features_dict


def generate_annotations(attributes_dict: dict,
                         data_path: str,
                         dataset_module="custom",
                         blip_model_name="Salesforce/blip-itm-large-coco",
                         owl_model_name="google/owlv2-large-patch14-ensemble",
                         device="cuda",
                         batch_size=1,
                         threshold=0.3,
                         base_text=True):
    # Load models
    print("Setting up models...")
    blip_model = BlipForImageTextRetrieval.from_pretrained(blip_model_name).to(device)
    blip_processor = AutoProcessor.from_pretrained(blip_model_name)
    owl_processor = Owlv2Processor.from_pretrained(owl_model_name)
    owl_model = Owlv2ForObjectDetection.from_pretrained(owl_model_name, device_map=device)

    # get text features
    print("Getting up text features...")
    text_features_dict = get_text_features_from_attrs(attributes_dict,
                                                      blip_processor,
                                                      owl_processor.tokenizer,
                                                      device,
                                                      base_text=base_text)

    # Dataset and dataloader
    print("Setting up dataset...")
    dataset = get_dataset(dataset_module, data_path)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )

    # collect attribute labels for each image
    results = dict(filenames=[],
                   config=dict(data=dict(dataset_module=dataset_module, data_path=data_path),
                               models=dict(blip_model_name=blip_model_name, owl_model_name=owl_model_name),
                               batch_size=batch_size,
                               base_text=base_text))
    print("Begin attribute labeling...")
    with torch.no_grad(), tqdm(total=len(dataset)) as pbar:
        for batch in data_loader:
            # get images
            images = batch['image']

            # save filenames
            results['filenames'].extend(copy.deepcopy(batch['filename']))

            # get BLIP image features
            image_blip_inputs = blip_processor.image_processor(images,
                                                               do_resize=False,  # already resized in transform
                                                               do_rescale=False,  # already rescaled in transform
                                                               do_normalize=False,  # already normalized in transform
                                                               return_tensors="pt").to(device)
            blip_vision_outputs = blip_model.vision_model(**image_blip_inputs)
            blip_image_embeds = blip_vision_outputs[0]
            blip_image_atts = torch.ones(blip_image_embeds.size()[:-1], dtype=torch.long).to(device)

            # get BLIP base score
            if base_text:
                base_text_outputs = blip_model.text_encoder(encoder_hidden_states=blip_image_embeds,
                                                            encoder_attention_mask=blip_image_atts,
                                                            **text_features_dict["base_prompt"])
                base_logits = blip_model.itm_head(base_text_outputs.last_hidden_state[:, 0, :])
                base_score = base_logits.softmax(dim=1)[:, 1].cpu().numpy()

            # Get image inputs for owl model
            image_owl_inputs = owl_processor.image_processor(images,
                                                             do_rescale=False,  # already rescaled in transform
                                                             do_normalize=False,  # already normalized in transform
                                                             return_tensors="pt").to(device)

            # Get predictions for each attribute category
            for attr_c, text_features in text_features_dict.items():
                if attr_c == "base_prompt":
                    continue

                attr_list = text_features["labels"]

                # Image-text matching using BLIP
                if isinstance(text_features['features'], dict):
                    # loop through each attribute label in category
                    for attr in attr_list:
                        inputs_text = text_features["features"][attr]
                        text_outputs = blip_model.text_encoder(encoder_hidden_states=blip_image_embeds,
                                                               encoder_attention_mask=blip_image_atts,
                                                               **inputs_text)
                        logits = blip_model.itm_head(text_outputs.last_hidden_state[:, 0, :])
                        scores = logits.softmax(dim=1)[:, 1].cpu().numpy()

                        if base_text:
                            scores = scores - base_score

                        # save scores
                        results.setdefault(f"{attr_c}_{attr}", []).extend(scores.tolist())

                # Object detection using OWL
                else:
                    inputs = text_features["features"].copy()
                    for key in inputs:  # repeat text features for each image in batch
                        inputs[key] = torch.concat([inputs[key]] * images.shape[0], dim=0)
                    inputs["pixel_values"] = image_owl_inputs.pixel_values

                    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
                    target_sizes = torch.Tensor([images.shape[2::]] * images.shape[0]).to(device)

                    # get predictions
                    outputs = owl_model(**inputs)
                    owl_results = owl_processor.post_process_object_detection(outputs=outputs,
                                                                              threshold=threshold,
                                                                              target_sizes=target_sizes)

                    # save detections
                    for i, attr in enumerate(attr_list):
                        results.setdefault(f"{attr_c}_{attr}", [])
                        for owl_result in owl_results:
                            bboxes = owl_result["boxes"][owl_result["labels"] == i].cpu().tolist()
                            scores = owl_result["scores"][owl_result["labels"] == i].cpu().tolist()
                            results[f"{attr_c}_{attr}"].append(dict(bboxes=bboxes, scores=scores))

            # update progress bar
            pbar.update(images.shape[0])

    return results
