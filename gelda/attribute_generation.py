from typing import Optional
from tqdm import tqdm

from .utils.chatgpt_utils import get_chatgpt_response, get_list_from_chatgpt, DEFAULT_CHATGPT_KWARGS


def generate_categories(caption: str, n_attrs=10, n_generations=5, n_attempts=10, **chat_kwargs):
    # Query attribute categories
    attribute_categories_query = [{
        "role": "user",
        "content": f"What are {n_attrs} attribute categories that can be used to visually distinguish images described by the caption '{caption}'?"
    },
        {
            "role": "user",
            "content": "Output the categories in one Python list."
        }]
    all_attr_categories = []
    for _ in tqdm(range(n_generations),
                  desc=f"generating {n_attrs} attribute categories"):  # repeat for n_generations
        attr_categories, responses = get_list_from_chatgpt(attribute_categories_query,
                                                           n_attempts=n_attempts,
                                                           **chat_kwargs)
        if attr_categories is not None:
            all_attr_categories.extend(attr_categories)
    return sorted(set(all_attr_categories), key=all_attr_categories.count, reverse=True)[:n_attrs]


def generate_labels(caption: str, attr_c: str, n_labels=10, n_generations=5, n_attempts=10, **chat_kwargs):
    # Query labels for each attribute category
    labels_query = [{
        "role": "user",
        "content": f"What are {n_labels} different examples of the category '{attr_c}' that can be used to distinguish images described by the caption '{caption}'?"
    },
        {
            "role": "user",
            "content": "Output the examples in one Python list."
        }]
    all_labels = []
    for _ in tqdm(range(n_generations),
                  desc=f"generating {n_labels} labels for {attr_c}"):  # repeat for n_generations
        labels, responses = get_list_from_chatgpt(labels_query,
                                                  n_attempts=n_attempts,
                                                  **chat_kwargs)
        if labels is not None:
            all_labels.extend(labels)
    return sorted(set(all_labels), key=all_labels.count, reverse=True)[:n_labels]


def generate_attribute_is_object(labels: list, n_attempts=10, **chat_kwargs):
    # Query if attributes are objects or items
    obj_query_message = [{
        "role": "user",
        "content": f"Are [{' '.join(labels)}] examples of objects or items? Answer with a yes or no. Explain your answer."
    }]
    response = get_chatgpt_response(n_attempts=n_attempts, messages=obj_query_message, **chat_kwargs)
    message = response['choices'][-1]['message']["content"]

    is_object = True if "yes" in message[:4].lower() else False
    return is_object


def generate_attribute_label_prompts(labels: list, attr_c: str, caption: str, n_attempts=10, **chat_kwargs):
    # Generate label prompts for each label
    label_prompts = []
    for lab in labels:
        if attr_c in lab:
            modified_caption_query = [{
                "role": "user",
                "content": f"Return a simple sentence that adds the description '{lab}' to the caption '{caption}'"
            }]
        else:
            modified_caption_query = [{
                "role": "user",
                "content": f"Return a simple sentence that adds the description '{lab} {attr_c}' to the caption '{caption}'"
            }]
        response = get_chatgpt_response(n_attempts=n_attempts, messages=modified_caption_query, **chat_kwargs)
        label_prompts.append(response['choices'][-1]['message']["content"])
    return label_prompts


def generate_attributes(
        caption: str,  # Caption used as context to generate attributes e.g. "a photo of a dog"
        n_attrs=10,  # Number of attribute categories to generate e.g. 10
        n_labels=10,  # Number of attribute labels to generate e.g. 10
        n_generations=5,  # Number of times to generate list of attribute categories and labels, from which top n_attrs or n_labels are kept.
        n_attempts=10,  # Number of tries to generate response from chatgpt
        chat_kwargs: Optional[dict] = None  # ChatGPT completion parameters e.g. {"model": "gpt-3.5-turbo", temperature": 0.1, "max_tokens": 1024}
):
    # Chat completion parameters
    if chat_kwargs is None:
        chat_kwargs = DEFAULT_CHATGPT_KWARGS

    # Initialize results dictionary
    results = {"caption": caption, "attributes": dict(), "chat_args": chat_kwargs}

    # Query attribute categories
    top_attr_categories = generate_categories(caption,
                                              n_attrs=n_attrs,
                                              n_generations=n_generations,
                                              n_attempts=n_attempts,
                                              **chat_kwargs)
    print("attribute categories: ", top_attr_categories)

    for attr_c in top_attr_categories:
        # Query labels for each attribute category
        top_labels = generate_labels(caption,
                                     attr_c,
                                     n_labels=n_labels,
                                     n_generations=n_generations,
                                     n_attempts=n_attempts,
                                     **chat_kwargs)

        if top_labels is not None:
            # Query if attributes are objects or items
            is_object = generate_attribute_is_object(top_labels,
                                                     n_attempts=n_attempts,
                                                     **chat_kwargs)

            # Generate label prompts for each label
            label_prompts = generate_attribute_label_prompts(top_labels,
                                                             attr_c,
                                                             caption,
                                                             n_attempts=n_attempts,
                                                             **chat_kwargs)

            # save and print data
            results["attributes"][attr_c] = {"labels": top_labels,
                                             "prompts": label_prompts,
                                             "is_object": is_object}
            print(f"attributes for {attr_c}: ", results["attributes"][attr_c])
        else:
            print(f"failed to generate labels for {attr_c}")

    return results
