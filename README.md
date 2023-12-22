# GELDA: A generative language annotation framework to reveal visual biases in datasets
### [Project Page](https://krishk97.github.io/publication/gelda) | [Paper](https://arxiv.org/abs/2311.18064)

<img src="assets/pipeline_gelda.png">

GELDA is an automatic framework that leverages large language models (LLMs) to propose and label various attributes for a domain. It takes a user-defined domain caption (e.g. "a photo of a living room") and uses an LLM to hierarchically generate attributes. It then uses the LLM to decide which of a set of vision-language models (VLMs) to use to annotate each attribute in images.

GELDA is designed as a tool to help researchers analyze datasets in a cheap, low-effort, and flexible manner. In particular, it is designed to combat annotator "blind spots", which can help reveal biases in dataset attributes such as confounding between class labels and background features.

## Installation

```bash
pip install gelda
```

## Usage

### 1. Generate attributes for your dataset

GELDA uses OpenAI's text generation models (i.e. GPT) to generate attributes. To access these models, you will require an OpenAI API key. Please refer to the OpenAI API documentation [here](https://openai.com/blog/openai-api). 

To generate attributes for your dataset, you need to provide: 
- \<caption\> a caption that describes the domain of your dataset. For example, if you are analyzing a dataset of living room images, you could provide the caption "a photo of a living room".
- \<n_attrs\> the number of attribute categories to generate (Default: 10)
- \<n_labels\> the number of labels to generate for each attribute category (Default: 10)
- \<n_generations\> Number of times to generate list of attribute categories and labels, from which top n_attrs or n_labels are kept. (Default: 5)
- \<n_attempts\> Number of tries to generate response for chatgpt (we use [wrapt_timeout_decorator](https://github.com/bitranox/wrapt_timeout_decorator) to timeout chatgpt after 100 seconds) (Default: 10)
- \<chatgpt_kwargs\> These are additional keyword arguments you can pass to OpenAI's chat completions API (e.g., model, temperature, max tokens, etc.). Please refer to the OpenAI API documentation [here](https://platform.openai.com/docs/api-reference/chat) for more details. (Default: {"model": "gpt-3.5-turbo", temperature": 0.1, "max_tokens": 1024})

Python code: 
```python
import openai 
import gelda

openai.api_key = <your_openai_api_key>
attribute_dict = gelda.generate_attributes(
    caption=<caption>,
    n_attrs=<n_attrs>,
    n_labels=<n_labels>,
    n_generations=<n_generations>,
    n_attempts=<n_attempts>,
    chatgpt_kwargs=<chatgpt_kwargs>
)
```

Python script: (saves attributes to `your_save_path.json`)
```bash
export OPENAI_API_KEY=<your_openai_api_key>
python scripts/run_gelda_attribute_gen.py \
  -c <caption> \
  -s <your_save_path.json> \
  -g <n_generations> \
  -a <n_attrs>  \
  -l <n_labels> \
  --model <model_name>
  -t <temperature> \
  --max_tokens <max_tokens> \
  --timeout <timeout>
```

### 2. Generate annotations for your dataset

GELDA uses a set of vision-language models (VLMs) to annotate images. We access these models through the [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) library. Specifically, we use [BLIP](https://huggingface.co/docs/transformers/model_doc/blip) to annotate image-level concepts (e.g. style, color scheme) and [OWLv2](https://huggingface.co/docs/transformers/main/model_doc/owlv2) to annotate object-level concepts (e.g. objects and parts). 

To generate annotations for your dataset, you need to provide:
- <attribute_dict> the attribute dictionary generated in step 1
- \<dataset_path\> path to your dataset
- \<dataset_module\> name of the dataset module to use (see below for pre-defined modules) (Default: "custom")
- \<blip_model_name\> name of the BLIP ITM model to use (Default: "Salesforce/blip-itm-large-coco"
- \<owl_model_name\> name of the OWLv2 model to use (Default: "google/owlv2-large-patch14-ensemble")
- \<device\> device to use for VLMs (Default: "cuda" if available, else "cpu")
- \<batch_size\> batch size for the VLMs (Default: 1)
- <\threshold \> threshold for OWLv2 object detection (Default: 0.3)
- \<base_text\> Boolean for whether to subtract BLIP ITM score for base caption, defined to be identical to the caption provided to generate attributes in step 1 (Default: True) 

Python code: 
```python
import gelda 

annotation_dict = gelda.generate_annotations(
    <attribute_dict_from_step_1>, 
    <dataset_path>,
    dataset_module=<dataset_module>,
    blip_model_name=<blip_model_name>,
    owl_model_name=<owl_model_name>,
    device=<device>,
    batch_size=<batch_size>,
    threshold=<threshold>,
    base_text=<base_text
)
```

Python script: (saves annotations to `your_save_path.pkl`) 
```bash
python scripts/run_gelda_annotations.py \
  -a <attribute_dict_from_step_1> \
  -p <dataset_path> \
  -s <your_save_path.pkl> \
  -d <dataset_module> \
  --blip_model <blip_model_name> \
  --owl_model <owl_model_name> \
  -t <threshold> \
  -bs <batch_size> \
  --base_text \ 
```

#### Pre-defined dataset modules
We provide pre-defined dataset modules for the following datasets:
- [Custom (ImageFolderDataset)](gelda/datasets/base.py) (for your own image dataset)
- [CelebA](gelda/datasets/celeba.py)
- [CUB-200](gelda/datasets/cub.py)
- [Stanford Cars](gelda/datasets/cars.py)
- [DeepFashion](gelda/datasets/deepfashion.py)

### 3. Analyze your dataset

We provide a set of tools to analyze your dataset.

- Convert annotations to a pandas dataframe
```python
from gelda.utils import annotations_to_df

annotation_df = annotations_to_df(
    <attribute_dict_from_step_1>,
    <annotation_dict_from_step_2>,
    blip_threshold=<blip_threshold>  # (default=0)
)
```

- Bar plot of attribute frequencies sorted by frequency in each category
```python
from gelda.utils.plotting_utils import plot_annotation_barplot

fig, ax = plot_annotation_barplot(
            <attribute_dict_from_step_1>,
            <annotation_dict_from_step_2>
)
fig.show()
```

- Example images for each attribute
```python
from gelda.utils.plotting_utils import plot_annotation_examples

annotation_example_figs_dict = plot_annotation_examples(
    <attribute_dict_from_step_1>,
    <annotation_dict_from_step_2>, 
    <attribute_category>,  # name of attribute category to plot 
    img_dir=<img_dir>,  # Optional path to image directory (default=None)
    n_examples=<n_examples>,  # number of examples to plot (default=10)
    n_rows=<n_rows>  # number of images per row (default=5)
)

# To show all figures
for attribute, fig_dict in annotation_example_figs_dict.items():
    fig, ax = fig_dict['fig'], fig_dict['ax']
    ax.set_title(attribute)
    fig.show()
```

### Citation
If you use this work, please cite: 
```
@article{kabra2023gelda,
  title={GELDA: A generative language annotation framework to reveal visual biases in datasets},
  author={Kabra, Krish and Lewis, Kathleen M and Balakrishnan, Guha},
  journal={arXiv preprint arXiv:2311.18064},
  year={2023}
}
```