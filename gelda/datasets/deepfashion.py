import os
import numpy as np

from .base import ImageFolderDataset


class DeepFashionDataset(ImageFolderDataset):
    def __init__(self,
                 data_dir,
                 split='train',
                 resolution=128,
                 use_labels=True,
                 attrs_only=False,
                 **dataset_base_kwargs):

        self._path = data_dir
        self._split = split
        assert os.path.exists(self._path), f'Path {self._path} does not exist'
        self._type = 'dir'
        self._attrs_only = attrs_only

        # Get image fnames from images.txt
        with open(os.path.join(self._path, f'Anno_fine/{self._split}.txt'), 'r') as f:
            self._image_fnames = np.array([line.strip() for line in f.readlines()])

        self.resolution_ = resolution
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        super(ImageFolderDataset, self).__init__(name="DeepFashion",
                                                 raw_shape=raw_shape,
                                                 use_labels=use_labels,
                                                 **dataset_base_kwargs)
        # Initialize labels
        if self._use_labels:
            _ = self._get_raw_labels()

    def _load_raw_labels(self):
        # Get clothing attribute names from list_attr_cloth.txt
        with open(os.path.join(self._path, f'Anno_fine/list_attr_cloth.txt'), 'r') as f:
            self.cloth_attr_names = [line.strip().split(' ')[0] for line in f.readlines()[2:]]
        # Get attribute labels
        with open(os.path.join(self._path, f'Anno_fine/{self._split}_attr.txt'), 'r') as f:
            attribute_labels = np.array([line.strip().strip('\n').split(' ') for line in f.readlines()],
                                        dtype=np.int64)

        if not self._attrs_only:
            # Get clothing categories from list_category_cloth.txt
            with open(os.path.join(self._path, f'Anno_fine/list_category_cloth.txt'), 'r') as f:
                self.cloth_categories = [line.strip().split(' ')[0] for line in f.readlines()[2:]]
            # Get clothing category labels
            with open(os.path.join(self._path, f'Anno_fine/{self._split}_cate.txt'), 'r') as f:
                cloth_cate_labels = np.array([line.strip().strip('\n').split(' ') for line in f.readlines()],
                                             dtype=np.int64).squeeze()
                # convert labels to one-hot
                cloth_cate_labels = np.eye(len(self.cloth_categories))[cloth_cate_labels - 1]

            # Combine species and attribute labels
            labels = np.concatenate([cloth_cate_labels, attribute_labels], axis=1).astype(np.float32)
        else:
            labels = attribute_labels.astype(np.float32)

        return labels

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            self._label_shape = raw_labels.shape
        return list(self._label_shape)
