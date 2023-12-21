import os
import numpy as np

from .base import ImageFolderDataset


class CUBDataset(ImageFolderDataset):
    def __init__(self,
                 data_dir,
                 split='train',
                 resolution=128,
                 use_labels=True,
                 attrs_only=False,
                 **dataset_base_kwargs):

        self._root = data_dir
        self._path = os.path.join(data_dir, 'images')
        self._split = split
        assert os.path.exists(self._path), f'Path {self._path} does not exist'
        self._type = 'dir'
        self._attrs_only = attrs_only

        # Get train_test_split.txt
        with open(os.path.join(self._root, 'train_test_split.txt'), 'r') as f:
            self.train_test_split = np.array([line.split(' ')[1].strip() for line in f.readlines()], dtype=int)

        # Get image fnames from images.txt
        with open(os.path.join(data_dir, 'images.txt'), 'r') as f:
            self._image_fnames = np.array([line.split(' ')[1].strip() for line in f.readlines()])
        if self._split != 'all':
            self._image_fnames = self._image_fnames[self.train_test_split == (self._split == 'train')]

        self.resolution_ = resolution
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        super(ImageFolderDataset, self).__init__(name="CUB-200",
                                                 raw_shape=raw_shape,
                                                 use_labels=use_labels,
                                                 **dataset_base_kwargs)
        # Initialize labels
        if self._use_labels:
            _ = self._get_raw_labels()

    def _load_raw_labels(self):
        # Get attribute categories and names from attributes.txt
        with open(os.path.join(self._root, 'attributes/attributes.txt'), 'r') as f:
            self.attribute_categories_and_names = [line.split(' ')[1].strip().split('::') for line in f.readlines()]

        # Get attribute labels
        with open(os.path.join(self._root, 'attributes/image_attribute_labels.txt'), 'r') as f:
            attribute_labels = np.array([line.split(' ')[2] for line in f.readlines()], dtype=np.int64).reshape(-1, 312)

        # Get bird species labels
        if not self._attrs_only:
            # Get classes from classes.txt
            with open(os.path.join(self._root, 'classes.txt'), 'r') as f:
                self.class_names = [line.split(' ')[1].strip() for line in f.readlines()]
            # Get species labels
            with open(os.path.join(self._root, 'image_class_labels.txt'), 'r') as f:
                species_labels = np.array([line.split(' ')[1].strip() for line in f.readlines()], dtype=np.int64)
            # convert species labels to one-hot
            species_labels = np.eye(len(self.class_names), dtype=np.int64)[species_labels - 1]
            # Combine species and attribute labels
            labels = np.concatenate([species_labels, attribute_labels], axis=1).astype(np.float32)
        else:
            labels = attribute_labels.astype(np.float32)

        # Return corresponding split labels
        if self._split != 'all':
            return labels[self.train_test_split == (self._split == 'train')]
        else:
            return labels

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            self._label_shape = raw_labels.shape
        return list(self._label_shape)
