import os
import numpy as np
from .base import ImageFolderDataset


class StanfordCars(ImageFolderDataset):
    def __init__(self,
                 data_dir,
                 split='train',
                 resolution=128,
                 use_labels=True,
                 **dataset_base_kwargs):

        self._root = data_dir
        self._split = split

        if split == 'train':
            path = os.path.join(data_dir, 'cars_train')
        elif split == 'test':
            path = os.path.join(data_dir, 'cars_test')
        else:
            raise ValueError(f"split must be 'train' or 'test'. User inputted split={split}")

        super().__init__(path,
                         name='stanford-cars',
                         resolution=resolution,
                         use_labels=use_labels,
                         **dataset_base_kwargs)

        # Initialize labels
        if self._use_labels:
            _ = self._get_raw_labels()

    def _load_raw_labels(self):
        # Get class names
        with open(os.path.join(self._root, 'names.csv'), 'r') as f:
            self.class_names = [line.strip('\n') for line in f.readlines()]

        # Get labels from annotations
        with open(os.path.join(self._root, f"anno_{self._split}.csv"), 'r') as f:
            fnames, labels = [], []
            for line in f.readlines():
                fname, _, _, _, _, label = line.strip('\n').split(',')
                fnames.append(fname)
                labels.append(label)
        assert set(fnames).intersection(self._image_fnames) == set(self._image_fnames), f"image filenames do not match filenames in annotation file"

        return np.array(labels, dtype=np.int64)

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            self._label_shape = raw_labels.shape
        return list(self._label_shape)
