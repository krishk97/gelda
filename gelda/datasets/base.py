"""
Dataset class for loading images and labels from a directory.
Inspired by https://github.com/NVlabs/stylegan2-ada/blob/main/training/dataset.py
"""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch

from ..utils import EasyDict

try:
    import pyspng
except ImportError:
    pyspng = None


# ----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 name,  # Name of the dataset.
                 raw_shape,  # Shape of the raw image data (NCHW).
                 transform=None,  # A function/transform that takes in an image and returns a transformed version.
                 target_transform=None,  # A function/transform that takes in the label and transforms it.
                 max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
                 use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
                 xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
                 random_seed=0,  # Random seed to use when applying max_size.
                 ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self.transform = transform
        self.target_transform = target_transform
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def _get_filename(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        # Load image.
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
        if self.transform is not None:
            image = self.transform(image)

        # Load labels.
        labels = self.get_label(idx)
        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return dict(image=image,
                    labels=labels,
                    filename=self._get_filename(self._raw_idx[idx]))

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


# ----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
                 path,  # Path to directory or zip.
                 name=None,  # Name of the dataset.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in
                                os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        if name is None:
            name = os.path.splitext(os.path.basename(self._path))[0]

        self.resolution_ = resolution
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        #         if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
        #             raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _get_filename(self, raw_idx):
        return self._image_fnames[raw_idx]

    def _load_raw_image(self, raw_idx):
        fname = self._get_filename(raw_idx)
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
                if self.resolution_ is not None and (
                        image.shape[0] != self.resolution_ or image.shape[1] != self.resolution_):
                    raise IOError(f'Cannot use pyspng to resize. Image {fname} does not match the specified resolution')
            else:
                image = PIL.Image.open(f)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                if self.resolution_ is not None and (
                        image.size[0] != self.resolution_ or image.size[1] != self.resolution_):
                    image = image.resize((self.resolution_, self.resolution_), resample=PIL.Image.BICUBIC)
                image = np.asarray(image).clip(0, 255).astype(np.uint8)
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    @property
    def path(self):
        return self._path

# ----------------------------------------------------------------------------

class ConditionalImageFolderDataset(ImageFolderDataset):
    def __init__(self,
                 image_path,  # Path to image directory or zip.
                 check_img_fname_cond_func,  # Function to check if image filenames meets some conditions. Return True if image is valid.
                 check_img_fname_cond_kwargs={}, # Keyword arguments for check_img_cond_func.
                 max_size=None,  # Maximum number of images to use, None = use all images.
                 xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
                 random_seed=0,  # Random seed to use when applying max_size.
                 **super_kwargs,  # Additional arguments for the ImageFolderDataset base class.
                 ):

        # Initialize ImageFolderDataset base class
        super().__init__(image_path,
                         max_size=max_size,
                         xflip=xflip,
                         random_seed=random_seed,
                         **super_kwargs)

        # Filter images meeting conditions
        image_fnames = []
        for img_fname in self._image_fnames:
            # check if image meets condition
            if check_img_fname_cond_func(img_fname, **check_img_fname_cond_kwargs):
                image_fnames.append(img_fname)

        # Sort filenames.
        self._image_fnames = sorted(image_fnames)

        # Redo raw shape
        self._raw_shape[0] = len(self._image_fnames)

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])
