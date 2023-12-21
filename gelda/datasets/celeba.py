import numpy as np
import torch
from torchvision.datasets import CelebA


class CelebADataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_root,
                 split: str = 'train',
                 transform=None,
                 target_transform=None,
                 max_imgs=None):
        super().__init__()
        self.name = "CelebA"

        # data_dir must be root directory containing "celeba" folder
        if os.path.basename(data_root) == "celeba":
            data_root = os.path.dirname(data_root)

        # load dataset
        self.dataset = CelebA(root=data_root,
                              split=split,
                              target_type='attr',
                              transform=transform,
                              target_transform=target_transform)
        self.dataset.attr_names = self.dataset.attr_names[:-1]  # remove empty string attr

        # edit "No_Beard" to "Beard"
        self.dataset.attr_names[self.dataset.attr_names.index("No_Beard")] = "Beard"

        # Limit dataset size
        if max_imgs is not None and max_imgs < len(self.dataset):
            subset_indxs = np.random.choice(len(self.dataset), size=max_imgs, replace=False)
            self.dataset.filename = np.array(self.dataset.filename)[subset_indxs]
            self.dataset.attr = np.array(self.dataset.attr)[subset_indxs]
            self.dataset.identity = np.array(self.dataset.identity)[subset_indxs]
            self.dataset.bbox = np.array(self.dataset.bbox)[subset_indxs]
            self.dataset.landmarks_align = np.array(self.dataset.landmarks_align)[subset_indxs]

    def __getitem__(self, index):
        # edit "No_Beard" to "Beard"
        labels = self.dataset.attr[index]
        labels[self.dataset.attr_names.index("Beard")].apply_(lambda x: 1 if x == 0 else 0)

        return dict(image=self.dataset[index][0],
                    labels=self.dataset[index][1],
                    filename=self.dataset.filename[index])

    def __len__(self):
        return len(self.dataset)

    @property
    def path(self):
        return self.dataset.root
