import torch
import pickle
import gdown
import os

from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from embedding import create_embedding


class IMCAG(InMemoryDataset):
    url = 'https://drive.google.com/uc?id=1foikoZwOJmEGFrYAd_DJrEEAu6MKTPcz'
    #url = 'https://drive.google.com/uc?id=1QH2WNnx4X7Qm6kDgG6Ry8c5fNSGnjZsI'

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # The name of the files in the self.raw_dir folder that must be present in order to skip downloading.
        return ['data_v2_0.pickle']

    @property
    def processed_file_names(self):
        # The name of the files in the self.processed_dir folder that must be present in order to skip processing.
        return ['data.pt']

    # def download(self):
    #     for f in self.raw_file_names:
    #         download_url(os.path.join(self.url, f), self.raw_dir)

    def download(self):
        gdown.download(self.url, os.path.join(self.raw_dir, self.raw_file_names[0]), quiet=True)

    def load_pickle(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data

    def process(self):
        data_list = []
        files = [f for f in os.listdir(self.raw_dir) if not os.path.isdir(f)]
        for f in files:
            data_list.extend(self.load_pickle(os.path.join(self.raw_dir, f)))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GraphDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = create_embedding('all-mpnet-base-v2')

    def preprocess(self, batch):
        x = [self.embedding(node) for node in batch.x]
        batch.x = torch.vstack(x)
        return batch

    def __iter__(self):
        self.iterator = super().__iter__()
        return self

    def __next__(self):
        batch = next(self.iterator)
        return self.preprocess(batch)
