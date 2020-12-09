from torch.utils.data import Dataset
import torch


class Dataset2(Dataset):
    def __init__(self, item_of_single_class=10, noise_scope_list=None, data_vector_dim=20, k=2):
        if noise_scope_list is None:
            noise_scope_list = [1, 1, 1, 1, 1, 1, 1]
        self.noise_scope_list = noise_scope_list
        self.item_of_single_class = item_of_single_class
        self.dataset_list = []
        self.data_vector_dim = data_vector_dim
        self.k = k
        self.creat_dataset_list()

    def __getitem__(self, item):
        return self.dataset_list[item][0], self.dataset_list[item][1]

    def __len__(self):
        return self.item_of_single_class * len(self.noise_scope_list)

    def creat_dataset_list(self):
        self.dataset_list = []
        for i, noise_scope in enumerate(self.noise_scope_list):
            for j in range(self.item_of_single_class):
                self.dataset_list.append(
                    [(i - 0.5) * self.k + noise_scope * torch.rand(self.data_vector_dim), i])

