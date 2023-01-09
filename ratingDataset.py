from surprise import Dataset as DS
from torch.utils.data import Dataset
from torch import Tensor, zeros


class RatingDataset(Dataset):
    def __init__(self):
        super(RatingDataset, self).__init__()
        self.data = DS.load_builtin()
        self.data = self.data.build_full_trainset()

    def __len__(self):
        return self.data.n_users

    def __getitem__(self, index) -> Tensor:
        t = zeros(self.data.n_items)

        for u, i, r in self.data.all_ratings():
            if u == index:
                t[i] = r / 5.0

        return t

    @property
    def n_items(self):
        return self.data.n_items
