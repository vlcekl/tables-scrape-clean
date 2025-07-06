from torch_geometric.loader.dataloader import DataLoader as _BaseLoader
from torch_geometric.data import Batch

class DataLoader(_BaseLoader):
    def __init__(self, *args, collate_fn=None, **kwargs):
        """
        Arguments:
        - collate_fn: function(List[Data]) -> batch
        All other args/kwargs as per torch_geometric.loader.DataLoader.
        """
        self._user_collate = collate_fn
        # Remove collate_fn from kwargs so base class won’t drop it again
        kwargs.pop('collate_fn', None)
        super().__init__(*args, **kwargs)

    def collate(self, data_list):
        # If user provided a custom collate, call it first
        if self._user_collate is not None:
            return self._user_collate(data_list)
        # Otherwise fall back to default behavior
        return super().collate(data_list)
