import utils
from torch_geometric import io as tgio
from spektral.data import Dataset, Graph 

tgio.parse_off = utils.parse_off

class InMemoryPhysicochemicalDataset(Dataset):
    def __init__(self, data_dir, list_examples, split, use_txt=False):
        
