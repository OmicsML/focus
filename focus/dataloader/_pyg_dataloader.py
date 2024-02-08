from functools import partial
from math import ceil
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from torch import optim
from torch_geometric import data
from torch_geometric.transforms import Constant
from torch_geometric.loader import DataLoader

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

