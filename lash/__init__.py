from .config import LASHConfig
from .env import A2ANegotiationEnv
from .data_collector import save_episode, collection_stats
from .types import BuyerType, SellerType, sample_types

# LASHDataset / collate_fn require torch — import only when needed
def __getattr__(name):
    if name in ("LASHDataset", "collate_fn"):
        from .dataset import LASHDataset, collate_fn
        globals()["LASHDataset"] = LASHDataset
        globals()["collate_fn"] = collate_fn
        return globals()[name]
    raise AttributeError(f"module 'lash' has no attribute {name!r}")
