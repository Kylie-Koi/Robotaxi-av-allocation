from src.rl.train_hierarchical import train_hierarchical
from src.utils.config import load_config

if __name__ == "__main__":
    train_hierarchical(load_config())
