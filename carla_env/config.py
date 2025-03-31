import yaml
from typing_extensions import TypedDict

CONFIG_FILE = "./carla_env/config.yaml"


class Config(TypedDict): ...


def read_config() -> Config:
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    return config
