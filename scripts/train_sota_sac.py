import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src/sota-implementations/discrete_sac"))

from discrete_sac import main
from omegaconf import OmegaConf

config = OmegaConf.load(os.path.join(os.path.dirname(__file__), "../src/sota-implementations/discrete_sac/config.yaml"))


main(config)
