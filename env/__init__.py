from .tailsitter import Tailsitter
from .quadcopter import Quadcopter


# Mappings from strings to environments
env_map = {
    "Tailsitter": Tailsitter,
    "Quadcopter": Quadcopter,
}
