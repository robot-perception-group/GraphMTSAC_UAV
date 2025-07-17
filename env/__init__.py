from .tailsitter import Tailsitter
from .blimp_rand import BlimpRand
from .quadcopter import Quadcopter


# Mappings from strings to environments
env_map = {
    "Tailsitter": Tailsitter,
    "Quadcopter": Quadcopter,
    "BlimpRand": BlimpRand,
}
