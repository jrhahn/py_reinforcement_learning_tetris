import random
from enum import Enum


class BlockType(Enum):
    TypeL = "typel"
    TypeLinv = "typelinv"
    TypeBlock = "typeblock"
    TypeTriangle = "typetriangle"
    TypeBar = "typebar"
    TypeZ = "typez"
    TypeS = "types"

    @staticmethod
    def random():
        return random.choice(list(BlockType))
