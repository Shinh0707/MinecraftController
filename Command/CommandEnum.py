
from enum import Enum

class ValueEnum(Enum):
    def __str__(self):
        return self.value

class Advancements(ValueEnum):
    Minecraft = "story/root"