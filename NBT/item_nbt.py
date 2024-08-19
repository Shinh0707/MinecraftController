from typing import Any, Dict, Optional, Union
from enum import Enum, auto
from dataclasses import dataclass, field

from NBT.MItems import MItems
from NBT.nbt import NBTCompound, NBTList, NBTTag, NBTType
from NBT.parameter import IntPosition, boolean, dimension, identifier

@dataclass
class MItem:
    id: identifier
    item_state: Optional[NBTList] = None
    tags: Optional[NBTCompound] = None

    def __str__(self):
        s = f"{self.id}"
        if self.item_state:
            s += f"{self.item_state}"
        if self.tags:
            s += f"{self.tags}"
        return s
    
    @staticmethod
    def convert_from(value: Any):
        print(f"convert from {value}")
        if isinstance(value, MItem):
            return value
        if isinstance(value, identifier):
            return MItem(id=value)
        if isinstance(value, (MItems,str)):
            return MItem(id=identifier(name=value))
        raise NotImplementedError
    

@dataclass
class CompassState:
    name: str = field(default="",init=False)
    lodestone_tracker: Optional[NBTCompound] = None
    states: Optional[list[NBTTag]] = field(default_factory=list)

    def __post_init__(self):
        if self.lodestone_tracker:
            self.states.append(NBTCompound(name="lodestone_tracker",tags=[self.lodestone_tracker]))
    
    def __str__(self):
        params = self.states
        if params:
            states = [f"{nbt_tag.name}={NBTTag.to_nbt_str(nbt_tag.value,nbt_tag.type)}" for nbt_tag in params if not nbt_tag.value is None]
            if len(states) > 0:
                return "["+','.join(states)+"]"
        return ""
            
@dataclass
class CompassTag(NBTCompound):
    name: str = field(default="target",init=False)
    LodestoneTracked: Optional[boolean] = None
    LodestoneDimension: Optional[dimension] = field(default=dimension.OVERWORLD)
    LodestonePos: Optional[IntPosition] = None

    def __post_init__(self):
        if not self.LodestoneTracked is None:
            self.tags.append(self.LodestoneTracked.to_nbt_tag("tracked"))
        if self.LodestoneDimension:
            self.tags.append(self.LodestoneDimension.to_nbt_tag("dimension"))
        if self.LodestonePos:
            self.tags.append(self.LodestonePos.to_IList("pos"))
        self.value = self.tags

@dataclass
class Compass(MItem):
    id: identifier = field(default_factory=lambda: identifier(name=MItems.compass),init=False)
    item_state: Optional[CompassState] = None
    tags : Optional[CompassTag] = None