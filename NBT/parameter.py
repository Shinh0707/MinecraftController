from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import warnings
from numbers import Number

from NBT.nbt import NBTCompound, NBTTag, NBTType


class Facing(Enum):
    NORTH = 0
    EAST = 90
    SOUTH = 180
    WEST = 270
    UP = -1
    DOWN = -2


class Rotation(Enum):
    N = 0
    NNE = 1
    NE = 2
    ENE = 3
    E = 4
    ESE = 5
    SE = 6
    SSE = 7
    S = 8
    SSW = 9
    SW = 10
    WSW = 11
    W = 12
    WNW = 13
    NW = 14
    NNW = 15


@dataclass
class angle:
    value: float = 0.0
    relative: bool = False

    def __post_init__(self):
        self.value = ((self.value + 180) % 360) - 180

    def __str__(self):
        if self.relative:
            return f"~{self.value}"
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, angle):
            return abs(self.value - other.value) < 1e-6
        return False

    def __add__(self, other):
        if isinstance(other, Number):
            return angle(self.value + other, self.relative)
        elif isinstance(other, angle):
            return angle(self.value + other.value, self.relative or other.relative)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Number):
            return angle(self.value - other, self.relative)
        elif isinstance(other, angle):
            return angle(self.value - other.value, self.relative or other.relative)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Number):
            return angle(self.value * other, self.relative)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Number):
            return angle(self.value / other, self.relative)
        return NotImplemented

    def facing(self):
        if -45 <= self.value < 45:
            return facing(Facing.SOUTH)
        elif 45 <= self.value < 135:
            return facing(Facing.WEST)
        elif -135 <= self.value < -45:
            return facing(Facing.EAST)
        else:
            return facing(Facing.NORTH)

    def rotation(self):
        index = round(((self.value + 180) % 360) / 22.5)
        return rotation(Rotation(index))


@dataclass
class rotation:
    value: Rotation = Rotation.N

    def __str__(self):
        return str(self.value.value)

    def __eq__(self, other):
        if isinstance(other, rotation):
            return self.value == other.value
        return False

    def __add__(self, other):
        if isinstance(other, Number):
            return rotation(Rotation(round((self.value.value + other) % 16)))
        elif isinstance(other, rotation):
            return rotation(Rotation(round((self.value.value + other.value.value) % 16)))
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Number):
            return rotation(Rotation(round((self.value.value - other) % 16)))
        elif isinstance(other, rotation):
            return rotation(Rotation(round((self.value.value - other.value.value) % 16)))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Number):
            return rotation(Rotation(round((self.value.value * other) % 16)))
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Number):
            return rotation(Rotation(round((self.value.value / other) % 16)))
        return NotImplemented

    def angle(self):
        return angle(self.value.value * 22.5 - 180)

    def facing(self):
        return self.angle().facing()


@dataclass
class facing:
    value: Facing = Facing.NORTH

    def __str__(self):
        return self.value.name.lower()

    def __eq__(self, other):
        if isinstance(other, facing):
            return self.value == other.value
        return False

    def __add__(self, other):
        if isinstance(other, Number):
            return facing(Facing(round((self.value.value + other * 90) % 360)))
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Number):
            return facing(Facing(round((self.value.value - other * 90) % 360)))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Number):
            return facing(Facing(round((self.value.value * other) % 360)))
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Number):
            return facing(Facing(round((self.value.value / other) % 360)))
        return NotImplemented

    def angle(self):
        if self.value in [Facing.UP, Facing.DOWN]:
            warnings.warn(
                f"{self.value.name} does not have a corresponding angle. Using NORTH instead.")
            return angle(Facing.NORTH.value)
        return angle(self.value.value)

    def rotation(self):
        if self.value in [Facing.UP, Facing.DOWN]:
            warnings.warn(
                f"{self.value.name} does not have a corresponding rotation. Using NORTH instead.")
            return rotation(Rotation.N)
        return self.angle().rotation()


@dataclass
class boolean:
    _value: bool = False

    def __str__(self):
        return "true" if self._value else "false"

    def __and__(self, other):
        if isinstance(other, boolean):
            return boolean(self._value and other._value)
        if isinstance(other, (bool, int)):
            return boolean(self._value and other)
        raise ValueError

    def __or__(self, other):
        if isinstance(other, boolean):
            return boolean(self._value or other._value)
        if isinstance(other, (bool, int)):
            return boolean(self._value or other)
        raise ValueError

    def __mul__(self, other):
        if isinstance(other, boolean):
            return boolean(self._value and other._value)
        if isinstance(other, (bool, int)):
            return boolean(self._value and other)
        raise ValueError

    def __add__(self, other):
        if isinstance(other, boolean):
            return boolean(self._value or other._value)
        if isinstance(other, (bool, int)):
            return boolean(self._value or other)
        raise ValueError

    def __eq__(self, other):
        if isinstance(other, boolean):
            return self._value == other._value
        if isinstance(other, (bool, int)):
            return bool(self._value == other)
        return False

    def __is__(self, other):
        if isinstance(other, boolean):
            return self._value == other._value
        return False

    def __xor__(self, other):
        if isinstance(other, boolean):
            return boolean(self._value.__xor__(other._value))
        if isinstance(other, (bool, int)):
            return boolean(self._value.__xor__(other))
        raise ValueError

    def __bool__(self):
        return self._value

    def __neg__(self):
        return boolean(not self._value)

    def __invert__(self):
        return boolean(not self._value)


@dataclass
class identifier:
    namespace: str = field(default="minecraft")
    name: str = field(default="stone")

    def __str__(self):
        return "{}:{}".format(self.namespace, self.name)
    
    def to_nbt_tag(self, name: str="id"):
        return NBTTag(name, NBTType.STRING, str(self))


@dataclass
class block_states:
    def __str__(self):
        return "["+','.join([f"{s}={v}" for s, v in vars(self)])+"]"

    def to_key(self):
        return ','.join([f"{s}:{v}" for s, v in vars(self)])


@dataclass
class block_predicate:
    _block_id: identifier = field(default_factory=identifier())
    _block_states: block_states = None
    _Data_tags: NBTCompound = None

    def __str__(self):
        return "{}{}{}".format(self._block_id, self._block_states, self._Data_tags)


class attribute_name(Enum):
    MAX_HEALTH = "generic.max_health"
    MAX_ABSORPTION = "generic.max_absorption"
    FOLLOW_RANGE = "generic.follow_range"
    # 残りも実装する

    def to_nbt_tag(self, name: str = "AttributeName"):
        return NBTTag(name, NBTType.STRING, self.value)

class dimension(Enum):
    THE_NETHER = identifier(name="the_nether")
    OVERWORLD = identifier(name="overworld")
    THE_END = identifier(name="the_end")

    @staticmethod
    def custom(namespace: str, dimension_name: str):
        return identifier(namespace=namespace,name=dimension_name)
    
    def to_nbt_tag(self, name: str="dimension"):
        return self.value.to_nbt_tag(name)


@dataclass
class IntPosition(NBTTag):
    x: int = 0
    y: int = 0
    z: int = 0
    name: str = field(default="pos")
    type: NBTType = field(default=NBTType.LIST, init=False)

    def __post_init__(self):
        self.value = [
            NBTTag("", NBTType.INT, self.x),
            NBTTag("", NBTType.INT, self.y),
            NBTTag("", NBTType.INT, self.z)
        ]

    def __str__(self):
        return f"{self.x} {self.y} {self.z}"

    def __add__(self, other):
        if isinstance(other, IntPosition):
            return IntPosition(self.x + other.x, self.y + other.y, self.z + other.z)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, IntPosition):
            return IntPosition(self.x - other.x, self.y - other.y, self.z - other.z)
        return NotImplemented

    @property
    def down(self):
        return IntPosition(self.x, self.y - 1, self.z)

    @property
    def up(self):
        return IntPosition(self.x, self.y + 1, self.z)

    @property
    def north(self):
        return IntPosition(self.x, self.y, self.z - 1)

    @property
    def south(self):
        return IntPosition(self.x, self.y, self.z + 1)

    @property
    def west(self):
        return IntPosition(self.x - 1, self.y, self.z)

    @property
    def east(self):
        return IntPosition(self.x + 1, self.y, self.z)
