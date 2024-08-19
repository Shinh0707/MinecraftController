from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import math
from typing import Any, Dict, List, Tuple, Union
import warnings
from numbers import Number

import numpy as np

from NBT.MItems import MItems
from NBT.MBlocks import MBlocks
from NBT.nbt import NBTCompound, NBTList, NBTTag, NBTType

INFINITE = 'infinite'

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
    
    def to_cardinal(self):
        """
        16方位から最も近い4方位（北、東、南、西）に変換する。
        """
        cardinal_map = {
            Rotation.N: Rotation.N,
            Rotation.NNE: Rotation.N,
            Rotation.NE: Rotation.N,
            Rotation.ENE: Rotation.E,
            Rotation.E: Rotation.E,
            Rotation.ESE: Rotation.E,
            Rotation.SE: Rotation.E,
            Rotation.SSE: Rotation.S,
            Rotation.S: Rotation.S,
            Rotation.SSW: Rotation.S,
            Rotation.SW: Rotation.S,
            Rotation.WSW: Rotation.W,
            Rotation.W: Rotation.W,
            Rotation.WNW: Rotation.W,
            Rotation.NW: Rotation.N,
            Rotation.NNW: Rotation.N
        }
        return cardinal_map[self]

class TimeSpec(Enum):
    DAY = 1000
    NIGHT = 13000
    NOON = 6000
    MIDNIGHT = 18000
    SUNRISE = 23000
    SUNSET = 12000

    def __str__(self):
        return f"{self.value}"

class WeatherSpec(Enum):
    CLEAR = auto()
    RAIN = auto()
    THUNDER = auto()

    def __str__(self):
        return self.name.lower()
    
class EffectType(Enum):
    SPEED = ("speed", 1)
    SLOWNESS = ("slowness", 2)
    HASTE = ("haste", 3)
    MINING_FATIGUE = ("mining_fatigue", 4)
    STRENGTH = ("strength", 5)
    INSTANT_HEALTH = ("instant_health", 6)
    INSTANT_DAMAGE = ("instant_damage", 7)
    JUMP_BOOST = ("jump_boost", 8)
    NAUSEA = ("nausea", 9)
    REGENERATION = ("regeneration", 10)
    RESISTANCE = ("resistance", 11)
    FIRE_RESISTANCE = ("fire_resistance", 12)
    WATER_BREATHING = ("water_breathing", 13)
    INVISIBILITY = ("invisibility", 14)
    BLINDNESS = ("blindness", 15)
    NIGHT_VISION = ("night_vision", 16)
    HUNGER = ("hunger", 17)
    WEAKNESS = ("weakness", 18)
    POISON = ("poison", 19)
    WITHER = ("wither", 20)
    HEALTH_BOOST = ("health_boost", 21)
    ABSORPTION = ("absorption", 22)
    SATURATION = ("saturation", 23)
    GLOWING = ("glowing", 24)
    LEVITATION = ("levitation", 25)
    LUCK = ("luck", 26)
    UNLUCK = ("unluck", 27)
    SLOW_FALLING = ("slow_falling", 28)
    CONDUIT_POWER = ("conduit_power", 29)
    DOLPHINS_GRACE = ("dolphins_grace", 30)
    BAD_OMEN = ("bad_omen", 31)
    HERO_OF_THE_VILLAGE = ("hero_of_the_village", 32)
    DARKNESS = ("darkness", 33)

    def __init__(self, minecraft_id: str, numeric_id: int):
        self.minecraft_id = minecraft_id
        self.numeric_id = numeric_id

    def to_identifier(self):
        return identifier(name=self.minecraft_id)
    
@dataclass
class Day:
    """
    ゲーム内での1日。24000ティックに相当する。
    """
    value: float

    def __str__(self):
        return f"{int(self.value)}d"
    
    def to_sec(self):
        return Seconds(self.value*24000/20)
    
    def to_tick(self):
        return Tick(self.value*24000)

@dataclass
class Seconds:
    """
    現実世界での1秒。20ティックに相当する。
    """
    value: float

    def __str__(self):
        return f"{int(self.value)}s"
    
    def to_day(self):
        return Day(self.value*20/24000)
    
    def to_tick(self):
        return Tick(self.value*20)

@dataclass
class Tick:
    """
    1ティック。デフォルトの単位である。 
    """
    value: float

    def __str__(self):
        return f"{int(self.value)}t"
    
    def to_sec(self):
        return Seconds(self.value/20)
    
    def to_day(self):
        return Tick(self.value/24000)

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
        
    def __inv__(self):
        return angle(-self.value, self.relative)

    def __neg__(self):
        return self.__inv__()

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
        index = round(((self.value + 180) % 360) / 22.5) % 16
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
    
    def __inv__(self):
        return rotation(Rotation((self.value.value + 8) % 16))

    def __neg__(self):
        return self.__inv__()

    def angle(self):
        return angle(self.value.value * 22.5 - 180)

    def facing(self):
        return self.angle().facing()
    
    def to_cardinal(self):
        return rotation(self.value.to_cardinal())
    
    @property
    def numpy(self):
        """
        回転に対応する3次元方向ベクトルをNumPy配列として返す。
        北を(0, 0, -1)、東を(1, 0, 0)、上を(0, 1, 0)とする。
        """
        azimuth = math.radians(self.value.value * 22.5)
        x = round(math.sin(azimuth))
        z = -round(math.cos(azimuth))  # 北が負のz軸方向なので、cosineの符号を反転
        return np.array([x, 0, z], dtype=np.int64)

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
    
    def __inv__(self):
        if self.value == Facing.UP:
            return facing(Facing.DOWN)
        elif self.value == Facing.DOWN:
            return facing(Facing.UP)
        else:
            return facing(Facing((self.value.value + 180) % 360))

    def __neg__(self):
        return self.__inv__()

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
    
    def to_nbt_tag(self, name: str=""):
        return NBTTag(name,NBTType.BYTE,self._value)


@dataclass
class identifier:
    namespace: str = field(default="minecraft")
    name: Union[MBlocks,MItems,str] = field(default=MBlocks.stone)

    def __str__(self):
        return "{}:{}".format(self.namespace, self.name.name if isinstance(self.name,(MBlocks,MItems)) else self.name)
    
    def to_nbt_tag(self, name: str="id"):
        return NBTTag(name, NBTType.STRING, str(self))
    
    @property
    def block(self):
        if isinstance(self.name,(str,MItems)):
            return MBlocks.stone
        return self.name


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
    KNOCKBACK_RESISTANCE = "generic.knockback_resistance"
    MOVEMENT_SPEED = "generic.movement_speed"
    ATTACK_DAMAGE = "generic.attack_damage"
    ARMOR = "generic.armor"
    ARMOR_TOUGHNESS = "generic.armor_toughness"
    ATTACK_KNOCKBACK = "generic.attack_knockback"
    ATTACK_SPEED = "generic.attack_speed"
    LUCK = "generic.luck"
    SCALE = "generic.scale"
    STEP_HEIGHT = "generic.step_height"
    GRAVITY = "generic.gravity"
    SAFE_FALL_DISTANCE = "generic.safe_fall_distance"
    FALL_DAMAGE_MULTIPLIER = "generic.fall_damage_multiplier"
    JUMP_STRENGTH = "generic.jump_strength"
    BURNING_TIME = "generic.burning_time"
    EXPLOSION_KNOCKBACK_RESISTANCE = "generic.explosion_knockback_resistance"
    MOVEMENT_EFFICIENCY = "generic.movement_efficiency"
    OXYGEN_BONUS = "generic.oxygen_bonus"
    WATER_MOVEMENT_EFFICIENCY = "generic.water_movement_efficiency"
    
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

class IntPosition(NBTTag):
    IntPositionInput = Union['IntPosition', Tuple[int, int, int], List[int], np.ndarray, Dict[Union[str, int], int], str, int]
    def __init__(self, x: IntPositionInput, y: int = None, z: int = None, name: str = "Pos"):
        if isinstance(x, (np.ndarray, tuple, list)):
            if len(x) != 3:
                raise ValueError("Input must have exactly 3 elements")
            x, y, z = map(int, x)
        elif isinstance(x, IntPosition):
            x, y, z = x.x, x.y, x.z
        elif y is None or z is None:
            raise ValueError("Must provide all three coordinates (x, y, z)")

        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

        # Create NBTTag structure
        value = [
            NBTTag("", NBTType.INT, self.x),
            NBTTag("", NBTType.INT, self.y),
            NBTTag("", NBTType.INT, self.z)
        ]
        super().__init__(name, NBTType.LIST, value)

    def copy(self):
        return IntPosition(self.x,self.y,self.z)
    
    def to_XYZ(self,name:str|None=None):
        if not name:
            name = self.name
        value = [
            NBTTag("X", NBTType.INT, self.x),
            NBTTag("Y", NBTType.INT, self.y),
            NBTTag("Z", NBTType.INT, self.z)
        ]
        return NBTCompound(name=name,tags=value)
    
    def to_IList(self,name:str|None=None):
        if not name:
            name = self.name
        return NBTList(name=name,value=self.value)
    
    def __repr__(self):
        return f"IntPosition({self.x}, {self.y}, {self.z})"
    
    @staticmethod
    def from_any(value: Any) -> IntPosition:
        if isinstance(value, IntPosition):
            return value
        elif isinstance(value, (list, tuple, np.ndarray)):
            if len(value) != 3:
                raise ValueError("Input must have exactly 3 elements")
            return IntPosition(*map(int, value))
        elif isinstance(value, dict):
            if all(k in value for k in ('x', 'y', 'z')):
                return IntPosition(value['x'], value['y'], value['z'])
            elif all(k in value for k in (0, 1, 2)):
                return IntPosition(value[0], value[1], value[2])
        elif isinstance(value, str):
            # Try to parse string of format "x,y,z" or "x y z"
            try:
                coords = value.replace(',', ' ').split()
                if len(coords) != 3:
                    raise ValueError
                return IntPosition(*map(int, coords))
            except ValueError:
                raise ValueError(f"Cannot convert string '{value}' to IntPosition")
        elif isinstance(value, int):
            return IntPosition(value, value, value)
        
        raise TypeError(f"Cannot convert {type(value)} to IntPosition")

    def __str__(self):
        return self.to_nbt()

    def __eq__(self, other):
        if isinstance(other, IntPosition):
            return self.x == other.x and self.y == other.y and self.z == other.z
        return False

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __add__(self, other):
        if isinstance(other, (np.ndarray, list, tuple)):
            return IntPosition(self.x + int(other[0]), self.y + int(other[1]), self.z + int(other[2]))
        elif isinstance(other, int):
            return IntPosition(self.x + other, self.y + other, self.z + other)
        elif isinstance(other, IntPosition):
            return IntPosition(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            raise TypeError("Unsupported operand type for +")

    def __sub__(self, other):
        if isinstance(other, (np.ndarray, list, tuple)):
            return IntPosition(self.x - int(other[0]), self.y - int(other[1]), self.z - int(other[2]))
        elif isinstance(other, int):
            return IntPosition(self.x - other, self.y - other, self.z - other)
        elif isinstance(other, IntPosition):
            return IntPosition(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            raise TypeError("Unsupported operand type for -")

    def __mul__(self, other):
        if isinstance(other, (np.ndarray, list, tuple)):
            return IntPosition(self.x * int(other[0]), self.y * int(other[1]), self.z * int(other[2]))
        elif isinstance(other, int):
            return IntPosition(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, IntPosition):
            return IntPosition(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            raise TypeError("Unsupported operand type for *")

    def __truediv__(self, other):
        if isinstance(other, (np.ndarray, list, tuple)):
            return IntPosition(self.x // int(other[0]), self.y // int(other[1]), self.z // int(other[2]))
        elif isinstance(other, int):
            return IntPosition(self.x // other, self.y // other, self.z // other)
        elif isinstance(other, IntPosition):
            return IntPosition(self.x // other.x, self.y // other.y, self.z // other.z)
        else:
            raise TypeError("Unsupported operand type for /")

    def __rsub__(self, other):
        return IntPosition(other, other, other) - self if isinstance(other, int) else NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def box(self, other: IntPosition):
        pos_set = np.stack([self.numpy,other.numpy])
        return np.abs(self.numpy-other.numpy), np.min(pos_set,axis=0), np.max(pos_set,axis=0)
    
    def distance_to(self, other: IntPosition) -> float:
        return np.linalg.norm(self.to_numpy() - other.to_numpy())

    def manhattan_distance(self, other: IntPosition) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)

    def chebyshev_distance(self, other: IntPosition) -> int:
        return max(abs(self.x - other.x), abs(self.y - other.y), abs(self.z - other.z))

    def __dot__(self, other: IntPosition) -> int:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross_product(self, other: IntPosition) -> IntPosition:
        return IntPosition(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def magnitude(self) -> float:
        return np.linalg.norm(self.to_numpy())

    def normalize(self) -> np.ndarray:
        return self.to_numpy() / self.magnitude()

    @property
    def down(self):
        return IntPosition(x=self.x, y=self.y - 1, z=self.z)

    @property
    def up(self):
        return IntPosition(x=self.x, y=self.y + 1, z=self.z)

    @property
    def north(self):
        return IntPosition(x=self.x, y=self.y, z=self.z - 1)

    @property
    def south(self):
        return IntPosition(x=self.x, y=self.y, z=self.z + 1)

    @property
    def west(self):
        return IntPosition(x=self.x - 1, y=self.y, z=self.z)

    @property
    def east(self):
        return IntPosition(x=self.x + 1, y=self.y, z=self.z)
    
    @property
    def numpy(self):
        return np.array([self.x,self.y,self.z],dtype=np.int64)
    
    @property
    def abs(self) -> str:
        """Returns the position formatted for use in Minecraft commands."""
        return f"{self.x} {self.y} {self.z}"

    @property
    def rel(self) -> str:
        """Returns the position in Minecraft's tilde notation."""
        return f"~{self.x} ~{self.y} ~{self.z}"

    @property
    def loc(self) -> str:
        """Returns the position in Minecraft's caret notation."""
        return f"^{self.x} ^{self.y} ^{self.z}"

    def mix(self, relative_axes: List[str] = None) -> str:
        """
        Returns the position in a mixed notation, allowing specific axes to be relative.
        :param relative_axes: List of axes to be made relative (e.g., ['x', 'z'])
        :return: Mixed notation string
        """
        relative_axes = relative_axes or []
        x = f"~{self.x}" if 'x' in relative_axes else str(self.x)
        y = f"~{self.y}" if 'y' in relative_axes else str(self.y)
        z = f"~{self.z}" if 'z' in relative_axes else str(self.z)
        return f"{x} {y} {z}"
