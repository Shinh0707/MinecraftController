from typing import Any, Dict, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
import uuid


class NBTType(Enum):
    BYTE = "TAG_Byte"
    SHORT = "TAG_Short"
    INT = "TAG_Int"
    LONG = "TAG_Long"
    FLOAT = "TAG_Float"
    DOUBLE = "TAG_Double"
    STRING = "TAG_String"
    LIST = "TAG_List"
    COMPOUND = "TAG_Compound"
    INVALID = ""

    @staticmethod
    def get_nbt_type(value: Any):
        nbt_type = NBTType.INVALID
        if isinstance(value, float):
            if 1.0e-38 <= abs(value) <= 1.0e+38:
                nbt_type = NBTType.FLOAT
            elif 1.0e-308 <= abs(value) <= 1.0e+308:
                nbt_type = NBTType.DOUBLE
        elif isinstance(value, str):
            nbt_type = NBTType.STRING
        elif isinstance(value, int):
            if -128 <= value <= 127:
                nbt_type = NBTType.BYTE
            elif -32768 <= value <= 32767:
                nbt_type = NBTType.SHORT
            elif -2147483648 <= value <= 2147483647:
                nbt_type = NBTType.INT
            elif -9223372036854775808 <= value <= 9223372036854775807:
                nbt_type = NBTType.LONG
        elif isinstance(value, bool):
            nbt_type = NBTType.BYTE
        elif isinstance(value, dict):
            nbt_type = NBTType.COMPOUND
        elif isinstance(value, (list, tuple)):
            nbt_type = NBTType.LIST
        return nbt_type


@dataclass
class NBTTag:
    name: str = field(default="")
    type: NBTType = field(default=NBTType.INVALID)
    value: Any = field(default=None)

    def to_nbt(self) -> str:
        if self.value is None:
            return ""
        if self.name is None or len(self.name) == 0:
            return NBTTag.to_nbt_str(self.value, self.type)
        return "{}:{}".format(self.name, NBTTag.to_nbt_str(self.value, self.type))

    @staticmethod
    def create_nbt_tag(name: str, value: Any, nbt_type: NBTType | None = None):
        if nbt_type is None:
            nbt_type = NBTType.get_nbt_type(value)
        return NBTTag(name=name, type=nbt_type,value=value)

    @staticmethod
    def to_nbt_str(value: Any, nbt_type: NBTType | None = None):
        if isinstance(value, NBTTag):
            return value.to_nbt()
        if nbt_type is None:
            nbt_type = NBTType.get_nbt_type(value)
        if nbt_type == NBTType.BYTE:
            if isinstance(value, bool):
                return "true" if value else "false"
            return f"{value}b"
        elif nbt_type == NBTType.SHORT:
            return f"{value}s"
        elif nbt_type == NBTType.INT:
            return f"{value}"
        elif nbt_type == NBTType.LONG:
            return f"{value}L"
        elif nbt_type == NBTType.FLOAT:
            return f"{value}f"
        elif nbt_type == NBTType.DOUBLE:
            return f"{value}d"
        elif nbt_type == NBTType.STRING:
            return f'"{value}"'
        elif nbt_type == NBTType.LIST:
            return f"[{','.join(NBTTag.to_nbt_str(v) for v in value if not v is None)}]"
        elif nbt_type == NBTType.COMPOUND:
            if isinstance(value, dict):
                valid_values = [v.to_nbt() if isinstance(v, NBTTag) else NBTTag.create_nbt_tag(
                    n, v).to_nbt() for n, v in value.items() if not v is None]
                if len(valid_values) > 0:
                    return "{" + ",".join(valid_values) + "}"
                else:
                    return "{}"
            elif isinstance(value, list):
                valid_values = [v.to_nbt() for v in value if (
                    not v is None) and isinstance(v, NBTTag)]
                if len(valid_values) > 0:
                    return "{" + ",".join(valid_values) + "}"
                else:
                    return "{}"
        else:
            raise ValueError(f"Unknown NBT type: {nbt_type}")
    
    def __str__(self):
        return self.to_nbt()
        
@dataclass
class NBTCompound(NBTTag):
    name: str = field(default="")
    value: Any = field(default=None,init=False)
    type: NBTType = field(default=NBTType.COMPOUND,init =False)
    tags: list[NBTTag] = field(default_factory=list)

    def __post_init__(self):
        self.value = self.tags

    def __add__(self,other):
        if isinstance(other,NBTCompound):
            return NBTCompound(name=self.name,tags=self.tags+other.tags)
        if isinstance(other,NBTTag):
            if other.type == NBTType.COMPOUND and hasattr(other,'tags') and isinstance(other.tags,list):
                return NBTCompound(name=self.name,tags=self.tags+other.tags)
        raise NotImplementedError


@dataclass
class NBTList(NBTTag):
    name: str = field(default=None)
    value: Any = field(default=None)
    type: NBTType = field(default=NBTType.LIST,init=False)

    def __post_init__(self):
        if not self.name:
            self.name = self.__class__.__name__


@dataclass
class UUID(NBTTag):
    name: str = field(default="UUID", init=False)
    _value: Union[str|list[int]] = field(default=None)
    use_int_array: bool = False

    @classmethod
    def random(cls, use_int_array: bool=False):
        # Generate a random UUID (version 4, variant 1)
        random_uuid = uuid.uuid4()
        return cls(_value=str(random_uuid), use_int_array=use_int_array)

    def to_nbt(self) -> str:
        if self.use_int_array:
            int_array = self._uuid_to_int_array(self._value)
            return f"UUID:[I;{','.join(map(str, int_array))}]"
        else:
            return f'UUID:"{self._value}"'

    @staticmethod
    def _uuid_to_int_array(uuid_str: str) -> list[int]:
        uuid_obj = uuid.UUID(uuid_str)
        return [
            uuid_obj.time_low,
            uuid_obj.time_mid,
            uuid_obj.time_hi_version,
            (uuid_obj.clock_seq_hi_variant << 8) | uuid_obj.clock_seq_low,
            uuid_obj.node >> 32,
            uuid_obj.node & 0xFFFFFFFF
        ]

    @staticmethod
    def _int_array_to_uuid(int_array: list[int]) -> str:
        if len(int_array) != 6:
            raise ValueError("Invalid int_array length for UUID")

        uuid_parts = [
            f"{int_array[0]:08x}",
            f"{int_array[1]:04x}",
            f"{int_array[2]:04x}",
            f"{int_array[3]:04x}",
            f"{(int_array[4] << 32 | int_array[5]):012x}"
        ]
        return "-".join(uuid_parts)

    def __post_init__(self):
        if isinstance(self._value, list):
            self.use_int_array = True
            self._value = self._int_array_to_uuid(self._value)
        elif not isinstance(self._value, str):
            raise ValueError(
                "UUID value must be either a string or a list of integers")
        self.type = NBTType.LIST if self.use_int_array else NBTType.STRING
        self.value = self._value

    def to_int_array(self) -> list[int]:
        return self._uuid_to_int_array(self._value)

    def to_string(self) -> str:
        return self._value
