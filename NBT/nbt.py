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
    name: str
    type: NBTType = field(default=NBTType.INVALID)
    value: Any = field(default=None)

    def to_nbt(self) -> str:
        if self.value is None:
            return ""
        if len(self.name) == 0:
            return NBTTag.to_nbt_str(self.value, self.type)
        return "{}:{}".format(self.name, NBTTag.to_nbt_str(self.value, self.type))

    @staticmethod
    def create_nbt_tag(name: str, value: Any, nbt_type: NBTType | None = None):
        if nbt_type is None:
            nbt_type = NBTType.get_nbt_type(value)
        return NBTTag(name, value, nbt_type)

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
    tags: list[NBTTag] = field(default_factory=[])

    @property
    def name(self):
        return ""

    @property
    def type(self):
        return NBTType.COMPOUND

    @property
    def value(self):
        return self.tags


@dataclass
class NBTList(NBTTag):
    
    @property
    def name(self):
        name = self.__class__.__name__
        return name

    @property
    def type(self):
        return NBTType.LIST


@dataclass
class UUID(NBTTag):
    value: Union[str|list[int]]
    use_int_array: bool = False

    @classmethod
    def random(cls, use_int_array: bool=False):
        # Generate a random UUID (version 4, variant 1)
        random_uuid = uuid.uuid4()
        return cls(str(random_uuid), use_int_array)

    @property
    def name(self) -> str:
        return "UUID"

    @property
    def type(self) -> NBTType:
        return NBTType.LIST if self.use_int_array else NBTType.STRING

    def to_nbt(self) -> str:
        if self.use_int_array:
            int_array = self._uuid_to_int_array(self.value)
            return f"UUID:[I;{','.join(map(str, int_array))}]"
        else:
            return f'UUID:"{self.value}"'

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
        if isinstance(self.value, list):
            self.use_int_array = True
            self.value = self._int_array_to_uuid(self.value)
        elif not isinstance(self.value, str):
            raise ValueError(
                "UUID value must be either a string or a list of integers")

    def to_int_array(self) -> list[int]:
        return self._uuid_to_int_array(self.value)

    def to_string(self) -> str:
        return self.value
