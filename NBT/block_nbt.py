from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from enum import Enum, auto
from dataclasses import dataclass, field

from NBT.MBlocks import MBlocks
from NBT.nbt import NBTCompound, NBTTag, NBTType
from NBT.parameter import IntPosition, boolean, facing, identifier, Facing


if TYPE_CHECKING:
    from Command.Command import Command

@dataclass
class BlockState:

    def __str__(self):
        params = vars(self)
        states = [f"{n}={v}" for n,v in params if not v is None]
        if len(states) > 0:
            return "["+','.join(states)+"]"
        return ""

@dataclass
class Block:
    id: Union[identifier,MBlocks]
    block_state: Optional[BlockState] = None
    tags: Optional[NBTCompound] = None

    def __post_init__(self):
        if isinstance(self.id,MBlocks):
            self.id = identifier(name=self.id)
        assert isinstance(self.id,identifier)

    def __str__(self):
        s = f"{self.id}"
        if self.block_state:
            s += f"{self.block_state}"
        if self.tags:
            s += f"{self.tags}"
        return s
    
    @staticmethod
    def convert_from(value: Any):
        if isinstance(value, Block):
            return value
        if isinstance(value, identifier):
            return Block(id=value)
        if isinstance(value,MBlocks):
            return Block(id=identifier(name=value))
        # エラー処理の改善
        error_message = f"Unsupported type: {type(value).__name__} for value: {repr(value)}"
        raise NotImplementedError(error_message)
    
    @property
    def block(self):
        if isinstance(self.id,identifier):
            return self.id.block
        return self.id
    
class Instrument(Enum):
    # (ノートナンバーの開始番号, 下に置くブロックのid, 対応するmidiのinstrument(推定))
    CHIME = (78, MBlocks.packed_ice, [14,15], 3)
    XYLOPHONE = (78, MBlocks.bone_block, [12,13], 2)
    BELL = (78, MBlocks.gold_block, [112,113], 1)
    COW_BELL = (0, MBlocks.soul_sand, [56], 2) # 66
    FLUTE = (0, MBlocks.clay, list(range(64,80)), 1) # 66
    HARP = (54, MBlocks.dirt, list(range(9))+list(range(16,24))+[46,108],1)
    BIT = (54, MBlocks.emerald_block, [80,81], 4)
    IRON_XYLOPHONE = (54, MBlocks.iron_block, [8,9,11], 3)
    PLING = (54, MBlocks.glowstone, list(range(82,96)), 2)
    BANJO = (54, MBlocks.hay_block, list(range(104,107)), 5)
    GUITAR = (42, MBlocks.white_wool, list(range(32,40))+[120], 1)
    DIDGERIDOO = (30, MBlocks.pumpkin,  list(range(104, 107)), 2)
    BASS = (30, MBlocks.oak_planks, list(range(32,40))+[43], 1)
    BASEDRUM = (0, MBlocks.stone, list(range(114,119)), 1)
    HAT = (0, MBlocks.glass, [98], 2)
    SNARE = (0, MBlocks.sand, [119], 3)

    @staticmethod
    def note_to_instruments(note: int,is_percussion: bool=False):
        """
        Return list[tuple[Instrument,int]]: 音ブロックの情報と操作回数を返す
        """
        preds = []
        for n in Instrument:
            if is_percussion:
                if n.value[0] == 0:
                    if note in n.value[2]:
                        preds.append((n,0))
            elif 0 <= note - n.value[0] <= 24:
                preds.append((n, round(note - n.value[0])))
        if is_percussion and len(preds) == 0:
            return [(Instrument.BASEDRUM,0)]
        return preds

@dataclass
class NoteBlockState(BlockState):
    note: int = 0
    powered: boolean = field(default_factory=lambda : boolean(_value=False))
    instrument: Instrument = Instrument.HARP

    def __str__(self):
        return "["+','.join([f"note={self.note}",f"powered={self.powered}",f"instrument={self.instrument.name.lower()}"])+"]"

@dataclass
class NoteBlock(Block):
    id: identifier = field(default_factory=lambda: identifier(name=MBlocks.note_block), init=False)
    block_state : NoteBlockState

@dataclass
class CommandBlockState(BlockState):
    conditional: Optional[boolean] = None
    facing: Optional['facing'] = None

@dataclass
class CommandBlockTag(NBTCompound):
    name: str = field(default="", init=False)
    Command: Optional[Union[str, 'Command']] = None
    CustomName: Optional[str] = None
    TrackOutput: Optional[boolean] = None
    auto: Optional[boolean] = None
    tags: Optional[list[NBTTag]] = field(default_factory=list)

    def __post_init__(self):
        if self.Command:
            if isinstance(self.Command, Command):
                command_str = str(self.Command)
            else:
                command_str = self.Command
            self.tags.append(NBTTag("Command", NBTType.STRING, command_str))
        if self.CustomName:
            self.tags.append(NBTTag("CustomName", NBTType.STRING, self.CustomName))
        if self.TrackOutput is not None:
            self.tags.append(self.TrackOutput.to_nbt_tag("TrackOutput"))
        if self.auto is not None:
            self.tags.append(self.auto.to_nbt_tag("auto"))
        if len(self.tags) > 0:
            self.value = self.tags
        else:
            self.value = None

@dataclass
class CommandBlock(Block):
    class CommandBlockType(Enum):
        IMPULSE = MBlocks.command_block
        CHAIN = MBlocks.chain_command_block
        REPEAT = MBlocks.repeating_command_block

    block_state: Optional[CommandBlockState] = None
    id: identifier = field(default_factory=lambda: identifier(name=MBlocks.command_block), init=False)
    command_block_type: CommandBlockType = field(default=CommandBlockType.IMPULSE)
    tags: Optional[CommandBlockTag] = None

    def __post_init__(self):
        super().__post_init__()
        self.id = identifier(name=self.command_block_type.value)

    def __command_name__(self):
        return "setblock"

    def __command_str__(self):
        pos = getattr(self, 'pos', IntPosition(0, 0, 0))
        return f"{pos.abs} {self} replace"
    
@dataclass
class RedstoneRepeaterState(BlockState):
    delay: int = 1
    facing: 'facing' = field(default_factory=lambda: facing(Facing.NORTH))
    locked: boolean = field(default_factory=lambda: boolean(False))
    powered: boolean = field(default_factory=lambda: boolean(False))

    def __post_init__(self):
        if self.delay not in [1, 2, 3, 4]:
            raise ValueError("Delay must be 1, 2, 3, or 4")
        if self.facing.value not in [Facing.NORTH, Facing.EAST, Facing.SOUTH, Facing.WEST]:
            raise ValueError("Facing must be NORTH, EAST, SOUTH, or WEST")

    def __str__(self):
        return f"[delay={self.delay},facing={self.facing},locked={self.locked},powered={self.powered}]"

@dataclass
class RedstoneRepeater(Block):
    id: identifier = field(default_factory=lambda: identifier(name=MBlocks.repeater), init=False)
    block_state: RedstoneRepeaterState = field(default_factory=RedstoneRepeaterState)

    def __post_init__(self):
        super().__post_init__()

    @classmethod
    def create(cls, facing_direction: facing|None=None, delay: int = 1, locked: bool = False, powered: bool = False):
        if not facing_direction:
            facing_direction = facing(Facing.NORTH)
        return cls(block_state=RedstoneRepeaterState(
            delay=delay,
            facing=facing_direction,
            locked=boolean(locked),
            powered=boolean(powered)
        ))

    def set_delay(self, new_delay: int):
        if new_delay in [1, 2, 3, 4]:
            self.block_state.delay = new_delay
        else:
            raise ValueError("Delay must be 1, 2, 3, or 4")
        return self

    def set_facing(self, new_facing: facing):
        if new_facing.value in [Facing.NORTH, Facing.EAST, Facing.SOUTH, Facing.WEST]:
            self.block_state.facing = new_facing
        else:
            raise ValueError("Facing must be NORTH, EAST, SOUTH, or WEST")
        return self

    def set_locked(self, is_locked: bool):
        self.block_state.locked = boolean(is_locked)
        return self

    def set_powered(self, is_powered: bool):
        self.block_state.powered = boolean(is_powered)
        return self
    
class SlabType(Enum):
    BOTTOM = "bottom"
    TOP = "top"
    DOUBLE = "double"

@dataclass
class SlabState(BlockState):
    type: SlabType = SlabType.BOTTOM
    waterlogged: boolean = field(default_factory=lambda: boolean(False))

    def __str__(self):
        return f"[type={self.type.value},waterlogged={self.waterlogged}]"

@dataclass
class Slab(Block):
    id: identifier = field(default_factory=lambda: identifier(name=MBlocks.stone_slab))
    block_state: SlabState = field(default_factory=SlabState)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.block_state, SlabState):
            self.block_state = SlabState(**self.block_state.__dict__)

    @classmethod
    def create(cls, slab_type: SlabType = SlabType.BOTTOM, waterlogged: bool = False):
        return cls(block_state=SlabState(
            type=slab_type,
            waterlogged=boolean(waterlogged)
        ))

    def set_type(self, slab_type: SlabType):
        self.block_state.type = slab_type
        return self

    def set_waterlogged(self, is_waterlogged: bool):
        self.block_state.waterlogged = boolean(is_waterlogged)
        return self