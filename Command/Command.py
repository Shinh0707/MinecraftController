from __future__ import annotations
import operator
from typing import Iterable, List, Dict, Literal, NamedTuple, Tuple, Union, Optional, Iterator
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, Flag, auto

import numpy as np
from Command.CommandEnum import Advancements
from NBT.MBlocks import MBlocks
from NBT.block_nbt import Block
from NBT.item_nbt import MItem
from NBT.parameter import Day, EffectType, IntPosition, Rotation, Seconds, Tick, WeatherSpec, angle, rotation, facing, boolean, identifier, dimension
from NBT.selector import Range, Target, SelectorType
import NBT.selector as nslc
from NBT.nbt import NBTCompound

class NamedEnum(Enum):
    def __str__(self):
        return self.name.lower()

@dataclass
class Command:
    def __str__(self):
        if not self.__command_str__() is None and len(self.__command_str__()) > 0:
            return f"{self.__command_name__()} {self.__command_str__()}"
        return f"{self.__command_name__()}"

    def __command_name__(self):
        return self.__class__.__name__.lower()
    
    def __command_str__(self) -> str:
        raise NotImplementedError
    
    def __call__(self, mc, **kwargs):
        return mc.execute(self,**kwargs)
    
    @staticmethod
    def c(command: str):
        return DirectCommand(command)

@dataclass
class DirectCommand(Command):
    command: str
    def __str__(self):
        return self.command

class Commands(Sequence):
    def __init__(self, *commands: Command) -> None:
        self.commands: List[Command] = list(commands)
    
    def execute(self):
        for command in self.commands:
            yield command
    
    def __call__(self, mc, **kwargs):
        return [command(mc,**kwargs) for command in self.execute()]

    def __getitem__(self, index: Union[int, slice]) -> Union[Command, 'Commands']:
        if isinstance(index, int):
            return self.commands[index]
        elif isinstance(index, slice):
            return Commands(*self.commands[index])
        raise TypeError("Index must be int or slice")

    def __len__(self) -> int:
        return len(self.commands)

    def __iter__(self) -> Iterator[Command]:
        return iter(self.commands)

    def __add__(self, other: Union[Command, 'Commands']) -> 'Commands':
        if isinstance(other, Command):
            return Commands(*self.commands, other)
        elif isinstance(other, Commands):
            return Commands(*self.commands, *other.commands)
        raise TypeError("Can only concatenate Command or Commands objects")

    def __iadd__(self, other: Union[Command, 'Commands']) -> 'Commands':
        if isinstance(other, Command):
            self.commands.append(other)
        elif isinstance(other, Commands):
            self.commands.extend(other.commands)
        else:
            raise TypeError("Can only concatenate Command or Commands objects")
        return self

    def append(self, command: Command) -> None:
        self.commands.append(command)

    def extend(self, commands: Union[List[Command], 'Commands']) -> None:
        if isinstance(commands, Commands):
            self.commands.extend(commands.commands)
        else:
            self.commands.extend(commands)

    def insert(self, index: int, command: Command) -> None:
        self.commands.insert(index, command)

    def remove(self, command: Command) -> None:
        self.commands.remove(command)

    def pop(self, index: int = -1) -> Command:
        return self.commands.pop(index)

    def clear(self) -> None:
        self.commands.clear()

    def index(self, command: Command, start: int = 0, end: int = None) -> int:
        return self.commands.index(command, start, end if end is not None else len(self.commands))

    def count(self, command: Command) -> int:
        return self.commands.count(command)

    def reverse(self) -> None:
        self.commands.reverse()

    def sort(self, key=None, reverse=False) -> None:
        self.commands.sort(key=key, reverse=reverse)

    def __str__(self) -> str:
        return "\n".join(str(command) for command in self.commands)

    def __repr__(self) -> str:
        return f"Commands({', '.join(repr(command) for command in self.commands)})"


@dataclass
class Advancement(Command):
    class Operation(NamedEnum):
        GRANT = auto()
        REVOKE = auto()
    
    operation: Operation
    target: Target

    def __command_name__(self):
        return "advancement"
    
    def __command_str__(self):
        return f"{self.operation} {self.target} {self.__advancement_str__()}"
    
    def __advancement_str__(self) -> str:
        raise NotImplementedError

@dataclass
class AdvancementEverything(Advancement):
    def __advancement_str__(self) -> str:
        return "everything"

@dataclass
class AdvancementOnly(Advancement):
    advancement: Advancements
    require: Optional[str] = None
    def __advancement_str__(self) -> str:
        return f"only {self.advancement}" + (f" \"{self.require}\"" if self.require else '')

@dataclass
class AdvancementRange(Advancement):
    mode: Literal['from','throgh','until']
    advancement: Advancements
    def __advancement_str__(self) -> str:
        return f"{self.mode} {self.advancement}"

SET_BLOCK_MODE = Literal['keep','replace','destroy']

@dataclass
class SetBlock(Command):
    pos: IntPosition
    block: Union[Block,MBlocks,identifier]
    mode: SET_BLOCK_MODE = field(default='replace')

    def __command_str__(self) -> str:
        return f"{self.pos.abs} {Block.convert_from(self.block)} {self.mode}"
    
    def get_block(self):
        if isinstance(self.block,(Block,identifier)):
            return self.block.block
        return self.block

@dataclass
class SetDirection:
    forward: float = 0
    right: float = 0
    up: float = 0

    def _operate(self, other: Union[float, 'SetDirection', np.ndarray, tuple, Iterable], op):
        if isinstance(other, (int, float)):
            return SetDirection(
                op(self.forward, other),
                op(self.right, other),
                op(self.up, other)
            )
        elif isinstance(other, SetDirection):
            return SetDirection(
                op(self.forward, other.forward),
                op(self.right, other.right),
                op(self.up, other.up)
            )
        elif isinstance(other, (np.ndarray, tuple, Iterable)):
            other = list(other)
            if len(other) != 3:
                raise ValueError("Iterable must have exactly 3 elements")
            return SetDirection(
                op(self.forward, other[0]),
                op(self.right, other[1]),
                op(self.up, other[2])
            )
        else:
            return NotImplemented

    def __add__(self, other):
        return self._operate(other, operator.add)

    def __sub__(self, other):
        return self._operate(other, operator.sub)

    def __mul__(self, other):
        return self._operate(other, operator.mul)

    def __truediv__(self, other):
        return self._operate(other, operator.truediv)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return SetDirection(
            operator.sub(other, self.forward),
            operator.sub(other, self.right),
            operator.sub(other, self.up)
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return SetDirection(
            operator.truediv(other, self.forward),
            operator.truediv(other, self.right),
            operator.truediv(other, self.up)
        )

@dataclass
class BlockDirectionPair:
    block: Union[Block, MBlocks, str] = None
    direction: SetDirection = field(default_factory=SetDirection)

    def __post_init__(self):
        if isinstance(self.block, str):
            self.block = Block(identifier(name=self.block))
        elif isinstance(self.block, MBlocks):
            self.block = Block(self.block)

@dataclass
class RelativeSetBlock(Command):
    block_direction_pair: RelativeSetBlockTYPE
    mode: str = 'replace'
    param: BlockDirectionPair = field(default_factory=BlockDirectionPair,init=False)

    def __post_init__(self):
        if isinstance(self.block_direction_pair, BlockDirectionPair):
            self.param = self.block_direction_pair
        if isinstance(self.block_direction_pair, (Block, MBlocks, str)):
            self.param = BlockDirectionPair(block=self.block_direction_pair)
        elif isinstance(self.block_direction_pair, tuple):
            if len(self.block_direction_pair) == 2:
                self.param = BlockDirectionPair(*self.block_direction_pair)
            else:
                self.param = BlockDirectionPair(block=self.block_direction_pair[0],direction=SetDirection(*self.block_direction_pair[1]))

    def __command_name__(self):
        return "setblock"
    
    def __command_str__(self) -> str:
        return self.to_absolute_command(IntPosition(0,0,0), 0).__command_str__()

    def to_absolute_command(self, agent_pos: IntPosition, agent_rotation: rotation):
        # 回転に基づいて相対座標を絶対座標に変換
        rotation_value = agent_rotation.value.value
        if rotation_value % 4 != 0:
            rotation_value = agent_rotation.to_cardinal().value.value
        
        rot = rotation(Rotation(rotation_value))
        if isinstance(self.block(), Block):
            if hasattr(self.block().block_state, 'facing'):
                self.param.block.block_state.facing = -rot.facing()
        dx, dz = 0, 0
        if rotation_value == 0:  # 北向き
            dx, dz = self.right(), -self.forward()
        elif rotation_value == 4:  # 東向き
            dx, dz = self.forward(), self.right()
        elif rotation_value == 8:  # 南向き
            dx, dz = -self.right(), self.forward()
        elif rotation_value == 12:  # 西向き
            dx, dz = -self.forward(), -self.right()
        
        absolute_pos = IntPosition(
            agent_pos.x + dx,
            agent_pos.y + self.up(),
            agent_pos.z + dz
        )

        return SetBlock(absolute_pos, self.block(), self.mode)

    def build(self, agent_pos: IntPosition, agent_rotation: rotation) -> SetBlock:
        return self.to_absolute_command(agent_pos, agent_rotation)

    def __repr__(self):
        return f"RelativeSetBlock(forward={self.forward()}, right={self.right()}, up={self.up()}, block={self.block()}, mode='{self.mode}')"
    
    def block(self):
        return self.param.block

    def direction(self):
        return self.param.direction

    def forward(self):
        return self.direction().forward
    
    def right(self):
        return self.direction().right
    
    def up(self):
        return self.direction().up


RelativeSetBlockTYPE = Union[RelativeSetBlock,BlockDirectionPair,Union[Block,MBlocks,str],Tuple[Union[Block,MBlocks,str],SetDirection]]

@dataclass
class Fill(Command):
    _from: IntPosition
    _to: IntPosition
    block: Union[Block,MBlocks,identifier]
    mode: Literal['keep','replace','destroy','outline','hollow'] = field(default='replace')

    def __command_str__(self) -> str:
        return f"{self._from.abs} {self._to.abs} {Block.convert_from(self.block)} {self.mode}"
    
    def get_block(self):
        if isinstance(self.block,(Block,identifier)):
            return self.block.block
        return self.block

@dataclass
class FillReplace(Fill):
    filter: Optional[Union[Block,MBlocks,identifier]] = None
    mode: str = field(default='replace',init=False)

    def __command_name__(self):
        return "fill"

    def __command_str__(self) -> str:
        if self.filter:
            return f"{super().__command_str__()} {Block.convert_from(self.filter)}"
        return super().__command_str__()

@dataclass
class DataTarget:
    target: Union[IntPosition,Target,identifier,str]

    def __str__(self):
        if isinstance(self.target,IntPosition):
            return f"block {self.target.abs}"
        elif isinstance(self.target,Target):
            return f"entity {self.target}"
        elif isinstance(self.target,identifier):
            return f"storage {self.target.to_nbt_tag(name='')}"
        return f"storage {self.target}"

@dataclass
class Data(Command):
    def __command_name__(self):
        return "data"
    
    def __command_str__(self):
        return f"{self.__operation_name__()} {self.__operation_str__()}"

    def __operation_name__(self):
        cls_name = self.__class__.__name__
        if cls_name.startswith("Data"):
            cls_name = cls_name[4:]
        return cls_name.lower()
    
    def __operation_str__(self):
        raise NotImplementedError
    
@dataclass
class DataGet(Data):
    target: DataTarget
    nbt_path: Optional[str] = None
    scale: Optional[float] = None

    def __operation_str__(self):
        o_str = f"{self.target}"
        if self.nbt_path:
            o_str += f" {self.nbt_path}"
        if self.scale:
            o_str += f" {self.scale}"
        return o_str
    
@dataclass
class Tp(Command):
    target: Target
    target_pos: IntPosition

    def __command_str__(self) -> str:
        return f"{self.target} {self.target_pos.abs}"
    
@dataclass
class Weather(Command):
    weather_spec: WeatherSpec
    duration: Optional[Union[Day,Seconds,Tick]] = None

    def __command_str__(self) -> str:
        return f"{self.weather_spec}" + ('' if self.duration is None else f" {self.duration}")

@dataclass
class Difficulty(Command):
    difficulty: nslc.Difficulty

    def __command_str__(self) -> str:
        return f"{self.difficulty}"

@dataclass
class GameMode(Command):
    game_mode: nslc.GameMode
    target: Optional[Target] = None

    def __command_str__(self) -> str:
        return f"{self.game_mode}" + (f" {self.target}" if self.target else '')
@dataclass
class Time(Command):
    operation: Literal['add','set']
    time: Union[Day,Seconds,Tick]

    def __command_str__(self) -> str:
        return f"{self.operation} {self.time}"
    
@dataclass
class Clear(Command):
    target: Optional[Target]
    item: Optional[MItem]
    max_count: Optional[int]

    def __command_str__(self) -> str:
        cmd_str = ""
        if self.target:
            cmd_str += f"{self.target}"
        if self.item:
            cmd_str += f" {self.item}"
        if self.max_count:
            cmd_str += f" {self.max_count}"
        return cmd_str
    
@dataclass
class Kill(Command):
    target: Optional[Target] = None

    def __command_str__(self) -> str:
        if self.target:
            return str(self.target)
        return ""

@dataclass
class Spawnpoint(Command):
    target: Optional[Target] = None
    pos: Optional[IntPosition] = None
    angle: Optional[float] = None

    def __command_str__(self) -> str:
        components = []
        if self.target:
            components.append(str(self.target))
        if self.pos:
            components.append(self.pos.abs)
        if self.angle is not None:
            components.append(str(self.angle))
        if len(components) > 0:
            return " ".join(components)
        return ""

@dataclass
class Effect(Command):
    class Operation(Enum):
        CLEAR = "clear"
        GIVE = "give"

    operation: Operation
    targets: Optional[Target] = None
    effect: Optional[EffectType] = None
    duration: Optional[Union[int, Seconds, Tick, str]] = None
    amplifier: Optional[int] = None
    hide_particles: Optional[Union[boolean,bool]] = None

    def __post_init__(self):
        if self.operation == Effect.Operation.GIVE and self.effect is None:
            raise ValueError("Effect must be specified for 'give' operation")
        
        if isinstance(self.duration, (int, float)):
            if self.effect and self.effect in [EffectType.INSTANT_DAMAGE, EffectType.INSTANT_HEALTH, EffectType.SATURATION]:
                self.duration = Tick(int(self.duration))
            else:
                self.duration = Seconds(int(self.duration))
        
        if isinstance(self.duration, Seconds) and (self.duration.value < 1 or self.duration.value > 1000000):
            raise ValueError("Duration must be between 1 and 1000000 seconds")
        
        if self.amplifier is not None and (self.amplifier < -2147483648 or self.amplifier > 2147483647):
            raise ValueError("Amplifier must be between -2147483648 and 2147483647")
        if self.hide_particles is not None and isinstance(self.hide_particles,bool):
            self.hide_particles = boolean(self.hide_particles)

    def __command_str__(self) -> str:
        command_parts = [self.operation.value]

        if self.targets:
            command_parts.append(str(self.targets))
        
        if self.effect:
            command_parts.append(str(self.effect.to_identifier()))
        
        if self.operation == Effect.Operation.GIVE:
            if self.duration:
                if self.duration == "infinite":
                    command_parts.append("infinite")
                else:
                    command_parts.append(str(self.duration.value))
            
            if self.amplifier is not None:
                command_parts.append(str(self.amplifier))
            
            if self.hide_particles is not None:
                command_parts.append(str(self.hide_particles))

        return " ".join(command_parts)

@dataclass
class Summon(Command):
    entity: Union[identifier, str]
    pos: Optional[IntPosition] = None
    nbt: Optional[NBTCompound] = None

    def __post_init__(self):
        if isinstance(self.entity, str):
            self.entity = identifier(name=self.entity)

    def __command_str__(self) -> str:
        command_parts = [str(self.entity)]
        
        if self.pos:
            command_parts.append(self.pos.abs)
        
        if self.nbt:
            command_parts.append(self.nbt.to_nbt())
        
        return " ".join(command_parts)

@dataclass
class Execute(Command):
    execute: Execute = field(default=None, metadata={"position": "last"},init=False)

    def __command_name__(self):
        return "execute"

    def __command_str__(self):
        additinal_execute = ""
        if self.execute:
            additinal_execute = f"{self.execute.__command_str__()}"
        ps = [p for p in [f"{self.__exec_name__()}",f"{self.__exec_str__()}",f"{additinal_execute}"] if len(p) > 0]
        if len(ps) > 0:
            return ' '.join(ps)
        return ''
    
    def __exec_name__(self):
        return self.__class__.__name__.lower()
    
    def __exec_str__(self):
        params = [str(v) for n,v in vars(self).items() if n != 'execute' and not v is None]
        if len(params) > 0:
            return ' '.join(params)
        return ''

class ExecuteChain:
    def __init__(self, *args):
        self.chain: List[Union[Execute, Command]] = list(args)

    def add(self, *args: Union[Execute, Command]):
        self.chain.extend(args)
        return self

    def build(self) -> Execute:
        if not self.chain:
            raise ValueError("ExecuteChain is empty")
        if isinstance(self.chain[-1], Execute):
            last_command = self.chain[-1]
        elif isinstance(self.chain[-1], Command):
            last_command = Run(self.chain[-1])
        else:
            raise NotImplementedError

        execute_command = last_command
        for exec_part in reversed(self.chain[:-1]):
            assert isinstance(exec_part,Execute)
            exec_part.execute = execute_command
            execute_command = exec_part

        return execute_command

    def __str__(self):
        return str(self.build())

@dataclass
class Align(Execute):
    class AXES(Flag):
        X = auto()
        Y = auto()
        Z = auto()
    axes: AXES = field(default_factory=lambda: Align.AXES.X|Align.AXES.Y|Align.AXES.Z)

    def __exec_str__(self):
        return ''.join([ax.name.lower() for ax in [Align.AXES.X,Align.AXES.Y,Align.AXES.Z] if ax in self.axes])
    
class Anchor(Enum):
    EYES = auto()
    FEET = auto()
    def __str__(self):
        return self.name.lower()

@dataclass
class Anchored(Execute):
    anchor: Anchor = field(default=Anchor.FEET)

@dataclass
class As(Execute):
    target: Target = None

@dataclass
class At(Execute):
    target: Target = None

@dataclass
class Facing(Execute):
    target: Union[IntPosition,tuple[Target,Anchor]] = None

    def __exec_str__(self):
        if isinstance(self.target,IntPosition):
            return self.target.abs
        targets, anchor = self.target 
        return f"{targets} {anchor}"

@dataclass
class In(Execute):
    dimension: dimension = None

    def __exec_str__(self):
        return f"{self.dimension.to_nbt_tag(name='')}"

@dataclass
class On(Execute):
    class Relation(Enum):
        ATTACKER = auto()
        CONTROLLER = auto()
        LEASHER = auto()
        ORIGIN = auto()
        OWNER = auto()
        PASSENGERS = auto()
        TARGET = auto()
        VEHICLE = auto()

        def __str__(self):
            return self.name.lower()
    relation: Relation = None

    def __exec_str__(self):
        return f"{self.relation}"

class Heightmap(Enum):
    WORLD_SURFACE = auto()
    MOTION_BLOCKING = auto()
    MOTION_BLOCKING_NO_LEAVES = auto()
    OCEAN_FLOOR = auto()

    def __str__(self):
        return self.name.lower()

@dataclass
class Positioned(Execute):
    target: Union[IntPosition,Target,Heightmap] = None

    def __exec_str__(self):
        if isinstance(self.target,IntPosition):
            return self.target.abs
        elif isinstance(self.target,Target):
            return f"as {self.target}"
        elif isinstance(self.target,Heightmap):
            return f"over {self.target}"
        raise NotImplementedError

@dataclass
class Rot:
    yaw: float
    pitch: float
    yaw_local: Optional[bool] = False
    pitch_local: Optional[bool] = False

    def __str__(self):
        return f"{'~' if self.yaw_local else ''}{self.yaw} {'~' if self.pitch_local else ''}{self.pitch}"

@dataclass
class Rotated(Execute):
    target: Union[Rot,Target] = None

    def __exec_str__(self):
        if isinstance(self.target,Target):
            return f"as {self.target}"
        return f"{self.target}"

@dataclass
class SummonExecute(Execute):
    entity: Union[identifier,str] = None

    def __exec_name__(self):
        return "summon"

    def __exec_str__(self):
        return f"{self.entity}"

@dataclass
class Checker:
    def __str__(self):
        cls_name = self.__class__.__name__.lower()
        if cls_name.endswith("checker"):
            cls_name = cls_name[:-6]
        return f"{cls_name} " + self.__param_str__()
    
    def __param_str__(self):
        return ' '.join([str(v) for _,v in vars(self)])

@dataclass
class ExecuteChecker(Execute):
    check: Checker = None

    def __exec_str__(self):
        return f"{self.check}"

@dataclass
class If(ExecuteChecker):
    pass

@dataclass
class Unless(ExecuteChecker):
    pass

@dataclass
class BlockChecker(Checker):
    pos: IntPosition = None
    block: MBlocks = None

    def __param_str__(self):
        return f"{self.pos.abs} {self.block}"

@dataclass
class BlocksChecker(Checker):
    class Mask(Enum):
        ALL = auto()
        MASKED = auto()

        def __str__(self):
            return self.name.lower()
    start: IntPosition = None
    end: IntPosition = None
    destination: IntPosition = None
    mask: Mask = None

    def __param_str__(self):
        return f"{self.start.abs} {self.end.abs} {self.destination.abs} {self.mask}"

@dataclass
class DataChecker(Checker):
    target: Union[IntPosition,Target,identifier,str] = None
    path: str = None

    def __param_str__(self):
        target_str = ""
        if isinstance(self.target,IntPosition):
            target_str = "block " + self.target.abs
        elif isinstance(self.target,Target):
            target_str = f"entity {self.target}"
        elif isinstance(self.target,identifier):
            target_str = f"storage {self.target.to_nbt_tag(name='')}"
        else:
            target_str = f"storage {self.target}"
        return f"{target_str} {self.path}"

@dataclass
class DimensionChecker(Checker):
    dimension: dimension = None

    def __param_str__(self):
        return f"{self.dimension.to_nbt_tag(name='')}"

@dataclass
class EntityChecker(Checker):
    entity: Target = None

    def __param_str__(self):
        return f"{self.entity}"
    
@dataclass
class LoadedChecker(Checker):
    pos: IntPosition = None

    def __param_str__(self):
        return self.pos.abs

class CompareOperator(Enum):
    LT = "<"
    LE = "<="
    EQ = "="
    GE = ">="
    GT = ">"

    def __str__(self):
        return self.value

class ScoreComparison(NamedTuple):
    target: Union[Target, str] = None
    target_objective: str = None
    operator: CompareOperator = None
    source: Union[Target, str] = None
    source_objective: str = None

class ScoreRange(NamedTuple):
    target: Union[Target, str] = None
    target_objective: str = None
    range: Range = None

@dataclass
class ScoreChecker(Checker):
    check: Union[ScoreComparison, ScoreRange] = None

    def __param_str__(self):
        if isinstance(self.check, ScoreComparison):
            return f"{self.check.target} {self.check.target_objective} {self.check.operator} {self.check.source} {self.check.source_objective}"
        elif isinstance(self.check, ScoreRange):
            return f"{self.check.target} {self.check.target_objective} matches {self.check.range}"
        raise ValueError("Invalid check type for ScoreChecker")

class StoreMode(Enum):
    RESULT = auto()
    SUCCESS = auto()

    def __str__(self):
        return self.name.lower()

class NBTDataType(Enum):
    BYTE = auto()
    SHORT = auto()
    INT = auto()
    LONG = auto()
    FLOAT = auto()
    DOUBLE = auto()

    def __str__(self):
        return self.name.lower()

@dataclass
class Store(Execute):
    mode: StoreMode = None

    def __exec_name__(self):
        return "store"

    def __exec_str__(self):
        return f"{self.mode} {self._store_specific_str()}"

    def _store_specific_str(self):
        raise NotImplementedError("Subclasses must implement this method")

@dataclass
class StoreBlock(Store):
    target_pos: IntPosition = None
    path: str = None
    data_type: NBTDataType = None
    scale: float = None

    def _store_specific_str(self):
        return f"block {self.target_pos} {self.path} {self.data_type} {self.scale}"

@dataclass
class StoreBossbar(Store):
    id: identifier = None
    value_type: Literal["value", "max"] = None

    def _store_specific_str(self):
        return f"bossbar {self.id} {self.value_type}"

@dataclass
class StoreEntity(Store):
    target: Target = None
    path: str = None
    data_type: NBTDataType = None
    scale: float = None

    def _store_specific_str(self):
        return f"entity {self.target} {self.path} {self.data_type} {self.scale}"

@dataclass
class StoreScore(Store):
    targets: Union[Target, str] = None
    objective: str = None

    def _store_specific_str(self):
        return f"score {self.targets} {self.objective}"

@dataclass
class StoreStorage(Store):
    target: identifier = None
    path: str = None
    data_type: NBTDataType = None
    scale: float = None

    def _store_specific_str(self):
        return f"storage {self.target} {self.path} {self.data_type} {self.scale}"

@dataclass
class Run(Execute):
    execute: Execute = field(default=None,init=False)
    command: Command = None

    def __exec_str__(self):
        return f"{self.command}"


    




