from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from itertools import repeat
import random
import re
from typing import Dict, Iterable, List, Optional, Tuple, Union
import uuid
import numpy as np
import Command.Command as cmd
from Command.Command import SET_BLOCK_MODE, As, At, BlockDirectionPair, Commands, Command, DataGet, DataTarget, Effect, ExecuteChain, Kill, Positioned, RelativeSetBlock, RelativeSetBlockTYPE, Rotated, SetBlock, Fill, Execute, Run, SetDirection, Spawnpoint, Summon, Weather
from Helper.helpers import divide_box_discrete, mode_val, partition_3d_space
from NBT.MBlocks import MBlocks
from NBT.block_nbt import Block, CommandBlock
from NBT.nbt import NBTCompound, NBTTag, NBTType
from NBT.parameter import EffectType, IntPosition, Rotation, Seconds, Tick, TimeSpec, WeatherSpec, angle, identifier, rotation
from NBT.selector import Difficulty, GameMode, SelectorType, Target

@dataclass
class CompoundCommand:
    commands: Commands = field(default_factory=Commands)

    def __post_init__(self):
        pass

    def add_command(self, command: Command|Commands|CompoundCommand):
        if isinstance(command,Command):
            self.commands.append(command)
        elif isinstance(command,Commands):
            self.commands += command
        elif isinstance(command,CompoundCommand):
            self.commands += command.commands

    def execute(self):
        return self.commands.execute()
    
    def __call__(self, mc, **kwargs):
        return self.commands(mc, **kwargs)

MAX_FILL_SIZE = 32

@dataclass
class AdjustFill(CompoundCommand):
    base_fill: Fill
    commands: Commands = field(default_factory=Commands,init=False)

    def __post_init__(self):
        _, box_start, box_end = self.base_fill._from.box(self.base_fill._to)
        result = divide_box_discrete(box_start,box_end,MAX_FILL_SIZE)
        for start, end in result:
            self.add_command(Fill(IntPosition(start),IntPosition(end),self.base_fill.block,self.base_fill.mode))

@dataclass
class SetBlocks(CompoundCommand):
    start_pos: IntPosition
    blocks_array: np.ndarray
    blocks_mapping: np.ndarray
    mode: SET_BLOCK_MODE = field(default='replace')
    do_prefill: bool = True
    empty_mask: np.ndarray = None
    commands: Commands = field(default_factory=Commands,init=False)
    
    def __post_init__(self):
        blocks = self.blocks_array.copy()
        block_mapping = [Block.convert_from(bm) for bm in self.blocks_mapping]
        if not self.empty_mask is None:
            blocks[self.empty_mask] = -1
        elif self.do_prefill:
            mode_block = mode_val(blocks)
            coords = np.argwhere(blocks==mode_block)
            p1, p2 = np.min(coords, axis=0), np.max(coords, axis=0)
            ps,pe = self.start_pos+IntPosition(p1), self.start_pos+IntPosition(p2)
            self.add_command(AdjustFill(Fill(
                ps,
                pe,
                block_mapping[mode_block],
                self.mode
            )))
            blocks[blocks==mode_block] = -1
        parts = partition_3d_space(blocks)
        for p1, p2, b in parts:
            if b == -1:
                continue
            p1, p2 = IntPosition(p1), IntPosition(p2)
            ps, pe = self.start_pos+p1,self.start_pos+p2
            self.add_command(AdjustFill(Fill(
                ps,
                pe,
                block_mapping[b],
                self.mode
            )))

@dataclass
class GetPosition(CompoundCommand):
    target: Target
    commands: Commands = field(default_factory=Commands,init=False)

    def __post_init__(self):
        self.add_command(DataGet(DataTarget(self.target), "Pos"))

    def __call__(self, mc, **kwargs):
        responses = super().__call__(mc,**kwargs)
        response = responses[-1]
        match = re.search(
            r'\[(-?\d+\.\d+)d\, (-?\d+\.\d+)d\, (-?\d+\.\d+)d\]', response)
        if match:
            x, y, z = map(int, map(float, match.groups()))
            return IntPosition(x=x, y=y, z=z)
        else:
            raise ValueError(
                f"Failed to parse position from response: {response}")

@dataclass
class GetRotation(CompoundCommand):
    target: Target
    commands: Commands = field(default_factory=Commands, init=False)

    def __post_init__(self):
        self.add_command(DataGet(DataTarget(self.target), "Rotation"))

    def __call__(self, mc, **kwargs):
        responses = super().__call__(mc, **kwargs)
        response = responses[-1]
        match = re.search(r'\[(-?\d+\.\d+)f\, (-?\d+\.\d+)f\]', response)
        if match:
            yaw, pitch = map(float, match.groups())
            # Convert Minecraft yaw to rotation
            # Minecraft yaw: 0 = south, 90 = west, 180/-180 = north, -90 = east
            # Our rotation: 0 = north, 4 = east, 8 = south, 12 = west
            angl = angle(yaw)
            return angl.rotation()
        else:
            raise ValueError(f"Failed to parse rotation from response: {response}")

@dataclass
class BaseSettings(CompoundCommand):
    target: Optional[Target] = None
    game_mode: Optional[GameMode] = None
    difficulty: Optional[Difficulty] = None
    time: Optional[TimeSpec] = None
    weather: Optional[WeatherSpec] = None
    clear_items: bool = False
    kill_others: bool = False
    set_spawnpoint: bool = False
    spawnpoint_pos: Optional[IntPosition] = None
    spawnpoint_angle: Optional[float] = None
    commands: Commands = field(default_factory=Commands, init=False)

    def __post_init__(self):
        if self.game_mode:
            self.add_command(cmd.GameMode(self.game_mode, self.target))
        if self.difficulty:
            self.add_command(cmd.Difficulty(self.difficulty))
        if self.time is not None:
            self.add_command(cmd.Time('set', self.time))
        if self.weather:
            self.add_command(cmd.Weather(self.weather))
        if self.clear_items:
            self.add_command(Kill(Target(SelectorType.ALL_ENTITIES, type="item")))
        if self.kill_others and self.target:
            self.add_command(Kill(Target(SelectorType.ALL_ENTITIES, type=f"!{self.target.type}")))
        if self.set_spawnpoint:
            self.add_command(Spawnpoint(self.target, self.spawnpoint_pos, self.spawnpoint_angle))

@dataclass
class SuperFlat(CompoundCommand):
    class LayerReference(Enum):
        TOP = "top"
        BOTTOM = "bottom"

    @dataclass
    class LayerStartPoint:
        layer_index: int
        reference: SuperFlat.LayerReference

    @dataclass
    class BottomOffset:
        offset: int

    surface_pos: IntPosition
    range: IntPosition
    layers: list[tuple[int, Block]]
    start_point: Union[LayerStartPoint, BottomOffset]
    commands: Commands = field(default_factory=Commands, init=False)

    def __post_init__(self):
        
        if isinstance(self.start_point, SuperFlat.LayerStartPoint):
            start_y = self._calculate_layer_start(self.start_point)
        elif isinstance(self.start_point, SuperFlat.BottomOffset):
            start_y = self.surface_pos.y - self.start_point.offset
        else:
            raise ValueError("Invalid start_point specification")

        current_y = start_y
        self.layers.reverse()
        for layer_height, block in self.layers:
            layer_end = current_y + layer_height - 1
            fill_command = AdjustFill(
                Fill(
                    IntPosition(self.surface_pos.x - self.range.x, current_y, self.surface_pos.z - self.range.z),
                    IntPosition(self.surface_pos.x + self.range.x, layer_end, self.surface_pos.z + self.range.z),
                    block
                )
            )
            self.add_command(fill_command)
            current_y = layer_end + 1

    def _calculate_layer_start(self, start_point: LayerStartPoint) -> int:
        layer_heights = [layer[0] for layer in self.layers]
        if start_point.layer_index < 0 or start_point.layer_index >= len(self.layers):
            raise ValueError("Invalid layer index")
        top_layer = start_point.layer_index
        # TOPに変換して計算
        if start_point.reference == SuperFlat.LayerReference.BOTTOM:
            top_layer += 1
        if top_layer >= len(self.layers):
            height = 0
        else:
            height = sum(layer_heights[top_layer:])
        return self.surface_pos.y - height
    
@dataclass
class SummonWithEffect(CompoundCommand):
    base_summon: cmd.Summon
    effects: List[tuple[EffectType, Union[int, Seconds, Tick], Optional[int], Optional[bool]]] = field(default_factory=list)
    commands: Commands = field(default_factory=Commands, init=False)

    def __post_init__(self):
        self.commands = Commands()
        # Summonコマンドを追加
        self.add_command(self.base_summon)

        # スポーンしたエンティティを選択するためのターゲットを作成
        spawned_entity_target = Target(
            SelectorType.ALL_ENTITIES, 
            type=str(self.base_summon.entity), 
            limit=1, 
            sort="nearest"
        )

        # 各エフェクトに対してEffectコマンドを作成
        for effect_type, duration, amplifier, hide_particles in self.effects:
            effect_cmd = Effect(
                Effect.Operation.GIVE,
                spawned_entity_target,
                effect_type,
                duration,
                amplifier,
                hide_particles
            )
            # ExecuteコマンドでSummonの直後にEffectを適用
            execute_cmd = Execute(
                As(spawned_entity_target),
                execute=Run(effect_cmd)
            )
            self.add_command(execute_cmd)

    def add_effect(self, effect_type: EffectType, duration: Union[int, Seconds, Tick], 
                   amplifier: Optional[int] = None, hide_particles: Optional[bool] = None):
        """エフェクトを追加するメソッド"""
        self.effects.append((effect_type, duration, amplifier, hide_particles))
        self.__post_init__()  # コマンドを再生成

    def remove_effect(self, effect_type: EffectType):
        """指定したタイプのエフェクトを削除するメソッド"""
        self.effects = [e for e in self.effects if e[0] != effect_type]
        self.__post_init__()  # コマンドを再生成

@dataclass
class MultipleSummonWithEffects(CompoundCommand):
    entity: Union[identifier, str]
    count: int
    pos: Optional[Union[IntPosition, tuple[IntPosition, IntPosition]]] = None
    mask: Optional[np.ndarray] = None
    nbt: Optional[NBTCompound] = None
    effects: List[tuple[EffectType, Union[int, Seconds, Tick], Optional[int], Optional[bool]]] = field(default_factory=list)
    commands: Commands = field(default_factory=Commands, init=False)
    group_tag: str = field(default_factory=lambda: f"multi_summon_{uuid.uuid4().hex[:8]}")

    def __post_init__(self):
        self.commands = Commands()
        
        # NBTにグループタグを追加
        if self.nbt is None:
            self.nbt = NBTCompound()
        if "Tags" not in self.nbt.tags:
            self.nbt.tags.append(NBTTag("Tags", NBTType.LIST, [self.group_tag]))
        else:
            tags = next(tag for tag in self.nbt.tags if tag.name == "Tags")
            if isinstance(tags.value, list):
                tags.value.append(self.group_tag)
            else:
                tags.value = [tags.value, self.group_tag]

        # 複数のSummonコマンドを生成
        for _ in range(self.count):
            spawn_pos = self._get_spawn_position()
            summon_cmd = Summon(self.entity, spawn_pos, self.nbt)
            self.add_command(summon_cmd)

        # エフェクトを適用
        self._apply_effects()

    def _get_spawn_position(self) -> IntPosition:
        if isinstance(self.pos, tuple):  # ランダムな位置
            pos1, pos2 = self.pos
            x = random.randint(min(pos1.x, pos2.x), max(pos1.x, pos2.x))
            y = random.randint(min(pos1.y, pos2.y), max(pos1.y, pos2.y))
            z = random.randint(min(pos1.z, pos2.z), max(pos1.z, pos2.z))
            return IntPosition(x=x, y=y, z=z)
        elif self.mask is not None:  # マスクを使用
            spawn_positions = np.argwhere(self.mask)
            if len(spawn_positions) == 0:
                raise ValueError("No valid spawn positions in the given mask")
            x, y, z = random.choice(spawn_positions)
            base_pos = self.pos or IntPosition(0, 0, 0)
            return IntPosition(x=base_pos.x + x, y=base_pos.y + y, z=base_pos.z + z)
        elif self.pos:  # 固定位置
            return self.pos
        else:
            return IntPosition(0, 0, 0)  # デフォルト位置

    def _apply_effects(self):
        spawned_entities_target = Target(SelectorType.ALL_ENTITIES, tag=self.group_tag)
        for effect_type, duration, amplifier, hide_particles in self.effects:
            effect_cmd = Effect(
                Effect.Operation.GIVE,
                spawned_entities_target,
                effect_type,
                duration,
                amplifier,
                hide_particles
            )
            execute_cmd = Execute(
                As(spawned_entities_target),
                execute=Run(effect_cmd)
            )
            self.add_command(execute_cmd)

    def add_effect(self, effect_type: EffectType, duration: Union[int, Seconds, Tick], 
                   amplifier: Optional[int] = None, hide_particles: Optional[bool] = None):
        self.effects.append((effect_type, duration, amplifier, hide_particles))
        self.__post_init__()
        return self

    def remove_effect(self, effect_type: EffectType):
        self.effects = [e for e in self.effects if e[0] != effect_type]
        self.__post_init__()
        return self

    def set_count(self, new_count: int):
        self.count = new_count
        self.__post_init__()
        return self

    def set_random_position(self, pos1: IntPosition, pos2: IntPosition):
        self.pos = (pos1, pos2)
        self.mask = None
        self.__post_init__()
        return self

    def set_mask(self, mask: np.ndarray, base_pos: Optional[IntPosition] = None):
        self.mask = mask
        self.pos = base_pos
        self.__post_init__()
        return self

@dataclass
class SetCommandBlock(Command):
    pos: IntPosition
    command_block: CommandBlock

    def __command_name__(self):
        return "setblock"

    def __command_str__(self):
        return f"{self.pos.abs} {self.command_block} replace"
    
@dataclass
class AgentCommander(CompoundCommand):
    pos: IntPosition = field(default_factory=lambda: IntPosition(0, 0, 0))
    rot: rotation = field(default_factory=lambda: rotation())
    target: Target = field(default_factory=lambda: Target(SelectorType.NEAREST_PLAYER))
    auto_clear: bool = True
    commands: Commands = field(default_factory=Commands, init=False)

    def __post_init__(self):
        self.memory_condition()
        super().__post_init__()
    
    def __call__(self, mc, **kwargs):
        res = super().__call__(mc, **kwargs)
        if self.auto_clear:
            self.clear()
        return res
    
    def move(self, delta: IntPosition):
        self.pos += delta
        return self

    def rotate(self, new_rot: rotation):
        self.rot = new_rot
        return self

    def add(self, cmd: Union[Command, RelativeSetBlock]):
        if isinstance(cmd, RelativeSetBlock):
            abs_cmd = cmd.build(self.pos, self.rot)
            self.add_command(abs_cmd)
        else:
            exec_chain = ExecuteChain(
                As(self.target),
                Positioned(self.pos),
                cmd
            )
            self.add_command(exec_chain.build())

    def forward(self, dist: int = 1):
        direction = self.rot.numpy
        self.move(IntPosition(*direction * dist))
        return self

    def back(self, dist: int = 1):
        direction = self.rot.numpy
        self.move(IntPosition(*(-direction * dist)))
        return self

    def up(self, dist: int = 1):
        self.move(IntPosition(0, dist, 0))
        return self

    def down(self, dist: int = 1):
        self.move(IntPosition(0, -dist, 0))
        return self
    
    def left(self, dist:int = 1):
        self.turn_left()
        self.forward(dist)
        self.turn_right()
        return self
    
    def right(self, dist:int = 1):
        self.turn_right()
        self.forward(dist)
        self.turn_left()
        return self

    def turn_left(self,count:int=1):
        self.rotate(self.rot - 4*count)
        return self

    def turn_right(self,count:int=1):
        self.rotate(self.rot + 4*count)
        return self

    def set_pos(self, new_pos: IntPosition):
        self.pos = new_pos
        return self
    
    def set_rot(self, new_rot: rotation):
        self.rot = new_rot
        return self
    
    def memory_condition(self):
        self.memory = (self.pos.copy(),rotation(self.rot.value))
        return self
    
    def tp(self, pos: IntPosition):
        return self.set_pos(pos)
    
    def comeback(self):
        return self.set_pos(self.memory[0]).set_rot(self.memory[1])
    
    def delta_pos(self):
        return self.memory[0] - self.pos

    def clear(self):
        self.commands = Commands()
        return self

    def place(self, block: RelativeSetBlockTYPE):
        self.add(RelativeSetBlock(block_direction_pair=block))
        return self
    
    def place_line(self, 
                   blocks: Union[list[Union[RelativeSetBlockTYPE,list[Union[RelativeSetBlockTYPE,list[RelativeSetBlockTYPE]]]]], 
                                 Tuple[Union[Dict[int, Union[RelativeSetBlockTYPE,list[Union[RelativeSetBlockTYPE,list[RelativeSetBlockTYPE]]]]], List[Union[RelativeSetBlockTYPE,list[Union[RelativeSetBlockTYPE,list[RelativeSetBlockTYPE]]]]]], List[int]]],
                   stay: bool = False):
        def block_gen():
            if isinstance(blocks, tuple):
                fb = blocks[0]
                if isinstance(fb, (dict,list)):
                    ib = blocks[1]
                    if isinstance(ib, list):
                        for i in ib:
                            block = fb[i]
                            yield block
                else:
                    raise ValueError("Invalid input format for blocks")
            elif isinstance(blocks, list):
                yield from blocks
            else:
                raise ValueError("Invalid input format for blocks")

        block_iter = block_gen()
        fwd_dist = 0
        for i, block in enumerate(block_iter):
            if isinstance(block,list):
                for bi,bs in enumerate(block):
                    if isinstance(bs,list):
                        if bi > 0:
                            self.forward(1)
                        for b in bs:
                            self.place(b)
                    else:
                        self.place(bs)
            else:
                self.place(block)
            self.forward(1)
            fwd_dist += 1
        if stay:
            self.back(fwd_dist)

        return self