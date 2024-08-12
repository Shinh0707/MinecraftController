from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Union, List
from NBT.nbt import NBTCompound

# 既存のクラスをインポート（実際のインポート文は省略）
from NBT.parameter import angle, rotation, facing, boolean, identifier, block_states, block_predicate


class SelectorType(Enum):
    NEAREST_PLAYER = "@p"
    RANDOM_PLAYER = "@r"
    ALL_PLAYERS = "@a"
    ALL_ENTITIES = "@e"
    EXECUTOR = "@s"


class GameMode(Enum):
    SURVIVAL = "survival"
    CREATIVE = "creative"
    ADVENTURE = "adventure"
    SPECTATOR = "spectator"


class SortMethod(Enum):
    NEAREST = "nearest"
    FURTHEST = "furthest"
    RANDOM = "random"
    ARBITRARY = "arbitrary"


@dataclass
class Position:
    x: float
    y: float
    z: float


@dataclass
class Range:
    min: Optional[float] = None
    max: Optional[float] = None

    def __str__(self):
        if self.min is None and self.max is None:
            return ""
        elif self.min is None:
            return f"..{self.max}"
        elif self.max is None:
            return f"{self.min}.."
        else:
            return f"{self.min}..{self.max}"


@dataclass
class Target:
    selector: SelectorType
    position: Optional[Position] = None
    distance: Optional[Range] = None
    volume: Optional[tuple[float, float, float]] = None
    scores: Dict[str, Range] = field(default_factory=dict)
    tag: Optional[str] = None
    team: Optional[str] = None
    limit: Optional[int] = None
    sort: Optional[SortMethod] = None
    level: Optional[Range] = None
    gamemode: Optional[GameMode] = None
    name: Optional[str] = None
    x_rotation: Optional[angle] = None
    y_rotation: Optional[angle] = None
    type: Optional[identifier] = None
    nbt: Optional[NBTCompound] = None
    advancements: Dict[str, Union[boolean, Dict[str, boolean]]] = field(
        default_factory=dict)
    predicate: Optional[str] = None

    def __str__(self):
        args = []
        if self.position:
            args.extend(
                [
                    f"x={self.position.x}",
                    f"y={self.position.y}",
                    f"z={self.position.z}"
                ]
            )
        if self.distance:
            args.append(f"distance={self.distance}")
        if self.volume:
            args.extend(
                [
                    f"dx={self.volume[0]}",
                    f"dy={self.volume[1]}",
                    f"dz={self.volume[2]}"
                ]
            )
        if self.scores:
            score_str = ",".join(f"{k}={v}" for k, v in self.scores.items())
            args.append(f"scores={{{score_str}}}")
        if self.tag:
            args.append(f"tag={self.tag}")
        if self.team:
            args.append(f"team={self.team}")
        if self.limit:
            args.append(f"limit={self.limit}")
        if self.sort:
            args.append(f"sort={self.sort.value}")
        if self.level:
            args.append(f"level={self.level}")
        if self.gamemode:
            args.append(f"gamemode={self.gamemode.value}")
        if self.name:
            args.append(f"name={self.name}")
        if self.x_rotation:
            args.append(f"x_rotation={self.x_rotation}")
        if self.y_rotation:
            args.append(f"y_rotation={self.y_rotation}")
        if self.type:
            args.append(f"type={self.type}")
        if self.nbt:
            args.append(f"nbt={self.nbt}")
        if self.advancements:
            adv_str = ",".join(
                f"{k}={v}" for k, v in self.advancements.items())
            args.append(f"advancements={{{adv_str}}}")
        if self.predicate:
            args.append(f"predicate={self.predicate}")

        if args:
            return f"{self.selector.value}[{','.join(args)}]"
        else:
            return self.selector.value
