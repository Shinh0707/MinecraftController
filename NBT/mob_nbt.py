from __future__ import annotations
from typing import Any, Dict, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from NBT.nbt import NBTCompound, NBTList, NBTTag, NBTType, UUID
from NBT.parameter import Seconds, Tick, block_predicate, attribute_name, boolean, dimension, IntPosition, identifier, rotation

@dataclass
class StatusEffect(NBTCompound):
    from Command.Command import EffectParameters
    effect_params: EffectParameters = field(default=None)
    hidden_effect: Optional[StatusEffect] = field(default=None, compare=False)

    def __post_init__(self):
        assert not self.effect_params
        tags = [
            NBTTag("id", NBTType.STRING, self.effect_params.effect_type.minecraft_id),
            boolean(self.effect_params.hide_particles).to_nbt_tag("ambient"),
            NBTTag("amplifier", NBTType.INT, self.effect_params.amplifier),
            NBTTag("duration", NBTType.INT, self.effect_params.duration.value if isinstance(self.effect_params.duration, (Seconds, Tick)) else self.effect_params.duration),
            boolean(True).to_nbt_tag("show_icon"),
            boolean(not self.effect_params.hide_particles).to_nbt_tag("show_particles")
        ]
        if self.hidden_effect:
            tags.append(self.hidden_effect)
        self.value = tags

    def to_command(self):
        return str(self.effect_params)

@dataclass
class ActiveEffects(NBTList):
    from Command.Command import EffectParameters
    effects: list[EffectParameters] = field(default_factory=list)

    def add_effect(self, effect: EffectParameters, replace: bool = True):
        if effect.effect_type in [e.effect_type for e in self.effects]:
            if replace:
                self.effects = [e for e in self.effects if e.effect_type != effect.effect_type] + [effect]
        else:
            self.effects.append(effect)

    def add_effects(self, *effects: EffectParameters, replace: bool = True):
        for effect in effects:
            self.add_effect(effect, replace)

    def __post_init__(self):
        self.value = [StatusEffect(effect) for effect in self.effects]


@dataclass
class ArmorDropChances(NBTList):
    feet: float = 0.0
    legs: float = 0.0
    chest: float = 0.0
    head: float = 0.0

    def __post_init__(self):
        self.value = [
            NBTTag('', NBTType.FLOAT, self.feet),
            NBTTag('', NBTType.FLOAT, self.legs),
            NBTTag('', NBTType.FLOAT, self.chest),
            NBTTag('', NBTType.FLOAT, self.head)
        ]


@dataclass
class CommonItemTags(NBTCompound):
    Count: int = 1
    Slot: Optional[int] = None
    id: Optional[str] = None
    tag: Optional[NBTCompound] = None

    def __post_init__(self):
        tags = [
            NBTTag("Count", NBTType.BYTE, self.Count)
        ]
        if self.Slot:
            tags.append(NBTTag("Slot", NBTType.STRING, self.id))
        if self.id:
            tags.append(NBTTag("id", NBTType.STRING, self.id))
        if self.tag:
            tags.append(NBTTag("tag", NBTType.COMPOUND, self.tag))
        self.value = tags


@dataclass
class DurableItemCommonTags(NBTCompound):
    Damage: int = 0
    Unbreakable: bool = False
    CanDestroy: Optional[list[block_predicate]] = None
    CustomModelData: Optional[int] = None

    @property
    def value(self):
        pass


@dataclass
class ArmorItems(NBTList):
    feet: Optional[CommonItemTags] = None
    legs: Optional[CommonItemTags] = None
    chest: Optional[CommonItemTags] = None
    head: Optional[CommonItemTags] = None

    def __post_init__(self):
        tags = []
        if self.feet:
            tags.append(self.feet)
        else:
            tags.append(NBTCompound())
        if self.legs:
            tags.append(self.legs)
        else:
            tags.append(NBTCompound())
        if self.chest:
            tags.append(self.chest)
        else:
            tags.append(NBTCompound())
        if self.head:
            tags.append(self.head)
        else:
            tags.append(NBTCompound())
        self.value = tags

class EquipSlot(Enum):
    MAINHAND = "mainhand"
    OFFHAND = "offhand"
    FEET = "feet"
    LEGS = "legs"
    CHEST = "chest"
    HEAD = "head"

    def to_nbt_tag(self):
        return NBTTag("Slot", NBTType.STRING, self.value)


class AttributeModifierOperation(Enum):
    ADD = 0
    MULTIPLY_BASE = 1
    MULTIPLY = 2

    def to_nbt_tag(self):
        return NBTTag("Operation", NBTType.INT, self.value)

@dataclass
class AttributeModifier(NBTCompound):
    AttributeName: attribute_name = attribute_name.MAX_HEALTH
    Name: str = field(default="")
    Slot: EquipSlot = EquipSlot.MAINHAND
    Operation: AttributeModifierOperation = AttributeModifierOperation.ADD
    Amount: float = 0.0
    UUID: UUID = field(default_factory=UUID.random(False))

    def __post_init__(self):
        tags = [
            self.AttributeName.to_nbt_tag(),
            NBTTag("Name", NBTType.STRING, self.Name),
            self.Slot.to_nbt_tag(),
            self.Operation.to_nbt_tag(),
            NBTTag("Amount", NBTType.DOUBLE, self.Amount),
            self.UUID
        ]
        self.value = tags


@dataclass
class Attribute(NBTCompound):
    name: str = field(default="",init=False)
    tags: list[NBTTag] = field(default=list,init=False)
    Name: attribute_name = field(default=attribute_name.MAX_HEALTH)
    Base: float = 0.0
    Modifiers: Optional[list[AttributeModifier]] = None

    def __post_init__(self):
        tags = [
            self.Name.to_nbt_tag("id"),
            NBTTag("base", NBTType.DOUBLE, self.Base)
        ]
        if self.Modifiers:
            modifiers = []
            for mod in self.Modifiers:
                mod.AttributeName = self.Name
                modifiers.append(mod)
            tags.append(NBTTag("Modifiers", NBTType.LIST, modifiers))
        self.value = tags

@dataclass
class DimensionPosition(NBTCompound):
    dimension: dimension = dimension.OVERWORLD
    pos: IntPosition = field(default_factory=IntPosition(0,0,0))

    def __post_init__(self):
        tags = [
            self.dimension.to_nbt_tag(),
            self.pos
        ]
        self.value = tags

@dataclass
class BrainMemory(NBTCompound):
    _value: Union[int,list[int],NBTCompound,bool] = 0
    ttl: Optional[int] = None

    def __post_init__(self):
        tags = [
            NBTTag.create_nbt_tag("value", self._value)
        ]
        if self.ttl:
            tags.append(NBTTag("ttl", NBTType.LONG, self.ttl))
        self.tags = tags
        self.value = tags
    
    def __call__(self):
        return self

class Memories(Enum):
    ADMIRING_DISABLED = BrainMemory(_value=False, ttl=0)
    ADMIRING_ITEM = BrainMemory(_value=False, ttl=0)
    ANGRY_AT = BrainMemory(_value=UUID.random(True), ttl=0)
    DIG_COOLDOWN = BrainMemory(_value=NBTCompound(), ttl=1200)
    GAZE_COOLDOWN_TICKS = BrainMemory(_value=0)
    GOLEN_DITECTED_RECENTLY = BrainMemory(_value=False, ttl=0)
    HAS_HUNTING_COOLDOWN = BrainMemory(_value=False, ttl=0)
    HOME = BrainMemory(_value=DimensionPosition(dimension=dimension.OVERWORLD,pos=IntPosition(x=0,y=0,z=0)))
    IS_EMERGING = BrainMemory(_value=NBTCompound())
    IS_IN_WATER = BrainMemory(_value=NBTCompound())
    IS_PREGNANT = BrainMemory(_value=NBTCompound())
    IS_SNIFFING = BrainMemory(_value=NBTCompound())
    IS_TEMPTED = BrainMemory(_value=False)
    # 他のも同様に実装

    def to_nbt_tag(self):
        memory = self.value
        memory.name = str(identifier(name=self.name.lower()))
        return memory

@dataclass
class Brain(NBTCompound):
    memories: Memories = field(default=None)

    def __post_init__(self):
        self.value = self.memories

@dataclass
class Attributes(NBTList):
    Attributes: list[Attribute] = field(default=None)

    def __post_init__(self):
        self.value = self.Attributes


@dataclass
class HandDropChances(NBTList):
    main_hand: float = 0.085
    off_hand: float = 0.085

    def __post_init__(self):
        self.value = [
            NBTTag('', NBTType.FLOAT, self.main_hand),
            NBTTag('', NBTType.FLOAT, self.off_hand)
        ]


@dataclass
class HandItems(NBTList):
    main_hand: Optional[CommonItemTags] = None
    off_hand: Optional[CommonItemTags] = None

    def __post_init__(self):
        items = []
        if self.main_hand:
            items.append(self.main_hand)
        else:
            items.append(NBTCompound())
        if self.off_hand:
            items.append(self.off_hand)
        else:
            items.append(NBTCompound())
        self.value = items


@dataclass
class LeashUUID(NBTCompound):
    UUID: UUID = field(default_factory=UUID.random(True))

    def __post_init__(self):
        self.value = [self.UUID]


@dataclass
class LeashPosition(NBTCompound):
    X: int = 0
    Y: int = 0
    Z: int = 0

    def __post_init__(self):
        self.value = [
            NBTTag('X', NBTType.INT, self.X),
            NBTTag('Y', NBTType.INT, self.Y),
            NBTTag('Z', NBTType.INT, self.Z)
        ]


@dataclass
class Leash(NBTCompound):
    data: Union[LeashUUID, LeashPosition] = field(default=None)

    def __post_init__(self):
        if self.data:
            self.value = self.data.value
        else:
            self.value = None

"""
@dataclass
class CommonEntityTags(NBTCompound):
    id: Optional[identifier] = None
    Pos: Optional[IntPosition] = None
    Motion: Optional[tuple[float,float,float]] = None
    Rotation: Optional[tuple[float,float]] = None
    FallDistance: Optional[float] = None
    Fire: Optional[float] = None
    Air: Optional[int] = None
"""


@dataclass
class CommonMobTags(NBTCompound):
    AbsorptionAmount: Optional[float] = None
    ActiveEffects: Optional[ActiveEffects] = None
    ArmorDropChances: Optional[ArmorDropChances] = None
    ArmorItems: Optional[ArmorItems] = None
    attributes: Optional[Attributes] = None
    Brain: Optional[Brain] = None
    CanPickUpLoot: Optional[boolean] = None
    DeathLootTable: Optional[identifier] = None
    DeathLootTableSeed: Optional[int] = None
    DeathTime: Optional[int] = None
    FallFlying: Optional[boolean] = None
    Health: Optional[float] = None
    HurtByTimestamp: Optional[int] = None
    HurtTime: Optional[int] = None
    HandDropChances: Optional[HandDropChances] = None
    HandItems: Optional[HandItems] = None
    Leash: Optional[Leash] = None
    LeftHanded: Optional[boolean] = None
    NoAI: Optional[boolean] = None
    PersistenceRequired: Optional[boolean] = None
    SleepingX: Optional[int] = None
    SleepingY: Optional[int] = None
    SleepingZ: Optional[int] = None
    Team: Optional[str] = None
    tags: Optional[list[NBTTag]] = field(default_factory=list)

    def __post_init__(self):
        for field, value in self.__dict__.items():
            if value is not None:
                if field == 'AbsorptionAmount':
                    self.tags.append(NBTTag(field, NBTType.FLOAT, value))
                elif field in ['ActiveEffects', 'ArmorDropChances', 'ArmorItems', 'attributes', 'Brain', 'HandDropChances', 'HandItems', 'Leash']:
                    self.tags.append(NBTTag(field, NBTType.COMPOUND, value))
                elif field in ['CanPickUpLoot', 'FallFlying', 'LeftHanded', 'NoAI', 'PersistenceRequired']:
                    assert isinstance(value,boolean)
                    self.tags.append(value)
                elif field == 'DeathLootTable':
                    assert isinstance(value,identifier)
                    self.tags.append(value.to_nbt_tag(field))
                elif field in ['DeathLootTableSeed', 'DeathTime', 'HurtByTimestamp', 'HurtTime', 'SleepingX', 'SleepingY', 'SleepingZ']:
                    self.tags.append(NBTTag(field, NBTType.INT, value))
                elif field == 'Health':
                    self.tags.append(NBTTag(field, NBTType.FLOAT, value))
                elif field == 'Team':
                    self.tags.append(NBTTag(field, NBTType.STRING, value))
        self.value = self.tags
