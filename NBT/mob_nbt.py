from __future__ import annotations
from typing import Any, Dict, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from nbt import NBTCompound, NBTList, NBTTag, NBTType, UUID
from parameter import block_predicate, attribute_name, boolean, dimension, IntPosition, identifier

@dataclass
class StatusEffect(NBTCompound):
    id: str = field(default="")
    ambient: bool = field(default=False, compare=False)
    amplifier: int = field(default=0, compare=False)
    duration: int = field(default=0, compare=False)
    show_icon: bool = field(default=True, compare=False)
    show_particles: bool = field(default=True, compare=False)
    hidden_effect: Optional[StatusEffect] = field(default=None, compare=False)

    @property
    def value(self):
        tags = [
            NBTTag("id", NBTType.STRING, self.id),
            NBTTag("ambient", NBTType.BYTE, self.ambient),
            NBTTag("amplifier", NBTType.BYTE, self.amplifier),
            NBTTag("duration", NBTType.INT, self.duration),
            NBTTag("show_icon", NBTType.BYTE, self.show_icon),
            NBTTag("show_particles", NBTType.BYTE, self.show_particles)
        ]
        if self.hidden_effect:
            tags.append(self.hidden_effect)
        return tags


@dataclass
class ActiveEffects(NBTList):
    effects: list[StatusEffect] = []

    def add_effect(self, effect: StatusEffect, replace: bool = True):
        if effect in self.effects:
            if replace:
                self.effects[self.effects.index(effect)] = effect
        else:
            self.effects.append(effect)

    def add_effects(self, *effects: StatusEffect, replace: bool = True):
        for effect in effects:
            self.add_effect(effect, replace)

    @property
    def value(self):
        return self.effects


@dataclass
class ArmorDropChances(NBTList):
    feet: float = 0.0
    legs: float = 0.0
    chest: float = 0.0
    head: float = 0.0

    @property
    def value(self):
        return [
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

    @property
    def value(self):
        tags = [
            NBTTag("Count", NBTType.BYTE, self.Count)
        ]
        if self.Slot:
            tags.append(NBTTag("Slot", NBTType.STRING, self.id))
        if self.id:
            tags.append(NBTTag("id", NBTType.STRING, self.id))
        if self.tag:
            tags.append(NBTTag("tag", NBTType.COMPOUND, self.tag))
        return tags


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

    @property
    def value(self):
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
        return tags

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
    UUID: UUID = field(default=UUID.random(False))

    @property
    def value(self):
        tags = [
            self.AttributeName.to_nbt_tag(),
            NBTTag("Name", NBTType.STRING, self.Name),
            self.Slot.to_nbt_tag(),
            self.Operation.to_nbt_tag(),
            NBTTag("Amount", NBTType.DOUBLE, self.Amount),
            self.UUID
        ]
        return tags


@dataclass
class Attribute(NBTCompound):
    Base: float = 0.0
    Name: attribute_name = field(default=attribute_name.MAX_HEALTH)
    Modifiers: Optional[list[AttributeModifier]] = None

    @property
    def value(self):
        tags = [
            NBTTag("Base", NBTType.DOUBLE, self.Base),
            self.Name.to_nbt_tag("Name")
        ]
        if self.Modifiers:
            modifiers = []
            for mod in self.Modifiers:
                mod.AttributeName = self.Name
                modifiers.append(mod)
            tags.append(NBTTag("Modifiers", NBTType.LIST, modifiers))
        return tags

@dataclass
class DimensionPosition(NBTCompound):
    dimension: dimension
    pos: IntPosition

    @property
    def value(self):
        tags = [
            self.dimension.to_nbt_tag(),
            self.pos
        ]
        return tags

@dataclass
class BrainMemory(NBTCompound):
    _value: Union[int,list[int],NBTCompound,bool]
    ttl: Optional[int] = None

    @property
    def value(self):
        tags = [
            NBTTag.create_nbt_tag("value", self._value)
        ]
        if self.ttl:
            tags.append(NBTTag("ttl", NBTType.LONG, self.ttl))
        return tags

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
    memories: Memories

    @property
    def value(self):
        return self.memories

@dataclass
class Attributes(NBTList):
    Attributes: list[Attribute]

    @property
    def value(self):
        return self.Attributes


@dataclass
class HandDropChances(NBTList):
    main_hand: float = 0.085
    off_hand: float = 0.085

    @property
    def value(self):
        return [
            NBTTag('', NBTType.FLOAT, self.main_hand),
            NBTTag('', NBTType.FLOAT, self.off_hand)
        ]


@dataclass
class HandItems(NBTList):
    main_hand: Optional[CommonItemTags] = None
    off_hand: Optional[CommonItemTags] = None

    @property
    def value(self):
        items = []
        if self.main_hand:
            items.append(self.main_hand)
        else:
            items.append(NBTCompound())
        if self.off_hand:
            items.append(self.off_hand)
        else:
            items.append(NBTCompound())
        return items


@dataclass
class LeashUUID(NBTCompound):
    UUID: UUID

    @property
    def value(self):
        return [self.UUID]


@dataclass
class LeashPosition(NBTCompound):
    X: int
    Y: int
    Z: int

    @property
    def value(self):
        return [
            NBTTag('X', NBTType.INT, self.X),
            NBTTag('Y', NBTType.INT, self.Y),
            NBTTag('Z', NBTType.INT, self.Z)
        ]


@dataclass
class Leash(NBTCompound):
    data: Union[LeashUUID, LeashPosition]

    @property
    def value(self):
        return self.data.value


@dataclass
class CommonMobTags(NBTCompound):
    AbsorptionAmount: Optional[float] = None
    ActiveEffects: Optional[ActiveEffects] = None
    ArmorDropChances: Optional[ArmorDropChances] = None
    ArmorItems: Optional[ArmorItems] = None
    Attributes: Optional[Attributes] = None
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

    @property
    def value(self):
        tags = []
        for field, value in self.__dict__.items():
            if value is not None:
                if field == 'AbsorptionAmount':
                    tags.append(NBTTag(field, NBTType.FLOAT, value))
                elif field in ['ActiveEffects', 'ArmorDropChances', 'ArmorItems', 'Attributes', 'Brain', 'HandDropChances', 'HandItems', 'Leash']:
                    tags.append(NBTTag(field, NBTType.COMPOUND, value))
                elif field in ['CanPickUpLoot', 'FallFlying', 'LeftHanded', 'NoAI', 'PersistenceRequired']:
                    tags.append(NBTTag(field, NBTType.BYTE, int(value)))
                elif field == 'DeathLootTable':
                    assert isinstance(value,identifier)
                    tags.append(value.to_nbt_tag(field))
                elif field in ['DeathLootTableSeed', 'DeathTime', 'HurtByTimestamp', 'HurtTime', 'SleepingX', 'SleepingY', 'SleepingZ']:
                    tags.append(NBTTag(field, NBTType.INT, value))
                elif field == 'Health':
                    tags.append(NBTTag(field, NBTType.FLOAT, value))
                elif field == 'Team':
                    tags.append(NBTTag(field, NBTType.STRING, value))

        return tags
