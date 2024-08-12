from typing import Any, Dict, Optional, Union
from enum import Enum, auto
from dataclasses import dataclass, field

from NBT.parameter import boolean, identifier

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
    id: identifier
    block_state: Optional[BlockState] = None

    def __str__(self):
        if self.block_state:
            return "{}{}".format(self.id,self.block_state)
        return f"{self.id}"
    
class Instrument(Enum):
    # (ノートナンバーの開始番号, 下に置くブロックのid, 対応するmidiのinstrument(推定))
    BELL = (78, identifier(name="gold_block"), [112,113])
    CHIME = (78, identifier(name="packed_ice"), [14,15])
    XYLOPHONE = (78, identifier(name="bone_block"), [12,13])
    COW_BELL = (66, identifier(name="soul_sand"), [56])
    FLUTE = (66, identifier(name="crey"), list(range(64,80)))
    HARP = (54, identifier(name="dirt"), list(range(9))+list(range(16,24))+[46,108])
    BIT = (54, identifier(name="emerald_block"), [80,81])
    IRON_XYLOPHONE = (54, identifier(name="iron_block"), [8,9,11])
    PLING = (54, identifier(name="glow_stone"), list(range(82,96)))
    BANJO = (54, identifier(name="hay_block"), list(range(104,107)))
    GUITAR = (42, identifier(name="white_wool"), list(range(32,40))+[120])
    BASS = (30, identifier(name="oak_planks"), list(range(32,40))+[43])
    DIDGERIDOO = (30, identifier(name="pumpkin"),  list(range(104, 107)))
    BASEDRUM = (0, identifier(name="stone"), list(range(114,119)))
    HAT = (0, identifier(name="glass_block"), [98])
    SNARE = (0, identifier(name="sand"), [119])

    @staticmethod
    def note_to_instruments(note: int):
        """
        Return list[tuple[Instrument,int]]: 音ブロックの情報と操作回数を返す
        """
        preds = []
        for n in Instrument:
            if 0 <= note - n.value[0] <= 24:
                preds.append((n, round(note - n.value[0])))
        return preds

@dataclass
class NoteBlockState(BlockState):
    note: int = 0
    powered: boolean = field(default_factory=boolean(False))
    instrument: Instrument = Instrument.HARP

    def __str__(self):
        return "["+','.join([f"note={self.note}",f"powered={self.powered}",f"instrument={self.instrument.name.lower()}"])+"]"

@dataclass
class NoteBlock(Block):
    id: identifier = field(default_factory=identifier(
        name="note_block"), init=False)
    block_state : NoteBlockState
