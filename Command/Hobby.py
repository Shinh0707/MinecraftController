from collections import defaultdict
import time
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

import mido
from Command.Command import Command, Commands, Fill, SetBlock, SET_BLOCK_MODE
from Command.CompoundCommand import AgentCommander, CompoundCommand, AdjustFill, SetBlocks
from Helper.helpers import pixelate_image
from NBT.parameter import IntPosition, rotation
from NBT.selector import Target
from NBT.block_nbt import Block, Instrument, NoteBlock, NoteBlockState, RedstoneRepeater
from NBT.MBlocks import MBlocks
from scipy.spatial import KDTree
import numpy as np
from PIL import Image

class BlockColorManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BlockColorManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.block_colors, self.color_array, self.block_array, self.color_tree = self._init_block_colors()

    @staticmethod
    @lru_cache(maxsize=1)
    def _init_block_colors():
        block_colors = {}
        for block in MBlocks:
            if 'color' in block.value and block.value.get('is_block', False):
                block_colors[block] = tuple(block.value['color'])
        
        color_array = np.array(list(block_colors.values()))
        block_array = np.array(list(block_colors.keys()))
        color_tree = KDTree(color_array)

        return block_colors, color_array, block_array, color_tree

    def find_closest_block(self, color: Tuple[int, int, int, int]) -> MBlocks:
        if len(color) > 3 and color[3] == 0:  # Fully transparent
            return MBlocks.air
        _, index = self.color_tree.query(color)
        return self.block_array[index]

@dataclass
class ImageCreator(CompoundCommand):
    image_path: str
    start_pos: IntPosition
    width: Optional[int] = None
    fill_mode: SET_BLOCK_MODE = 'replace'
    color_level: int = 150
    commands: Commands = field(default_factory=Commands, init=False)
    _blocks_3d: np.ndarray = field(default=None, init=False)
    block_mapping: List[MBlocks] = field(default_factory=list, init=False)
    _rotation: Tuple[int, int, int] = field(default=(0, 0, 0))

    def __post_init__(self):
        if self._blocks_3d is None:
            self._create_image_blocks()

    def _create_image_blocks(self):
        block_color_manager = BlockColorManager()
        img_array = np.array(pixelate_image(self.image_path, self.width, self.color_level))

        blocks = np.empty(img_array.shape[:2], dtype=object)
        self.block_mapping = []
        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                color = tuple(img_array[y, x])
                block = block_color_manager.find_closest_block(color)
                if block in self.block_mapping:
                    blocks[y, x] = self.block_mapping.index(block)
                else:
                    blocks[y, x] = len(self.block_mapping)
                    self.block_mapping.append(block)

        # Convert 2D array to 3D
        self._blocks_3d = np.expand_dims(blocks, axis=1)

    def _rotate_3d(self, blocks: np.ndarray, rotation_x: int, rotation_y: int, rotation_z: int) -> np.ndarray:
        def rotate_2d(points, angle):
            angle = np.deg2rad(angle)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            return np.dot(points, rotation_matrix.T)

        # Get non-zero indices
        indices = np.array(np.nonzero(blocks)).T
        center = np.mean(indices, axis=0)

        # Rotate around X-axis
        yz_coords = indices[:, [1, 2]] - center[[1, 2]]
        rotated_yz = rotate_2d(yz_coords, rotation_x)
        indices[:, [1, 2]] = np.round(rotated_yz + center[[1, 2]]).astype(int)

        # Rotate around Y-axis
        xz_coords = indices[:, [0, 2]] - center[[0, 2]]
        rotated_xz = rotate_2d(xz_coords, rotation_y)
        indices[:, [0, 2]] = np.round(rotated_xz + center[[0, 2]]).astype(int)

        # Rotate around Z-axis
        xy_coords = indices[:, [0, 1]] - center[[0, 1]]
        rotated_xy = rotate_2d(xy_coords, rotation_z)
        indices[:, [0, 1]] = np.round(rotated_xy + center[[0, 1]]).astype(int)

        # Calculate new dimensions
        min_coords = np.min(indices, axis=0)
        max_coords = np.max(indices, axis=0)
        new_shape = max_coords - min_coords + 1

        # Shift indices to ensure all are non-negative
        indices -= min_coords

        # Create new array with rotated dimensions
        rotated_blocks = np.zeros(new_shape, dtype=blocks.dtype)

        # Place blocks in new positions
        for old_idx, new_idx in zip(np.array(np.nonzero(blocks)).T, indices):
            if np.all(new_idx >= 0) and np.all(new_idx < new_shape):
                rotated_blocks[tuple(new_idx)] = blocks[tuple(old_idx)]

        return rotated_blocks

    def rotate(self, rotation_x: int = 0, rotation_y: int = 0, rotation_z: int = 0):
        self._rotation = (
            (self._rotation[0] + rotation_x) % 360,
            (self._rotation[1] + rotation_y) % 360,
            (self._rotation[2] + rotation_z) % 360
        )
        return self

    def _get_rotated_blocks(self) -> np.ndarray:
        return self._rotate_3d(self._blocks_3d, *self._rotation)

    def place_blocks(self, do_prefill:bool=True, append:bool=False):
        rotated_blocks = self._get_rotated_blocks()
        # Use SetBlocks to efficiently place all blocks
        set_blocks = SetBlocks(
            start_pos=self.start_pos,
            blocks_array=rotated_blocks,
            blocks_mapping=self.block_mapping,
            mode=self.fill_mode,
            do_prefill=do_prefill
        )

        # Overwrite existing commands with new SetBlocks command
        if append:
            self.add_command(set_blocks)
        else:
            self.commands = Commands(set_blocks)

    def rotation_animation(self, mc, delta_rotation: tuple[float,float,float],count: int,interval: float):
        if interval > 0:
            for _ in range(count):
                self.rotate(*delta_rotation)
                self.place_blocks(False)
                super().__call__(mc)
                time.sleep(interval)
        else:
            for _ in range(count):
                self.rotate(*delta_rotation)
                self.place_blocks(False, True)
                print(len(self.commands))
            super().__call__(mc)

    def __call__(self, mc, **kwargs):
        self.place_blocks()
        return super().__call__(mc, **kwargs)
    
BASE_TICK_PER_BEAT = 480
BASE_BPM = 120
BASE_VELOCITY = 80
BASE_TIME_SIGNATURE = (4,4)
BASE_TEMPO = mido.bpm2tempo(BASE_BPM,BASE_TIME_SIGNATURE)
NOTE_BLOCK_RANGE = [30,102]

def adjust_note_to_range(note: int, range_min: int, range_max: int) -> int:
    if range_min <= note <= range_max:
        return note
    return ((note - range_min) % 12) + range_min

def extract_notes_for_note_block(midi_file_path: str, base_bpm: int=BASE_BPM, speed_modifier: float=1.0) -> List[Tuple[int, int, int]]:
    """
    ノートオンメッセージを抽出するメソッド

    :param midi_file_path: MIDIファイルのパス
    :return: (ノート番号(30~102(Minecraftのノートブロックで表現可能な範囲)に変換済み), 前の音からのRedStoneTick, インストゥルメント番号)のリスト
    """
    midi_file = mido.MidiFile(midi_file_path)
    
    notes = []
    current_tempo = mido.bpm2tempo(base_bpm,BASE_TIME_SIGNATURE)  # デフォルトのテンポ (120 BPM)
    time_signature = BASE_TIME_SIGNATURE  # Default time signature
    current_instrument = 0
    current_time = 0
    note_range = [255,0]
    #print(f"Tick per Beat: {midi_file.ticks_per_beat}")
    
    for msg in midi_file.merged_track:
        current_time += msg.time
        if msg.type == 'pitchwheel':
            pass
            #print(f"ピッチホイールの値: {msg.pitch}")
        elif msg.type == 'program_change':
            current_instrument = msg.program
        elif msg.type == 'set_tempo':
            current_tempo = msg.tempo
            #current_tempo = mido.bpm2tempo(mido.tempo2bpm(msg.tempo,time_signature),time_signature)
            #print(f"  New tempo: {mido.tempo2bpm(current_tempo,time_signature):.2f} BPM")
        elif msg.type == 'time_signature':
            #old_signature = time_signature
            time_signature = (msg.numerator, msg.denominator)
            #print(f"  Old time signature: {old_signature[0]}/{old_signature[1]}")
            print(f"  New time signature: {time_signature[0]}/{time_signature[1]}")
            #current_tempo = mido.bpm2tempo(mido.tempo2bpm(current_tempo,time_signature),time_signature)
        elif msg.type == 'note_on' and msg.velocity > 0:
            if note_range[0] > msg.note:
                note_range[0] = msg.note
            elif note_range[1] < msg.note:
                note_range[1] = msg.note
            #(base_time_signature[1]*time_signature[0]/(base_time_signature[0]*time_signature[1]))
            sec = mido.tick2second(current_time*speed_modifier, midi_file.ticks_per_beat, current_tempo)
            #print((msg.note,msg.velocity,msg.time, sec, current_instrument))
            note = adjust_note_to_range(msg.note,*NOTE_BLOCK_RANGE)
            notes.append((note, max(0,int(sec/0.1)), current_instrument))
            current_time = 0
    #print("Note Range:",note_range)
    return notes

NoteBlockInstruction = Tuple[MBlocks, int]
GroupedNoteBlockInstructions = List[Tuple[int, List[NoteBlockInstruction]]]

def convert_midi_to_grouped_noteblocks(midi_file_path: str, base_bpm: int=BASE_BPM,speed_modifier: float=1.0) -> GroupedNoteBlockInstructions:
    """
    Converts a MIDI file to Minecraft note block instructions, grouping simultaneous notes.

    :param midi_file_path: Path to the MIDI file
    :return: List of tuples (delay in Redstone ticks, List of (block to place under note block, number of times to operate the note block))
    """
    notes = extract_notes_for_note_block(midi_file_path,base_bpm,speed_modifier)
    grouped_instructions: GroupedNoteBlockInstructions = []
    current_group: List[NoteBlockInstruction] = []
    last_delay = 0
    
    # Keep track of the last used instrument for each MIDI instrument
    last_used_instrument: Dict[int, Instrument] = defaultdict(lambda: None)

    for note, delay, midi_instrument in notes:
        # Get all possible Minecraft instruments for this note
        minecraft_instruments = Instrument.note_to_instruments(note)
        
        if minecraft_instruments:
            # First, try to use the same instrument as the last note with this MIDI instrument
            if last_used_instrument[midi_instrument] in [instr for instr, _ in minecraft_instruments]:
                chosen_instrument = last_used_instrument[midi_instrument]
                operation_count = next(count for instr, count in minecraft_instruments if instr == chosen_instrument)
            else:
                # If we can't use the same instrument, choose the first available
                chosen_instrument, operation_count = minecraft_instruments[0]
            
            # Update the last used instrument for this MIDI instrument
            last_used_instrument[midi_instrument] = chosen_instrument
            
            # Get the block to place under the note block
            block_under = chosen_instrument.value[1]
        else:
            # If no suitable instrument is found, use a default (e.g., HARP with stone block)
            block_under = MBlocks.stone
            operation_count = note - Instrument.HARP.value[0]
        
        if delay > 0 and current_group:
            # If there's a delay and we have a current group, add it to the grouped instructions
            grouped_instructions.append((last_delay, current_group))
            current_group = []
            last_delay = delay
        
        current_group.append((block_under, operation_count))

    # Add the last group if it exists
    if current_group:
        grouped_instructions.append((last_delay, current_group))

    return grouped_instructions
