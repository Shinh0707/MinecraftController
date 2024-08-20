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

def process_track(track, ticks_per_beat: int, base_bpm: int) -> Tuple[List[Tuple[int, int, int, float, bool]],float]:
    notes = []
    current_tempo = mido.bpm2tempo(base_bpm, BASE_TIME_SIGNATURE)
    time_signature = BASE_TIME_SIGNATURE
    channel_volume = 127
    expression = 127
    current_instrument = 0
    current_time = 0
    end_time = 0
    pitch_bend = 0  # ピッチベンドの初期値（0 = 変更なし）

    for msg in track:
        current_time += msg.time
        end_time += mido.tick2second(msg.time, ticks_per_beat, current_tempo)
        if msg.is_cc(7):
            channel_volume = msg.value
        elif msg.is_cc(11):
            expression = msg.value
        elif msg.type == 'program_change':
            current_instrument = msg.program
        elif msg.type == 'set_tempo':
            current_tempo = msg.tempo
        elif msg.type == 'time_signature':
            time_signature = (msg.numerator, msg.denominator)
        elif msg.type == 'pitchwheel':
            pitch_bend = msg.pitch
        elif msg.type == 'note_on' and msg.velocity > 0:
            sec = mido.tick2second(current_time, ticks_per_beat, current_tempo)
            is_percussion = msg.channel == 9  # MIDI ではチャンネル 10 (0-indexed で 9) がパーカッション
            
            if is_percussion:
                note = msg.note  # パーカッションの場合、ノート番号をそのまま使用
            else:
                note = adjust_note_to_range(msg.note+round(pitch_bend / 8192 * 2), *NOTE_BLOCK_RANGE)
            
            notes.append((note, sec, current_instrument, channel_volume*expression*msg.velocity/(127*127), is_percussion))

    return notes,end_time

def extract_notes_for_note_block(midi_file_path: str, base_bpm: int = BASE_BPM, speed_modifier: float = 1.0) -> List[Tuple[int, int, int, bool, int]]:
    if speed_modifier == 0:
        return []
    midi_file = mido.MidiFile(midi_file_path)
    end_time = 0.0
    all_notes = []

    for i, track in enumerate(midi_file.tracks):
        track_notes, track_end_time = process_track(track, midi_file.ticks_per_beat, base_bpm)
        all_notes.extend(track_notes)
        end_time = max(end_time, track_end_time)
    if len(all_notes) == 0:
        return []
    # ノートをタイムスタンプでソート
    all_notes.sort(key=lambda x: np.sign(speed_modifier)*x[1])
    min_velocity = max(64,min([x[3] for x in all_notes]))
    scale = 10.0/speed_modifier
    # 重複を除去し、タイムスタンプを相対時間に変換
    final_notes = []
    queues = []
    last_time = 0.0
    for note, time, instrument, velocity, is_percussion in all_notes:
        if time < last_time:
            queues.append((note, 0, instrument, is_percussion))
        else:
            rtick = max(0, round((time - last_time)*scale))
            if rtick > 0:
                if len(queues) > 0:
                    final_notes.extend(queues)
                last_time = time
                queues = [(note, rtick, instrument, is_percussion)]
            else:
                queues.append((note, 0, instrument, is_percussion))
        if velocity > min_velocity:
            for _ in range(int(np.ceil(velocity/min_velocity)-1)):
                queues.append((note, 0, instrument, is_percussion))
    if len(queues) > 0:
        final_notes.extend(queues)
    print(f"Duration: {end_time}")
    return final_notes

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

    for note, delay, midi_instrument, is_percussion in notes:
        # Get all possible Minecraft instruments for this note
        minecraft_instruments = Instrument.note_to_instruments(note, is_percussion)
        
        if minecraft_instruments:
            # First, try to use the same instrument as the last note with this MIDI instrument
            if last_used_instrument[midi_instrument] in [instr for instr, _ in minecraft_instruments]:
                chosen_instrument = last_used_instrument[midi_instrument]
                operation_count = next(count for instr, count in minecraft_instruments if instr == chosen_instrument)
            else:
                if len(minecraft_instruments) > 1:
                    if any([(midi_instrument in instr.value[-2]) for instr, _ in minecraft_instruments]):
                        minecraft_instruments = [(instr,oc) for instr, oc in minecraft_instruments if (midi_instrument in instr.value[-2])]
                    # If we can't use the same instrument, choose the first available
                    minecraft_instruments.sort(key=lambda x: (abs(x[1]-12), x[0].value[-1]))
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
