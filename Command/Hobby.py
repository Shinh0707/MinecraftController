from collections import defaultdict
from itertools import repeat
import math
import os
from pathlib import Path
import time
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

import mido
from tqdm import tqdm
from Command.Command import Command, Commands, Fill, SetBlock, SET_BLOCK_MODE, SetDirection, Tp
from Command.CompoundCommand import AgentCommander, CompoundCommand, AdjustFill, SetBlocks
from Helper.helpers import adaptive_optimize_midi_timing, optimize_midi_timing, optimize_midi_timing_dynamic, optimize_midi_timing_fast,group_notes, optimize_midi_timing_uniform, pixelate_image
from NBT.parameter import IntPosition, boolean, rotation
from NBT.selector import SelectorType, Target
from NBT.block_nbt import Block, CommandBlock, CommandBlockState, CommandBlockTag, Instrument, NoteBlock, NoteBlockState, RedstoneRepeater, Slab, SlabState, SlabType
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
    

def time_grouping(time_stamps: List[float], is_percussion: List[bool], min_interval: float) -> List[Tuple[int, List[int]]]:
    if not time_stamps:
        return []
    result = []
    ts = np.array(sorted(time_stamps))
    pc = np.array([x[0] for x in sorted(zip(is_percussion,time_stamps),key=lambda x: x[1])])
    # あえて細かくとって、あとで結合させる
    grid = np.arange(ts[0],ts[-1],step=min_interval/4)
    distances = np.abs(ts[:, np.newaxis] - grid[np.newaxis, :])
    valid_dists = (distances < min_interval)
    # For non-percussion
    non_perc_mask = ~pc[:, np.newaxis] & valid_dists
    non_perc_dists = distances[non_perc_mask]
    median_dist_non_perc = np.median([np.min(row) for row in non_perc_dists if row.size > 0]) if non_perc_dists.size > 0 else np.inf

    # For percussion
    perc_mask = pc[:, np.newaxis] & valid_dists
    perc_dists = distances[perc_mask]
    median_dist_perc = np.min([np.mean(row) for row in perc_dists if row.size > 0]) if perc_dists.size > 0 else np.inf

    # Create the groups
    groups = (~pc[:, np.newaxis] & (distances < median_dist_non_perc)) | (pc[:, np.newaxis] & (distances < median_dist_perc))
    last_t = 0.0
    # ２つをマージして１つのグループにする。マージする必要のないものはそのまま
    # グリッドの間の中間辺りにある音は２つに複製して、その各グリッド上にあるとして扱う
    queue = []
    for g,t in enumerate(grid):
        check = groups[:,g]
        if np.any(check):
            group_stamps = np.argwhere(check).reshape(-1)
            queue.extend(group_stamps.tolist())
        time_delay = (t-last_t)/min_interval
        delay = int(time_delay)
        if delay >= 1 and queue:
            result.append((delay,queue))
            queue = []
            #queue.extend(group_stamps[(ts[group_stamps]-last_t)/min_interval - delay >= min_interval/2].tolist())
            last_t = t
    if queue:
        result.append((delay,queue))
    return result

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

def process_track(track, ticks_per_beat: int) -> Tuple[List[Tuple[int, int, int, float, bool, float]],float]:
    notes = []
    current_tempo = mido.bpm2tempo(BASE_BPM, BASE_TIME_SIGNATURE)
    time_signature = BASE_TIME_SIGNATURE
    channel_volume = 127
    expression = 127
    current_instrument = 0
    current_time = 0
    additional_secounds = 0.0
    pitch_bend = 0  # ピッチベンドの初期値（0 = 変更なし）
    queues = []
    for msg in track:
        current_time += msg.time
        if msg.is_cc(7):
            channel_volume = msg.value
        elif msg.is_cc(11):
            expression = msg.value
        elif msg.type == 'program_change':
            current_instrument = msg.program
        elif msg.type == 'set_tempo':
            additional_secounds += mido.tick2second(current_time, ticks_per_beat, current_tempo)
            current_time = 0.0
            current_tempo = msg.tempo
            print(f"Tempo changed => {current_tempo}, {mido.tempo2bpm(current_tempo,time_signature)}")
        elif msg.type == 'time_signature':
            time_signature = (msg.numerator, msg.denominator)
        elif msg.type == 'pitchwheel':
            pitch_bend = msg.pitch
        elif msg.type == 'note_on' or msg.type == 'note_off':
            is_percussion = msg.channel == 9  # MIDI ではチャンネル 10 (0-indexed で 9) がパーカッション
            current_sec = additional_secounds + mido.tick2second(current_time, ticks_per_beat, current_tempo)
            if is_percussion:
                note = msg.note  # パーカッションの場合、ノート番号をそのまま使用
            else:
                note = adjust_note_to_range(msg.note+round(pitch_bend / 8192 * 2), *NOTE_BLOCK_RANGE)
            
            if msg.velocity > 0 and (msg.type == 'note_on'):
                queues.append([note, current_sec, current_instrument, channel_volume*expression*msg.velocity/(127*127), is_percussion])
            else:
                dels = []
                for i,q in enumerate(queues):
                    if q[0] == note and not (q[4] ^ is_percussion):
                        notes.append(q+[current_sec])
                        dels.append(i)
                queues = [q for i,q in enumerate(queues) if not i in dels]
        elif msg.is_cc(123) or msg.is_cc(120):
            current_sec = additional_secounds + mido.tick2second(current_time, ticks_per_beat, current_tempo)
            for q in queues:
                notes.append(q+[current_sec])
            queues.clear()
    current_sec = additional_secounds + mido.tick2second(current_time, ticks_per_beat, current_tempo)
    for q in queues:
        notes.append(q+[current_sec])
    queues.clear()
    return notes,current_sec

def extract_notes_for_note_block(midi_file_path: str, speed_modifier: float = 1.0) -> List[Tuple[int, int, int, float, bool]]:
    if speed_modifier == 0:
        return []
    midi_file = mido.MidiFile(midi_file_path)
    end_time = 0.0
    all_notes = []

    for i, track in enumerate(midi_file.tracks):
        track_notes, track_end_time = process_track(track, midi_file.ticks_per_beat)
        all_notes.extend(track_notes)
        end_time = max(end_time, track_end_time)
    if len(all_notes) == 0:
        return []
    min_velocity = max(50,min([x[3] for x in all_notes]))
    # ノートをタイムスタンプでソート
    all_notes.sort(key=lambda x: np.sign(speed_modifier)*x[1])
    time_stamps = [n[1] for n in all_notes] if speed_modifier > 0 else [end_time-n[1] for n in all_notes]
    speed_modifier = abs(speed_modifier)
    time_stamps = [t/speed_modifier for t in time_stamps]
    data = time_grouping(time_stamps, [n[4] for n in all_notes],0.1) # (delay(int),notes(list))のリスト
    result = []
    next_del = 0
    auto_decay = 10
    decay_end = float(min_velocity)
    decay_start = 127.0
    decay_rate = (decay_end/decay_start)**(1.0/auto_decay) if auto_decay > 0 else 0.0
    decaies = []
    for (d, ns),(next_d, next_n) in zip(data,data[1:]+[(0,[0])]):
        dly = d-next_del
        current_decaies = []
        for n in ns:
            result.append((dly,n))
            if all_notes[n][3] > min_velocity:
                for _ in range(int(math.ceil(all_notes[n][3]/min_velocity) - 1)):
                    result.append((0,n))
            dly = 0
            """
            note_decay = round((all_notes[n][5] - all_notes[n][1])*10/speed_modifier) - 1
            if note_decay > 0:
                current_decaies.append((0, note_decay, n))
                print(f"assign decay: {n}, {note_decay}")
            """
        """
        if next_d > 1 and len(decaies) > 0:
            for ndi in range(next_d-1):
                ddels = []
                dly = 1
                decaies = [(dc+1,mdc,dn) for dc,mdc,dn in decaies]
                for di,dc in enumerate(decaies):
                    dcn = dc[2]
                    if int(all_notes[dcn][3]*(decay_rate**dc[0])) >= min_velocity:
                        result.append((dly, dcn))
                        dly = 0
                        if dc[0] > dc[1]:
                            ddels.append(di)
                    else:
                        ddels.append(di)
                decaies = [dc for di,dc in enumerate(decaies) if not di in ddels]
                if len(decaies) == 0:
                    next_del = ndi
                    break
        else:
            next_del = 0
        if current_decaies:
            decaies.extend(current_decaies)
        """
    """
    while len(decaies) > 0:
        ddels = []
        dly = 1
        decaies = [(dc+1,mdc,dn) for dc,mdc,dn in decaies]
        for di,dc in enumerate(decaies):
            dcn = dc[2]
            if int(all_notes[dcn][3]*(decay_rate**dc[0])) >= min_velocity:
                result.append((dly, dcn))
                dly = 0
                if dc[0] > dc[1]:
                    ddels.append(di)
            else:
                ddels.append(di)
        decaies = [dc for di,dc in enumerate(decaies) if not di in ddels]
    """
    final_notes = [(all_notes[n][0],delay,all_notes[n][2],all_notes[n][3],all_notes[n][4]) for delay,n in result]
    print(f"Duration: {end_time}")
    return final_notes

NoteBlockInstruction = Tuple[MBlocks, int, Instrument, float]
GroupedNoteBlockInstructions = List[Tuple[int, List[NoteBlockInstruction]]]

def convert_midi_to_grouped_noteblocks(midi_file_path: str,speed_modifier: float=1.0) -> GroupedNoteBlockInstructions:
    """
    Converts a MIDI file to Minecraft note block instructions, grouping simultaneous notes.

    :param midi_file_path: Path to the MIDI file
    :return: List of tuples (delay in Redstone ticks, List of (block to place under note block, number of times to operate the note block))
    """
    notes = extract_notes_for_note_block(midi_file_path,speed_modifier)
    grouped_instructions: GroupedNoteBlockInstructions = []
    current_group: List[NoteBlockInstruction] = []
    last_delay = 0
    
    # Keep track of the last used instrument for each MIDI instrument
    last_used_instrument: Dict[int, Instrument] = defaultdict(lambda: None)

    for note, delay, midi_instrument, volume, is_percussion in notes:
        
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
        else:
            # If no suitable instrument is found, use a default (e.g., HARP with stone block)
            chosen_instrument = Instrument.HARP
            operation_count = note - Instrument.HARP.value[0]
        # Get the block to place under the note block
        block_under = chosen_instrument.value[1]
        if delay > 0 and current_group:
            # If there's a delay and we have a current group, add it to the grouped instructions
            grouped_instructions.append((last_delay, current_group))
            current_group = []
            last_delay = delay
        current_group.append((block_under, operation_count, chosen_instrument, volume))
        
    # Add the last group if it exists
    if current_group:
        grouped_instructions.append((last_delay, current_group))

    return grouped_instructions

def playmidisound(mc, pos: IntPosition, midi_file_path: str,speed_modifier: float=1.0):
    sound_groups = convert_midi_to_grouped_noteblocks(midi_file_path, speed_modifier)
    last_run_time = time.perf_counter()
    for delay, notes in sound_groups:
        cmds = Commands()
        for _, oc, inst, vol in notes[:30]:
            # Rconの中身が0.03秒のsleepを入れているため
            cmds.append(inst.playsound_command(oc, Target(SelectorType.NEAREST_PLAYER), pos, vol/127))
        delay_time = delay * 0.1
        now_time = time.perf_counter()
        if delay_time > (now_time-last_run_time):
            time.sleep(delay_time-(now_time-last_run_time))
        last_run_time = time.perf_counter()
        cmds(mc)
        

@dataclass
class MusicBoxCreator(CompoundCommand):
    midi_file_path: str
    start_pos: IntPosition
    start_rotation: rotation
    execute_as: Target = field(default_factory=lambda :Target(SelectorType.NEAREST_PLAYER))
    speed_modifier: float = 1.0
    commands: Commands = field(default_factory=Commands, init=False)

    def __post_init__(self):
        self.agent = AgentCommander(self.start_pos,self.start_rotation.to_cardinal(),self.execute_as)
        self.sound_groups = convert_midi_to_grouped_noteblocks(self.midi_file_path, self.speed_modifier)
        self.midi_name = Path(self.midi_file_path).stem

    def set_sounds(self, total_delay: int, sounds: List[NoteBlockInstruction], is_middle: bool, mc):
        sounds = sounds[:80] # 上限80音
        static_len = 4
        sound_count = len(sounds)
        full_delays = total_delay//4
        remain_delay = total_delay%4
        last_delay = 0
        if remain_delay > 0:
            last_delay = remain_delay
            remain_delay = 0
        else:
            full_delays -= 1
            last_delay = 4
        pre_length = max(static_len, full_delays+(1 if remain_delay > 0 else 0))
        full_delays = min(static_len, full_delays)
        layers = int(np.ceil(sound_count/9))
        B_count = layers//2
        A_count = layers-B_count
        LH = B_count*3
        MH = (A_count-B_count)*3+LH
        self.agent.fill(
            SetDirection(forward=1,right=-3),
            SetDirection(forward=pre_length+4,right=3,up=MH),
            MBlocks.air
        )(mc)
        self.agent.fill(
            SetDirection(right=-1),
            SetDirection(forward=pre_length+3,right=1),
            MBlocks.stone
        )(mc)
        if is_middle:
            self.agent.forward(5)
            tp_pos = self.agent.pos.copy()+IntPosition(0,MH+10,0)
            self.agent.back(5)
            self.agent.place(CommandBlock(CommandBlockState(boolean(False)),CommandBlockTag(Command=Tp(Target(SelectorType.NEAREST_PLAYER),tp_pos)),CommandBlock.CommandBlockType.IMPULSE))
        else:
            self.agent.place(MBlocks.stone)
        self.agent.up().place(MBlocks.redstone_wire)
        self.agent.forward().place(MBlocks.redstone_wire)
        for _ in range(pre_length-static_len):
            self.agent.forward().place(RedstoneRepeater.create(delay=4))
        self.agent.back()
        pre_length = static_len
        self.agent.fill(
            SetDirection(forward=2,right=-1),
            SetDirection(forward=7,right=1,up=LH),
            MBlocks.stone
        )(mc)
        self.agent.fill(
            SetDirection(forward=2,right=-1),
            SetDirection(forward=7,right=1,up=LH),
            MBlocks.stone
        )(mc)
        self.agent.fill(
            SetDirection(forward=5,right=-1),
            SetDirection(forward=5,right=1,up=LH),
            MBlocks.glass
        )(mc)
        l = -1
        while l < MH:
            self.agent.fill(
                SetDirection(forward=4,right=-1,up=l),
                SetDirection(forward=4,right=1,up=l),
                MBlocks.glass
            )(mc)
            l += 5
        def generate_sequence(n):
            x = 0
            step = 1
            while x <= n:
                yield x, 6 if x % 3 == 0 else 3
                x += step
                step = 3 - step  # 1と2を交互に

        n = LH
        is_first = True
        self.agent.memory_condition()
        for h,f in generate_sequence(n):
            v = ((f//3)-1)*2-1
            self.agent.fill(
                SetDirection(forward=f,right=-1,up=1+h),
                SetDirection(forward=f+v,right=1,up=2+h),
                MBlocks.air
            )(mc)
            self.agent.fill(
                SetDirection(forward=f,right=-1,up=1+h),
                SetDirection(forward=f,right=1,up=1+h),
                MBlocks.redstone_wire
            )(mc)
            self.agent.fill(
                SetDirection(forward=f,up=1+h),
                SetDirection(forward=f+v,up=1+h),
                MBlocks.redstone_wire
            )(mc)
            self.agent.up(h).forward(f).turn_right(1-v)
            for er in [1,2,-1]:
                self.agent.forward(2)
                if is_first:
                    self.agent.back().up()
                    for lm in [1,-2]:
                        self.agent.left(lm).place(MBlocks.stone).up().place(MBlocks.redstone_wire)
                        self.agent.back().place(MBlocks.stone)
                        self.agent.down(2).forward(2)
                        if len(sounds) > 0:
                            s,oc,_, _ = sounds.pop()
                            self.agent.place(s).place((NoteBlock(NoteBlockState(note=oc)),SetDirection(up=1)))
                        self.agent.back().up()
                    self.agent.left().down().forward()
                    is_first = False
                else:
                    for r in [-1,2,-1]:
                        if len(sounds) > 0:
                            s,oc,_, _ = sounds.pop()
                            self.agent.place(s).place((NoteBlock(NoteBlockState(note=oc)),SetDirection(up=1)))
                            self.agent.right(r)
                        else:
                            break
                self.agent.back(2).turn_right(er)
            self.agent.comeback()
        h = 0
        while h < MH:
            f = 5-h % 2
            self.agent.fill(
                SetDirection(forward=f,up=h),
                SetDirection(forward=f,up=h),
                MBlocks.redstone_wire
            )(mc)
            h += 1
        self.agent.forward()
        for i in range(pre_length-1):
            self.agent.forward()
            if full_delays+i >= pre_length-1:
                self.agent.place(RedstoneRepeater.create(delay=4))
            else:
                self.agent.place(MBlocks.redstone_wire)
        self.agent.forward().place(RedstoneRepeater.create(delay=last_delay))
        self.agent.forward(3).down(1)
        self.agent(mc)

    def create_music_box(self, mc, clip: int|None=None):
        self.agent.forward(2)
        max_length = 5
        next_col_len = 6
        self.agent.fill(
            SetDirection(forward=-3, right=-3),
            SetDirection(forward=4, up=3),
            MBlocks.air
        )(mc)
        self.agent.place(
            CommandBlock(
                CommandBlockState(boolean(False)),
                CommandBlockTag(
                    Command='title @p title {"text":"{}","color":"gold"}'.format(self.midi_name)
                ),
                CommandBlock.CommandBlockType.IMPULSE
            )
        ).place((MBlocks.heavy_weighted_pressure_plate, SetDirection(up=1)))
        self.agent.forward().place(MBlocks.red_wool).place((RedstoneRepeater.create(delay=4),SetDirection(up=1)))
        self.agent.back(2)
        start_pos = self.agent.pos.copy()
        self.agent.forward(3)
        if not clip:
            clip = len(self.sound_groups)
        else:
            clip = min(len(self.sound_groups),clip)
        for i in tqdm(range(clip)):
            total_delay, sounds = self.sound_groups[i]
            self.set_sounds(total_delay, sounds, (i+1) % max_length == np.ceil(max_length/2), mc)
            if (i+1) % max_length == 0:
                Tp(Target(SelectorType.NEAREST_PLAYER), self.agent.pos+IntPosition(0,10,0))(mc)
                turn = 1 - 2 * ((i // max_length) % 2)
                self.agent.up().place(MBlocks.stone).up().place(MBlocks.redstone_wire).forward().down()
                self.agent.fill(
                    SetDirection(right=-turn),
                    SetDirection(forward=2, right=turn * (next_col_len+1), up=2),
                    MBlocks.air
                )(mc)
                self.agent.turn_right(turn)
                self.agent.fill(
                    SetDirection(),
                    SetDirection(forward=next_col_len),
                    MBlocks.stone
                )(mc)
                self.agent.fill(
                    SetDirection(up=1),
                    SetDirection(forward=next_col_len, up=1),
                    MBlocks.redstone_wire
                )(mc)
                self.agent.forward(next_col_len)
                self.agent.turn_right(turn).down()
            self.agent(mc)
        self.agent.forward(3).place(
            CommandBlock(
                CommandBlockState(boolean(False)),
                CommandBlockTag(
                    Command=Tp(Target(SelectorType.NEAREST_PLAYER), start_pos)
                ),
                CommandBlock.CommandBlockType.IMPULSE
            )
        )
        self.agent.up().place(MBlocks.heavy_weighted_pressure_plate)
        self.agent(mc)
        Tp(Target(SelectorType.NEAREST_PLAYER), start_pos)(mc)

    def __call__(self, mc, clip: int|None=None, **kwargs):
        self.create_music_box(mc, clip, **kwargs)
        return super().__call__(mc, **kwargs)