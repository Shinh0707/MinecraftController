from __future__ import absolute_import, annotations
import random
import re
import sys
from typing import List, Literal, Tuple, Optional, Union, Dict, Any
from abc import ABC, abstractmethod
from enum import Enum, auto
import os
import uuid
from PIL import Image
import numpy as np
from mcrcon import MCRcon
from scipy.spatial import KDTree

from Command import Command
from Command.CompoundCommand import AgentCommander, BaseSettings, GetPosition, GetRotation, SuperFlat
from Command.Hobby import ImageCreator,convert_midi_to_grouped_noteblocks
from NBT.MBlocks import MBlocks
from NBT.block_nbt import Block, BlockState, CommandBlock, CommandBlockTag, NoteBlock, NoteBlockState, RedstoneRepeater, Slab, SlabState, SlabType
from NBT.item_nbt import Compass, CompassState, CompassTag, MItem
from NBT.mob_nbt import ActiveEffects, Attribute, Attributes, CommonItemTags, CommonMobTags, HandItems, StatusEffect
from NBT.nbt import NBTTag, NBTType, NBTCompound, UUID
from NBT.parameter import INFINITE, EffectType, Rotation, Seconds, angle, attribute_name, dimension, rotation, facing, boolean, identifier, IntPosition
from NBT.selector import Difficulty, Target, SelectorType, GameMode, SortMethod, Range
from MShape.MShape import MShape, Cube, Sphere, Pyramid, Plane, Cylinder

class MinecraftController:
    def __init__(self, server_properties_path: str):
        self.server_properties_path = server_properties_path
        self.rcon = None
        self.host = None
        self.host_port = None
        self.port = None
        self.password = None
        self.BLOCK_COLORS = None
        self.print_command_result = False
        self._read_server_properties()

    def _read_server_properties(self):
        if os.path.exists(self.server_properties_path):
            with open(self.server_properties_path) as f:
                props = f.readlines()
            for prop in props:
                prop = prop.strip()
                if prop.startswith('server-ip='):
                    self.host = prop[10:] or 'localhost'
                elif prop.startswith('server-port='):
                    self.host_port = int(prop[12:])
                elif prop.startswith('rcon.port='):
                    self.port = int(prop[10:])
                elif prop.startswith('rcon.password='):
                    self.password = prop[14:]

    def __enter__(self):
        self.rcon = MCRcon(self.host, self.password, port=self.port)
        self.rcon.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.rcon:
            self.rcon.disconnect()

    def execute(self, command: Command):
        return self.rcon.command(str(command))

    def execute_commands(self, commands: Command.Commands):
        return [self.execute(cmd) for cmd in commands]


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    from Dungeon.DungeonMaker import Analyzer
    from NBT.MBlocks import MBlocks
    from NBT.parameter import IntPosition, identifier
    from NBT.block_nbt import Block
    from MShape.MShape import Cube, Plane

    with MinecraftController("MinecraftServer/server.properties") as mc:
        image_size = 100

        mc.print_command_result = True
        bs_cmd = BaseSettings(
            target=Target(SelectorType.NEAREST_PLAYER),
            game_mode=GameMode.CREATIVE,
            difficulty=Difficulty.PEACEFUL,
            clear_items=True
        )
        print(bs_cmd(mc))

        get_pos_cmd = GetPosition(target=Target(SelectorType.NEAREST_PLAYER))
        # プレイヤーの現在位置を取得
        player_pos = get_pos_cmd(mc)
        print(player_pos)
        get_rot_cmd = GetRotation(Target(SelectorType.NEAREST_PLAYER))
        player_rot = get_rot_cmd(mc)
        print(player_rot)

        sf_cmd = SuperFlat(
            player_pos,
            IntPosition(image_size,0,image_size),
            layers=[
                (image_size, Block(MBlocks.air)),
                (1, Block(MBlocks.grass_block)),
                (2, Block(MBlocks.dirt)),
                (1, Block(MBlocks.bedrock))
            ],
            start_point=SuperFlat.LayerStartPoint(1,SuperFlat.LayerReference.TOP)
        )
        sf_cmd(mc)
        """
        imc_cmd = ImageCreator(
            image_path="ch_img_miku.png",
            start_pos=player_pos,
            width=60,
            _rotation=(0,0,-90)
        )
        print(imc_cmd(mc))
        """

        effect_cmd = Command.Effect(
            Command.Effect.Operation.GIVE,
            targets=Target(SelectorType.NEAREST_PLAYER),
            effect=EffectType.ABSORPTION,
            duration=Seconds(20),
            amplifier=2,
            hide_particles=False
        )
        effect_cmd(mc)
        cube = Cube(initial_position=player_pos+IntPosition(0,10,0),size=6, block=MBlocks.stone)
        cube.place()(mc)
        cube.rotate_axis((0,1,0),45)
        cube.rotate_axis((0,0,1),45)
        cube.translate(IntPosition(10,0,0))
        cube.place()(mc)

        agent = AgentCommander(
            player_pos,
            player_rot.to_cardinal(),
            Target(SelectorType.NEAREST_PLAYER)
        )
        agent.forward(1).place(MBlocks.stone_pressure_plate)
        agent.forward(1).place(MBlocks.redstone_wire)
        agent.forward(1).place((MBlocks.gold_block,Command.SetDirection(up=-1))).place(MBlocks.note_block)
        agent.forward(1).place(RedstoneRepeater.create(delay=2))
        print(agent(mc))
        agent.forward()
        agent.forward()
        
        def set_sounds(total_delay:int, sounds: list[tuple[MBlocks,int]]):

            def set_sound_blocks(delay:int,last_skip:bool=False):
                if len(sounds) == 0:
                    return False
                agent.forward()
                agent.place(MBlocks.stone).place((RedstoneRepeater.create(delay=delay),Command.SetDirection(up=1)))
                agent.forward()
                s, t = sounds.pop()
                agent.place(s).place((NoteBlock(NoteBlockState(note=t)),Command.SetDirection(up=1)))
                agent.left()
                if len(sounds) == 0:
                    return False
                s, t = sounds.pop()
                agent.place(s).place((NoteBlock(NoteBlockState(note=t)),Command.SetDirection(up=1)))
                agent.right(2)
                if not last_skip:
                    if len(sounds) == 0:
                        return False
                    s, t = sounds.pop()
                    agent.place(s).place((NoteBlock(NoteBlockState(note=t)),Command.SetDirection(up=1)))
                agent.left().back(2)
                return True

            def setting_sound_blocks(delay:int,last_skip:bool=False):
                if set_sound_blocks(delay):
                    agent.turn_left()
                    if set_sound_blocks(delay):
                        agent.turn_right(2)
                        if set_sound_blocks(delay,last_skip):
                            agent.turn_left()
                            return True
                return False

            def set_additional_sound_blocks(delay:int):
                if len(sounds) == 0:
                    return False
                agent.up()
                slab = Slab(block_state=SlabState(SlabType.TOP))
                agent.up().place(MBlocks.redstone_wire)
                agent.up().place(slab).down()
                agent.turn_left().forward()
                agent.place(slab).place((MBlocks.redstone_wire,Command.SetDirection(up=1)))
                if setting_sound_blocks(delay,True):
                    agent.back().turn_left(2).forward()
                    agent.place(slab).place((MBlocks.redstone_wire,Command.SetDirection(up=1)))
                    if setting_sound_blocks(delay,True):
                        agent.back().turn_left()
                        return True
                return False
            
            start_len = 3
            if total_delay == 0:
                total_delay = 20
            if total_delay > 4:
                for _ in range(total_delay//4):
                    agent.place(MBlocks.stone).place((RedstoneRepeater.create(delay=4),Command.SetDirection(up=1)))
                    agent.forward()
                    start_len -= 1
                if total_delay % 4 == 0:
                    delay = 4
                else:
                    delay = total_delay % 4
            else:
                delay = total_delay
            
            print(delay)
            if start_len > 0:
                for _ in range(start_len):
                    agent.place(MBlocks.stone).place((MBlocks.redstone_wire,Command.SetDirection(up=1)))
                    agent.forward()
            agent.up()
            agent.place(MBlocks.stone)
            agent.down().memory_condition()
            success = setting_sound_blocks(delay)
            while success:
                success = set_additional_sound_blocks(delay)
            agent.comeback()
            agent.forward(3)
        
        """
        
        # テスト用
        sound_groups = [
            [random.randint(1,20), random.choices([(MBlocks.emerald_block,10),
                (MBlocks.stone,3),
                (MBlocks.diamond_block,23)],k=random.randint(1,40))] for _ in range(20)]
        """
        max_length = 10
        next_col_len = 9
        sound_groups = convert_midi_to_grouped_noteblocks("Liszt_lacampanella.mid",100,1/0.7)
        cols = np.ceil(len(sound_groups) / max_length)
        col_length = (cols - 1) * (next_col_len-1)
        print(f"col length: {col_length}")
        agent.right(col_length//2)
        agent.memory_condition()
        start_pos = agent.pos
        for c in range(int(cols//2)+2):
            Command.Tp(Target(SelectorType.NEAREST_PLAYER),agent.pos)(mc)
            sf_cmd = SuperFlat(
                agent.pos,
                IntPosition(max_length*10,0,max_length*10),
                layers=[
                    (20, Block(MBlocks.air)),
                    (1, Block(MBlocks.grass_block)),
                    (2, Block(MBlocks.dirt)),
                    (1, Block(MBlocks.bedrock))
                ],
                start_point=SuperFlat.LayerStartPoint(1,SuperFlat.LayerReference.TOP)
            )
            sf_cmd(mc)
            agent.left(next_col_len*2)
        agent.comeback()
        do_place = True
        if do_place:
            for i, (total_delay, sounds) in enumerate(sound_groups,1):
                set_sounds(total_delay, sounds)
                if i % max_length == 0:
                    Command.Tp(Target(SelectorType.NEAREST_PLAYER),agent.pos)(mc)
                    if (i // max_length) % 2 == 0:
                        agent.turn_right()
                    else:
                        agent.turn_left()
                    for _ in range(next_col_len-1):
                        agent.place(MBlocks.stone).place((MBlocks.redstone_wire,Command.SetDirection(up=1)))
                        agent.forward()
                    if (i // max_length) % 2 == 0:
                        agent.turn_right()
                    else:
                        agent.turn_left()
                    agent.place(MBlocks.stone).place((MBlocks.redstone_wire,Command.SetDirection(up=1)))
                    agent.forward()
                #print(agent.commands)
                agent(mc)
        Command.Tp(Target(SelectorType.NEAREST_PLAYER),start_pos)(mc)
        """
        
        monster_id = "magma_cube"
        # プレイヤーの位置から少し離れた場所に迷路を生成
        maze_start_pos = IntPosition(
            x=player_pos.x + image_size//2 + 10, y=player_pos.y, z=player_pos.z)

        # 迷路の生成
        maze, labels, start_goal_candidates = Analyzer.create_maze(
            (30, 30), 50)

        # 迷路の3D配列を作成
        maze_block_map = [
            MBlocks.air,
            MBlocks.bedrock
        ]
        m_maze = np.tile(np.expand_dims(np.pad(np.array(maze, dtype=np.int64), ((
            1, 1), (1, 1)), 'constant', constant_values=1), 1), (1, 3, 1))
        m_maze_floor = np.ones_like(m_maze)[:,[0],:]
        m_maze = np.concatenate([m_maze_floor,m_maze],axis=1)

        # 迷路の配置
        mc.setblocks(maze_start_pos-IntPosition(0,1,0), m_maze, maze_block_map)

        # スタートとゴールの設定
        region = random.choice(list(start_goal_candidates.keys()))
        start_pos = random.choice(start_goal_candidates[region][0])
        goal_pos = random.choice(start_goal_candidates[region][1])

        # ゴールの設置
        goal_m_pos = IntPosition(x=maze_start_pos.x +
                        goal_pos[0] + 1, y=maze_start_pos.y, z=maze_start_pos.z + goal_pos[1] + 1)
        mc.give(
            Target(selector=SelectorType.NEAREST_PLAYER),
            Compass(item_state=CompassState(
                lodestone_tracker=CompassTag(
                    LodestoneTracked=boolean(True),
                    LodestoneDimension=dimension.OVERWORLD,
                    LodestonePos=goal_m_pos
                )))
        )
        mc.setblock(
            MBlocks.lodestone,
            goal_m_pos
        )
        mc.setblock(
            MBlocks.heavy_weighted_pressure_plate,
            goal_m_pos + IntPosition(0,1,0)
        )
        mc.setblock(
            CommandBlock(
                tags=CommandBlockTag(Command=mc.setblock(
                        MBlocks.redstone_block,
                        goal_m_pos + IntPosition(0,-2,0),
                        execute=False
                    ), auto=boolean(_value=False))
            ),
            goal_m_pos + IntPosition(0,-1,0)
        )
        mc.setblock(
            CommandBlock(
                tags=CommandBlockTag(Command="gamemode creative @p", auto=boolean(_value=False))
            ),
            goal_m_pos + IntPosition(1,-2,0)
        )
        mc.setblock(
            CommandBlock(
                tags=CommandBlockTag(Command=mc.setblock(
                        MBlocks.air,
                        goal_m_pos + IntPosition(0,-2,0),
                        execute=False
                    ), auto=boolean(_value=False))
            ),
            goal_m_pos + IntPosition(2,-2,0)
        )

        # プレイヤーをスタート地点にテレポート
        mc.tp(
            "@p",
            IntPosition(x=maze_start_pos.x +
                        start_pos[0] + 1, y=maze_start_pos.y, z=maze_start_pos.z + start_pos[1] + 1)
        )

        print(f"Maze generated at {maze_start_pos}")
        print(f"Player teleported to start position: {start_pos}")
        print(f"Goal position: {goal_pos}")
        mc.command("difficulty normal")

        # 敵の生成
        region_mask = labels==region
        walkable_positions = np.argwhere(region_mask)
        distances = np.sum(np.abs(walkable_positions-start_pos),axis=-1) > 2
        if np.sum(distances) > 0:
            walkable_positions= walkable_positions[distances]
            monster_count = 0*np.sum(distances) // 12 + 1  # 迷路の通行可能領域サイズの1/12+1体

            monster_nbt = CommonMobTags(
                Health=1000000.0,  # 実質的に無限のHP
                attributes=Attributes(Attributes=[
                    Attribute(Name=attribute_name.MAX_HEALTH,Base=1000000.0),
                    Attribute(Name=attribute_name.FOLLOW_RANGE,Base=2.0),
                    Attribute(Name=attribute_name.MOVEMENT_SPEED,Base=0.3),
                    Attribute(Name=attribute_name.ATTACK_SPEED,Base=0.05),
                    Attribute(Name=attribute_name.ATTACK_DAMAGE,Base=8.0),
                    Attribute(Name=attribute_name.STEP_HEIGHT,Base=0.0),
                    Attribute(Name=attribute_name.SCALE,Base=1.5)
                ]),
                NoAI=boolean(False),
                PersistenceRequired=boolean(True),
                tags=[
                    NBTTag("Size",NBTType.INT,0)
                ]
            )
            monster_effect = StatusEffect(id="mining_fatigue",ambient=boolean(False),amplifier=5,duration=INFINITE,show_icon=boolean(False),show_particles=boolean(True))
            
            for x,z in random.choices(walkable_positions,k=monster_count):
                monster_pos = IntPosition(
                    x=maze_start_pos.x + x + 1,
                    y=maze_start_pos.y,
                    z=maze_start_pos.z + z + 1
                )
                mc.spawn(monster_id, monster_pos, nbt=monster_nbt, effects=[monster_effect.to_command()])

            print(f"Spawned {monster_count} {monster_id} in the maze")
        mc.command(f"spawnpoint @p {player_pos.abs}")    
        mc.command("gamemode survival @p")
        """
