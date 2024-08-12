from __future__ import absolute_import, annotations
import random
import re
import sys
from typing import List, Tuple, Optional, Union, Dict, Any
from abc import ABC, abstractmethod
from enum import Enum, auto
import os
import uuid
from PIL import Image
import numpy as np
from mcrcon import MCRcon

from NBT.block_nbt import Block, BlockState, NoteBlock, NoteBlockState
from NBT.nbt import NBTTag, NBTType, NBTCompound, UUID
from NBT.parameter import angle, rotation, facing, boolean, identifier, IntPosition
from NBT.selector import Target, SelectorType, GameMode, SortMethod, Position, Range


class MinecraftController:
    """
    Minecraft サーバーを制御するためのクラスです。
    server.properties ファイルから設定を読み込み、RCON を使用してコマンドを実行します。
    もし、サーバーを起動していない場合は、server.jarファイルがあるディレクトリで、
    ```
    java -Xmx1024M -Xms1024M -jar server.jar nogui
    ```
    を実行しましょう
    """
    BLOCK_COLORS: Dict[str, Tuple[int, int, int]] = {
        # Concrete
        "white_concrete": (207, 213, 214),
        "orange_concrete": (224, 97, 1),
        "magenta_concrete": (169, 48, 159),
        "light_blue_concrete": (36, 137, 199),
        "yellow_concrete": (241, 175, 21),
        "lime_concrete": (94, 169, 24),
        "pink_concrete": (214, 101, 143),
        "gray_concrete": (55, 58, 62),
        "light_gray_concrete": (125, 125, 115),
        "cyan_concrete": (21, 119, 136),
        "purple_concrete": (100, 32, 156),
        "blue_concrete": (45, 47, 143),
        "brown_concrete": (96, 59, 31),
        "green_concrete": (73, 91, 36),
        "red_concrete": (142, 33, 33),
        "black_concrete": (8, 10, 15),

        # Wood Planks
        "oak_planks": (162, 130, 78),
        "spruce_planks": (104, 78, 47),
        "birch_planks": (216, 203, 155),
        "jungle_planks": (160, 115, 80),
        "acacia_planks": (168, 90, 50),
        "dark_oak_planks": (66, 43, 21),

        # Stone types
        "stone": (125, 125, 125),
        "granite": (153, 114, 99),
        "diorite": (180, 180, 183),
        "andesite": (130, 130, 130),

        # Ores
        "coal_ore": (115, 115, 115),
        "iron_ore": (135, 130, 126),
        "gold_ore": (143, 140, 125),
        "diamond_ore": (129, 140, 143),
        "emerald_ore": (110, 129, 116),
        "lapis_ore": (102, 112, 134),
        "redstone_ore": (133, 107, 107),

        # Nature
        "grass_block": (127, 178, 56),
        "dirt": (134, 96, 67),
        "sand": (219, 211, 160),
        "gravel": (136, 126, 126),
        "oak_leaves": (60, 192, 41),
        "birch_leaves": (128, 167, 85)
    }

    def __init__(self, server_properties_path: str):
        self.server_properties_path = server_properties_path
        self.rcon = None
        self.host = None
        self.host_port = None
        self.port = None
        self.password = None
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

    def clear(self, pos1: IntPosition, pos2: Optional[IntPosition] = None) -> None:
        if pos2:
            command = f"fill {pos1} {pos2} minecraft:air"
        else:
            command = f"setblock {pos1} minecraft:air"
        self.rcon.command(command)

    def get_position(self, entity: str) -> IntPosition:
        command = f"data get entity {entity} Pos"
        response = self.rcon.command(command)
        match = re.search(
            r'\[(-?\d+\.\d+)d\, (-?\d+\.\d+)d\, (-?\d+\.\d+)d\]', response)
        if match:
            x, y, z = map(int, map(float, match.groups()))
            return IntPosition(x=x, y=y, z=z)
        else:
            raise ValueError(
                f"Failed to parse position from response: {response}")

    def setblock(self, block: Block, pos: IntPosition) -> None:
        command = f"setblock {pos} {block}"
        self.rcon.command(command)

    def place_note_block(self, pos: IntPosition, note_block: NoteBlock, block_below: Block):
        self.setblock(block_below, IntPosition(x=pos.x, y=pos.y - 1, z=pos.z))
        self.setblock(note_block, pos)

    def tp(self, target: str, pos: IntPosition) -> None:
        command = f"tp {target} {pos}"
        self.rcon.command(command)

    def setblocks(self, pos: IntPosition, blocks: np.ndarray, start_edge: str = "northwest") -> None:
        directions = {
            "northwest": (1, 1, 1),
            "northeast": (-1, 1, 1),
            "southwest": (1, 1, -1),
            "southeast": (-1, 1, -1)
        }
        dx, dy, dz = directions[start_edge]

        for i in range(blocks.shape[0]):
            for j in range(blocks.shape[1]):
                for k in range(blocks.shape[2]):
                    x = pos.x + i * dx
                    y = pos.y + j * dy
                    z = pos.z + k * dz
                    block = blocks[i, j, k]
                    self.setblock(block, IntPosition(x=x, y=y, z=z))

    def fill(self, pos1: IntPosition, pos2: IntPosition, block: Block) -> None:
        max_fill_size = 32
        size_x = abs(pos2.x - pos1.x) + 1
        size_y = abs(pos2.y - pos1.y) + 1
        size_z = abs(pos2.z - pos1.z) + 1

        segments_x = (size_x - 1) // max_fill_size + 1
        segments_y = (size_y - 1) // max_fill_size + 1
        segments_z = (size_z - 1) // max_fill_size + 1

        step_x = 1 if pos2.x >= pos1.x else -1
        step_y = 1 if pos2.y >= pos1.y else -1
        step_z = 1 if pos2.z >= pos1.z else -1

        for sx in range(segments_x):
            for sy in range(segments_y):
                for sz in range(segments_z):
                    start_x = pos1.x + sx * max_fill_size * step_x
                    start_y = pos1.y + sy * max_fill_size * step_y
                    start_z = pos1.z + sz * max_fill_size * step_z

                    end_x = min(start_x + (max_fill_size - 1) *
                                step_x, pos2.x, key=lambda x: abs(x - pos1.x))
                    end_y = min(start_y + (max_fill_size - 1) *
                                step_y, pos2.y, key=lambda y: abs(y - pos1.y))
                    end_z = min(start_z + (max_fill_size - 1) *
                                step_z, pos2.z, key=lambda z: abs(z - pos1.z))

                    command = f"fill {start_x} {start_y} {
                        start_z} {end_x} {end_y} {end_z} {block}"
                    self.rcon.command(command)

    def fill_relative(self, pos1: IntPosition, relative_pos2: IntPosition, block: Block) -> None:
        absolute_pos2 = IntPosition(
            x=pos1.x + relative_pos2.x,
            y=pos1.y + relative_pos2.y,
            z=pos1.z + relative_pos2.z
        )
        self.fill(pos1, absolute_pos2, block)

    def fill_centered(self, center: IntPosition, dx_pos: int, dx_neg: int, dy_pos: int, dy_neg: int, dz_pos: int, dz_neg: int, block: Block) -> None:
        pos1 = IntPosition(
            x=center.x - dx_neg,
            y=center.y - dy_neg,
            z=center.z - dz_neg
        )
        pos2 = IntPosition(
            x=center.x + dx_pos,
            y=center.y + dy_pos,
            z=center.z + dz_pos
        )
        self.fill(pos1, pos2, block)

    def _find_closest_block(self, color: Tuple[int, int, int]) -> str:
        distances = {block: np.linalg.norm(np.array(color) - np.array(block_color))
                     for block, block_color in self.BLOCK_COLORS.items()}
        return min(distances, key=distances.get)

    def create_image(self, image_path: str, start_pos: IntPosition, width: int = None, facing: str = 'south', orientation: str = 'horizontal', item_hold: bool = False):
        img = Image.open(image_path)
        img = img.convert('RGB')
        if width:
            height = int((width / img.width) * img.height)
            img = img.resize((width, height), Image.LANCZOS)

        img_array = np.array(img)

        blocks = np.empty(img_array.shape[:2], dtype=object)
        block_counts = {}
        for y in range(img_array.shape[0]):
            for x in range(img_array.shape[1]):
                color = tuple(img_array[y, x][:3])
                block_name = self._find_closest_block(color)
                blocks[y, x] = Block(id=identifier(name=block_name))
                block_counts[block_name] = block_counts.get(block_name, 0) + 1

        if facing == 'north':
            blocks = np.rot90(blocks, 2)
        elif facing == 'east':
            blocks = np.rot90(blocks, 1)
        elif facing == 'west':
            blocks = np.rot90(blocks, 3)

        for y in range(blocks.shape[0]):
            for x in range(blocks.shape[1]):
                if orientation == 'vertical':
                    if facing in ['north', 'south']:
                        pos = IntPosition(
                            x=start_pos.x + x, y=start_pos.y + blocks.shape[0] - y - 1, z=start_pos.z)
                    else:  # east or west
                        pos = IntPosition(
                            x=start_pos.x, y=start_pos.y + blocks.shape[0] - y - 1, z=start_pos.z + x)
                else:  # horizontal
                    if facing in ['north', 'south']:
                        pos = IntPosition(x=start_pos.x + x,
                                          y=start_pos.y, z=start_pos.z + y)
                    else:  # east or west
                        pos = IntPosition(x=start_pos.x + y,
                                          y=start_pos.y, z=start_pos.z + x)

                self.setblock(blocks[y, x], pos)

        print(f"Image created at {start_pos} facing {
              facing}, orientation: {orientation}")

        if item_hold:
            self._fill_player_hotbar(block_counts)

    def _fill_player_hotbar(self, block_counts: Dict[str, int]):
        sorted_blocks = sorted(block_counts.items(),
                               key=lambda x: x[1], reverse=True)
        for i, (block_name, _) in enumerate(sorted_blocks[:9]):
            self.rcon.command(
                f"item replace entity @p hotbar.{i} with {block_name} 1")
        print("Player's hotbar filled with the most used blocks.")

    def kill(self, entity_selector: Target, invert: bool = False) -> None:
        if invert:
            command = f"kill @e[type=!player,type=!{entity_selector}]"
        else:
            command = f"kill {entity_selector}"
        self.rcon.command(command)

    def _apply_effects_with_tag(self, tag: str, effects: List[str]):
        for effect in effects:
            self.apply_effect(f"@e[tag={tag}]", effect)
        self.rcon.command(f"tag @e[tag={tag}] remove {tag}")

    def spawn(self, entity_type: str, pos: IntPosition, nbt: Optional[NBTCompound] = None, effects: Optional[List[str]] = None) -> None:
        tag = f"temp_{uuid.uuid4().hex[:8]}" if effects else None

        command = f"summon {entity_type} {pos}"
        if nbt:
            command += f" {nbt}"
        if tag:
            command += f" {{Tags:[\"{tag}\"]}}"

        self.rcon.command(command)

        if effects:
            self._apply_effects_with_tag(tag, effects)

    def spawn_random(self, entity_type: str, pos1: IntPosition, pos2: IntPosition, count: int = 1, nbt: Optional[NBTCompound] = None, effects: Optional[List[str]] = None) -> None:
        tag = f"temp_{uuid.uuid4().hex[:8]}" if effects else None

        for _ in range(count):
            x = random.randint(min(pos1.x, pos2.x), max(pos1.x, pos2.x))
            y = random.randint(min(pos1.y, pos2.y), max(pos1.y, pos2.y))
            z = random.randint(min(pos1.z, pos2.z), max(pos1.z, pos2.z))
            random_pos = IntPosition(x=x, y=y, z=z)

            command = f"summon {entity_type} {random_pos}"
            if nbt:
                command += f" {nbt}"
            if tag:
                command += f" {{Tags:[\"{tag}\"]}}"

            self.rcon.command(command)

        if effects:
            self._apply_effects_with_tag(tag, effects)

    def spawn_with_mask(self, entity_type: str, pos: IntPosition, mask: np.ndarray, count: int = 1, nbt: Optional[NBTCompound] = None, effects: Optional[List[str]] = None) -> None:
        tag = f"temp_{uuid.uuid4().hex[:8]}" if effects else None

        spawn_positions = np.argwhere(mask)
        for spawn_pos in random.choices(spawn_positions, k=count):
            x, y, z = spawn_pos
            spawn_pos = IntPosition(x=pos.x + x, y=pos.y + y, z=pos.z + z)

            command = f"summon {entity_type} {spawn_pos}"
            if nbt:
                command += f" {nbt}"
            if tag:
                command += f" {{Tags:[\"{tag}\"]}}"

            self.rcon.command(command)

        if effects:
            self._apply_effects_with_tag(tag, effects)
    
    def apply_effect(self, target: str, effect: str) -> None:
        command = f"effect give {target} {effect}"
        self.rcon.command(command)

    def remove_effect(self, target: str, effect_id: Union[str, None] = None) -> None:
        if effect_id:
            command = f"effect clear {target} {effect_id}"
        else:
            command = f"effect clear {target}"
        self.rcon.command(command)

    def get_active_effects(self, target: str) -> str:
        command = f"data get entity {target} ActiveEffects"
        return self.rcon.command(command)


class ShapeAlignment(Enum):
    CENTER = auto()
    TOPLEFT = auto()
    TOPRIGHT = auto()
    BOTTOMLEFT = auto()
    BOTTOMRIGHT = auto()
    TOP = auto()
    BOTTOM = auto()
    LEFT = auto()
    RIGHT = auto()


class Shape(ABC):
    def __init__(self, mc: MinecraftController, pos: IntPosition, alignment: ShapeAlignment = ShapeAlignment.CENTER):
        self.mc = mc
        self.pos = pos
        self.alignment = alignment

    @property
    @abstractmethod
    def center(self) -> IntPosition:
        pass

    @property
    @abstractmethod
    def size(self) -> Tuple[int, int, int]:
        pass

    @property
    def left(self) -> int:
        return self.center.x - self.size[0] // 2

    @property
    def right(self) -> int:
        return self.center.x + self.size[0] // 2

    @property
    def top(self) -> int:
        return self.center.y + self.size[1] // 2

    @property
    def bottom(self) -> int:
        return self.center.y - self.size[1] // 2

    @property
    def front(self) -> int:
        return self.center.z - self.size[2] // 2

    @property
    def back(self) -> int:
        return self.center.z + self.size[2] // 2

    @abstractmethod
    def get_blocks(self) -> List[Tuple[IntPosition, Block]]:
        pass

    def place(self):
        for pos, block in self.get_blocks():
            self.mc.setblock(block, pos)

    def clear(self):
        for pos, _ in self.get_blocks():
            self.mc.setblock(Block(id=identifier(name="air")), pos)


class Cube(Shape):
    def __init__(self, mc: MinecraftController, pos: IntPosition, size: Tuple[int, int, int], block: Block, alignment: ShapeAlignment = ShapeAlignment.CENTER):
        super().__init__(mc, pos, alignment)
        self._size = size
        self.block = block
        self._calculate_center()

    def _calculate_center(self):
        if self.alignment == ShapeAlignment.CENTER:
            self._center = self.pos
        elif self.alignment == ShapeAlignment.TOPLEFT:
            self._center = IntPosition(
                x=self.pos.x + self._size[0] // 2,
                y=self.pos.y - self._size[1] // 2,
                z=self.pos.z + self._size[2] // 2
            )
        elif self.alignment == ShapeAlignment.BOTTOM:
            self._center = IntPosition(
                x=self.pos.x,
                y=self.pos.y + self._size[1] // 2,
                z=self.pos.z
            )
        # その他のアライメントケースも同様に実装

    @property
    def center(self) -> IntPosition:
        return self._center

    @property
    def size(self) -> Tuple[int, int, int]:
        return self._size

    def get_blocks(self) -> List[Tuple[IntPosition, Block]]:
        blocks = []
        for x in range(self.left, self.right + 1):
            for y in range(self.bottom, self.top + 1):
                for z in range(self.front, self.back + 1):
                    blocks.append((IntPosition(x=x, y=y, z=z), self.block))
        return blocks

    def place(self):
        max_fill_size = 32
        segments_x = (self.size[0] - 1) // max_fill_size + 1
        segments_y = (self.size[1] - 1) // max_fill_size + 1
        segments_z = (self.size[2] - 1) // max_fill_size + 1

        for sx in range(segments_x):
            for sy in range(segments_y):
                for sz in range(segments_z):
                    start_x = self.left + sx * max_fill_size
                    start_y = self.bottom + sy * max_fill_size
                    start_z = self.front + sz * max_fill_size

                    end_x = min(start_x + max_fill_size - 1, self.right)
                    end_y = min(start_y + max_fill_size - 1, self.top)
                    end_z = min(start_z + max_fill_size - 1, self.back)

                    self.mc.rcon.command(f"fill {start_x} {start_y} {start_z} {
                                         end_x} {end_y} {end_z} {self.block} replace")

    def clear(self):
        max_fill_size = 32
        segments_x = (self.size[0] - 1) // max_fill_size + 1
        segments_y = (self.size[1] - 1) // max_fill_size + 1
        segments_z = (self.size[2] - 1) // max_fill_size + 1

        for sx in range(segments_x):
            for sy in range(segments_y):
                for sz in range(segments_z):
                    start_x = self.left + sx * max_fill_size
                    start_y = self.bottom + sy * max_fill_size
                    start_z = self.front + sz * max_fill_size

                    end_x = min(start_x + max_fill_size - 1, self.right)
                    end_y = min(start_y + max_fill_size - 1, self.top)
                    end_z = min(start_z + max_fill_size - 1, self.back)

                    self.mc.rcon.command(f"fill {start_x} {start_y} {start_z} {
                                         end_x} {end_y} {end_z} air replace")


class Sphere(Shape):
    def __init__(self, mc: MinecraftController, pos: IntPosition, radius: int, block: Block, alignment: ShapeAlignment = ShapeAlignment.CENTER):
        super().__init__(mc, pos, alignment)
        self.radius = radius
        self.block = block
        self._calculate_center()

    def _calculate_center(self):
        if self.alignment == ShapeAlignment.CENTER:
            self._center = self.pos
        elif self.alignment == ShapeAlignment.TOPLEFT:
            self._center = IntPosition(
                x=self.pos.x + self.radius,
                y=self.pos.y - self.radius,
                z=self.pos.z + self.radius
            )
        # その他のアライメントケースも同様に実装

    @property
    def center(self) -> IntPosition:
        return self._center

    @property
    def size(self) -> Tuple[int, int, int]:
        diameter = self.radius * 2 + 1
        return (diameter, diameter, diameter)

    def get_blocks(self) -> List[Tuple[IntPosition, Block]]:
        blocks = []
        for x in range(self.left, self.right + 1):
            for y in range(self.bottom, self.top + 1):
                for z in range(self.front, self.back + 1):
                    if (x - self.center.x) ** 2 + (y - self.center.y) ** 2 + (z - self.center.z) ** 2 <= self.radius ** 2:
                        blocks.append((IntPosition(x=x, y=y, z=z), self.block))
        return blocks

    def place(self):
        for pos, _ in self.get_blocks():
            self.mc.setblock(self.block, pos)

    def clear(self):
        air_block = Block(id=identifier(name="air"))
        for pos, _ in self.get_blocks():
            self.mc.setblock(air_block, pos)


if __name__ == "__main__":
    # プロジェクトのルートディレクトリをPythonパスに追加
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from Dungeon.DungeonMaker import Analyzer
    with MinecraftController("MinecraftServer/server.properties") as mc:
        image_size = 100
        
        # プレイヤーの現在位置を取得
        player_pos = mc.get_position("@p")

        # プレイヤーの周囲をクリア
        mc.fill_centered(player_pos,
                         image_size*2, image_size*2,
                         30, 0,
                         image_size*2, image_size*2,
                         Block(id=identifier(name="air")))

        # 地面を作成
        mc.fill_centered(player_pos,
                         image_size*2, image_size*2,
                         -1, 1,
                         image_size*2, image_size*2,
                         Block(id=identifier(name="grass_block")))
        
        # 画像を作成する位置を設定
        image_pos = IntPosition(x=player_pos.x - image_size//2,
                                y=player_pos.y, z=player_pos.z - image_size//2)

        # 画像を作成
        mc.create_image(
            image_path="kazehaya.jpg",  # 画像ファイルのパスを指定
            start_pos=image_pos,
            width=image_size,  # 画像の幅（ブロック数）
            facing='south',  # 画像の向き
            orientation='horizontal',  # 画像の配置方向
            item_hold=True  # プレイヤーのホットバーに使用したブロックを配置
        )

        print(f"Image created at {image_pos}")

        # プレイヤーを画像の前にテレポート
        teleport_pos = IntPosition(
            x=player_pos.x, y=player_pos.y, z=image_pos.z - 5)
        mc.tp("@p", teleport_pos)

        # 画像を見るためのエフェクトを付与
        mc.apply_effect("@p", "minecraft:night_vision 60 1")

        print("Player teleported and given night vision effect.")

        # プレイヤーの位置から少し離れた場所に迷路を生成
        maze_start_pos = IntPosition(
            x=player_pos.x + image_size//2 + 10, y=player_pos.y, z=player_pos.z)

        # 迷路の生成
        maze, labels, start_goal_candidates = Analyzer.create_maze(
            (30, 30), 50)

        # 迷路の基本ブロック
        base_block = Block(id=identifier(name="bedrock"))

        # 迷路の3D配列を作成
        m_maze = [[base_block] * (2 + maze.shape[1])]
        for i in range(maze.shape[0]):
            i_maze = [base_block]
            for j in range(maze.shape[1]):
                if maze[i, j] == 0:
                    i_maze.append(Block(id=identifier(name="air")))
                else:
                    i_maze.append(base_block)
            m_maze.append(i_maze + [base_block])
        m_maze.append([base_block] * (2 + maze.shape[1]))
        m_maze = np.tile(np.expand_dims(np.array(m_maze), 1), (1, 3, 1))

        # 迷路の配置
        mc.setblocks(maze_start_pos, m_maze)

        # 迷路の床を設置
        mc.fill(
            IntPosition(x=maze_start_pos.x, y=maze_start_pos.y -
                        1, z=maze_start_pos.z),
            IntPosition(x=maze_start_pos.x +
                        maze.shape[0] + 1, y=maze_start_pos.y - 1, z=maze_start_pos.z + maze.shape[1] + 1),
            Block(id=identifier(name="bedrock"))
        )

        # スタートとゴールの設定
        region = random.choice(list(start_goal_candidates.keys()))
        start_pos = random.choice(start_goal_candidates[region][0])
        goal_pos = random.choice(start_goal_candidates[region][1])

        # ゴールの設置
        mc.setblock(
            Block(id=identifier(name="gold_block")),
            IntPosition(x=maze_start_pos.x +
                        goal_pos[0] + 1, y=maze_start_pos.y, z=maze_start_pos.z + goal_pos[1] + 1)
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
