import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Union, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Command.Command import Command, Commands, Fill, SetBlock
from Command.CompoundCommand import CompoundCommand, AdjustFill, SetBlocks
from NBT.MBlocks import MBlocks
from NBT.parameter import IntPosition

class MinecraftCommandVisualizer:
    def __init__(self, base_position: IntPosition = IntPosition(0, 0, 0)):
        self.base_position = base_position
        self.array = None
        self.block_mapping = [MBlocks.air]
        self.block_colors = [np.zeros(4,dtype=np.float32)]
        self.min_pos = None
        self.max_pos = None

    def parse_commands(self, commands: Union[Command,Commands, CompoundCommand, List[Union[Command,Commands, CompoundCommand]]]):
        parsed_commands: list[tuple[IntPosition,IntPosition,MBlocks]] = []
        print(f"Checking for {commands.__class__.__name__}")
        if isinstance(commands, (list,Commands)):
            for cmd in commands:
                parsed_commands.extend(self.parse_commands(cmd))
        elif isinstance(commands, CompoundCommand):
            parsed_commands.extend(self.parse_commands(commands.commands))
        elif isinstance(commands, Fill):
            parsed_commands.append((commands._from, commands._to, commands.get_block()))
        elif isinstance(commands, SetBlock):
            parsed_commands.append((commands.pos, commands.pos, commands.get_block()))
        return parsed_commands

    def create_3d_array(self, parsed_commands:list[tuple[IntPosition,IntPosition,MBlocks]]):
        min_pos = self.base_position.copy()
        max_pos = self.base_position.copy()
        
        for cmd in parsed_commands:
            start, end, _ = cmd
            min_pos = IntPosition(min(min_pos.x, start.x, end.x),
                                  min(min_pos.y, start.y, end.y),
                                  min(min_pos.z, start.z, end.z))
            max_pos = IntPosition(max(max_pos.x, start.x, end.x),
                                  max(max_pos.y, start.y, end.y),
                                  max(max_pos.z, start.z, end.z))
        
        shape = (max_pos.x - min_pos.x + 1, max_pos.y - min_pos.y + 1, max_pos.z - min_pos.z + 1)
        array = np.zeros((*shape,1),dtype=np.int32)
        for cmd in parsed_commands:
            start, end, block = cmd
            x_slice = slice(start.x - min_pos.x, end.x - min_pos.x + 1)
            y_slice = slice(start.y - min_pos.y, end.y - min_pos.y + 1)
            z_slice = slice(start.z - min_pos.z, end.z - min_pos.z + 1)
            if not block in self.block_mapping:
                block_id = len(self.block_mapping)
                self.block_mapping.append(block)
                self.block_colors.append(np.clip(np.array(block.value.get('color',[0,0,0,0]),dtype=np.float32)/255,0.0,1.0))
            else:
                block_id = self.block_mapping.index(block)
            array[x_slice, y_slice, z_slice] = block_id
        self.array = array
        self.min_pos = min_pos
        self.max_pos = max_pos

    def visualize(self, commands: Union[Command, CompoundCommand, List[Union[Command, CompoundCommand]]], 
                  figsize: tuple = (12, 10), elev: int = 20, azim: int = 45):
        parsed_commands = self.parse_commands(commands)
        self.create_3d_array(parsed_commands)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        indices = np.indices(self.array.shape[:-1])
        cols = np.stack(self.block_colors)
        f_a = self.array.reshape((-1))
        f_i = indices.transpose(3,2,1,0).reshape((-1,3))
        a_c = cols[f_a]
        exist_mask = a_c[:,3] > 0
        f_i = f_i[exist_mask]
        ax.bar3d(f_i[:,0],f_i[:,2],f_i[:,1],1,1,1,color=a_c[exist_mask])
        ax.set_xlabel('X axis')
        ax.set_ylabel('Z axis')
        ax.set_zlabel('Y axis')
        ax.set_title('Minecraft Command Visualization')
        
        max_shape = max(self.array.shape[:4])
        ax.set_xlim(0, max_shape)
        ax.set_ylim(0, max_shape)
        ax.set_zlim(0, max_shape)
        
        
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    def get_block_at(self, position: IntPosition) -> Optional[MBlocks]:
        if self.array is None:
            return None
        relative_pos = position - self.min_pos
        if (0 <= relative_pos.x < self.array.shape[0] and
            0 <= relative_pos.y < self.array.shape[1] and
            0 <= relative_pos.z < self.array.shape[2]):
            return self.array[relative_pos.x, relative_pos.y, relative_pos.z]
        return None
    
if __name__ == "__main__":
    from Command.CompoundCommand import SuperFlat
    from MShape.MShape import Cube,Sphere
    p_pos = IntPosition(1,5,10)
    visualizer = MinecraftCommandVisualizer(p_pos)
    cmd = SuperFlat(
        p_pos,
        range=IntPosition(5,0,5),
        layers=[
            (1,MBlocks.grass_block),
            (2,MBlocks.dirt),
            (1,MBlocks.bedrock)
        ],
        start_point=SuperFlat.LayerStartPoint(
            0,SuperFlat.LayerReference.TOP
        )
    )
    #cube = Cube(p_pos+IntPosition(0,3,0),3)
    #cube.rotate_axis((0,1,0),45)
    sphere = Sphere(p_pos+IntPosition(0,8,0),4)
    sphere.translate((2,0,0))
    sphere.translate((-2,0,0))
    cmd.add_command(sphere.place())
    visualizer.visualize(cmd)
    