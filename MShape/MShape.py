import copy
from dataclasses import dataclass, field
import numpy as np
from typing import Callable, List, Optional, Tuple, Union

from scipy import ndimage
from Helper.helpers import Quaternion, rotate_3d_array
from NBT.MBlocks import MBlocks
from NBT.parameter import IntPosition
from Command.CompoundCommand import SetBlocks

DEFAULT_BLOCK = MBlocks.stone

@dataclass
class MShape:
    block_mapping: List[MBlocks] = field(default_factory=list)
    initial_position: IntPosition = field(default_factory=lambda: IntPosition(0, 0, 0))
    shape: Optional[np.ndarray] = None
    empty_mask: Optional[np.ndarray] = None
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3), init=False)
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    center: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.shape is None:
            raise ValueError("Shape must be specified")
        if self.empty_mask is None:
            self.empty_mask = np.zeros_like(self.shape, dtype=bool)
        elif self.empty_mask.shape != self.shape.shape:
            raise ValueError("Shape and empty_mask must have the same dimensions")
        self.center = np.array(self.shape.shape) / 2
        self.rotation_quat = Quaternion(1, 0, 0, 0)  # Identity quaternion
    
    def copy(self):
        new_instance = copy.deepcopy(self)
        
        if type(self) != MShape:
            return new_instance
        
        return MShape(
            block_mapping=new_instance.block_mapping,
            initial_position=new_instance.initial_position,
            shape=new_instance.shape.copy() if new_instance.shape is not None else None,
            empty_mask=new_instance.empty_mask.copy() if new_instance.empty_mask is not None else None,
            rotation=new_instance.rotation.copy(),
            translation=new_instance.translation.copy(),
            center=new_instance.center.copy()
        )
    
    def rotate(self, angles: Tuple[float, float, float]):
        """
        Rotate the shape using Unity-like Euler angles (in degrees).
        Internally uses quaternions to avoid gimbal lock.
        """
        rotation_quat = Quaternion.from_euler(*angles)
        self.rotation_quat = rotation_quat * self.rotation_quat
        self.rotation = self.rotation_quat.to_rotation_matrix()

    def rotate_around(self, point: Union[Tuple[float, float, float], IntPosition], angles: Tuple[float, float, float]):
        """
        Rotate the shape around a specific point using Unity-like Euler angles (in degrees).
        """
        if isinstance(point, IntPosition):
            point = point.numpy
        else:
            point = np.array(point)

        # Translate to origin
        self.translate(-point)
        
        # Rotate
        self.rotate(angles)
        
        # Translate back
        self.translate(point)

    def rotate_axis(self, axis: Tuple[float, float, float], angle: float):
        """
        Rotate the shape around a specified axis by the given angle (in degrees).
        """
        rotation_quat = Quaternion.from_axis_angle(axis, angle)
        self.rotation_quat = rotation_quat * self.rotation_quat
        self.rotation = self.rotation_quat.to_rotation_matrix()

    def translate(self, offset: Union[Tuple[int, int, int], IntPosition]):
        if isinstance(offset,IntPosition):
            self.translation += offset.numpy
        else:
            self.translation += np.array(offset)

    def set_center(self, new_center: Union[Tuple[int, int, int], IntPosition]):
        if isinstance(new_center,IntPosition):
            self.center = new_center.numpy
        else:
            self.center = np.array(new_center)

    def place(self):
        rotated_shape, rotated_mask = self.get_rotated()
        
        new_center = np.array(rotated_shape.shape) / 2
        center_diff = new_center - self.center
        
        final_position = self.initial_position + IntPosition(self.translation) - IntPosition(center_diff)
        return SetBlocks(final_position, rotated_shape.astype(int), self.block_mapping, empty_mask=~rotated_mask,do_prefill=True)
    
    def get_rotated(self):
        if self.rotation_quat.is_close_to_identity():
            return self.shape, ~self.empty_mask
        conv = rotate_3d_array(self.shape.shape, self.rotation, self.center, ~self.empty_mask)
        return conv(self.shape)

@dataclass
class Cube(MShape):
    size: int = 1
    block: MBlocks = DEFAULT_BLOCK
    shape: np.ndarray = field(init=False)
    block_mapping: list[MBlocks] = field(default_factory=list,init=False)
    empty_mask: np.ndarray = field(init=False)

    def __post_init__(self):
        self.shape = np.zeros((self.size, self.size, self.size))
        self.block_mapping = [self.block]
        self.empty_mask = np.zeros_like(self.shape,dtype=np.bool_)
        super().__post_init__()

@dataclass
class Sphere(MShape):
    radius: int = 1
    block: MBlocks = DEFAULT_BLOCK
    shape: np.ndarray = field(init=False)
    block_mapping: list[MBlocks] = field(default_factory=list,init=False)
    empty_mask: np.ndarray = field(init=False)

    def __post_init__(self):
        size = 2 * self.radius + 1
        self.shape = np.zeros((size, size, size))
        self.empty_mask = np.ones_like(self.shape,dtype=np.bool_)
        center = np.array([self.radius, self.radius, self.radius])
        for index in np.ndindex(self.shape.shape):
            if np.linalg.norm(np.array(index) - center) - self.radius < 0.5:
                self.empty_mask[index] = False
        self.block_mapping = [self.block]
        super().__post_init__()

@dataclass
class Cylinder(MShape):
    radius: int = 1
    height: int = 1
    block: MBlocks = DEFAULT_BLOCK
    shape: np.ndarray = field(init=False)
    block_mapping: list[MBlocks] = field(default_factory=list,init=False)
    empty_mask: np.ndarray = field(init=False)

    def __post_init__(self):
        self.shape = np.zeros((2*self.radius+1, self.height, 2*self.radius+1))
        self.empty_mask = np.ones_like(self.shape,dtype=np.bool_)
        center = np.array([self.radius, 0, self.radius])
        for x, z in np.ndindex(self.shape.shape[0], self.shape.shape[2]):
            if np.linalg.norm(np.array([x, z]) - center[[0,2]]) <= self.radius:
                self.empty_mask[x, :, z] = False
        self.block_mapping = [self.block]
        super().__post_init__()

@dataclass
class Pyramid(MShape):
    base_size: int = 1
    block: MBlocks = DEFAULT_BLOCK
    shape: np.ndarray = field(init=False)
    block_mapping: list[MBlocks] = field(default_factory=list,init=False)
    empty_mask: np.ndarray = field(init=False)

    def __post_init__(self):
        self.shape = np.zeros((self.base_size, self.base_size, self.base_size))
        self.empty_mask = np.ones_like(self.shape,dtype=np.bool_)
        for y in range(self.base_size):
            size = self.base_size - y
            self.shape[y, (self.base_size-size)//2:(self.base_size+size)//2, (self.base_size-size)//2:(self.base_size+size)//2] = False
        self.block_mapping = [self.block]
        super().__post_init__()

@dataclass
class Plane(MShape):
    width: int = 1
    height: int = 1
    block: MBlocks = DEFAULT_BLOCK
    shape: np.ndarray = field(init=False)
    block_mapping: list[MBlocks] = field(default_factory=list,init=False)
    empty_mask: np.ndarray = field(init=False)

    def __post_init__(self):
        self.shape = np.zeros((self.width, 1, self.height))
        self.empty_mask = np.ones_like(self.shape,dtype=np.bool_)
        self.empty_mask[:, 0, :] = 0
        self.block_mapping = [self.block]
        super().__post_init__()