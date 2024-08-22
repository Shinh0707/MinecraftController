from collections import defaultdict
from enum import Enum
from math import cos, radians, sin
from typing import Dict, List, Tuple
import cv2
import mido
import numpy as np
from scipy import ndimage
from Helper.math import flexible_round
from NBT.MBlocks import MBlocks
from NBT.block_nbt import Instrument

class NamedEnum(Enum):
    def __str__(self):
        return self.name.lower()

class ValuedEnum(Enum):
    def __str__(self):
        return f"{self.value}"

class Quaternion:
    def __init__(self, w=1, x=0, y=0, z=0):
        self.q = complex(w, x)
        self.r = complex(y, z)

    @classmethod
    def from_axis_angle(cls, axis, angle):
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)
        angle_rad = radians(angle)
        s = sin(angle_rad / 2)
        return cls(cos(angle_rad / 2), axis[0] * s, axis[1] * s, axis[2] * s)

    @classmethod
    def from_euler(cls, roll, pitch, yaw):
        cr, cp, cy = cos(radians(roll/2)), cos(radians(pitch/2)), cos(radians(yaw/2))
        sr, sp, sy = sin(radians(roll/2)), sin(radians(pitch/2)), sin(radians(yaw/2))
        return cls(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        )

    def to_rotation_matrix(self):
        a, b, c, d = self.q.real, self.q.imag, self.r.real, self.r.imag
        return np.array([
            [1 - 2*(c**2 + d**2), 2*(b*c - a*d), 2*(b*d + a*c)],
            [2*(b*c + a*d), 1 - 2*(b**2 + d**2), 2*(c*d - a*b)],
            [2*(b*d - a*c), 2*(c*d + a*b), 1 - 2*(b**2 + c**2)]
        ])

    def __str__(self):
        return f"Quaternion({self.q.real}, {self.q.imag}, {self.r.real}, {self.r.imag})"

    def __add__(self, other):
        return Quaternion(
            self.q.real + other.q.real,
            self.q.imag + other.q.imag,
            self.r.real + other.r.real,
            self.r.imag + other.r.imag
        )

    def __sub__(self, other):
        return Quaternion(
            self.q.real - other.q.real,
            self.q.imag - other.q.imag,
            self.r.real - other.r.real,
            self.r.imag - other.r.imag
        )

    def __mul__(self, other):
        if isinstance(other, Quaternion):
            a1, b1, c1, d1 = self.q.real, self.q.imag, self.r.real, self.r.imag
            a2, b2, c2, d2 = other.q.real, other.q.imag, other.r.real, other.r.imag
            return Quaternion(
                a1*a2 - b1*b2 - c1*c2 - d1*d2,
                a1*b2 + b1*a2 + c1*d2 - d1*c2,
                a1*c2 - b1*d2 + c1*a2 + d1*b2,
                a1*d2 + b1*c2 - c1*b2 + d1*a2
            )
        elif isinstance(other, (int, float)):
            return Quaternion(
                self.q.real * other,
                self.q.imag * other,
                self.r.real * other,
                self.r.imag * other
            )
        else:
            raise TypeError("Multiplication is only defined for Quaternions and scalars")

    def __rmul__(self, other):
        return self.__mul__(other)

    def conjugate(self):
        return Quaternion(self.q.real, -self.q.imag, -self.r.real, -self.r.imag)

    def norm(self):
        return (self.q.real**2 + self.q.imag**2 + self.r.real**2 + self.r.imag**2)**0.5

    def inverse(self):
        norm_squared = self.norm()**2
        conjugate = self.conjugate()
        return Quaternion(
            conjugate.q.real / norm_squared,
            conjugate.q.imag / norm_squared,
            conjugate.r.real / norm_squared,
            conjugate.r.imag / norm_squared
        )

    def __truediv__(self, other):
        if isinstance(other, Quaternion):
            return self * other.inverse()
        elif isinstance(other, (int, float)):
            return Quaternion(
                self.q.real / other,
                self.q.imag / other,
                self.r.real / other,
                self.r.imag / other
            )
        else:
            raise TypeError("Division is only defined for Quaternions and scalars")
    
    @classmethod
    def identity(cls):
        return cls(1, 0, 0, 0)

    def is_close_to_identity(self, tolerance=1e-6):
        return (abs(self.q.real - 1) < tolerance and
                abs(self.q.imag) < tolerance and
                abs(self.r.real) < tolerance and
                abs(self.r.imag) < tolerance)
        
def divide_box_discrete(p1, p2, m):
    # p1とp2を整数に丸めて正規化（p1の座標が常に小さくなるようにする）
    start = np.floor(np.minimum(p1, p2)).astype(int)
    end = np.floor(np.maximum(p1, p2)).astype(int) + 1
    
    # 大きさ0の軸を1として扱う
    size = np.maximum(end - start, 1)
    end = start + size
    
    # 全てのブロックの開始点を生成
    ranges = [np.arange(0, s, m) for s in size]
    start_points = np.array(np.meshgrid(*ranges, indexing='ij')).reshape(3, -1).T + start
    
    # 終点を計算（開始点 + m - 1、ただしend座標を超えないようにする）
    end_points = np.minimum(start_points + m - 1, end - 1)
    
    # 結果を整形
    result = np.stack([start_points, end_points], axis=1)
    return result

def find_largest_cuboid(space: np.ndarray, value: int, start: tuple[int, int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    x, y, z = start
    max_x, max_y, max_z = space.shape

    # Expand in x direction
    dx = 1
    while x + dx < max_x and space[x + dx, y, z] == value:
        dx += 1

    # Expand in y direction
    dy = 1
    while y + dy < max_y and np.all(space[x:x+dx, y+dy, z] == value):
        dy += 1

    # Expand in z direction
    dz = 1
    while z + dz < max_z and np.all(space[x:x+dx, y:y+dy, z+dz] == value):
        dz += 1

    return (x, y, z), (x+dx-1, y+dy-1, z+dz-1)


def partition_3d_space(space: np.ndarray) -> list[tuple[tuple[int, int, int], tuple[int, int, int], int]]:
    partitions = []
    remaining = np.ones_like(space, dtype=bool)

    def recursive_partition(x_start, y_start, z_start):
        for x in range(x_start, space.shape[0]):
            for y in range(y_start if x == x_start else 0, space.shape[1]):
                for z in range(z_start if x == x_start and y == y_start else 0, space.shape[2]):
                    if remaining[x, y, z]:
                        value = space[x, y, z]
                        start, end = find_largest_cuboid(
                            space, value, (x, y, z))
                        partitions.append((start, end, value))

                        # Mark the cuboid as processed
                        x1, y1, z1 = start
                        x2, y2, z2 = end
                        remaining[x1:x2+1, y1:y2+1, z1:z2+1] = False

                        # Recursively process remaining parts
                        if x2 + 1 < space.shape[0]:
                            recursive_partition(x2 + 1, y1, z1)
                        if y2 + 1 < space.shape[1]:
                            recursive_partition(x1, y2 + 1, z1)
                        if z2 + 1 < space.shape[2]:
                            recursive_partition(x1, y1, z2 + 1)
                        return

    recursive_partition(0, 0, 0)
    return partitions

def mode_val(arr: np.ndarray):
    unique, freq = np.unique(arr, return_counts=True) #return_counts=Trueが肝
    return unique[np.argmax(freq)] #freqの最も頻度が多い引数を取得して、uniqueから引っ張ってくる

def pixelate_image(image_path, target_long_side, color_reduction=32):
    # 画像を読み込む
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # チャンネル数を確認
    if img.ndim == 2:  # グレースケール画像の場合
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 3:  # BGRの場合
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.shape[2] == 4:  # BGRAの場合
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    
    # 元の画像サイズとアスペクト比を取得
    height, width = img.shape[:2]
    aspect_ratio = width / height
    
    # 新しいサイズを計算（長辺が target_long_side になるように）
    if width > height:
        new_width = target_long_side
        new_height = int(round(new_width / aspect_ratio))
    else:
        new_height = target_long_side
        new_width = int(round(new_height * aspect_ratio))
    
    # ステップ1: ダウンサンプリング（エリア補間を使用）
    small = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # アルファチャンネルを分離（存在する場合）
    if small.shape[2] == 4:
        small_rgb = small[:,:,:3]
        alpha = small[:,:,3]
    else:
        small_rgb = small
        alpha = None
    
    # ステップ2: カラーの削減（k-means法を使用）
    pixels = small_rgb.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, color_reduction, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(small_rgb.shape)
    
    # ステップ3: 穏やかなエッジの強調
    edge_preserved = cv2.edgePreservingFilter(quantized, flags=1, sigma_s=60, sigma_r=0.4)
    
    # アルファチャンネルを再結合（存在する場合）
    if alpha is not None:
        edge_preserved = np.dstack((edge_preserved, alpha))
    
    # RGBからBGRに戻す（OpenCVで保存するため）
    if edge_preserved.shape[2] == 3:
        result = cv2.cvtColor(edge_preserved, cv2.COLOR_RGB2BGR)
    else:
        result = cv2.cvtColor(edge_preserved, cv2.COLOR_RGBA2BGRA)
    
    return cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA)

from scipy import ndimage

def compress_to_center(array_3d,mask):
    """
    フラグ値を持つ3次元配列の各要素を、中心からの距離が遠いものから順に
    中心に向かって1単位（1ブロック）だけ移動させる。
    ベクトル計算に基づいて移動方向を決定し、必要な場合のみ移動する。
    元の位置は-1で埋め、maskも同時に更新する。

    Parameters:
    array_3d (np.ndarray): 入力の3次元配列（フラグ値を含む）
    mask (np.ndarray): 空でない部分をTrueで表す3次元ブール配列

    Returns:
    tuple: (移動後の配列, 更新されたマスク)
    """
    # 配列の形状を取得
    shape = np.array(array_3d.shape)
    
    # 中心座標を計算
    center = (shape-1) / 2
    
    # 結果を格納する配列を初期化（-1で埋める）
    result = np.full_like(array_3d, fill_value=-1)
    
    # 新しいマスクを初期化（全てFalse）
    new_mask = np.zeros_like(mask)
    
    # 各点の座標を取得
    coords = np.array(np.where(mask)).T
    
    # 中心からの距離を計算
    distances = np.sqrt(np.sum((coords - center) ** 2, axis=1))
    
    # 距離でソート（降順）
    sorted_indices = np.argsort(distances)
    
    # 中心からの距離が遠いものから順に処理
    for idx in sorted_indices:
        x, y, z = coords[idx]
        value = array_3d[x, y, z]
        
        # 中心へのベクトル
        to_center = center - np.array([x, y, z])
        
        # ベクトルの長さが1未満なら移動しない
        if np.linalg.norm(to_center) < 1:
            result[x, y, z] = value
            new_mask[x, y, z] = True
            continue
        
        # 移動方向（最も中心に近づく整数座標）
        move_direction = to_center / np.linalg.norm(to_center)
        
        # 新しい位置
        new_pos = np.round(np.array([x, y, z]) + move_direction).astype(int)
        
        # 範囲内にクリップ
        new_pos = np.clip(new_pos, 0, shape - 1)
        result[x, y, z] = -1
        new_mask[x, y, z] = False
        result[tuple(new_pos)] = value
        new_mask[tuple(new_pos)] = True    
    
    return result, new_mask

def erosion(binary_volume, kernel_size=3):
    """3D binary volumeに対してerosion操作を行う"""
    kernel = np.ones((kernel_size, kernel_size, kernel_size))
    return ndimage.binary_erosion(binary_volume, structure=kernel).astype(binary_volume.dtype)

def dilation(binary_volume, kernel_size=3):
    """3D binary volumeに対してdilation操作を行う"""
    kernel = np.ones((kernel_size, kernel_size, kernel_size))
    return ndimage.binary_dilation(binary_volume, structure=kernel).astype(binary_volume.dtype)

def opening(binary_volume, kernel_size=3):
    """3D binary volumeに対してopening操作を行う"""
    return dilation(erosion(binary_volume, kernel_size), kernel_size)

def closing(binary_volume, kernel_size=3):
    """3D binary volumeに対してclosing操作を行う"""
    return erosion(dilation(binary_volume, kernel_size), kernel_size)

def erosion(binary_volume, kernel_size=3):
    """3D binary volumeに対してerosion操作を行う"""
    kernel = np.ones((kernel_size, kernel_size, kernel_size))
    return ndimage.binary_erosion(binary_volume, structure=kernel).astype(binary_volume.dtype)

def dilation(binary_volume, kernel_size=3):
    """3D binary volumeに対してdilation操作を行う"""
    kernel = np.ones((kernel_size, kernel_size, kernel_size))
    return ndimage.binary_dilation(binary_volume, structure=kernel).astype(binary_volume.dtype)

def opening(binary_volume, kernel_size=3, iterations=1):
    """3D binary volumeに対してopening操作を指定回数行う"""
    result = binary_volume
    for _ in range(iterations):
        result = dilation(erosion(result, kernel_size), kernel_size)
    return result

def closing(binary_volume, kernel_size=3, iterations=1):
    """3D binary volumeに対してclosing操作を指定回数行う"""
    result = binary_volume
    for _ in range(iterations):
        result = erosion(dilation(result, kernel_size), kernel_size)
    return result

def fill_empty_voxels_morphological(X, mask, method='closing', kernel_size=3, iterations=2, return_mask: bool=False):
    """
    3次元配列Xの「空」のボクセルを、形態学的操作を用いて埋める。
    
    Parameters:
    X (np.ndarray): 3次元の入力配列
    mask (np.ndarray): Xと同じ形状のブール配列。Trueは「空でない」ボクセルを示す。
    method (str): 使用する形態学的操作。'erosion', 'dilation', 'opening', 'closing'のいずれか。
    kernel_size (int): 形態学的操作に使用するカーネルのサイズ。
    iterations (int): opening または closing 操作を繰り返す回数。
    
    Returns:
    np.ndarray: 空のボクセルが埋められた新しい3次元配列
    """
    # 形態学的操作の辞書
    morphological_ops = {
        'erosion': lambda m: erosion(m, kernel_size),
        'dilation': lambda m: dilation(m, kernel_size),
        'opening': lambda m: opening(m, kernel_size, iterations),
        'closing': lambda m: closing(m, kernel_size, iterations)
    }
    
    if method not in morphological_ops:
        raise ValueError("Invalid method. Choose from 'erosion', 'dilation', 'opening', 'closing'.")
    
    # 選択された形態学的操作を適用
    processed_mask = morphological_ops[method](mask)
    
    # 新しく埋めるべきボクセルを特定
    fill_mask = processed_mask & (~mask)
    
    # 結果を格納する配列を初期化
    result = X.copy()
    
    # 近傍の値の最頻値を計算
    def masked_mode(values):
        center_value = values[values.size // 2]
        values = np.delete(values, values.size // 2)
        not_nan = values[~np.isnan(values)]
        if not_nan.size == 0:
            return center_value
        
        unique, counts = np.unique(not_nan, return_counts=True)
        return unique[np.argmax(counts)]
    
    mode_values = ndimage.generic_filter(
        np.where(mask, X, np.nan),
        masked_mode,
        size=kernel_size,
        mode='constant',
        cval=np.nan
    )
    
    # 新しく特定されたボクセルを埋める
    result[fill_mask] = mode_values[fill_mask]
    if return_mask:
        return result, fill_mask | mask
    return result

def rotate_3d_array(shape, rot_matrix, origin=(0, 0, 0), addidional_mask: np.ndarray|None=None):
    # 3D配列の各要素の座標を生成
    coords = np.indices(shape).reshape(3, -1).T

    # 原点を中心に回転するように座標を調整
    center = np.array(shape) / 2
    coords = coords - center + np.array(origin)

    # 座標を回転
    rotated_coords = np.dot(coords, rot_matrix.T)

    # 回転後の座標を元の範囲に戻す
    rotated_coords = rotated_coords + center - np.array(origin)

    # 回転後の座標を整数に丸める
    scale = 2
    # rotated_coords = np.round(rotated_coords).astype(int)
    to_indices = rotated_coords.T.reshape((3,) + shape).transpose(3,2,1,0)*scale
    flattened = flexible_round(to_indices.reshape((-1,3))).astype(int)
    to_range_min = np.min(flattened,axis=0)
    to_range_max = np.max(flattened,axis=0)
    to_box = to_range_max - to_range_min + 1
    flattend_indices = flattened - to_range_min
    selector = (flattend_indices[:,0],flattend_indices[:,1],flattend_indices[:,2])
    indice_mask = np.zeros(to_box,dtype=bool)
    if not addidional_mask is None:
        indice_mask[*selector] = addidional_mask.ravel()
    else:
        indice_mask[*selector] = True
    def converter(x:np.ndarray,default_value=0.0):
        new_array = np.full(to_box,default_value)
        new_array[*selector] = x.ravel() # remove_false_planes(new_array,indice_mask)
        return fill_empty_voxels_morphological(*compress_to_center(fill_empty_voxels_morphological(new_array,indice_mask),indice_mask),return_mask=True)
    return converter

def minecraft_command_escape(text):
    # Minecraftコマンドでエスケープが必要な文字
    escape_chars = {
        '\\': '\\\\',  # バックスラッシュ
        '"': '\\"',    # ダブルクォーテーション
    }
    
    # 文字列内の各文字をチェックし、必要に応じてエスケープ
    escaped_text = ''.join(escape_chars.get(char, char) for char in text)
    
    return escaped_text

import re

def to_title_case(input_string):
    # 単語を分割（アンダースコア、空白、ドットを区切り文字として使用）
    words = re.split(r'([_\s.]+)', input_string)
    
    # 各単語を処理
    processed_words = []
    for word in words:
        if re.match(r'[_\s.]+', word):  # 区切り文字はスペースに置換
            processed_words.append(' ')
        else:
            # キャメルケース、パスカルケース、数字、日本語を分割
            subwords = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+|[ぁ-んァ-ン一-龥]+|[-ー]', word)
            for i, subword in enumerate(subwords):
                if re.match(r'^[A-Za-z]+$', subword):  # 英語の単語
                    processed_words.append(subword.capitalize())
                elif re.match(r'^\d+$', subword):  # 数字
                    processed_words.append(subword)
                elif subword in ['-', 'ー']:  # ハイフンとマイナス記号
                    processed_words.append(subword)
                else:  # 日本語
                    processed_words.append(subword)
                
                # 数字、英語、日本語の間にスペースを挿入
                if i < len(subwords) - 1:
                    next_subword = subwords[i+1]
                    if (re.match(r'^[A-Za-z]+$', subword) and (re.match(r'^\d+$', next_subword) or re.match(r'[ぁ-んァ-ン一-龥]', next_subword))) or \
                       (re.match(r'^\d+$', subword) and (re.match(r'^[A-Za-z]+$', next_subword) or re.match(r'[ぁ-んァ-ン一-龥]', next_subword))) or \
                       (re.match(r'[ぁ-んァ-ン一-龥]', subword) and (re.match(r'^[A-Za-z]+$', next_subword) or re.match(r'^\d+$', next_subword))):
                        processed_words.append(' ')
    
    # 処理した単語を結合し、連続するスペースを1つに短縮
    result = re.sub(r'\s+', ' ', ''.join(processed_words))
    
    return result.strip()