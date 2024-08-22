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

def adaptive_optimize_midi_timing(X, dt, tempo_preservation_factor=0.9, length_flexibility=0.05):
    # X: note-onのタイミングを秒単位で表した配列
    # dt: 出力可能な最小時間間隔
    # window_size: 密度計算のウィンドウサイズ（秒）
    # tempo_preservation_factor: テンポ感維持の強さを制御するファクター (0-1)
    # length_flexibility: 全体の長さの許容変動幅 (0-1)

    # Step 1: Xをソートし、同時刻のノートをグループ化
    sorted_events = defaultdict(list)
    for i, x in enumerate(X):
        sorted_events[x].append(i)

    # Step 2: 局所的なテンポの計算
    times = sorted(sorted_events.keys())
    local_tempos = []
    for i in range(1, len(times)):
        interval = times[i] - times[i-1]
        local_tempos.append(1 / interval if interval > 0 else float('inf'))

    # Step 3: テンポ変化に基づく適応的な調整係数の計算
    adjustment_factors = []
    avg_tempo = np.mean([t for t in local_tempos if t != float('inf')])
    for tempo in local_tempos:
        if tempo == float('inf'):
            factor = 1.0
        else:
            tempo_ratio = tempo / avg_tempo
            factor = 1 + (tempo_ratio - 1) * (1 - tempo_preservation_factor)
        adjustment_factors.append(factor)

    # Step 4: テンポ感を維持しつつタイミングを最適化
    optimized_delays = []
    current_time = 0
    total_original_time = times[-1] - times[0]

    for i in range(len(times)):
        if i == 0:
            target_delay = times[0]
        else:
            target_delay = (times[i] - times[i-1]) * adjustment_factors[i-1]
        
        optimal_delay = round(target_delay / dt) * dt
        optimized_delays.append((optimal_delay, sorted_events[times[i]]))
        current_time += optimal_delay

    # Step 5: 全体の長さを調整（許容範囲内で）
    total_optimized_time = sum(delay for delay, _ in optimized_delays)
    length_ratio = total_optimized_time / total_original_time

    if abs(length_ratio - 1) > length_flexibility:
        scale_factor = (1 + np.sign(1 - length_ratio) * length_flexibility) / length_ratio
        optimized_delays = [(delay * scale_factor, notes) for delay, notes in optimized_delays]

    # Step 6: 結果を整形
    result = []
    for delay, notes in optimized_delays:
        result.append((max(1, int(round(delay / dt))), notes))  # 最小遅延を1に設定

    return result

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist

def optimize_histogram_offset(data: np.ndarray, bin_width: float, n_offsets: int = 100) -> Tuple[float, np.ndarray, np.ndarray]:
    data_min, data_max = np.min(data), np.max(data)
    n_bins = int(np.ceil((data_max - data_min) / bin_width))
    
    max_variance = -np.inf
    best_offset = 0
    best_hist = None
    best_bin_edges = None
    
    for offset in np.linspace(0, bin_width, n_offsets):
        # オフセットを適用してbin edgesを計算
        bin_edges = np.arange(data_min - offset, data_max + bin_width, bin_width)
        hist, _ = np.histogram(data, bins=bin_edges)
        
        # binのデータ点数の分散を計算
        variance = np.var(hist)
        
        if variance > max_variance:
            max_variance = variance
            best_offset = offset
            best_hist = hist
            best_bin_edges = bin_edges
    
    return best_offset, best_hist, best_bin_edges

def optimize_midi_timing(X: List[float], dt: float) -> List[Tuple[int, List[int]]]:
    X = np.array(sorted(X))
    
    # optimize_histogram_offset 関数を使用してヒストグラムを最適化
    best_offset, best_hist, best_bin_edges = optimize_histogram_offset(X, dt)
    
    # 最適化されたビンにノートを割り当て
    assigned_bins = np.digitize(X+best_offset, best_bin_edges)
    
    # 結果を格納するリスト
    result = []
    
    # 各ビンに割り当てられたノートをグループ化
    for bin_index in range(len(best_hist)):
        notes = np.where(assigned_bins == bin_index)[0].tolist()
        if notes:
            time = int(round((best_bin_edges[bin_index] + best_offset) / dt))
            result.append((time, notes))
    
    # 相対的な遅延時間に変換
    fin = []
    last_time = 0
    for time, notes in result:
        fin.append((time - last_time, notes))
        last_time = time
    
    return fin

def optimize_midi_timing_fast(X, dt):
    X = np.sort(np.array(X))
    max_clusters = min(len(X), int(X[-1] / dt) + 1)

    # Use pdist to pre-compute distances
    distances = pdist(X.reshape(-1, 1))
    Z = linkage(distances, method='ward')

    # Binary search for optimal threshold
    low, high = 0, X[-1]
    while high - low > dt / 100:
        threshold = (low + high) / 2
        clusters = fcluster(Z, threshold, criterion='distance')
        n_clusters = len(np.unique(clusters))
        
        if n_clusters > max_clusters:
            low = threshold
        else:
            high = threshold
    
    clusters = fcluster(Z, high, criterion='distance')
    n_clusters = len(np.unique(clusters))

    # Vectorized operations for cluster statistics
    cluster_ids = np.arange(1, n_clusters + 1)
    cluster_mins = np.array([np.min(X[clusters == i]) for i in cluster_ids])
    
    # Snap to grid and assign in one step
    snapped_times = np.round(cluster_mins / dt) * dt
    assigned_times = snapped_times[np.searchsorted(cluster_mins, X) - 1]

    # Efficient result creation
    unique_times, inverse = np.unique(assigned_times, return_inverse=True)
    delays = np.round(unique_times / dt).astype(int)
    result = [(int(delays[i] - (0 if i == 0 else delays[i-1])),
               np.where(inverse == i)[0].tolist()) for i in range(len(unique_times))]

    return result

from scipy.signal import find_peaks

def optimize_midi_timing_dynamic(X, dt):
    X = np.sort(np.array(X))
    
    def initial_grouping(X, dt):
        diffs = np.diff(X)
        peaks, _ = find_peaks(diffs, height=dt/2, distance=1)
        return np.split(X, peaks + 1)
    
    def group_cost(group, target_time):
        return np.sum((group - target_time)**2)
    
    def optimize_groups(groups, dt):
        optimized = []
        current_group = np.array([])
        current_time = groups[0][0] if isinstance(groups[0], np.ndarray) else groups[0][1][0]
        
        for group in groups:
            group_array = group if isinstance(group, np.ndarray) else group[1]
            if current_group.size == 0:
                current_group = group_array
                continue
            
            combined = np.concatenate([current_group, group_array])
            combined_center = np.mean(combined)
            next_time = current_time + dt
            
            cost_separate = (group_cost(current_group, current_time) + 
                             group_cost(group_array, next_time))
            cost_combined = group_cost(combined, next_time)
            
            if cost_combined < cost_separate:
                current_group = combined
            else:
                optimized.append((current_time, current_group))
                current_group = group_array
                current_time = next_time
        
        if current_group.size > 0:
            optimized.append((current_time, current_group))
        
        return optimized
    
    def refine_groups(groups, dt):
        refined = []
        for time, group in groups:
            if group.size > 1:
                subgroups = initial_grouping(group, dt/2)
                if len(subgroups) > 1:
                    refined.extend([(time, sg) for sg in subgroups])
                else:
                    refined.append((time, group))
            else:
                refined.append((time, group))
        return optimize_groups(refined, dt)
    
    initial_groups = initial_grouping(X, dt)
    optimized_groups = optimize_groups([(0, group) for group in initial_groups], dt)
    refined_groups = refine_groups(optimized_groups, dt)
    
    result = []
    last_time = refined_groups[0][0] - dt
    for time, group in refined_groups:
        delay = int(round((time - last_time) / dt))
        notes = np.where((X >= group[0]) & (X <= group[-1]))[0].tolist()
        result.append((delay, notes))
        last_time = time
    
    return result

from scipy.optimize import minimize

def optimize_midi_timing_uniform(X, dt):
    X = np.sort(np.array(X))
    
    # ピーク検出（前回と同じ）
    max_diff = np.max(np.diff(X))
    peaks, _ = find_peaks(-np.diff(X), height=-max_diff, distance=int(dt/np.mean(np.diff(X))))
    
    # 初期グループ化
    groups = np.split(X, peaks + 1)
    
    # 各グループの中央値を計算
    group_times = np.array([np.median(group) for group in groups if len(group) > 0])
    
    # 最適化関数
    def objective(offsets):
        adjusted_times = group_times + offsets
        intervals = np.diff(adjusted_times)
        
        # dtからの逸脱に対するペナルティ（二乗誤差）
        interval_penalty = np.sum((intervals - dt)**2)
        
        # 元のタイミングからの逸脱に対するペナルティ
        timing_penalty = np.sum(offsets**2)
        
        # 全体の時間範囲を保持するためのペナルティ
        range_penalty = (adjusted_times[-1] - adjusted_times[0] - (len(adjusted_times) - 1) * dt)**2
        
        return interval_penalty + 0.00 * timing_penalty + 0*range_penalty
    
    # 制約条件：隣接するグループ間の順序を保持
    def constraint(offsets):
        adjusted_times = group_times + offsets
        return np.diff(adjusted_times)  # すべての差が正であることを保証
    
    constraints = ({'type': 'ineq', 'fun': constraint})
    
    # 最適化
    initial_offsets = np.zeros_like(group_times)
    result = minimize(objective, initial_offsets, method='SLSQP', constraints=constraints)
    
    optimized_offsets = result.x
    adjusted_times = group_times + optimized_offsets
    
    # 結果の生成
    result = []
    last_time = adjusted_times[0] - dt  # 最初のグループの前に dt の間隔を設定
    for i, adj_time in enumerate(adjusted_times):
        delay = int(round((adj_time - last_time) / dt))
        notes = np.where((X >= groups[i][0]) & (X <= groups[i][-1]))[0].tolist()
        result.append((delay, notes))
        last_time = adj_time
    
    return result

import numpy as np

def group_notes(X, dt):
    """
    X: note-onの時間系列（秒単位）
    dt: 目標とする時間分解能（秒単位）
    
    返り値: (int(dtの数), list(ノートのインデックス))のリスト
    """
    result = []
    X = np.array(X)
    
    # 最初のグループを作成
    current_group = []
    current_time =0.0
    
    for i in range(len(X)):
        # 現在のノートと前のグループの開始時間との差
        time_diff = X[i] - current_time
        
        # 時間差がdtの0.5倍未満の場合、現在のグループに追加
        if time_diff < 0.5 * dt:
            current_group.append(i)
        else:
            # 新しいグループを開始する時間かどうかを判断
            if time_diff > dt:
                # 長い間隔がある場合、現在のグループを終了し、新しいグループを開始
                result.append((int(round(time_diff / dt)), current_group))
            else:
                # dtに近い間隔の場合、現在のグループを終了し、新しいグループを開始
                result.append((1, current_group))
            current_group = [i]
            current_time = X[i]
    
    # 最後のグループを追加
    if current_group:
        result.append((1, current_group))
    
    return result