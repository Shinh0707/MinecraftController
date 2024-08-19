import numpy as np
from typing import List, Tuple


def find_largest_cuboid(space: np.ndarray, value: int, start: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
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


def partition_3d_space(space: np.ndarray) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int], int]]:
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


def reconstruct_space(partitions: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], int]], shape: Tuple[int, int, int]) -> np.ndarray:
    reconstructed = np.zeros(shape, dtype=int)
    for (x1, y1, z1), (x2, y2, z2), value in partitions:
        reconstructed[x1:x2+1, y1:y2+1, z1:z2+1] = value
    return reconstructed


def verify_reconstruction(original: np.ndarray, reconstructed: np.ndarray) -> bool:
    return np.array_equal(original, reconstructed)


# Example usage with L-shaped region
space = np.array([
    [[1, 1, 2],
     [1, 3, 2],
     [1, 3, 2]],
    [[1, 1, 2],
     [1, 1, 2],
     [1, 3, 2]]
])

print("Original space:")
print(space)

result = partition_3d_space(space)
print(f"\nNumber of partitions: {len(result)}")
for i, box in enumerate(result):
    print(f"Partition {i}: from {box[0]} to {box[1]}, value: {box[2]}")

reconstructed_space = reconstruct_space(result, space.shape)

print("\nReconstructed space:")
print(reconstructed_space)

is_correct = verify_reconstruction(space, reconstructed_space)
print(f"\nReconstruction is correct: {is_correct}")

if not is_correct:
    print("\nDifferences:")
    diff = space - reconstructed_space
    print(diff)
    print("\nPositions where original and reconstructed spaces differ:")
    for x in range(space.shape[0]):
        for y in range(space.shape[1]):
            for z in range(space.shape[2]):
                if space[x, y, z] != reconstructed_space[x, y, z]:
                    print(f"Position ({x},{y},{z}): Original={space[x, y,z]}, Reconstructed={reconstructed_space[x,y,z]}")