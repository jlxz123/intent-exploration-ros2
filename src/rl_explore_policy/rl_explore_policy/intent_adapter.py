from collections import deque

import cv2
import numpy as np


FREE_PIXEL = 0
UNKNOWN_PIXEL = 128
OCCUPIED_PIXEL = 255
INTENT_PIXEL = 255
AGENT_PIXEL = 128

RELATIVE_DIRECTION_NAMES = ("front", "front_right", "right", "back_right", "back", "back_left", "left", "front_left")
WORLD_DIRECTION_NAMES = ("right", "down_right", "down", "down_left", "left", "up_left", "up", "up_right")
WORLD_DIRECTION_DELTAS = np.asarray(
    (
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    ),
    dtype=np.int32,
)
MOVE_DELTAS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _resize_nearest(patch, size=24):
    return cv2.resize(patch, dsize=(size, size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def _project_to_size(value, dim_min, dim_max, size=24):
    span = max(int(dim_max) - int(dim_min), 1)
    scaled = int(round((float(value) - float(dim_min)) * (size - 1) / float(span)))
    return int(np.clip(scaled, 0, size - 1))


def _extract_local_patch(source, center_row, center_col, half_size=12, fill_value=0):
    padded = np.pad(
        source,
        ((half_size, half_size), (half_size, half_size)),
        mode="constant",
        constant_values=fill_value,
    )
    row = int(center_row) + half_size
    col = int(center_col) + half_size
    return padded[row - half_size : row + half_size, col - half_size : col + half_size].astype(np.uint8)


def _crop_bounds(explore_map, active_intent_mask=None):
    if active_intent_mask is not None and np.any(active_intent_mask):
        crop_mask = (explore_map > 0) | active_intent_mask
    else:
        crop_mask = explore_map > 0

    rows, cols = np.nonzero(crop_mask)
    if rows.size == 0:
        return 0, explore_map.shape[0] - 1, 0, explore_map.shape[1] - 1
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


class CcrlIntentAdapter:
    def __init__(
        self,
        full_map_shape,
        valid_area_mask=None,
        map_size=24,
        local_half_size=12,
        intent_square_size=6,
        intent_max_cells=36,
        intent_field_decay_radius=12,
        intent_done_threshold=0.90,
    ):
        self.full_map_shape = tuple(int(v) for v in full_map_shape)
        self.map_size = int(map_size)
        self.local_half_size = int(local_half_size)
        self.intent_square_size = int(intent_square_size)
        self.intent_max_cells = int(intent_max_cells)
        self.intent_field_decay_radius = int(intent_field_decay_radius)
        self.intent_done_threshold = float(intent_done_threshold)

        if valid_area_mask is None:
            self.valid_area_mask = np.ones(self.full_map_shape, dtype=bool)
        else:
            valid = np.asarray(valid_area_mask, dtype=bool)
            if valid.shape != self.full_map_shape:
                raise ValueError("valid_area_mask shape must match full_map_shape")
            self.valid_area_mask = valid

        self.active_intent_mask = np.zeros(self.full_map_shape, dtype=bool)
        self.active_frontier_mask = np.zeros(self.full_map_shape, dtype=bool)
        self.active_relative_direction = None
        self.active_world_direction = None
        self.active_seed = None

    def clear(self):
        self.active_intent_mask.fill(False)
        self.active_frontier_mask.fill(False)
        self.active_relative_direction = None
        self.active_world_direction = None
        self.active_seed = None

    def has_active_intent(self):
        return self.active_relative_direction is not None and np.any(self.active_intent_mask)

    def _relative_to_world_direction_index(self, orientation, relative_direction_index):
        return (int(orientation) * 2 + int(relative_direction_index)) % 8

    def _compute_known_reachable_free_mask(self, grid, robot_cell):
        reachable = np.zeros(self.full_map_shape, dtype=bool)
        start_row = int(robot_cell[0])
        start_col = int(robot_cell[1])
        if not self._in_bounds(start_row, start_col) or int(grid[start_row, start_col]) != FREE_PIXEL:
            return reachable

        queue = deque([(start_row, start_col)])
        reachable[start_row, start_col] = True
        while queue:
            row, col = queue.popleft()
            for delta_row, delta_col in MOVE_DELTAS:
                next_row = row + int(delta_row)
                next_col = col + int(delta_col)
                if not self._in_bounds(next_row, next_col):
                    continue
                if reachable[next_row, next_col]:
                    continue
                if int(grid[next_row, next_col]) != FREE_PIXEL:
                    continue
                reachable[next_row, next_col] = True
                queue.append((next_row, next_col))
        return reachable

    def _compute_frontier_belt(self, grid, robot_cell, region_mask):
        belt_mask = np.zeros(self.full_map_shape, dtype=bool)
        region_unknown_mask = region_mask & (grid == UNKNOWN_PIXEL) & self.valid_area_mask
        if not np.any(region_unknown_mask):
            return belt_mask

        reachable_mask = self._compute_known_reachable_free_mask(grid, robot_cell)
        if not np.any(reachable_mask):
            return belt_mask

        visited = np.zeros(self.full_map_shape, dtype=bool)
        queue = deque()
        for row, col in np.argwhere(region_unknown_mask):
            row = int(row)
            col = int(col)
            visited[row, col] = True
            queue.append((row, col, 0))

        goal_layer = None
        frontier_cells = []
        while queue:
            row, col, distance = queue.popleft()
            if goal_layer is not None and distance >= goal_layer:
                continue

            for delta_row, delta_col in MOVE_DELTAS:
                next_row = row + int(delta_row)
                next_col = col + int(delta_col)
                if not self._in_bounds(next_row, next_col):
                    continue
                if visited[next_row, next_col]:
                    continue
                if not self.valid_area_mask[next_row, next_col]:
                    continue
                if int(grid[next_row, next_col]) == OCCUPIED_PIXEL:
                    continue

                visited[next_row, next_col] = True
                next_distance = int(distance) + 1
                if reachable_mask[next_row, next_col]:
                    if goal_layer is None:
                        goal_layer = next_distance
                    if next_distance == goal_layer:
                        frontier_cells.append((next_row, next_col))
                    continue

                if goal_layer is None:
                    queue.append((next_row, next_col, next_distance))

        if frontier_cells:
            frontier_cells = np.asarray(frontier_cells, dtype=np.int32)
            belt_mask[frontier_cells[:, 0], frontier_cells[:, 1]] = True
        return belt_mask

    def _compute_frontier_unknown_mask(self, grid):
        unknown_mask = (grid == UNKNOWN_PIXEL) & self.valid_area_mask
        known_free_mask = grid == FREE_PIXEL
        adjacent_known_free = np.zeros_like(unknown_mask, dtype=bool)
        adjacent_known_free[1:, :] |= known_free_mask[:-1, :]
        adjacent_known_free[:-1, :] |= known_free_mask[1:, :]
        adjacent_known_free[:, 1:] |= known_free_mask[:, :-1]
        adjacent_known_free[:, :-1] |= known_free_mask[:, 1:]
        return unknown_mask & adjacent_known_free

    def _find_ray_seed(self, grid, robot_cell, world_direction_index):
        delta = WORLD_DIRECTION_DELTAS[int(world_direction_index)]
        row = int(robot_cell[0]) + int(delta[0])
        col = int(robot_cell[1]) + int(delta[1])

        while self._in_bounds(row, col):
            if not self.valid_area_mask[row, col]:
                row += int(delta[0])
                col += int(delta[1])
                continue
            if int(grid[row, col]) == UNKNOWN_PIXEL:
                return row, col
            row += int(delta[0])
            col += int(delta[1])
        return None

    def _find_sector_seed(self, grid, robot_cell, world_direction_index, candidate_mask):
        candidate_rows, candidate_cols = np.nonzero(candidate_mask)
        if candidate_rows.size == 0:
            return None

        direction = WORLD_DIRECTION_DELTAS[int(world_direction_index)].astype(np.float32)
        direction = direction / np.linalg.norm(direction)
        row_delta = candidate_rows.astype(np.float32) - float(robot_cell[0])
        col_delta = candidate_cols.astype(np.float32) - float(robot_cell[1])
        distance = np.sqrt(row_delta**2 + col_delta**2)

        valid_distance = distance > 1e-6
        if not np.any(valid_distance):
            return None

        candidate_rows = candidate_rows[valid_distance]
        candidate_cols = candidate_cols[valid_distance]
        row_delta = row_delta[valid_distance]
        col_delta = col_delta[valid_distance]
        distance = distance[valid_distance]

        cosine = (row_delta * direction[0] + col_delta * direction[1]) / distance
        within_sector = cosine >= np.cos(np.pi / 8.0)
        if not np.any(within_sector):
            return None

        candidate_rows = candidate_rows[within_sector]
        candidate_cols = candidate_cols[within_sector]
        row_delta = row_delta[within_sector]
        col_delta = col_delta[within_sector]
        distance = distance[within_sector]

        lateral_distance = np.abs(row_delta * direction[1] - col_delta * direction[0])
        order = np.lexsort((lateral_distance, distance))
        best_index = int(order[0])
        return int(candidate_rows[best_index]), int(candidate_cols[best_index])

    def _find_frontier_seed(self, grid, robot_cell, world_direction_index):
        ray_seed = self._find_ray_seed(grid, robot_cell, world_direction_index)
        if ray_seed is not None:
            return ray_seed

        frontier_seed = self._find_sector_seed(
            grid,
            robot_cell,
            world_direction_index,
            self._compute_frontier_unknown_mask(grid),
        )
        if frontier_seed is not None:
            return frontier_seed

        fallback_unknown_mask = (grid == UNKNOWN_PIXEL) & self.valid_area_mask
        return self._find_sector_seed(grid, robot_cell, world_direction_index, fallback_unknown_mask)

    def _clamp_window(self, start, size, limit):
        if size >= limit:
            return 0, limit
        start = int(np.clip(start, 0, limit - size))
        return start, start + size

    def _preferred_window_origin(self, seed_row, seed_col, size, delta_row, delta_col):
        if delta_row > 0:
            row_start = seed_row
        elif delta_row < 0:
            row_start = seed_row - size + 1
        else:
            row_start = seed_row - size // 2

        if delta_col > 0:
            col_start = seed_col
        elif delta_col < 0:
            col_start = seed_col - size + 1
        else:
            col_start = seed_col - size // 2
        return row_start, col_start

    def _select_square_window(self, grid, seed_row, seed_col, size, delta_row, delta_col):
        row_min = max(0, seed_row - size + 1)
        row_max = min(seed_row, self.full_map_shape[0] - size)
        col_min = max(0, seed_col - size + 1)
        col_max = min(seed_col, self.full_map_shape[1] - size)

        preferred_row_start, preferred_col_start = self._preferred_window_origin(
            seed_row=seed_row,
            seed_col=seed_col,
            size=size,
            delta_row=delta_row,
            delta_col=delta_col,
        )

        unknown_mask = (grid == UNKNOWN_PIXEL) & self.valid_area_mask
        best_score = None
        best_window = None
        for row_start in range(row_min, row_max + 1):
            row_end = row_start + size
            for col_start in range(col_min, col_max + 1):
                col_end = col_start + size
                valid_count = int(self.valid_area_mask[row_start:row_end, col_start:col_end].sum())
                unknown_count = int(unknown_mask[row_start:row_end, col_start:col_end].sum())
                anchor_deviation = abs(row_start - preferred_row_start) + abs(col_start - preferred_col_start)
                score = (valid_count, unknown_count, -anchor_deviation)
                if best_score is None or score > best_score:
                    best_score = score
                    best_window = (row_start, row_end, col_start, col_end)

        if best_window is not None:
            return best_window

        row_start, row_end = self._clamp_window(preferred_row_start, size, self.full_map_shape[0])
        col_start, col_end = self._clamp_window(preferred_col_start, size, self.full_map_shape[1])
        return row_start, row_end, col_start, col_end

    def _build_directional_intent_mask(self, grid, world_direction_index, seed):
        mask = np.zeros(self.full_map_shape, dtype=bool)
        if seed is None:
            return mask

        seed_row, seed_col = int(seed[0]), int(seed[1])
        if not self._in_bounds(seed_row, seed_col) or not self.valid_area_mask[seed_row, seed_col]:
            return mask

        delta_row, delta_col = WORLD_DIRECTION_DELTAS[int(world_direction_index)]
        row_start, row_end, col_start, col_end = self._select_square_window(
            grid,
            seed_row=seed_row,
            seed_col=seed_col,
            size=self.intent_square_size,
            delta_row=int(delta_row),
            delta_col=int(delta_col),
        )

        mask[row_start:row_end, col_start:col_end] = self.valid_area_mask[row_start:row_end, col_start:col_end]
        if int(mask.sum()) > self.intent_max_cells:
            mask_indices = np.argwhere(mask)
            distance_sq = (mask_indices[:, 0].astype(np.float32) - float(seed_row)) ** 2 + (
                mask_indices[:, 1].astype(np.float32) - float(seed_col)
            ) ** 2
            selected_indices = mask_indices[np.argsort(distance_sq)[: self.intent_max_cells]]
            mask.fill(False)
            mask[selected_indices[:, 0], selected_indices[:, 1]] = True
        return mask

    def set_direction(self, relative_direction_index, grid, robot_cell, orientation):
        relative_direction_index = int(relative_direction_index) % 8
        world_direction_index = self._relative_to_world_direction_index(orientation, relative_direction_index)
        seed = self._find_frontier_seed(grid, robot_cell, world_direction_index)
        intent_mask = self._build_directional_intent_mask(grid, world_direction_index, seed)
        if not np.any(intent_mask):
            return None
        if not np.any(intent_mask & (grid == UNKNOWN_PIXEL)):
            return None

        frontier_mask = self._compute_frontier_belt(grid, robot_cell, intent_mask)
        if not np.any(frontier_mask):
            return None

        self.active_intent_mask = intent_mask.copy()
        self.active_frontier_mask = frontier_mask.copy()
        self.active_relative_direction = int(relative_direction_index)
        self.active_world_direction = int(world_direction_index)
        self.active_seed = None if seed is None else (int(seed[0]), int(seed[1]))
        return {
            "relative_index": int(relative_direction_index),
            "relative_name": RELATIVE_DIRECTION_NAMES[int(relative_direction_index)],
            "world_index": int(world_direction_index),
            "world_name": WORLD_DIRECTION_NAMES[int(world_direction_index)],
            "seed": self.active_seed,
            "frontier_size": int(frontier_mask.sum()),
            "intent_size": int(intent_mask.sum()),
        }

    def apply_to_s_map(self, s_map, grid, robot_cell):
        s_map = np.asarray(s_map, dtype=np.uint8).copy()
        if not self.has_active_intent():
            s_map[2].fill(0)
            s_map[3].fill(0)
            return s_map

        progress = self.intent_progress(grid)
        if progress >= self.intent_done_threshold:
            self.clear()
            s_map[2].fill(0)
            s_map[3].fill(0)
            return s_map

        self.active_frontier_mask = self._compute_frontier_belt(grid, robot_cell, self.active_intent_mask)
        explore_map = (grid != UNKNOWN_PIXEL).astype(np.uint8) * INTENT_PIXEL
        row_min, row_max, col_min, col_max = _crop_bounds(explore_map, self.active_intent_mask)

        global_map = _resize_nearest(explore_map[row_min : row_max + 1, col_min : col_max + 1], self.map_size)
        pos_row = _project_to_size(robot_cell[0], row_min, row_max, self.map_size)
        pos_col = _project_to_size(robot_cell[1], col_min, col_max, self.map_size)
        global_map[
            np.clip(pos_row - 1, 0, self.map_size) : np.clip(pos_row + 2, 0, self.map_size),
            np.clip(pos_col - 1, 0, self.map_size) : np.clip(pos_col + 2, 0, self.map_size),
        ] = AGENT_PIXEL
        s_map[0] = global_map

        global_intent_source = self._build_continuous_intent_field(self.active_intent_mask)
        global_intent_patch = global_intent_source[row_min : row_max + 1, col_min : col_max + 1]
        s_map[2] = _resize_nearest(global_intent_patch, self.map_size)

        frontier_source = self.active_frontier_mask.astype(np.uint8) * INTENT_PIXEL
        s_map[3] = _extract_local_patch(
            frontier_source,
            robot_cell[0],
            robot_cell[1],
            half_size=self.local_half_size,
            fill_value=0,
        )
        return s_map

    def intent_progress(self, grid):
        if not np.any(self.active_intent_mask):
            return 0.0
        total_cells = int(self.active_intent_mask.sum())
        if total_cells == 0:
            return 0.0
        explored_cells = int(np.count_nonzero(grid[self.active_intent_mask] != UNKNOWN_PIXEL))
        return explored_cells / float(total_cells)

    def _build_continuous_intent_field(self, intent_mask):
        if not np.any(intent_mask):
            return np.zeros(self.full_map_shape, dtype=np.uint8)

        rows = np.arange(self.full_map_shape[0], dtype=np.int32)[:, None]
        cols = np.arange(self.full_map_shape[1], dtype=np.int32)[None, :]
        distance = np.full(self.full_map_shape, fill_value=self.intent_field_decay_radius + 1, dtype=np.int32)
        for mask_row, mask_col in np.argwhere(intent_mask):
            distance = np.minimum(distance, np.abs(rows - int(mask_row)) + np.abs(cols - int(mask_col)))

        field = np.clip(1.0 - distance.astype(np.float32) / float(self.intent_field_decay_radius), 0.0, 1.0)
        field[intent_mask] = 1.0
        return (field * INTENT_PIXEL).astype(np.uint8)

    def _in_bounds(self, row, col):
        return 0 <= int(row) < self.full_map_shape[0] and 0 <= int(col) < self.full_map_shape[1]
