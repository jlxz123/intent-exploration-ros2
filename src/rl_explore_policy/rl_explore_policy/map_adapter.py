import math

import cv2
import numpy as np


FREE_PIXEL = 0
UNKNOWN_PIXEL = 128
OCCUPIED_PIXEL = 255
EXPLORED_PIXEL = 255
AGENT_PIXEL = 128


def yaw_to_ccrl_orientation(yaw):
    ros_quadrant = int(round(float(yaw) / (math.pi * 0.5))) % 4
    return {0: 0, 1: 3, 2: 2, 3: 1}[ros_quadrant]


def _normalize_full_map_shape(shape):
    if shape is None or len(shape) < 2:
        return None
    rows = int(shape[0])
    cols = int(shape[1])
    if rows <= 0 or cols <= 0:
        return None
    return rows, cols


def _resize_nearest(patch, size=24):
    return cv2.resize(patch, dsize=(size, size), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def _project_to_size(value, dim_min, dim_max, size=24):
    span = max(int(dim_max) - int(dim_min), 1)
    scaled = int(round((float(value) - float(dim_min)) * (size - 1) / float(span)))
    return int(np.clip(scaled, 0, size - 1))


def _extract_local_patch(source, center_row, center_col, half_size=12, fill_value=UNKNOWN_PIXEL):
    padded = np.pad(
        source,
        ((half_size, half_size), (half_size, half_size)),
        mode="constant",
        constant_values=fill_value,
    )
    row = int(center_row) + half_size
    col = int(center_col) + half_size
    return padded[row - half_size : row + half_size, col - half_size : col + half_size].astype(np.uint8)


def _crop_bounds(explore_map):
    if not np.any(explore_map > 0):
        return 0, explore_map.shape[0] - 1, 0, explore_map.shape[1] - 1
    rows, cols = np.nonzero(explore_map > 0)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _map_point_to_anchor_cell(x, y, anchor_map_x, anchor_map_y, anchor_row, anchor_col, anchor_yaw, cell_size_m):
    dx = float(x) - float(anchor_map_x)
    dy = float(y) - float(anchor_map_y)
    cos_yaw = math.cos(float(anchor_yaw))
    sin_yaw = math.sin(float(anchor_yaw))
    anchor_x = cos_yaw * dx + sin_yaw * dy
    anchor_y = -sin_yaw * dx + cos_yaw * dy
    col = int(math.floor(float(anchor_col) + 0.5 + anchor_x / float(cell_size_m)))
    row = int(math.floor(float(anchor_row) + 0.5 - anchor_y / float(cell_size_m)))
    return row, col


def _scan_distance_at_angle(scan_msg, angle):
    ranges = np.asarray(scan_msg.ranges, dtype=np.float32)
    if ranges.size == 0:
        return math.inf
    increment = float(scan_msg.angle_increment)
    if abs(increment) < 1e-9:
        return math.inf
    index = int(round((float(angle) - float(scan_msg.angle_min)) / increment))
    if index < 0 or index >= ranges.size:
        return math.inf
    return float(ranges[index])


def _bounded_increment(array, row, col, limit):
    array[row, col] = min(int(limit), int(array[row, col]) + 1)


def _bounded_decrement(array, row, col):
    if int(array[row, col]) > 0:
        array[row, col] = int(array[row, col]) - 1


class CcrlRayStateBuilder:
    def __init__(
        self,
        cell_size_m=0.65,
        map_size=24,
        local_half_size=12,
        full_map_shape=None,
        laser_range_max_cells=50.0,
    ):
        shape = _normalize_full_map_shape(full_map_shape)
        if shape is None:
            raise ValueError("full_map_shape must contain positive rows and cols")
        self.cell_size_m = float(cell_size_m)
        self.map_size = int(map_size)
        self.local_half_size = int(local_half_size)
        self.full_map_shape = shape
        self.laser_range_max_cells = float(laser_range_max_cells)
        self.laser_range_max_m = self.laser_range_max_cells * self.cell_size_m
        self.relative_ray_angles = np.arange(
            -0.75 * math.pi,
            0.75 * math.pi + 1e-5,
            0.05 * math.pi,
            dtype=np.float32,
        )
        if self.relative_ray_angles.size != 31:
            self.relative_ray_angles = np.linspace(-0.75 * math.pi, 0.75 * math.pi, 31, dtype=np.float32)
        self.grid = np.full(shape, UNKNOWN_PIXEL, dtype=np.uint8)
        self.free_counts = np.zeros(shape, dtype=np.uint8)
        self.occupied_counts = np.zeros(shape, dtype=np.uint8)
        self.observation_count_limit = 3
        self.occupied_confirm_count = 2
        self.hit_endpoint_push_m = 0.15 * self.cell_size_m

    def _robot_cell(self, robot_x, robot_y, anchor_map_x, anchor_map_y, anchor_row, anchor_col, anchor_yaw):
        row, col = _map_point_to_anchor_cell(
            robot_x,
            robot_y,
            anchor_map_x,
            anchor_map_y,
            anchor_row,
            anchor_col,
            anchor_yaw,
            self.cell_size_m,
        )
        rows, cols = self.full_map_shape
        return int(np.clip(row, 0, rows - 1)), int(np.clip(col, 0, cols - 1))

    def _normalised_scan_distances(self, scan_msg):
        values = []
        sensor_max = min(float(scan_msg.range_max), self.laser_range_max_m)
        sensor_max = max(sensor_max, 1e-6)
        for ccrl_angle in self.relative_ray_angles:
            ros_angle = -float(ccrl_angle)
            distance = _scan_distance_at_angle(scan_msg, ros_angle)
            if not math.isfinite(distance):
                distance = sensor_max
            distance = float(np.clip(distance, float(scan_msg.range_min), sensor_max))
            values.append(float(np.clip(distance / max(self.laser_range_max_m, 1e-6), 0.0, 1.0)))
        return np.asarray(values, dtype=np.float32)

    def _refresh_cell(self, row, col):
        free_count = int(self.free_counts[row, col])
        occupied_count = int(self.occupied_counts[row, col])
        if occupied_count >= self.occupied_confirm_count or (occupied_count > 0 and free_count == 0):
            self.grid[row, col] = OCCUPIED_PIXEL
        elif free_count > 0:
            self.grid[row, col] = FREE_PIXEL
        else:
            self.grid[row, col] = UNKNOWN_PIXEL

    def _observe_free(self, row, col):
        _bounded_increment(self.free_counts, row, col, self.observation_count_limit)
        _bounded_decrement(self.occupied_counts, row, col)
        self._refresh_cell(row, col)

    def _observe_occupied(self, row, col):
        _bounded_increment(self.occupied_counts, row, col, self.observation_count_limit)
        self._refresh_cell(row, col)

    def _cell_from_map_point(self, x, y, anchor_map_x, anchor_map_y, anchor_row, anchor_col, anchor_yaw):
        row, col = _map_point_to_anchor_cell(
            x,
            y,
            anchor_map_x,
            anchor_map_y,
            anchor_row,
            anchor_col,
            anchor_yaw,
            self.cell_size_m,
        )
        rows, cols = self.full_map_shape
        if not (0 <= row < rows and 0 <= col < cols):
            return None
        return int(row), int(col)

    def _update_ray_from_sensor_pose(
        self,
        sensor_x,
        sensor_y,
        ray_yaw,
        distance_m,
        has_hit,
        anchor_map_x,
        anchor_map_y,
        anchor_row,
        anchor_col,
        anchor_yaw,
    ):
        distance_cells = float(np.clip(distance_m / max(self.cell_size_m, 1e-6), 0.0, self.laser_range_max_cells))
        end_step = max(1, int(math.ceil(distance_cells)))
        sin_yaw = math.sin(float(ray_yaw))
        cos_yaw = math.cos(float(ray_yaw))
        visited_free = set()

        for step in range(1, end_step + 1):
            ray_distance = float(step) * self.cell_size_m
            if has_hit and ray_distance >= float(distance_m):
                break
            ray_distance = min(ray_distance, float(distance_m))
            cell = self._cell_from_map_point(
                float(sensor_x) + ray_distance * cos_yaw,
                float(sensor_y) + ray_distance * sin_yaw,
                anchor_map_x,
                anchor_map_y,
                anchor_row,
                anchor_col,
                anchor_yaw,
            )
            if cell is None:
                break
            if cell in visited_free:
                continue
            visited_free.add(cell)
            self._observe_free(cell[0], cell[1])

        if has_hit:
            endpoint_distance = min(
                self.laser_range_max_m,
                max(0.0, float(distance_m) + self.hit_endpoint_push_m),
            )
            cell = self._cell_from_map_point(
                float(sensor_x) + endpoint_distance * cos_yaw,
                float(sensor_y) + endpoint_distance * sin_yaw,
                anchor_map_x,
                anchor_map_y,
                anchor_row,
                anchor_col,
                anchor_yaw,
            )
            if cell is not None:
                self._observe_occupied(cell[0], cell[1])

    def _update_grid(
        self,
        scan_msg,
        robot_cell,
        sensor_x,
        sensor_y,
        sensor_yaw,
        anchor_map_x,
        anchor_map_y,
        anchor_row,
        anchor_col,
        anchor_yaw,
    ):
        self._observe_free(robot_cell[0], robot_cell[1])
        sensor_max = min(float(scan_msg.range_max), self.laser_range_max_m)
        sensor_max = max(sensor_max, 1e-6)
        hit_limit = sensor_max - max(0.02, 0.02 * self.cell_size_m)

        for ccrl_angle in self.relative_ray_angles:
            ros_angle = -float(ccrl_angle)
            distance = _scan_distance_at_angle(scan_msg, ros_angle)
            has_hit = math.isfinite(distance) and float(scan_msg.range_min) <= distance < hit_limit
            if not math.isfinite(distance):
                distance = sensor_max
            distance = float(np.clip(distance, float(scan_msg.range_min), sensor_max))
            self._update_ray_from_sensor_pose(
                sensor_x,
                sensor_y,
                float(sensor_yaw) + ros_angle,
                distance,
                has_hit,
                anchor_map_x,
                anchor_map_y,
                anchor_row,
                anchor_col,
                anchor_yaw,
            )

    def _build_s_map(self, robot_cell):
        explore_map = (self.grid != UNKNOWN_PIXEL).astype(np.uint8) * EXPLORED_PIXEL
        row_min, row_max, col_min, col_max = _crop_bounds(explore_map)
        global_map = _resize_nearest(explore_map[row_min : row_max + 1, col_min : col_max + 1], self.map_size)
        pos_row = _project_to_size(robot_cell[0], row_min, row_max, self.map_size)
        pos_col = _project_to_size(robot_cell[1], col_min, col_max, self.map_size)
        global_map[
            np.clip(pos_row - 1, 0, self.map_size) : np.clip(pos_row + 2, 0, self.map_size),
            np.clip(pos_col - 1, 0, self.map_size) : np.clip(pos_col + 2, 0, self.map_size),
        ] = AGENT_PIXEL

        local_map = _extract_local_patch(
            self.grid,
            robot_cell[0],
            robot_cell[1],
            half_size=self.local_half_size,
            fill_value=UNKNOWN_PIXEL,
        )
        if local_map.shape != (self.map_size, self.map_size):
            local_map = _resize_nearest(local_map, self.map_size)

        zero_map = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
        return np.stack([global_map, local_map, zero_map, zero_map.copy()], axis=0).astype(np.uint8)

    def build_from_scan(
        self,
        scan_msg,
        robot_x,
        robot_y,
        yaw,
        anchor_map_x,
        anchor_map_y,
        anchor_row,
        anchor_col,
        anchor_yaw=0.0,
        sensor_x=None,
        sensor_y=None,
        sensor_yaw=None,
    ):
        robot_cell = self._robot_cell(
            robot_x,
            robot_y,
            anchor_map_x,
            anchor_map_y,
            anchor_row,
            anchor_col,
            anchor_yaw,
        )
        orientation = yaw_to_ccrl_orientation(float(yaw) - float(anchor_yaw))
        if sensor_x is None:
            sensor_x = robot_x
        if sensor_y is None:
            sensor_y = robot_y
        if sensor_yaw is None:
            sensor_yaw = yaw
        self._update_grid(
            scan_msg,
            robot_cell,
            sensor_x,
            sensor_y,
            sensor_yaw,
            anchor_map_x,
            anchor_map_y,
            anchor_row,
            anchor_col,
            anchor_yaw,
        )
        s_map = self._build_s_map(robot_cell)
        s_sensor = np.concatenate(
            [
                self._normalised_scan_distances(scan_msg),
                np.asarray([float(orientation) / 4.0], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)
        return s_map, self.grid.copy(), robot_cell, orientation, s_sensor
