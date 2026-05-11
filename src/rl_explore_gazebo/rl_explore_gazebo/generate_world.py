import argparse
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


DEFAULT_CELL_SIZE_M = 0.65
DEFAULT_OBSTACLE_HEIGHT_M = 0.50


def _rectangles_from_mask(mask):
    """Greedy rectangle decomposition for occupied cells."""
    remaining = mask.copy()
    rows, cols = remaining.shape
    rectangles = []

    for row in range(rows):
        col = 0
        while col < cols:
            if not remaining[row, col]:
                col += 1
                continue

            col_end = col
            while col_end < cols and remaining[row, col_end]:
                col_end += 1

            row_end = row + 1
            while row_end < rows and np.all(remaining[row_end, col:col_end]):
                row_end += 1

            remaining[row:row_end, col:col_end] = False
            rectangles.append((row, row_end, col, col_end))
            col = col_end

    return rectangles


def _rect_pose(row_start, row_end, col_start, col_end, rows, cols, cell_size, obstacle_height):
    center_col = (float(col_start) + float(col_end)) * 0.5
    center_row = (float(row_start) + float(row_end)) * 0.5
    x = (center_col - float(cols) * 0.5) * cell_size
    y = (float(rows) * 0.5 - center_row) * cell_size
    z = obstacle_height * 0.5
    size_x = float(col_end - col_start) * cell_size
    size_y = float(row_end - row_start) * cell_size
    return x, y, z, size_x, size_y, obstacle_height


def build_world_text(map_array, world_name, cell_size, obstacle_height):
    rows, cols = map_array.shape
    occupied_mask = map_array != 0
    rectangles = _rectangles_from_mask(occupied_mask)
    ground_size_x = max(float(cols) * cell_size + 2.0, 5.0)
    ground_size_y = max(float(rows) * cell_size + 2.0, 5.0)

    parts = [
        '<?xml version="1.0"?>',
        '<sdf version="1.9">',
        f'  <world name="{escape(world_name)}">',
        "    <plugin filename=\"ignition-gazebo-physics-system\" name=\"ignition::gazebo::systems::Physics\"/>",
        "    <plugin filename=\"ignition-gazebo-user-commands-system\" name=\"ignition::gazebo::systems::UserCommands\"/>",
        "    <plugin filename=\"ignition-gazebo-scene-broadcaster-system\" name=\"ignition::gazebo::systems::SceneBroadcaster\"/>",
        "    <plugin filename=\"ignition-gazebo-sensors-system\" name=\"ignition::gazebo::systems::Sensors\">",
        "      <render_engine>ogre2</render_engine>",
        "    </plugin>",
        "    <light name=\"sun\" type=\"directional\">",
        "      <cast_shadows>true</cast_shadows>",
        "      <pose>0 0 10 0 0 0</pose>",
        "      <diffuse>0.8 0.8 0.8 1</diffuse>",
        "      <specular>0.2 0.2 0.2 1</specular>",
        "      <direction>-0.5 0.1 -0.9</direction>",
        "    </light>",
        "    <model name=\"ground_plane\">",
        "      <static>true</static>",
        "      <link name=\"link\">",
        "        <collision name=\"collision\">",
        "          <geometry>",
        f"            <box><size>{ground_size_x:.6f} {ground_size_y:.6f} 0.02</size></box>",
        "          </geometry>",
        "        </collision>",
        "        <visual name=\"visual\">",
        "          <geometry>",
        f"            <box><size>{ground_size_x:.6f} {ground_size_y:.6f} 0.02</size></box>",
        "          </geometry>",
        "          <material><ambient>0.9 0.9 0.9 1</ambient><diffuse>0.9 0.9 0.9 1</diffuse></material>",
        "        </visual>",
        "      </link>",
        "    </model>",
        "    <model name=\"rl_explore_obstacles\">",
        "      <static>true</static>",
    ]

    for index, (row_start, row_end, col_start, col_end) in enumerate(rectangles):
        x, y, z, size_x, size_y, size_z = _rect_pose(
            row_start,
            row_end,
            col_start,
            col_end,
            rows,
            cols,
            cell_size,
            obstacle_height,
        )
        parts.extend(
            [
                f"      <link name=\"wall_{index:04d}\">",
                f"        <pose>{x:.6f} {y:.6f} {z:.6f} 0 0 0</pose>",
                "        <collision name=\"collision\">",
                "          <geometry>",
                f"            <box><size>{size_x:.6f} {size_y:.6f} {size_z:.6f}</size></box>",
                "          </geometry>",
                "        </collision>",
                "        <visual name=\"visual\">",
                "          <geometry>",
                f"            <box><size>{size_x:.6f} {size_y:.6f} {size_z:.6f}</size></box>",
                "          </geometry>",
                "          <material><ambient>0.25 0.25 0.25 1</ambient><diffuse>0.35 0.35 0.35 1</diffuse></material>",
                "        </visual>",
                "      </link>",
            ]
        )

    parts.extend(["    </model>", "  </world>", "</sdf>", ""])
    return "\n".join(parts)


def generate_world(map_path, output_path, cell_size=DEFAULT_CELL_SIZE_M, obstacle_height=DEFAULT_OBSTACLE_HEIGHT_M):
    map_path = Path(map_path)
    output_path = Path(output_path)
    map_array = np.load(map_path)
    if map_array.ndim != 2:
        raise ValueError(f"Expected a 2D npy map, got shape {map_array.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    world_text = build_world_text(
        map_array=map_array,
        world_name=output_path.stem,
        cell_size=float(cell_size),
        obstacle_height=float(obstacle_height),
    )
    output_path.write_text(world_text, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate a Gazebo world from a CCRL npy map.")
    parser.add_argument("map_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--cell-size", type=float, default=DEFAULT_CELL_SIZE_M)
    parser.add_argument("--obstacle-height", type=float, default=DEFAULT_OBSTACLE_HEIGHT_M)
    args = parser.parse_args()

    output_path = generate_world(
        map_path=args.map_path,
        output_path=args.output_path,
        cell_size=args.cell_size,
        obstacle_height=args.obstacle_height,
    )
    print(output_path)


if __name__ == "__main__":
    main()
