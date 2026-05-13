import math
from argparse import Namespace
from pathlib import Path

from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
import cv2
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool, Int32
from tf2_ros import Buffer, TransformException, TransformListener
import torch

from .intent_adapter import CcrlIntentAdapter, RELATIVE_DIRECTION_NAMES
from .map_adapter import CcrlRayStateBuilder
from .network import load_actor_critic_checkpoint

CCRL_YAWS = (0.0, -0.5 * math.pi, math.pi, 0.5 * math.pi)


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def parameter_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def yaw_to_quaternion(yaw):
    q = Quaternion()
    half_yaw = float(yaw) * 0.5
    q.z = math.sin(half_yaw)
    q.w = math.cos(half_yaw)
    return q


def map_cell_to_world(row, col, rows, cols, cell_size_m):
    x = (float(col) + 0.5 - float(cols) * 0.5) * float(cell_size_m)
    y = (float(rows) * 0.5 - float(row) - 0.5) * float(cell_size_m)
    return x, y


def world_to_map_cell(x, y, rows, cols, cell_size_m):
    col = int(math.floor(float(x) / float(cell_size_m) + float(cols) * 0.5))
    row = int(math.floor(float(rows) * 0.5 - float(y) / float(cell_size_m)))
    return row, col


def map_index_for_eval_seed(map_name):
    try:
        level = int(str(map_name).rsplit("_l", 1)[1])
    except (IndexError, ValueError):
        return 0
    return max(0, level - 1)


def ccrl_spawn_seed(map_name, base_seed, eval_index):
    return int(base_seed) + map_index_for_eval_seed(map_name) * 1000 + int(eval_index)


def ccrl_random_spawn_pose(grid, map_name, base_seed, eval_index, cell_size_m):
    free_cells = np.argwhere(grid == 0)
    seed = ccrl_spawn_seed(map_name, base_seed, eval_index)
    if free_cells.size == 0:
        row, col = 0, 0
        orientation = 0
    else:
        rng = np.random.default_rng(seed)
        orientation = int(rng.integers(4))
        row, col = free_cells[int(rng.integers(len(free_cells)))]
        row, col = int(row), int(col)
    x, y = map_cell_to_world(row, col, int(grid.shape[0]), int(grid.shape[1]), cell_size_m)
    return float(x), float(y), float(CCRL_YAWS[orientation]), int(row), int(col), int(orientation), int(seed)


class CcrlPolicyNode(Node):
    def __init__(self):
        super().__init__("rl_explore_policy_node")

        default_checkpoint = str(
            Path(get_package_share_directory("rl_explore_policy")) / "checkpoints" / "final_dagger.pth"
        )
        self.declare_parameter("checkpoint_path", default_checkpoint)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("cell_size_m", 0.65)
        self.declare_parameter("plain_laser_range_max_cells", 50.0)
        self.declare_parameter("forward_speed", 0.22)
        self.declare_parameter("turn_speed", 1.20)
        self.declare_parameter("distance_tolerance", 0.025)
        self.declare_parameter("angle_tolerance", 0.025)
        self.declare_parameter("action_timeout", 8.0)
        self.declare_parameter("post_action_pause", 0.03)
        self.declare_parameter("front_stop_distance", 0.45)
        self.declare_parameter("min_forward_speed", 0.06)
        self.declare_parameter("drive_yaw_tolerance", 0.04)
        self.declare_parameter("lateral_tolerance", 0.04)
        self.declare_parameter("turn_gain", 2.8)
        self.declare_parameter("heading_gain", 2.4)
        self.declare_parameter("lateral_gain", 1.4)
        self.declare_parameter("max_lateral_yaw_correction", 0.22)
        self.declare_parameter("world_name", "")
        self.declare_parameter("model_name", "mbot")
        self.declare_parameter("snap_z", 0.0)
        self.declare_parameter("static_map_package", "rl_explore_gazebo")
        self.declare_parameter("static_map_name", "")
        self.declare_parameter("static_map_path", "")
        self.declare_parameter("world_spawn_x", "auto")
        self.declare_parameter("world_spawn_y", "auto")
        self.declare_parameter("world_spawn_yaw", "auto")
        self.declare_parameter("world_spawn_base_seed", 1)
        self.declare_parameter("world_spawn_eval_index", 0)
        self.declare_parameter("ccrl_map_rows", 80)
        self.declare_parameter("ccrl_map_cols", 80)
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("cmd_vel_topic", "/rl_explore/manual_cmd_vel")
        self.declare_parameter("nav_action_name", "navigate_to_pose")
        self.declare_parameter("debug_dump_dir", "")
        self.declare_parameter("show_ccrl_map_window", True)
        self.declare_parameter("ccrl_map_window_refresh_hz", 2.0)
        self.declare_parameter("ccrl_map_window_scale", 10)
        self.declare_parameter("map_image_topic", "/rl_explore/gui/map_image")
        self.declare_parameter("action_mode", "argmax")
        self.declare_parameter("action_execution_mode", "nav2")
        self.declare_parameter("intent_topic", "/rl_explore/intent_direction")
        self.declare_parameter("log_actions", True)

        requested_device = str(self.get_parameter("device").value)
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            self.get_logger().warn("CUDA is unavailable, falling back to CPU.")
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        self.cell_size_m = float(self.get_parameter("cell_size_m").value)
        self.plain_laser_range_max_cells = float(self.get_parameter("plain_laser_range_max_cells").value)
        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.turn_speed = float(self.get_parameter("turn_speed").value)
        self.distance_tolerance = float(self.get_parameter("distance_tolerance").value)
        self.angle_tolerance = float(self.get_parameter("angle_tolerance").value)
        self.action_timeout = float(self.get_parameter("action_timeout").value)
        self.post_action_pause = float(self.get_parameter("post_action_pause").value)
        self.front_stop_distance = float(self.get_parameter("front_stop_distance").value)
        self.min_forward_speed = float(self.get_parameter("min_forward_speed").value)
        self.drive_yaw_tolerance = float(self.get_parameter("drive_yaw_tolerance").value)
        self.lateral_tolerance = float(self.get_parameter("lateral_tolerance").value)
        self.turn_gain = float(self.get_parameter("turn_gain").value)
        self.heading_gain = float(self.get_parameter("heading_gain").value)
        self.lateral_gain = float(self.get_parameter("lateral_gain").value)
        self.max_lateral_yaw_correction = float(self.get_parameter("max_lateral_yaw_correction").value)
        self.world_name = str(self.get_parameter("world_name").value)
        self.model_name = str(self.get_parameter("model_name").value)
        self.snap_z = float(self.get_parameter("snap_z").value)
        self.static_map_package = str(self.get_parameter("static_map_package").value)
        self.static_map_name = str(self.get_parameter("static_map_name").value) or self.world_name
        self.static_map_path = str(self.get_parameter("static_map_path").value)
        self.world_spawn_x_arg = str(self.get_parameter("world_spawn_x").value)
        self.world_spawn_y_arg = str(self.get_parameter("world_spawn_y").value)
        self.world_spawn_yaw_arg = str(self.get_parameter("world_spawn_yaw").value)
        self.world_spawn_base_seed = int(self.get_parameter("world_spawn_base_seed").value)
        self.world_spawn_eval_index = int(self.get_parameter("world_spawn_eval_index").value)
        self.ccrl_map_rows_arg = int(self.get_parameter("ccrl_map_rows").value)
        self.ccrl_map_cols_arg = int(self.get_parameter("ccrl_map_cols").value)
        self.action_mode = str(self.get_parameter("action_mode").value)
        self.action_execution_mode = str(self.get_parameter("action_execution_mode").value)
        self.intent_topic = str(self.get_parameter("intent_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.nav_action_name = str(self.get_parameter("nav_action_name").value)
        self.log_actions = bool(self.get_parameter("log_actions").value)
        self.debug_dump_dir = str(self.get_parameter("debug_dump_dir").value)
        self.show_ccrl_map_window = parameter_to_bool(self.get_parameter("show_ccrl_map_window").value)
        self.ccrl_map_window_refresh_hz = float(self.get_parameter("ccrl_map_window_refresh_hz").value)
        self.ccrl_map_window_scale = int(self.get_parameter("ccrl_map_window_scale").value)
        self.map_image_topic = str(self.get_parameter("map_image_topic").value)
        self.ccrl_map_window_available = True
        self.ccrl_map_window_name = "CCRL s_map channels"
        self.ccrl_map_window_next_update = 0.0
        self.ccrl_map_window_last_image = None
        self.last_s_map = None
        self.debug_dump_index = 0
        self.action_index = 0
        self.static_grid = None
        self.static_grid_shape = None
        self.world_spawn_pose = (0.0, 0.0)
        self.world_spawn_yaw = 0.0
        self.world_spawn_cell = (-1, -1)
        self.world_spawn_orientation = -1
        self.world_spawn_seed = ccrl_spawn_seed(self.static_map_name or self.world_name, self.world_spawn_base_seed, self.world_spawn_eval_index)
        self.load_static_world()
        self.ccrl_full_map_shape = self.static_grid_shape or (
            max(1, int(self.ccrl_map_rows_arg)),
            max(1, int(self.ccrl_map_cols_arg)),
        )
        self.ccrl_anchor_cell = self.resolve_ccrl_anchor_cell()
        self.ccrl_map_anchor = None

        self.state_builder = CcrlRayStateBuilder(
            cell_size_m=self.cell_size_m,
            full_map_shape=self.ccrl_full_map_shape,
            laser_range_max_cells=self.plain_laser_range_max_cells,
        )
        valid_area_mask = None if self.static_grid is None else np.asarray(self.static_grid) != 128
        self.intent_adapter = CcrlIntentAdapter(
            full_map_shape=self.ccrl_full_map_shape,
            valid_area_mask=valid_area_mask,
        )
        self.pending_intent_direction = None
        self.pending_intent_request_index = 0
        self.scan_msg = None
        self.ccrl_scan_msg = None
        self.odom_msg = None
        self.current_pose = None
        self.current_z = self.snap_z
        self.current_yaw = 0.0
        self.warned_tf_fallback = False
        self.warned_nav_tf = False
        self.warned_nav_server = False
        self.executing_action = None
        self.action_start_pose = None
        self.action_start_yaw = 0.0
        self.action_start_time = None
        self.action_target_pose = None
        self.action_target_yaw = 0.0
        self.action_phase = "idle"
        self.motion_anchor_pose = None
        self.motion_anchor_yaw = 0.0
        self.discrete_target_pose = None
        self.discrete_target_yaw = 0.0
        self.pause_until = None
        self.nav_goal_handle = None
        self.nav_send_goal_future = None
        self.nav_result_future = None
        self.nav_goal_target_pose = None
        self.nav_goal_target_yaw = 0.0
        self.nav_goal_action = None

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.action_pub = self.create_publisher(Int32, "/rl_explore/action", 10)
        self.target_pose_pub = self.create_publisher(PoseStamped, "/rl_explore/target_pose", 10)
        self.state_ready_pub = self.create_publisher(Bool, "/rl_explore/state_ready", 10)
        self.map_image_pub = None
        if self.map_image_topic:
            self.map_image_pub = self.create_publisher(Image, self.map_image_topic, 2)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_client = ActionClient(self, NavigateToPose, self.nav_action_name)

        self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.create_subscription(LaserScan, "/ccrl_scan", self.ccrl_scan_callback, 10)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 20)
        self.create_subscription(Int32, self.intent_topic, self.intent_direction_callback, 10)

        checkpoint_path = str(self.get_parameter("checkpoint_path").value)
        config = Namespace(
            s_map_dim=(4, 24, 24),
            s_sensor_dim=(32,),
            action_dim=3,
            hidden_dim=32,
        )
        self.net = load_actor_critic_checkpoint(config, checkpoint_path, self.device)
        self.net.eval()
        self.get_logger().info(f"Loaded checkpoint: {checkpoint_path}")
        self.get_logger().info(f"device={self.device}")
        self.get_logger().info(f"action_execution_mode={self.action_execution_mode}")
        self.get_logger().info(f"cmd_vel_topic={self.cmd_vel_topic}")
        self.get_logger().info(f"intent_topic={self.intent_topic}")
        if self.map_image_pub is not None:
            self.get_logger().info(f"map_image_topic={self.map_image_topic}")
        if self.action_execution_mode == "nav2":
            self.get_logger().info(f"nav_action_name={self.nav_action_name}")
        self.get_logger().info(
            "world_spawn=({:.3f},{:.3f}) yaw={:.3f} cell=({},{}) orientation={} seed={} base_seed={} eval_index={}".format(
                float(self.world_spawn_pose[0]),
                float(self.world_spawn_pose[1]),
                float(self.world_spawn_yaw),
                int(self.world_spawn_cell[0]),
                int(self.world_spawn_cell[1]),
                int(self.world_spawn_orientation),
                int(self.world_spawn_seed),
                int(self.world_spawn_base_seed),
                int(self.world_spawn_eval_index),
            )
        )

        rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(rate_hz, 1.0), self.timer_callback)

    def scan_callback(self, msg):
        self.scan_msg = msg

    def ccrl_scan_callback(self, msg):
        self.ccrl_scan_msg = msg

    def intent_direction_callback(self, msg):
        direction = int(msg.data)
        if direction < -1 or direction > 7:
            self.get_logger().warn(f"Ignoring invalid intent direction: {direction}")
            return
        self.pending_intent_direction = direction
        self.pending_intent_request_index += 1
        if direction == -1:
            self.get_logger().info("Queued intent clear for the next policy decision.")
        else:
            self.get_logger().info(
                "Queued intent direction {} ({}) for the next policy decision.".format(
                    direction,
                    RELATIVE_DIRECTION_NAMES[direction],
                )
            )

    def odom_callback(self, msg):
        self.odom_msg = msg
        position = msg.pose.pose.position
        self.current_pose = (float(position.x), float(position.y))
        self.current_z = float(position.z)
        self.current_yaw = quaternion_to_yaw(msg.pose.pose.orientation)

    def resolve_static_map_path(self):
        if self.static_map_path:
            return Path(self.static_map_path)
        if not self.static_map_name:
            return None
        try:
            package_dir = Path(get_package_share_directory(self.static_map_package))
        except Exception as error:
            self.get_logger().warn(f"Static map package unavailable: {self.static_map_package}. {error}")
            return None
        return package_dir / "maps" / f"{self.static_map_name}.npy"

    def parse_auto_float(self, value):
        text = str(value)
        if text == "auto" or not text:
            return None
        return float(text)

    def load_static_world(self):
        map_path = self.resolve_static_map_path()
        if map_path is not None and map_path.exists():
            self.static_grid = np.load(map_path)
            self.static_grid_shape = tuple(int(v) for v in self.static_grid.shape)
            self.get_logger().info(f"Loaded static Gazebo grid for pose guard: {map_path}")
        else:
            self.get_logger().warn("No static Gazebo grid loaded; discrete pose guard will use scan only.")

        spawn_x = self.parse_auto_float(self.world_spawn_x_arg)
        spawn_y = self.parse_auto_float(self.world_spawn_y_arg)
        spawn_yaw = self.parse_auto_float(self.world_spawn_yaw_arg)
        if self.static_grid is not None and (spawn_x is None or spawn_y is None or spawn_yaw is None):
            auto_x, auto_y, auto_yaw, row, col, orientation, seed = ccrl_random_spawn_pose(
                self.static_grid,
                self.static_map_name or self.world_name,
                self.world_spawn_base_seed,
                self.world_spawn_eval_index,
                self.cell_size_m,
            )
            self.world_spawn_cell = (int(row), int(col))
            self.world_spawn_orientation = int(orientation)
            self.world_spawn_seed = int(seed)
            if spawn_x is None:
                spawn_x = auto_x
            if spawn_y is None:
                spawn_y = auto_y
            if spawn_yaw is None:
                spawn_yaw = auto_yaw
        self.world_spawn_pose = (0.0 if spawn_x is None else float(spawn_x), 0.0 if spawn_y is None else float(spawn_y))
        self.world_spawn_yaw = 0.0 if spawn_yaw is None else float(spawn_yaw)
        if self.static_grid is not None and self.world_spawn_cell == (-1, -1):
            self.world_spawn_cell = self.static_world_cell(self.world_spawn_pose)

    def odom_pose_to_world(self, pose_xy, yaw):
        cos_yaw = math.cos(self.world_spawn_yaw)
        sin_yaw = math.sin(self.world_spawn_yaw)
        x = float(pose_xy[0])
        y = float(pose_xy[1])
        world_x = self.world_spawn_pose[0] + cos_yaw * x - sin_yaw * y
        world_y = self.world_spawn_pose[1] + sin_yaw * x + cos_yaw * y
        return (world_x, world_y), normalize_angle(self.world_spawn_yaw + float(yaw))

    def static_world_cell(self, world_pose):
        if self.static_grid is None:
            return None
        rows, cols = self.static_grid.shape
        return world_to_map_cell(world_pose[0], world_pose[1], rows, cols, self.cell_size_m)

    def static_world_pose_is_free(self, world_pose):
        if self.static_grid is None:
            return True
        cell = self.static_world_cell(world_pose)
        row, col = cell
        rows, cols = self.static_grid.shape
        if not (0 <= row < rows and 0 <= col < cols):
            return False
        return int(self.static_grid[row, col]) == 0

    def resolve_ccrl_anchor_cell(self):
        rows, cols = self.ccrl_full_map_shape
        if self.static_grid is not None:
            row, col = self.static_world_cell(self.world_spawn_pose)
            if 0 <= row < rows and 0 <= col < cols:
                return int(row), int(col)
        return int(rows // 2), int(cols // 2)

    def ensure_ccrl_map_anchor(self, map_pose):
        if self.ccrl_map_anchor is not None:
            return
        anchor_yaw = -float(self.world_spawn_yaw) if self.static_grid is not None else 0.0
        self.ccrl_map_anchor = (float(map_pose[0]), float(map_pose[1]), anchor_yaw)
        self.get_logger().info(
            "CCRL map anchor: map=({:.3f},{:.3f}) cell=({},{}) shape=({}, {}) "
            "anchor_yaw={:.3f} source=/ccrl_scan".format(
                float(self.ccrl_map_anchor[0]),
                float(self.ccrl_map_anchor[1]),
                int(self.ccrl_anchor_cell[0]),
                int(self.ccrl_anchor_cell[1]),
                int(self.ccrl_full_map_shape[0]),
                int(self.ccrl_full_map_shape[1]),
                float(anchor_yaw),
            )
        )

    def ready(self):
        state_ready = self.ccrl_scan_msg is not None and self.current_pose is not None
        if self.action_execution_mode == "nav2":
            return state_ready and self.nav2_ready() and self.lookup_map_robot_pose(log_warning=False) is not None
        return state_ready

    def nav2_ready(self):
        ready = self.nav_client.server_is_ready()
        if not ready and not self.warned_nav_server:
            self.get_logger().info(f"Waiting for Nav2 action server: {self.nav_action_name}")
            self.warned_nav_server = True
        if ready:
            self.warned_nav_server = False
        return ready

    def publish_status(self):
        ready_msg = Bool()
        ready_msg.data = self.ready()
        self.state_ready_pub.publish(ready_msg)

    def decision_pose(self):
        map_pose = self.lookup_map_robot_pose(log_warning=False)
        if map_pose is None:
            map_pose = self.map_frame_pose()
        self.ensure_motion_anchor((map_pose[0], map_pose[1]), map_pose[2])
        return self.motion_anchor_pose[0], self.motion_anchor_pose[1], self.motion_anchor_yaw

    def build_ccrl_map_components(self):
        robot_x, robot_y, map_yaw = self.decision_pose()
        sensor_pose = self.lookup_map_robot_pose(log_warning=False)
        if sensor_pose is None:
            sensor_pose = self.map_frame_pose()
        self.ensure_ccrl_map_anchor((robot_x, robot_y, map_yaw))
        anchor_x, anchor_y, anchor_yaw = self.ccrl_map_anchor
        return self.state_builder.build_from_scan(
            self.ccrl_scan_msg,
            robot_x=robot_x,
            robot_y=robot_y,
            yaw=map_yaw,
            anchor_map_x=anchor_x,
            anchor_map_y=anchor_y,
            anchor_row=self.ccrl_anchor_cell[0],
            anchor_col=self.ccrl_anchor_cell[1],
            anchor_yaw=anchor_yaw,
            sensor_x=sensor_pose[0],
            sensor_y=sensor_pose[1],
            sensor_yaw=sensor_pose[2],
        )

    def consume_pending_intent(self, grid, robot_cell, orientation):
        if self.pending_intent_direction is None:
            return

        direction = int(self.pending_intent_direction)
        self.pending_intent_direction = None
        if direction == -1:
            self.intent_adapter.clear()
            self.get_logger().info("Applied intent clear at policy decision boundary.")
            return

        intent_info = self.intent_adapter.set_direction(direction, grid, robot_cell, orientation)
        if intent_info is None:
            self.get_logger().warn(
                "Intent direction {} ({}) has no valid frontier region at this decision; keeping previous intent.".format(
                    direction,
                    RELATIVE_DIRECTION_NAMES[direction],
                )
            )
            return

        self.get_logger().info(
            "Applied intent direction {} ({}) -> world {} seed={} intent_cells={} frontier_cells={}".format(
                intent_info["relative_index"],
                intent_info["relative_name"],
                intent_info["world_name"],
                intent_info["seed"],
                intent_info["intent_size"],
                intent_info["frontier_size"],
            )
        )

    def build_state(self):
        s_map, grid, robot_cell, orientation, s_sensor = self.build_ccrl_map_components()
        self.consume_pending_intent(grid, robot_cell, orientation)
        s_map = self.intent_adapter.apply_to_s_map(s_map, grid, robot_cell)
        self.last_s_map = s_map
        return {"s_map": s_map, "s_sensor": s_sensor}

    def render_s_map_channels_image(self, s_map):
        scale = max(2, int(self.ccrl_map_window_scale))
        title_h = 22
        gap = 8
        labels = (
            "ch0 global built",
            "ch1 local built",
            "ch2 global intent",
            "ch3 local frontier",
        )
        panels = []
        for index, label in enumerate(labels):
            channel = np.asarray(s_map[index], dtype=np.uint8)
            panel = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)
            panel = cv2.resize(
                panel,
                (channel.shape[1] * scale, channel.shape[0] * scale),
                interpolation=cv2.INTER_NEAREST,
            )
            for y in range(0, panel.shape[0] + 1, scale):
                cv2.line(panel, (0, y), (panel.shape[1], y), (55, 55, 55), 1)
            for x in range(0, panel.shape[1] + 1, scale):
                cv2.line(panel, (x, 0), (x, panel.shape[0]), (55, 55, 55), 1)

            titled = np.zeros((panel.shape[0] + title_h, panel.shape[1], 3), dtype=np.uint8)
            titled[title_h:, :] = panel
            cv2.putText(
                titled,
                label,
                (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
            panels.append(titled)

        panel_h, panel_w = panels[0].shape[:2]
        image = np.full((panel_h * 2 + gap, panel_w * 2 + gap, 3), 25, dtype=np.uint8)
        image[0:panel_h, 0:panel_w] = panels[0]
        image[0:panel_h, panel_w + gap : panel_w * 2 + gap] = panels[1]
        image[panel_h + gap : panel_h * 2 + gap, 0:panel_w] = panels[2]
        image[panel_h + gap : panel_h * 2 + gap, panel_w + gap : panel_w * 2 + gap] = panels[3]
        return image

    def show_ccrl_built_map(self, force=False):
        if not self.show_ccrl_map_window and self.map_image_pub is None:
            return

        now = self.now_seconds()
        if not force and now < self.ccrl_map_window_next_update:
            if self.show_ccrl_map_window and self.ccrl_map_window_available and self.ccrl_map_window_last_image is not None:
                cv2.waitKey(1)
            return

        if self.last_s_map is None:
            if self.show_ccrl_map_window and self.ccrl_map_window_available and self.ccrl_map_window_last_image is not None:
                cv2.waitKey(1)
            return

        image = self.render_s_map_channels_image(self.last_s_map)
        self.ccrl_map_window_last_image = image
        refresh_hz = max(float(self.ccrl_map_window_refresh_hz), 0.1)
        self.ccrl_map_window_next_update = now + 1.0 / refresh_hz
        self.publish_map_image(image)

        if not self.show_ccrl_map_window or not self.ccrl_map_window_available:
            return
        try:
            cv2.imshow(self.ccrl_map_window_name, image)
            cv2.waitKey(1)
        except Exception as error:
            self.ccrl_map_window_available = False
            self.get_logger().warn(f"CCRL map debug window disabled: {error}")

    def publish_map_image(self, image):
        if self.map_image_pub is None:
            return
        if image.ndim != 3 or image.shape[2] != 3:
            return
        if not image.flags["C_CONTIGUOUS"]:
            image = image.copy()

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "ccrl_map"
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = int(image.shape[1] * 3)
        msg.data = image.tobytes()
        self.map_image_pub.publish(msg)

    def lookup_robot_pose(self, target_frame, log_warning=False):
        try:
            transform = self.tf_buffer.lookup_transform(target_frame, "base_footprint", Time())
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            return float(translation.x), float(translation.y), quaternion_to_yaw(rotation)
        except TransformException as error:
            if log_warning and not self.warned_nav_tf:
                self.get_logger().warn(f"TF {target_frame}->base_footprint unavailable for Nav2 goal. {error}")
                self.warned_nav_tf = True
            return None

    def lookup_map_robot_pose(self, log_warning=False):
        pose = self.lookup_robot_pose("map", log_warning=log_warning)
        if pose is not None:
            self.warned_nav_tf = False
        return pose

    def map_frame_pose(self):
        pose = self.lookup_robot_pose("map")
        if pose is not None:
            return pose
        else:
            if not self.warned_tf_fallback:
                self.get_logger().warn("TF map->base_footprint unavailable; using /odom pose for policy input.")
                self.warned_tf_fallback = True
            return self.current_pose[0], self.current_pose[1], self.current_yaw

    def maybe_dump_state(self, state):
        if not self.debug_dump_dir:
            return
        output_dir = Path(self.debug_dump_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / f"state_{self.debug_dump_index:05d}_s_map.npy", state["s_map"])
        np.save(output_dir / f"state_{self.debug_dump_index:05d}_s_sensor.npy", state["s_sensor"])
        self.debug_dump_index += 1

    def infer_action(self, state):
        with torch.no_grad():
            s_map = torch.from_numpy(state["s_map"]).unsqueeze(0).to(self.device)
            s_sensor = torch.from_numpy(state["s_sensor"]).unsqueeze(0).to(self.device)
            logits = self.net.actor(s_map, s_sensor)
            if self.action_mode == "sample":
                action = int(torch.distributions.Categorical(logits=logits).sample().item())
            else:
                action = int(torch.argmax(logits, dim=-1).item())
        msg = Int32()
        msg.data = action
        self.action_pub.publish(msg)
        if self.log_actions:
            logits_np = logits.detach().cpu().numpy()[0]
            self.get_logger().info(
                "policy step={} action={} logits=[{:.3f}, {:.3f}, {:.3f}] "
                "pose=({:.2f},{:.2f}) yaw={:.2f}".format(
                    self.action_index,
                    action,
                    float(logits_np[0]),
                    float(logits_np[1]),
                    float(logits_np[2]),
                    float(self.current_pose[0]),
                    float(self.current_pose[1]),
                    float(self.current_yaw),
                )
            )
            self.action_index += 1
        return action

    def snap_cardinal_yaw(self, yaw):
        quadrant = int(round(float(yaw) / (math.pi * 0.5)))
        return normalize_angle(float(quadrant) * math.pi * 0.5)

    def cardinal_direction(self, yaw):
        quadrant = int(round(float(yaw) / (math.pi * 0.5))) % 4
        return ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))[quadrant]

    def ensure_motion_anchor(self, pose_xy=None, yaw=None):
        if self.motion_anchor_pose is None:
            if pose_xy is None:
                pose_xy = self.current_pose
            if yaw is None:
                yaw = self.current_yaw
            self.motion_anchor_pose = (float(pose_xy[0]), float(pose_xy[1]))
            self.motion_anchor_yaw = self.snap_cardinal_yaw(yaw)

    def clamp_signed(self, value, limit):
        limit = abs(float(limit))
        return max(-limit, min(limit, float(value)))

    def now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def publish_target_pose(self, pose_xy, yaw, frame_id="odom"):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.pose.position.x = float(pose_xy[0])
        msg.pose.position.y = float(pose_xy[1])
        msg.pose.position.z = self.current_z if math.isfinite(self.current_z) else self.snap_z
        msg.pose.orientation = yaw_to_quaternion(yaw)
        self.target_pose_pub.publish(msg)

    def plan_action_target(self, action):
        self.ensure_motion_anchor()

        action = int(action)
        if action == 1:
            direction_x, direction_y = self.cardinal_direction(self.motion_anchor_yaw)
            return (
                (
                    self.motion_anchor_pose[0] + self.cell_size_m * direction_x,
                    self.motion_anchor_pose[1] + self.cell_size_m * direction_y,
                ),
                self.motion_anchor_yaw,
            )
        if action == 0:
            return self.motion_anchor_pose, normalize_angle(self.motion_anchor_yaw + math.pi * 0.5)
        if action == 2:
            return self.motion_anchor_pose, normalize_angle(self.motion_anchor_yaw - math.pi * 0.5)
        return self.motion_anchor_pose, self.motion_anchor_yaw

    def start_action(self, action):
        self.executing_action = int(action)
        self.action_start_pose = self.current_pose
        self.action_start_yaw = self.current_yaw
        self.action_start_time = self.get_clock().now()
        self.action_target_pose, self.action_target_yaw = self.plan_action_target(self.executing_action)

        if self.executing_action == 1:
            self.action_phase = "align"
        elif self.executing_action in (0, 2):
            self.action_phase = "turn"
        else:
            self.action_phase = "idle"

    def start_nav2_action(self, action):
        map_pose = self.lookup_map_robot_pose(log_warning=True)
        if map_pose is None:
            return

        self.ensure_motion_anchor((map_pose[0], map_pose[1]), map_pose[2])
        action = int(action)
        if action == 1 and self.front_is_blocked():
            self.get_logger().warn("Nav2 forward action blocked by front scan; re-planning from current policy state.")
            self.motion_anchor_pose = (float(map_pose[0]), float(map_pose[1]))
            self.motion_anchor_yaw = self.snap_cardinal_yaw(map_pose[2])
            if self.post_action_pause > 0.0:
                self.pause_until = self.now_seconds() + self.post_action_pause
            return

        target_pose, target_yaw = self.plan_action_target(action)
        frame_id = "map"
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = frame_id
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(target_pose[0])
        goal.pose.pose.position.y = float(target_pose[1])
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation = yaw_to_quaternion(target_yaw)

        self.executing_action = action
        self.action_start_pose = (float(map_pose[0]), float(map_pose[1]))
        self.action_start_yaw = float(map_pose[2])
        self.action_start_time = self.get_clock().now()
        self.nav_goal_action = action
        self.nav_goal_target_pose = target_pose
        self.nav_goal_target_yaw = target_yaw
        self.publish_target_pose(target_pose, target_yaw, frame_id=frame_id)

        self.nav_send_goal_future = self.nav_client.send_goal_async(goal)
        self.nav_send_goal_future.add_done_callback(self.handle_nav_goal_response)
        if self.log_actions:
            self.get_logger().info(
                "nav target action={} frame={} pose=({:.3f},{:.3f}) yaw={:.3f}".format(
                    action,
                    frame_id,
                    float(target_pose[0]),
                    float(target_pose[1]),
                    float(target_yaw),
                )
            )

    def handle_nav_goal_response(self, future):
        if future is not self.nav_send_goal_future:
            return
        self.nav_send_goal_future = None
        try:
            goal_handle = future.result()
        except Exception as error:
            self.get_logger().warn(f"Nav2 goal send failed: {error}")
            self.finish_nav2_action(completed=False, status_text="send_failed")
            return

        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 goal was rejected.")
            self.finish_nav2_action(completed=False, status_text="rejected")
            return

        self.nav_goal_handle = goal_handle
        self.nav_result_future = goal_handle.get_result_async()
        self.nav_result_future.add_done_callback(self.handle_nav_result)
        if self.log_actions:
            self.get_logger().info("Nav2 goal accepted.")

    def handle_nav_result(self, future):
        if future is not self.nav_result_future:
            return
        try:
            result = future.result()
            status = int(result.status)
        except Exception as error:
            self.get_logger().warn(f"Nav2 result failed: {error}")
            self.finish_nav2_action(completed=False, status_text="result_failed")
            return

        completed = status == GoalStatus.STATUS_SUCCEEDED
        self.finish_nav2_action(completed=completed, status_text=str(status))

    def finish_nav2_action(self, completed, status_text=""):
        action = self.nav_goal_action
        target_pose = self.nav_goal_target_pose
        target_yaw = self.nav_goal_target_yaw

        if completed and target_pose is not None:
            self.motion_anchor_pose = (float(target_pose[0]), float(target_pose[1]))
            self.motion_anchor_yaw = float(target_yaw)
        else:
            map_pose = self.lookup_map_robot_pose(log_warning=False)
            if map_pose is not None:
                self.motion_anchor_pose = (float(map_pose[0]), float(map_pose[1]))
                self.motion_anchor_yaw = self.snap_cardinal_yaw(map_pose[2])

        if self.log_actions:
            outcome = "succeeded" if completed else "failed"
            self.get_logger().info(f"Nav2 goal {outcome}: action={action}, status={status_text}")

        self.executing_action = None
        self.action_start_pose = None
        self.action_start_time = None
        self.action_target_pose = None
        self.action_phase = "idle"
        self.nav_goal_handle = None
        self.nav_send_goal_future = None
        self.nav_result_future = None
        self.nav_goal_target_pose = None
        self.nav_goal_action = None
        if self.post_action_pause > 0.0:
            self.pause_until = self.now_seconds() + self.post_action_pause

    def step_nav2_action(self):
        if self.executing_action is None:
            return
        if self.action_timed_out():
            self.get_logger().warn(f"Nav2 action {self.executing_action} timed out; canceling goal.")
            if self.nav_goal_handle is not None:
                self.nav_goal_handle.cancel_goal_async()
            self.finish_nav2_action(completed=False, status_text="timeout")

    def start_action_continuous(self, action):
        self.executing_action = int(action)
        self.action_start_pose = self.current_pose
        self.action_start_yaw = self.current_yaw
        self.action_start_time = self.get_clock().now()
        self.ensure_motion_anchor()

        if self.executing_action == 1:
            direction_x, direction_y = self.cardinal_direction(self.motion_anchor_yaw)
            self.action_target_pose = (
                self.motion_anchor_pose[0] + self.cell_size_m * direction_x,
                self.motion_anchor_pose[1] + self.cell_size_m * direction_y,
            )
            self.action_target_yaw = self.motion_anchor_yaw
            self.action_phase = "align"
        elif self.executing_action == 0:
            self.action_target_pose = self.motion_anchor_pose
            self.action_target_yaw = normalize_angle(self.motion_anchor_yaw + math.pi * 0.5)
            self.action_phase = "turn"
        elif self.executing_action == 2:
            self.action_target_pose = self.motion_anchor_pose
            self.action_target_yaw = normalize_angle(self.motion_anchor_yaw - math.pi * 0.5)
            self.action_phase = "turn"
        else:
            self.action_target_pose = self.motion_anchor_pose
            self.action_target_yaw = self.motion_anchor_yaw
            self.action_phase = "idle"

    def stop_action(self, completed=False):
        completed_action = self.executing_action
        target_pose = self.action_target_pose
        target_yaw = self.action_target_yaw
        self.cmd_pub.publish(Twist())
        if completed and completed_action == 1 and target_pose is not None:
            self.motion_anchor_pose = target_pose
            self.motion_anchor_yaw = target_yaw
        elif completed and completed_action in (0, 2):
            self.motion_anchor_yaw = target_yaw
        elif not completed and self.current_pose is not None:
            self.motion_anchor_pose = self.current_pose
            self.motion_anchor_yaw = self.snap_cardinal_yaw(self.current_yaw)

        if completed and self.post_action_pause > 0.0:
            self.pause_until = self.now_seconds() + self.post_action_pause
        self.executing_action = None
        self.action_start_pose = None
        self.action_start_time = None
        self.action_target_pose = None
        self.action_phase = "idle"

    def action_timed_out(self):
        if self.action_start_time is None:
            return False
        elapsed = (self.get_clock().now() - self.action_start_time).nanoseconds * 1e-9
        return elapsed > self.action_timeout

    def front_is_blocked(self):
        if self.scan_msg is None:
            return False
        scan = self.scan_msg
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return False
        half_width = math.radians(8.0)
        idx0 = int(round((0.0 - half_width - float(scan.angle_min)) / float(scan.angle_increment)))
        idx1 = int(round((0.0 + half_width - float(scan.angle_min)) / float(scan.angle_increment)))
        idx0 = max(0, min(idx0, ranges.size - 1))
        idx1 = max(0, min(idx1, ranges.size - 1))
        if idx0 > idx1:
            idx0, idx1 = idx1, idx0
        front = ranges[idx0 : idx1 + 1]
        front = front[np.isfinite(front)]
        if front.size == 0:
            return False
        return float(np.min(front)) < self.front_stop_distance

    def step_current_action(self):
        if self.executing_action is None:
            return
        if self.action_timed_out():
            self.get_logger().warn(f"Action {self.executing_action} timed out; stopping.")
            self.stop_action(completed=False)
            return

        twist = Twist()
        if self.executing_action == 1:
            if self.front_is_blocked():
                self.get_logger().warn("Forward action blocked by front scan; stopping.")
                self.stop_action(completed=False)
                return

            direction_x, direction_y = self.cardinal_direction(self.action_target_yaw)
            normal_x, normal_y = -direction_y, direction_x
            dx = self.current_pose[0] - self.motion_anchor_pose[0]
            dy = self.current_pose[1] - self.motion_anchor_pose[1]
            progress = dx * direction_x + dy * direction_y
            lateral_error = dx * normal_x + dy * normal_y
            remaining = self.cell_size_m - progress

            if remaining <= self.distance_tolerance:
                self.stop_action(completed=True)
                return

            yaw_error = normalize_angle(self.action_target_yaw - self.current_yaw)
            if self.action_phase == "align" or abs(yaw_error) > self.drive_yaw_tolerance:
                if abs(yaw_error) <= self.angle_tolerance:
                    self.action_phase = "drive"
                    self.cmd_pub.publish(Twist())
                    return
                twist.angular.z = self.clamp_signed(self.turn_gain * yaw_error, self.turn_speed)
                self.action_phase = "align"
                self.cmd_pub.publish(twist)
                return

            target_dx = self.action_target_pose[0] - self.current_pose[0]
            target_dy = self.action_target_pose[1] - self.current_pose[1]
            target_distance = math.sqrt(target_dx * target_dx + target_dy * target_dy)
            if remaining <= self.distance_tolerance or target_distance <= self.distance_tolerance:
                self.stop_action(completed=True)
                return

            twist.linear.x = min(self.forward_speed, max(self.min_forward_speed, remaining * 1.4))
            correction = self.heading_gain * yaw_error - self.lateral_gain * lateral_error
            if abs(lateral_error) <= self.lateral_tolerance:
                correction = self.heading_gain * yaw_error
            twist.angular.z = self.clamp_signed(correction, self.max_lateral_yaw_correction)
        elif self.executing_action in (0, 2):
            yaw_error = normalize_angle(self.action_target_yaw - self.current_yaw)
            if abs(yaw_error) <= self.angle_tolerance:
                self.stop_action(completed=True)
                return
            twist.angular.z = self.clamp_signed(self.turn_gain * yaw_error, self.turn_speed)
        else:
            self.stop_action(completed=False)
            return

        self.cmd_pub.publish(twist)

    def timer_callback(self):
        self.publish_status()
        if not self.ready():
            if self.action_execution_mode != "nav2":
                self.cmd_pub.publish(Twist())
            self.show_ccrl_built_map()
            return

        if self.action_execution_mode == "nav2":
            if self.executing_action is not None:
                self.step_nav2_action()
                self.show_ccrl_built_map()
                return
            if self.pause_until is not None:
                if self.now_seconds() < self.pause_until:
                    self.show_ccrl_built_map()
                    return
                self.pause_until = None

            state = self.build_state()
            self.maybe_dump_state(state)
            self.show_ccrl_built_map(force=True)
            action = self.infer_action(state)
            self.start_nav2_action(action)
            return

        if self.executing_action is not None:
            self.step_current_action()
            self.show_ccrl_built_map()
            return
        if self.pause_until is not None:
            if self.now_seconds() < self.pause_until:
                self.show_ccrl_built_map()
                return
            self.pause_until = None

        state = self.build_state()
        self.maybe_dump_state(state)
        self.show_ccrl_built_map(force=True)
        action = self.infer_action(state)
        self.start_action_continuous(action)


def main(args=None):
    rclpy.init(args=args)
    node = CcrlPolicyNode()
    try:
        rclpy.spin(node)
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
