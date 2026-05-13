import math

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformException, TransformListener


def parameter_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class IntentExploreGuiNode(Node):
    def __init__(self):
        super().__init__("rl_explore_intent_gui_node")
        self.declare_parameter("camera_topic", "/rl_explore/gui/camera_image")
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("robot_frame", "base_footprint")
        self.declare_parameter("fallback_robot_frame", "base_link")
        self.declare_parameter("window_name", "RL Explore Intent GUI")
        self.declare_parameter("show_window", True)
        self.declare_parameter("refresh_hz", 20.0)
        self.declare_parameter("panel_width", 640)
        self.declare_parameter("panel_height", 480)

        self.camera_topic = str(self.get_parameter("camera_topic").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.robot_frame = str(self.get_parameter("robot_frame").value)
        self.fallback_robot_frame = str(self.get_parameter("fallback_robot_frame").value)
        self.window_name = str(self.get_parameter("window_name").value)
        self.show_window = parameter_to_bool(self.get_parameter("show_window").value)
        self.refresh_hz = max(1.0, float(self.get_parameter("refresh_hz").value))
        self.panel_width = max(160, int(self.get_parameter("panel_width").value))
        self.panel_height = max(120, int(self.get_parameter("panel_height").value))

        self.camera_image = None
        self.map_image = None
        self.map_frame = ""
        self.map_resolution = 0.0
        self.map_width = 0
        self.map_height = 0
        self.map_origin_x = 0.0
        self.map_origin_y = 0.0
        self.map_origin_yaw = 0.0
        self.window_created = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, self.camera_topic, self.camera_callback, 2)
        self.create_subscription(OccupancyGrid, self.map_topic, self.map_callback, 2)
        self.create_timer(1.0 / self.refresh_hz, self.timer_callback)

        self.get_logger().info(f"camera_topic={self.camera_topic}")
        self.get_logger().info(f"map_topic={self.map_topic}")
        self.get_logger().info(f"robot_frame={self.robot_frame}")

    def camera_callback(self, msg):
        image = self.image_msg_to_bgr(msg)
        if image is not None:
            self.camera_image = image

    def map_callback(self, msg):
        image = self.occupancy_grid_to_bgr(msg)
        if image is None:
            return
        self.map_image = image
        self.map_frame = str(msg.header.frame_id)
        self.map_resolution = float(msg.info.resolution)
        self.map_width = int(msg.info.width)
        self.map_height = int(msg.info.height)
        self.map_origin_x = float(msg.info.origin.position.x)
        self.map_origin_y = float(msg.info.origin.position.y)
        self.map_origin_yaw = quaternion_to_yaw(msg.info.origin.orientation)

    def occupancy_grid_to_bgr(self, msg):
        width = int(msg.info.width)
        height = int(msg.info.height)
        if width <= 0 or height <= 0:
            return None

        data = np.asarray(msg.data, dtype=np.int16)
        if data.size != width * height:
            self.get_logger().warn(
                f"Invalid occupancy grid size: data={data.size}, expected={width * height}"
            )
            return None

        grid = data.reshape((height, width))
        gray = np.full((height, width), 128, dtype=np.uint8)
        known = grid >= 0
        if np.any(known):
            values = np.clip(grid[known], 0, 100)
            gray[known] = np.asarray(255 - np.rint(values * 2.55), dtype=np.uint8)

        gray = np.flipud(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def lookup_robot_pose(self):
        if not self.map_frame:
            return None

        for robot_frame in (self.robot_frame, self.fallback_robot_frame):
            if not robot_frame:
                continue
            try:
                tf_msg = self.tf_buffer.lookup_transform(self.map_frame, robot_frame, rclpy.time.Time())
                translation = tf_msg.transform.translation
                rotation = tf_msg.transform.rotation
                return float(translation.x), float(translation.y), quaternion_to_yaw(rotation)
            except TransformException:
                continue
        return None

    def map_xy_to_pixel(self, x, y):
        if self.map_resolution <= 0.0 or self.map_width <= 0 or self.map_height <= 0:
            return None

        dx = x - self.map_origin_x
        dy = y - self.map_origin_y
        cos_yaw = math.cos(self.map_origin_yaw)
        sin_yaw = math.sin(self.map_origin_yaw)
        grid_x = cos_yaw * dx + sin_yaw * dy
        grid_y = -sin_yaw * dx + cos_yaw * dy
        col = grid_x / self.map_resolution
        row = grid_y / self.map_resolution
        if col < 0.0 or row < 0.0 or col >= self.map_width or row >= self.map_height:
            return None

        pixel_x = int(round(col))
        pixel_y = int(round(self.map_height - 1 - row))
        return pixel_x, pixel_y

    def draw_robot_marker(self, image):
        pose = self.lookup_robot_pose()
        if pose is None:
            return image

        x, y, yaw = pose
        pixel = self.map_xy_to_pixel(x, y)
        if pixel is None:
            return image

        pixel_x, pixel_y = pixel
        marker = image.copy()
        radius = 8
        arrow_length = 30
        marker_yaw = yaw - self.map_origin_yaw
        tip_x = int(round(pixel_x + math.cos(marker_yaw) * arrow_length))
        tip_y = int(round(pixel_y - math.sin(marker_yaw) * arrow_length))

        cv2.circle(marker, (pixel_x, pixel_y), radius + 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(
            marker,
            (pixel_x, pixel_y),
            (tip_x, tip_y),
            (255, 255, 255),
            8,
            cv2.LINE_AA,
            tipLength=0.35,
        )
        cv2.circle(marker, (pixel_x, pixel_y), radius, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.arrowedLine(
            marker,
            (pixel_x, pixel_y),
            (tip_x, tip_y),
            (0, 0, 255),
            4,
            cv2.LINE_AA,
            tipLength=0.35,
        )
        return marker

    def image_msg_to_bgr(self, msg):
        encoding = str(msg.encoding).lower()
        if encoding in ("bgr8", "rgb8"):
            channels = 3
        elif encoding in ("mono8", "8uc1"):
            channels = 1
        else:
            self.get_logger().warn(f"Unsupported image encoding: {msg.encoding}")
            return None

        expected_step = int(msg.width) * channels
        if int(msg.step) < expected_step:
            self.get_logger().warn(f"Invalid image step for {msg.encoding}: {msg.step}")
            return None

        data = np.frombuffer(msg.data, dtype=np.uint8)
        try:
            rows = data.reshape((int(msg.height), int(msg.step)))
            rows = rows[:, :expected_step]
            if channels == 1:
                image = rows.reshape((int(msg.height), int(msg.width)))
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = rows.reshape((int(msg.height), int(msg.width), channels))
        except ValueError as error:
            self.get_logger().warn(f"Failed to decode image: {error}")
            return None

        if encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image.copy()

    def make_placeholder(self, title, message):
        image = np.full((self.panel_height, self.panel_width, 3), 35, dtype=np.uint8)
        cv2.putText(image, title, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.putText(image, message, (24, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 170, 170), 1, cv2.LINE_AA)
        return image

    def fit_panel(self, image, title):
        if image is None:
            return self.make_placeholder(title, "Waiting for topic...")

        source_h, source_w = image.shape[:2]
        if source_h <= 0 or source_w <= 0:
            return self.make_placeholder(title, "Invalid image")

        panel = np.full((self.panel_height, self.panel_width, 3), 22, dtype=np.uint8)
        title_h = 34
        available_h = max(1, self.panel_height - title_h)
        scale = min(float(self.panel_width) / float(source_w), float(available_h) / float(source_h))
        target_w = max(1, int(round(source_w * scale)))
        target_h = max(1, int(round(source_h * scale)))
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        x0 = (self.panel_width - target_w) // 2
        y0 = title_h + (available_h - target_h) // 2
        panel[y0 : y0 + target_h, x0 : x0 + target_w] = resized
        cv2.putText(panel, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)
        return panel

    def render_window(self):
        camera_panel = self.fit_panel(self.camera_image, "Camera")
        map_image = self.draw_robot_marker(self.map_image) if self.map_image is not None else None
        map_panel = self.fit_panel(map_image, "SLAM Map (/map)")
        gap = 10
        height = max(camera_panel.shape[0], map_panel.shape[0])
        width = camera_panel.shape[1] + gap + map_panel.shape[1]
        canvas = np.full((height, width, 3), 12, dtype=np.uint8)
        canvas[:, : camera_panel.shape[1]] = camera_panel
        canvas[:, camera_panel.shape[1] + gap :] = map_panel
        return canvas

    def timer_callback(self):
        if not self.show_window:
            return
        image = self.render_window()
        if not self.window_created:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, image.shape[1], image.shape[0])
            self.window_created = True
        cv2.imshow(self.window_name, image)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = IntentExploreGuiNode()
    try:
        rclpy.spin(node)
    except (ExternalShutdownException, KeyboardInterrupt):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
