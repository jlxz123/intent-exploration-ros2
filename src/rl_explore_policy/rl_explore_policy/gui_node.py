import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image


def parameter_to_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class IntentExploreGuiNode(Node):
    def __init__(self):
        super().__init__("rl_explore_intent_gui_node")
        self.declare_parameter("camera_topic", "/rl_explore/gui/camera_image")
        self.declare_parameter("map_topic", "/rl_explore/gui/map_image")
        self.declare_parameter("window_name", "RL Explore Intent GUI")
        self.declare_parameter("show_window", True)
        self.declare_parameter("refresh_hz", 20.0)
        self.declare_parameter("panel_width", 640)
        self.declare_parameter("panel_height", 480)

        self.camera_topic = str(self.get_parameter("camera_topic").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.window_name = str(self.get_parameter("window_name").value)
        self.show_window = parameter_to_bool(self.get_parameter("show_window").value)
        self.refresh_hz = max(1.0, float(self.get_parameter("refresh_hz").value))
        self.panel_width = max(160, int(self.get_parameter("panel_width").value))
        self.panel_height = max(120, int(self.get_parameter("panel_height").value))

        self.camera_image = None
        self.map_image = None
        self.window_created = False

        self.create_subscription(Image, self.camera_topic, self.camera_callback, 2)
        self.create_subscription(Image, self.map_topic, self.map_callback, 2)
        self.create_timer(1.0 / self.refresh_hz, self.timer_callback)

        self.get_logger().info(f"camera_topic={self.camera_topic}")
        self.get_logger().info(f"map_topic={self.map_topic}")

    def camera_callback(self, msg):
        image = self.image_msg_to_bgr(msg)
        if image is not None:
            self.camera_image = image

    def map_callback(self, msg):
        image = self.image_msg_to_bgr(msg)
        if image is not None:
            self.map_image = image

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
            return self.make_placeholder(title, "Waiting for image topic...")

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
        map_panel = self.fit_panel(self.map_image, "CCRL Map")
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
