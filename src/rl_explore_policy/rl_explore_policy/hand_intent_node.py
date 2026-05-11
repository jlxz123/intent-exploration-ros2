from collections import deque
from pathlib import Path
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32


DEFAULT_MODEL_PATH = "/home/zbf/achievement/graduate/hand_dir_detect/hand_landmarker.task"

DIRECTION_TEXT = {
    0: "front",
    1: "front_right",
    2: "right",
    3: "back_right",
    4: "back",
    5: "back_left",
    6: "left",
    7: "front_left",
}


def parameter_to_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in ("1", "true", "yes", "on")


def classify_intent_from_index_finger(landmarks, tip_idx=8, base_idx=6):
    tip = landmarks[tip_idx]
    base = landmarks[base_idx]
    dx = float(tip.x) - float(base.x)
    dy = float(tip.y) - float(base.y)
    angle_deg = math.degrees(math.atan2(-dy, dx))
    if angle_deg < 0.0:
        angle_deg += 360.0

    if angle_deg >= 337.5 or angle_deg < 22.5:
        return 2, "right"
    if angle_deg < 67.5:
        return 1, "front_right"
    if angle_deg < 112.5:
        return 0, "front"
    if angle_deg < 157.5:
        return 7, "front_left"
    if angle_deg < 202.5:
        return 6, "left"
    if angle_deg < 247.5:
        return 5, "back_left"
    if angle_deg < 292.5:
        return 4, "back"
    return 3, "back_right"


class HandIntentNode(Node):
    def __init__(self):
        super().__init__("rl_explore_hand_intent_node")
        self.declare_parameter("intent_topic", "/rl_explore/intent_direction")
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("model_path", DEFAULT_MODEL_PATH)
        self.declare_parameter("mirror_image", True)
        self.declare_parameter("show_window", True)
        self.declare_parameter("window_name", "RL Explore Hand Intent")
        self.declare_parameter("buffer_size", 5)
        self.declare_parameter("cooldown_ms", 1000.0)
        self.declare_parameter("motion_threshold", 0.02)
        self.declare_parameter("min_hand_detection_confidence", 0.5)
        self.declare_parameter("min_hand_presence_confidence", 0.5)
        self.declare_parameter("min_tracking_confidence", 0.5)

        self.intent_topic = str(self.get_parameter("intent_topic").value)
        self.camera_index = int(self.get_parameter("camera_index").value)
        self.model_path = Path(str(self.get_parameter("model_path").value)).expanduser()
        self.mirror_image = parameter_to_bool(self.get_parameter("mirror_image").value)
        self.show_window = parameter_to_bool(self.get_parameter("show_window").value)
        self.window_name = str(self.get_parameter("window_name").value)
        self.buffer_size = max(1, int(self.get_parameter("buffer_size").value))
        self.cooldown_ms = float(self.get_parameter("cooldown_ms").value)
        self.motion_threshold = float(self.get_parameter("motion_threshold").value)
        self.min_hand_detection_confidence = float(self.get_parameter("min_hand_detection_confidence").value)
        self.min_hand_presence_confidence = float(self.get_parameter("min_hand_presence_confidence").value)
        self.min_tracking_confidence = float(self.get_parameter("min_tracking_confidence").value)

        self.publisher = self.create_publisher(Int32, self.intent_topic, 10)
        self.direction_buffer = deque(maxlen=self.buffer_size)
        self.last_command_time_ms = 0.0
        self.previous_wrist = None
        self.last_timestamp_ms = 0

        self.get_logger().info(f"Hand intent topic: {self.intent_topic}")
        self.get_logger().info(f"Hand landmarker model: {self.model_path}")

    def run(self):
        runtime = self.load_runtime_dependencies()
        if runtime is None:
            return
        cv2, mp, mp_python, vision = runtime

        if not self.model_path.exists():
            self.get_logger().error(f"Hand landmarker model file does not exist: {self.model_path}")
            return

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.get_logger().error(
                f"Cannot open camera index {self.camera_index}. "
                "This does not affect package build; connect a camera before running this node."
            )
            return

        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self.get_logger().info("Hand intent node is running. Press q in the image window to quit.")
        try:
            with vision.HandLandmarker.create_from_options(options) as detector:
                while rclpy.ok():
                    rclpy.spin_once(self, timeout_sec=0.0)
                    ok, frame = cap.read()
                    if not ok:
                        self.get_logger().warn("Failed to read camera frame.")
                        time.sleep(0.03)
                        continue

                    if self.mirror_image:
                        frame = cv2.flip(frame, 1)

                    direction_index, direction_name, hand_landmarks = self.detect_direction(cv2, mp, detector, frame)
                    if hand_landmarks is None:
                        self.direction_buffer.clear()
                        self.previous_wrist = None
                    else:
                        self.update_stability_and_publish(direction_index, direction_name, hand_landmarks)
                        self.draw_debug_overlay(cv2, frame, hand_landmarks, direction_name)

                    if self.show_window:
                        cv2.imshow(self.window_name, frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        finally:
            cap.release()
            if self.show_window:
                cv2.destroyAllWindows()

    def load_runtime_dependencies(self):
        try:
            import cv2  # type: ignore
            import mediapipe as mp  # type: ignore
            from mediapipe.tasks import python as mp_python  # type: ignore
            from mediapipe.tasks.python import vision  # type: ignore
        except ImportError as error:
            self.get_logger().error(
                "Missing hand intent runtime dependency. Install mediapipe/opencv in the ROS Python environment. "
                f"Import error: {error}"
            )
            return None
        return cv2, mp, mp_python, vision

    def detect_direction(self, cv2, mp, detector, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        timestamp_ms = int(time.monotonic() * 1000.0)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms

        result = detector.detect_for_video(mp_image, timestamp_ms)
        if not result.hand_landmarks:
            return None, None, None

        hand_landmarks = result.hand_landmarks[0]
        direction_index, direction_name = classify_intent_from_index_finger(hand_landmarks)
        return direction_index, direction_name, hand_landmarks

    def update_stability_and_publish(self, direction_index, direction_name, hand_landmarks):
        wrist = hand_landmarks[0]
        motion = 0.0
        if self.previous_wrist is not None:
            dx = float(wrist.x) - self.previous_wrist[0]
            dy = float(wrist.y) - self.previous_wrist[1]
            motion = math.sqrt(dx * dx + dy * dy)
        self.previous_wrist = (float(wrist.x), float(wrist.y))

        if motion < self.motion_threshold:
            self.direction_buffer.append(int(direction_index))
        else:
            self.direction_buffer.clear()

        current_time_ms = time.monotonic() * 1000.0
        if len(self.direction_buffer) != self.buffer_size:
            return
        if len(set(self.direction_buffer)) != 1:
            return
        if current_time_ms - self.last_command_time_ms <= self.cooldown_ms:
            return

        stable_direction = int(self.direction_buffer[0])
        self.last_command_time_ms = current_time_ms
        self.direction_buffer.clear()

        msg = Int32()
        msg.data = stable_direction
        self.publisher.publish(msg)
        self.get_logger().info(
            f"Published hand intent {stable_direction}: {DIRECTION_TEXT.get(stable_direction, direction_name)}"
        )

    def draw_debug_overlay(self, cv2, frame, hand_landmarks, direction_name):
        height, width = frame.shape[:2]
        points = [(int(float(lm.x) * width), int(float(lm.y) * height)) for lm in hand_landmarks]
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        x_min = max(0, min(x_values))
        x_max = min(width - 1, max(x_values))
        y_min = max(0, min(y_values))
        y_max = min(height - 1, max(y_values))

        for point in points:
            cv2.circle(frame, point, 3, (0, 255, 0), -1)
        cv2.line(frame, points[6], points[8], (0, 255, 255), 3)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(
            frame,
            str(direction_name),
            (x_min, max(y_min - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main(args=None):
    rclpy.init(args=args)
    node = HandIntentNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
