import sys
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32


DIRECTION_ALIASES = {
    "w": 0,
    "f": 0,
    "front": 0,
    "forward": 0,
    "e": 1,
    "fr": 1,
    "front_right": 1,
    "right_front": 1,
    "d": 2,
    "r": 2,
    "right": 2,
    "c": 3,
    "br": 3,
    "back_right": 3,
    "right_back": 3,
    "x": 4,
    "b": 4,
    "back": 4,
    "z": 5,
    "bl": 5,
    "back_left": 5,
    "left_back": 5,
    "a": 6,
    "l": 6,
    "left": 6,
    "q": 7,
    "fl": 7,
    "front_left": 7,
    "left_front": 7,
    "s": -1,
    "clear": -1,
    "none": -1,
}

DIRECTION_TEXT = {
    -1: "clear",
    0: "front",
    1: "front_right",
    2: "right",
    3: "back_right",
    4: "back",
    5: "back_left",
    6: "left",
    7: "front_left",
}


class IntentDirectionNode(Node):
    def __init__(self):
        super().__init__("rl_explore_intent_node")
        self.declare_parameter("intent_topic", "/rl_explore/intent_direction")
        self.intent_topic = str(self.get_parameter("intent_topic").value)
        self.publisher = self.create_publisher(Int32, self.intent_topic, 10)
        self._running = True
        self._thread = threading.Thread(target=self._stdin_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(f"Publishing intent directions on {self.intent_topic}")
        self._print_help()

    def destroy_node(self):
        self._running = False
        super().destroy_node()

    def _print_help(self):
        print("")
        print("CCRL intent direction input:")
        print("  q/front_left   w/front        e/front_right")
        print("  a/left         s/clear        d/right")
        print("  z/back_left    x/back         c/back_right")
        print("Type a direction and press Enter. The policy will consume the latest value at the next decision.")
        print("", flush=True)

    def _stdin_loop(self):
        while self._running and rclpy.ok():
            line = sys.stdin.readline()
            if line == "":
                time.sleep(0.1)
                continue
            text = line.strip().lower().replace("-", "_").replace(" ", "_")
            if not text:
                continue
            if text in ("help", "h", "?"):
                self._print_help()
                continue
            if text not in DIRECTION_ALIASES:
                self.get_logger().warn(f"Unknown intent direction: {text}")
                continue
            direction = int(DIRECTION_ALIASES[text])
            msg = Int32()
            msg.data = direction
            self.publisher.publish(msg)
            self.get_logger().info(f"Published intent direction {direction}: {DIRECTION_TEXT[direction]}")


def main(args=None):
    rclpy.init(args=args)
    node = IntentDirectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
