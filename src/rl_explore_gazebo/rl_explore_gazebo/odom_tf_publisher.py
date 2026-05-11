from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class OdomTfPublisher(Node):
    def __init__(self):
        super().__init__("rl_explore_odom_tf_publisher")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("normalized_odom_topic", "")
        self.declare_parameter("parent_frame", "odom")
        self.declare_parameter("child_frame", "base_footprint")

        self.parent_frame = str(self.get_parameter("parent_frame").value)
        self.child_frame = str(self.get_parameter("child_frame").value)
        odom_topic = str(self.get_parameter("odom_topic").value)
        normalized_odom_topic = str(self.get_parameter("normalized_odom_topic").value)

        self.broadcaster = TransformBroadcaster(self)
        self.odom_pub = None
        if normalized_odom_topic:
            self.odom_pub = self.create_publisher(Odometry, normalized_odom_topic, 20)
        self.create_subscription(Odometry, odom_topic, self.odom_callback, 20)

    def odom_callback(self, msg):
        if self.odom_pub is not None:
            odom = Odometry()
            odom.header = msg.header
            odom.header.frame_id = self.parent_frame
            odom.child_frame_id = self.child_frame
            odom.pose = msg.pose
            odom.twist = msg.twist
            self.odom_pub.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = self.parent_frame
        transform.child_frame_id = self.child_frame
        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z
        transform.transform.rotation = msg.pose.pose.orientation
        self.broadcaster.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = OdomTfPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
