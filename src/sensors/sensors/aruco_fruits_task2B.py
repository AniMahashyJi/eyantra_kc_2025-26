import rclpy
from rclpy.node import Node
from control_msgs.msg import JointJog
from sensor_msgs.msg import JointState
import numpy as np

class JointPIDController(Node):
    def __init__(self):
        super().__init__('joint_pid_controller')
        
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                           'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        
        self.target_angles = [0.0] * 6
        self.current_angles = [0.0] * 6
        
        self.kp = 1.0
        self.ki = 0.0
        self.kd = 0.1
        
        self.integral = [0.0] * 6
        self.prev_error = [0.0] * 6
        
        self.timer = self.create_timer(0.01, self.control_loop)
        
    def joint_state_callback(self, msg):
        positions = msg.position
        velocities = msg.velocity
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                self.current_angles[i] = positions[idx]
    
    def control_loop(self):
        velocities = []
        for i in range(6):
            error = self.target_angles[i] - self.current_angles[i]
            self.integral[i] += error * 0.01
            derivative = (error - self.prev_error[i]) / 0.01
            velocity = self.kp * error + self.ki * self.integral[i] + self.kd * derivative
            velocities.append(np.clip(velocity, -1.0, 1.0))
            self.prev_error[i] = error
        
        joint_cmd = JointJog()
        joint_cmd.joint_names = self.joint_names
        joint_cmd.velocities = velocities
        self.joint_pub.publish(joint_cmd)
    
    def set_target_angles(self, angles):
        self.target_angles = angles

def main(args=None):
    rclpy.init(args=args)
    controller = JointPIDController()
    controller.set_target_angles([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()