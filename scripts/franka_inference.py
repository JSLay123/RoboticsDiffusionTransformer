#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import argparse
import sys
import threading
import time
import yaml
from collections import deque

# import numpy as np
# import rospy
import torch
# from cv_bridge import CvBridge
# from geometry_msgs.msg import Twist
# from nav_msgs.msg import Odometry
from PIL import Image as PImage
# from sensor_msgs.msg import Image, JointState
# from std_msgs.msg import Header
# import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import numpy as np
import threading
from collections import deque
import cv2

from scripts.agilex_model import create_model

# sys.path.append("./")

# cam_left为占位，实则为空
CAMERA_NAMES = ['cam_high', 'cam_left', 'cam_wrist']

# 观察窗口：双端队列，存储最近的观测数据
# 包含图像和关节位置，用于时序建模
observation_window = None

lang_embeddings = None

# debug
preload_images = None


# Initialize the model
def make_policy(args):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config
    
    # pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=args.config, 
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        # pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )

    return model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# Interpolate the actions to make the robot move smoothly
def interpolate_action(args, prev_action, cur_action):
    steps = np.array(args.arm_steps_length)
    diff = np.abs(cur_action - prev_action)
    # 计算每个关节需要的步数 = 距离/最大步长
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


def get_config(args):
    config = {
        'episode_len': args.max_publish_step,
        'state_dim': 8,
        'chunk_size': args.chunk_size,
        'camera_names': CAMERA_NAMES,
    }
    return config


# Get the observation from the ROS topic
def get_ros_observation(args,ros_operator):
    print_flag = True

    while True and rclpy.ok():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print("syn fail when get_ros_observation")
                print_flag = False
            time.sleep(1.0 / max(args.publish_rate, 1))
            continue
        print_flag = True
        (img_high, img_wrist, arm_state) = result
        # print(f"sync success when get_ros_observation")
        return (img_high, img_wrist, arm_state)


# Update the observation window buffer
def update_observation_window(args, config, ros_operator):
    # JPEG transformation
    # Align with training
    def jpeg_mapping(img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img
    
    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)
    
        # Append the first dummy image
        observation_window.append(
            {
                'qpos': None,
                'images':
                    {
                        config["camera_names"][0]: None,
                        config["camera_names"][1]: None,
                        config["camera_names"][2]: None,
                    },
            }
        )
        
    img_high, img_wrist, arm_state = get_ros_observation(args,ros_operator)
    img_high = jpeg_mapping(img_high)
    img_wrist = jpeg_mapping(img_wrist)
    # 占位的空相机数据
    dummy_left = np.zeros_like(img_high)
    
    qpos = np.array(arm_state.position[:8])
    qpos = torch.from_numpy(qpos).float().cuda()
    observation_window.append(
        {
            'qpos': qpos,
            'images':
                {
                    config["camera_names"][0]: img_high,
                    config["camera_names"][1]: dummy_left,
                    config["camera_names"][2]: img_wrist,
                },
        }
    )


# RDT inference
def inference_fn(args, config, policy, t):
    global observation_window
    global lang_embeddings
    
    # print(f"Start inference_thread_fn: t={t}")
    while True and rclpy.ok():
        time1 = time.time()     

        # fetch images in sequence [high, wrist, high, wrist]
        image_arrs = [
            observation_window[-2]['images'][config['camera_names'][0]],
            observation_window[-2]['images'][config['camera_names'][2]],
            observation_window[-2]['images'][config['camera_names'][1]],
            
            observation_window[-1]['images'][config['camera_names'][0]],
            observation_window[-1]['images'][config['camera_names'][2]],
            observation_window[-2]['images'][config['camera_names'][1]],
        ]
        
        images = [PImage.fromarray(arr) if arr is not None else None
                  for arr in image_arrs]
        
        # for i, pos in enumerate(['f', 'r', 'l'] * 2):
        #     images[i].save(f'{t}-{i}-{pos}.png')
        
        # get last qpos in shape [8, ]
        proprio = observation_window[-1]['qpos']
        # unsqueeze to [1, 8]
        proprio = proprio.unsqueeze(0)
        
        # actions shaped as [1, 64, 8]
        actions = policy.step(
            proprio=proprio,
            images=images,
            text_embeds=lang_embeddings 
        ).squeeze(0).cpu().numpy()
        # print(f"inference_actions: {actions.squeeze()}")
        
        print(f"Model inference time: {time.time() - time1} s")
        
        # print(f"Finish inference_thread_fn: t={t}")
        return actions


# Main loop for the manipulation task
def model_inference(args, config, ros_operator):
    global lang_embeddings
    
    # Load rdt model
    policy = make_policy(args)
    
    lang_dict = torch.load(args.lang_embeddings_path)
    print(lang_dict.shape)
    # print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    # lang_embeddings = lang_dict["embeddings"]
    lang_embeddings = lang_dict.unsqueeze(0)
    
    max_publish_step = config['episode_len']
    chunk_size = config['chunk_size']

    # Initialize position of the puppet arm
    init_qpos_0 = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04]  # replace with real safe init
    init_qpos_1 = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04]
    ros_operator.arm_publish_continuous(init_qpos_0)
    input("Press enter to continue")
    ros_operator.arm_publish_continuous(init_qpos_1)
    print("[DEBUG]1")
    # Initialize the previous action to be the initial robot state
    pre_action = np.zeros(config['state_dim'])
    pre_action[:8] = np.array([0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04])
    action = None
    # Inference loop
    with torch.inference_mode():
        while True and rclpy.ok():
            # The current time step
            t = 0
            publish_interval = 1.0 / max(args.publish_rate, 1)
    
            action_buffer = np.zeros([chunk_size, config['state_dim']])
            
            while t < max_publish_step and rclpy.ok():
                # Update observation window
                update_observation_window(args, config, ros_operator)
                
                # When coming to the end of the action chunk
                if t % chunk_size == 0:
                    # Start inference
                    action_buffer = inference_fn(args, config, policy, t).copy()
                
                raw_action = action_buffer[t % chunk_size]
                action = raw_action
                # print("[DEBUG] action_buffer.shape:", action_buffer.shape)
                # print("[DEBUG] raw_action.shape:", raw_action.shape)
                # print("[DEBUG] action.shape:", action.shape)

                # Interpolate the original action sequence
                if args.use_actions_interpolation:
                    # print(f"Time {t}, pre {pre_action}, act {action}")
                    interp_actions = interpolate_action(args, pre_action, action)
                    # print("[DEBUG] interp_actions.shape", interp_actions.shape)
                else:
                    interp_actions = action[np.newaxis, :]
                # Execute the interpolated actions one by one
                for act in interp_actions:
                    # act is shape (8,)
                    print(act)        
                    if not args.disable_puppet_arm:
                        ros_operator.arm_publish(act)  # puppet_arm_publish_continuous_thread
            
                    time.sleep(publish_interval)
                    # print(f"doing action: {act}")
                t += 1
                
                print("Published Step", t)
                pre_action = action.copy()


# ROS operator class
class RosOperator(Node):
    def __init__(self, args):
        super().__init__("rdt_franka_operator")
        self.img_high_deque = None
        self.img_wrist_deque = None
        self.arm_state_deque = None

        self.bridge = None

        self.arm_publisher = None
        self.arm_publish_thread = None
        self.arm_publish_lock = None
        
        self.args = args
        self.init()
        self.init_ros()


    def init(self):
        # 创建 CvBridge 实例，用于 ROS Image 消息和 OpenCV 图像的转换
        self.bridge = CvBridge()
        self.img_high_deque = deque()
        self.img_wrist_deque = deque()
        self.arm_state_deque = deque()
        # 创建线程锁，用于控制机械臂发布的线程同步
        self.arm_publish_lock = threading.Lock()
        # 立即获取锁，初始状态为锁定状态
        # 这确保首次发布时会阻塞，直到有需要发布的数据
        self.arm_publish_lock.acquire()

    def arm_publish(self, qpos):
        """直接发布 8 关节位置"""
        joint_state_msg = JointState()
        # 设置消息头，Header 包含时间戳和坐标系信息
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        # Set timestep
        joint_state_msg.name = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "panda_finger_joint1",
            # "panda_finger_joint2",
            ]  # 设置关节名称
        joint_state_msg.position = [float(x) for x in qpos]
        self.arm_publisher.publish(joint_state_msg)

    def arm_publish_continuous(self, target):
        """
        连续插值移动 Franka 8 关节
        """
        rate = self.create_rate(self.args.publish_rate)
        # 当前关节状态
        current = None
        print("Waiting for arm state...0")
        while rclpy.ok():
            if len(self.arm_state_deque) != 0:
                current = list(self.arm_state_deque[-1].position)
                break
            # print("Waiting for arm state...")
            time.sleep(1.0 / self.args.publish_rate)

        symbol = [1 if target[i] - current[i] > 0 else -1 for i in range(8)]

        flag = True     # 所有关节都到达目标位置时才为 False
        step = 0
        while flag and rclpy.ok():
            if self.arm_publish_lock.acquire(False):
                return

            diff = [abs(target[i] - current[i]) for i in range(8)]
            flag = False

            for i in range(8):
                # 如果当前位置已经很接近目标位置，直接设为目标值
                if diff[i] < self.args.arm_steps_length[i]:
                    current[i] = target[i]
                else:
                    current[i] += symbol[i] * self.args.arm_steps_length[i]
                    flag = True

            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = self.get_clock().now().to_msg()
            joint_state_msg.name = [
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
                "panda_finger_joint1",
                # "panda_finger_joint2",
                ]
            joint_state_msg.position = current
            self.arm_publisher.publish(joint_state_msg)
            step += 1
            print("arm_publish_continuous:", step)
            rate.sleep()

    def arm_publish_continuous_thread(self, target):
        """启动连续插值线程"""
        if self.arm_publish_thread is not None:
            self.arm_publish_lock.release()
            self.arm_publish_thread.join()
            self.arm_publish_lock.acquire(False)
            self.arm_publish_thread = None

        self.arm_publish_thread = threading.Thread(
            target=self.arm_publish_continuous,
            args=(target,)
        )
        self.arm_publish_thread.start()

    # 数据获取
    def get_frame(self):
        # 检查所有序列是否完整，不完整则返回False
        if len(self.img_high_deque) == 0 or len(self.img_wrist_deque) == 0 or len(self.arm_state_deque) == 0:
            return False
        # 确定时间同步
        frame_time = min([self.img_high_deque[-1].header.stamp.sec, 
                          self.img_wrist_deque[-1].header.stamp.sec, 
                          self.arm_state_deque[-1].header.stamp.sec])
        # 检查每个队列中是否有时间戳早于 frame_time 的数据
        # 如果没有，说明还没有同步的数据
        if len(self.img_high_deque) == 0 or self.img_high_deque[-1].header.stamp.sec < frame_time:
            return False
        if len(self.img_wrist_deque) == 0 or self.img_wrist_deque[-1].header.stamp.sec < frame_time:
            return False
        if len(self.arm_state_deque) == 0 or self.arm_state_deque[-1].header.stamp.sec < frame_time:
            return False

        # 删除时间戳早于 frame_time 的旧数据
        while self.img_high_deque[0].header.stamp.sec < frame_time:
            self.img_high_deque.popleft()
        # 取出时间戳等于 frame_time 的图像
        img_high = self.bridge.imgmsg_to_cv2(self.img_high_deque.popleft(), 'passthrough')

        while self.img_wrist_deque[0].header.stamp.sec < frame_time:
            self.img_wrist_deque.popleft()
        img_wrist = self.bridge.imgmsg_to_cv2(self.img_wrist_deque.popleft(), 'passthrough')

        while self.arm_state_deque[0].header.stamp.sec < frame_time:
            self.arm_state_deque.popleft()
        arm_state = self.arm_state_deque.popleft()

        return (img_high, img_wrist, arm_state)

    # 回调函数
    def img_high_callback(self, msg):
        # print("img_high_callback")
        if len(self.img_high_deque) >= 2000:
            self.img_high_deque.popleft()
        self.img_high_deque.append(msg)

    def img_wrist_callback(self, msg):
        # print("img_wrist_callback")
        if len(self.img_wrist_deque) >= 2000:
            self.img_wrist_deque.popleft()
        self.img_wrist_deque.append(msg)

    def arm_state_callback(self, msg):
        # print("arm_state_callback")
        if len(self.arm_state_deque) >= 2000:
            self.arm_state_deque.popleft()
        # print(msg)
        self.arm_state_deque.append(msg)


    def init_ros(self):
        # rospy.init_node('joint_state_publisher', anonymous=True)
        # camera subscribers
        print("[DEBUG] init ros")
        self.create_subscription(Image, self.args.img_high_topic, self.img_high_callback, 10)
        self.create_subscription(Image, self.args.img_wrist_topic,self.img_wrist_callback, 10)

        # Franka joint states
        self.create_subscription(JointState, self.args.arm_state_topic,self.arm_state_callback, 10)

        # Franka command publisher
        self.arm_publisher = self.create_publisher(JointState, self.args.arm_cmd_topic, 10)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int, 
                        help='Maximum number of action publishing steps', default=10000, required=False)
    parser.add_argument('--seed', action='store', type=int, 
                        help='Random seed', default=None, required=False)

    parser.add_argument('--img_high_topic', action='store', type=str, help='img_high_topic',
                        default='/rgb_high', required=False)
    parser.add_argument('--img_wrist_topic', action='store', type=str, help='img_wrist_topic',
                        default='/rgb_wrist', required=False)


    parser.add_argument('--arm_cmd_topic', action='store', type=str, help='arm_cmd_topic',
                        default='/joint_command', required=False)
    parser.add_argument('--arm_state_topic', action='store', type=str, help='arm_state_topic',
                        default='/joint_states', required=False)

    
    parser.add_argument('--publish_rate', action='store', type=int, 
                        help='The rate at which to publish the actions',
                        default=30, required=False)
    parser.add_argument('--ctrl_freq', action='store', type=int, 
                        help='The control frequency of the robot',
                        default=25, required=False)
    
    parser.add_argument('--chunk_size', action='store', type=int, 
                        help='Action chunk size',
                        default=64, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, 
                        help='The maximum change allowed for each joint per timestep',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01], required=False)

    parser.add_argument('--use_actions_interpolation', action='store_true',
                        help='Whether to interpolate the actions if the difference is too large',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store_true', 
                        help='Whether to use depth images',
                        default=False, required=False)
    
    parser.add_argument('--disable_puppet_arm', action='store_true',
                        help='Whether to disable the puppet arm. This is useful for safely debugging',
                        default=False)
    
    parser.add_argument('--config_path', type=str, default="configs/base_test.yaml", 
                        help='Path to the config file')
    # parser.add_argument('--cfg_scale', type=float, default=2.0,
    #                     help='the scaling factor used to modify the magnitude of the control features during denoising')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help='Name or path to the pretrained model')
    
    parser.add_argument('--lang_embeddings_path', type=str, required=True, 
                        help='Path to the pre-encoded language instruction embeddings')
    
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    rclpy.init()
    ros_operator = RosOperator(args)
    if args.seed is not None:
        set_seed(args.seed)
    config = get_config(args)
    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_operator,), daemon=True)
    spin_thread.start()
    model_inference(args, config, ros_operator)


if __name__ == '__main__':
    main()



# python -m scripts.franka_inference \
#       --use_actions_interpolation \
#       --pretrained_model_name_or_path=/home/silei/WorkSpace_git/RoboticsDiffusionTransformer/checkpoints/rdt-finetune-1b/checkpoint-46000/model.safetensors \
#       --lang_embeddings_path=/home/silei/WorkSpace_git/RoboticsDiffusionTransformer/data/datasets/my_franka/rdt_data/task1/lang_embed_23.pt \
#       --ctrl_freq=25