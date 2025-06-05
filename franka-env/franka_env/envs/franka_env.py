import pickle
import time

import cv2
import gym
import numpy as np
from frankateach.constants import CAM_PORT, CONTROL_PORT, GRIPPER_OPEN, HOST
from frankateach.messages import FrankaAction, FrankaState
from frankateach.network import ZMQCameraSubscriber, create_request_socket

INTERNET_HOST = "10.19.143.251"  # One of IP on local computer

left_camera_calibs = np.load(
    "/nas/projectaria/0331_calib_left_demo_7.npy",
    allow_pickle=True,
)[()]


K = {
    3: left_camera_calibs["cam_3"]["int"],
    4: left_camera_calibs["cam_4"]["int"],
    # aria
    # 6: np.array(
    #     [
    #         [610.8692627, 0.0, 703.5],
    #         [0.0, 610.8692627, 703.5],
    #         [0.0, 0.0, 1.0],
    #     ]
    # ),
    # iphone
    6: np.array(
        [
            [706.01969952, 0.0, 360.86504065],
            [0.0, 706.15628068, 490.34852859],
            [0.0, 0.0, 1.0],
        ],
    ),
}
D = {
    3: left_camera_calibs["cam_3"]["dist_coeff"],
    4: left_camera_calibs["cam_4"]["dist_coeff"],
    # aria
    # 6: np.zeros(5, dtype=np.float32),
    # iphone
    6: np.array(
        [
            [
                2.97673215e-01,
                -1.69844695e00,
                1.65368204e-03,
                3.61532041e-05,
                3.03597517e00,
            ],
        ]
    ),
}
T_robot_to_camera = {
    3: left_camera_calibs["cam_3"]["ext"],
    4: left_camera_calibs["cam_4"]["ext"],
}
T_aruco_to_camera = {
    2: np.array(
        [
            [0.87833182, 0.47753213, -0.0222771, -0.00709056],
            [0.2047396, -0.41787251, -0.88513516, 0.28966185],
            [-0.43198947, 0.77288138, -0.46480047, 1.12155578],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    4: np.array(
        [
            [-0.90335721, 0.42750082, -0.03447882, 0.08824751],
            [0.21219395, 0.3756293, -0.90215096, 0.2580675],
            [-0.37271902, -0.82228078, -0.43004052, 1.20200965],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
}


class FrankaEnv(gym.Env):
    def __init__(
        self,
        width=640,
        height=480,
        use_robot=True,
        use_gt_depth=False,
        crop_h=None,
        crop_w=None,
        cam_ids=[3, 4, 6],
    ):
        super(FrankaEnv, self).__init__()
        self.width = width
        self.height = height
        self.crop_h = crop_h
        self.crop_w = crop_w

        self.feature_dim = 8
        self.action_dim = 7
        self.use_robot = use_robot
        self.use_gt_depth = use_gt_depth
        self.n_channels = 3
        self.reward = 0

        self.franka_state = None
        self.curr_images = None

        self.action_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(self.action_dim,)
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, self.n_channels), dtype=np.uint8
        )

        if self.use_robot:
            self.cam_ids = cam_ids
            self.image_subscribers = {}
            if self.use_gt_depth:
                self.depth_subscribers = {}
            for cam_idx in self.cam_ids:
                port = CAM_PORT + cam_idx
                self.image_subscribers[cam_idx] = ZMQCameraSubscriber(
                    host=INTERNET_HOST if cam_idx == 6 else HOST,
                    port=port,
                    topic_type="RGB",
                )

                if self.use_gt_depth:
                    depth_port = CAM_PORT + cam_idx + 1000  # depth offset =1000
                    self.depth_subscribers[cam_idx] = ZMQCameraSubscriber(
                        host=INTERNET_HOST if cam_idx == 6 else HOST,
                        port=depth_port,
                        topic_type="Depth",
                    )
            self.action_request_socket = create_request_socket(HOST, CONTROL_PORT)

    def get_state(self):
        self.action_request_socket.send(b"get_state")
        franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())
        self.franka_state = franka_state
        return franka_state

    def step(self, abs_action):
        pos = abs_action[:3]
        quat = abs_action[3:7]
        gripper = abs_action[-1]
        # print("gripper prediction float", gripper)

        # Send action to the robot
        franka_action = FrankaAction(
            pos=pos,
            quat=quat,
            gripper=gripper,
            reset=False,
            timestamp=time.time(),
        )

        self.action_request_socket.send(bytes(pickle.dumps(franka_action, protocol=-1)))
        franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())
        self.franka_state = franka_state

        image_list = {}
        for cam_idx, subscriber in self.image_subscribers.items():
            image, _ = subscriber.recv_rgb_image()

            # crop the image
            if self.crop_h is not None and self.crop_w is not None:
                h, w, _ = image.shape
                image = image[
                    int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                    int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                ]

            image_list[cam_idx] = image

        if self.use_gt_depth:
            depth_list = {}
            for cam_idx, subscriber in self.depth_subscribers.items():
                depth, _ = subscriber.recv_depth_image()

                if self.crop_h is not None and self.crop_w is not None:
                    h, w = depth.shape
                    depth = depth[
                        int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                        int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                    ]

                depth_list[cam_idx] = depth

        self.curr_images = image_list

        obs = {
            "features": np.concatenate(
                (franka_state.pos, franka_state.quat, [franka_state.gripper])
            ),
        }

        for cam_idx, image in image_list.items():
            obs[f"pixels{cam_idx}"] = cv2.resize(image, (self.width, self.height))
        if self.use_gt_depth:
            for cam_idx, depth in depth_list.items():
                obs[f"depth{cam_idx}"] = cv2.resize(depth, (self.width, self.height))

        return obs, self.reward, False, None

    def reset(self):
        if self.use_robot:
            print("resetting")
            franka_action = FrankaAction(
                pos=np.array([0.4579441, 0.0321529, 0.56579893]),
                quat=np.array([0.99984777, 0.00877362, 0.01497245, 0.00180537]),
                gripper=GRIPPER_OPEN,
                reset=False,
                timestamp=time.time(),
            )
            self.action_request_socket.send(
                bytes(pickle.dumps(franka_action, protocol=-1))
            )
            franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())
            self.franka_state = franka_state
            print("reset done: ", franka_state)

            image_list = {}
            for cam_idx, subscriber in self.image_subscribers.items():
                image, _ = subscriber.recv_rgb_image()

                # crop the image
                if self.crop_h is not None and self.crop_w is not None:
                    h, w, _ = image.shape
                    image = image[
                        int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                        int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                    ]

                image_list[cam_idx] = image

            if self.use_gt_depth:
                depth_list = {}
                for cam_idx, subscriber in self.depth_subscribers.items():
                    depth, _ = subscriber.recv_depth_image()

                    if self.crop_h is not None and self.crop_w is not None:
                        h, w = depth.shape
                        depth = depth[
                            int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                            int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                        ]

                    depth_list[cam_idx] = depth

            self.curr_images = image_list

            obs = {
                "features": np.concatenate(
                    (franka_state.pos, franka_state.quat, [franka_state.gripper])
                ),
            }
            for cam_idx, image in image_list.items():
                obs[f"pixels{cam_idx}"] = cv2.resize(image, (self.width, self.height))
            if self.use_gt_depth:
                for cam_idx, depth in depth_list.items():
                    obs[f"depth{cam_idx}"] = cv2.resize(
                        depth, (self.width, self.height)
                    )

            return obs

        else:
            obs = {}
            obs["features"] = np.zeros(self.feature_dim)
            obs["pixels"] = np.zeros((self.height, self.width, self.n_channels))
            if self.use_gt_depth:
                obs["depth"] = np.zeros((self.height, self.width))

            return obs

    def render(self, mode="rgb_array", cam_idx=None, width=640, height=480):
        assert self.curr_images is not None, "Must call reset() before render()"
        if mode == "rgb_array":
            if cam_idx is not None:
                return self.curr_images[cam_idx]

            image_list = []
            for key, im in self.curr_images.items():
                h, w = im.shape[:2]
                aspect_ratio = w / h
                new_height = height  # desired height
                new_width = int(aspect_ratio * new_height)
                image_list.append(cv2.resize(im, (new_width, new_height)))

            return np.concatenate(image_list, axis=1)
        else:
            raise NotImplementedError
