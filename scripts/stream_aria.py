import argparse
import base64
import pickle
import signal
import subprocess
import sys
import time
from contextlib import contextmanager

import aria.sdk as aria
import cv2
import numpy as np
import zmq
from franka_env.envs.franka_env import INTERNET_HOST
from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.image import InterpolationMethod
from projectaria_tools.core.sensor_data import ImageDataRecord


def update_iptables() -> None:
    """
    Update firewall to permit incoming UDP connections for DDS
    """
    update_iptables_cmd = [
        "sudo",
        "iptables",
        "-A",
        "INPUT",
        "-p",
        "udp",
        "-m",
        "udp",
        "--dport",
        "7000:8000",
        "-j",
        "ACCEPT",
    ]
    print("Running the following command to update iptables:")
    print(update_iptables_cmd)
    subprocess.run(update_iptables_cmd)


@contextmanager
def ctrl_c_handler(signal_handler=None):
    class ctrl_c_state:
        def __init__(self):
            self._caught_ctrl_c = False

        def __bool__(self):
            return self._caught_ctrl_c

    state = ctrl_c_state()

    def _handler(sig, frame):
        state._caught_ctrl_c = True
        if signal_handler:
            signal_handler()

    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handler)

    try:
        yield state
    finally:
        signal.signal(signal.SIGINT, original_sigint_handler)


def quit_keypress():
    key = cv2.waitKey(1)
    # Press ESC, 'q'
    return key == 27 or key == ord("q")


# TODO: Run on Local Computer
# TODO: Run 'aria auth pair' to pair aria
# TODO: Activate the python venv. Run Python -m run_aria
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        required=False,
        default="usb",
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile22",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux",
    )
    parser.add_argument(
        "--device-ip", help="IP address to connect to the device over wifi"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    aria.set_log_level(aria.Level.Info)

    device_client = aria.DeviceClient()

    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    else:
        print(client_config.ip_v4_address)
    device_client.set_client_config(client_config)

    device = device_client.connect()

    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile_name

    if args.streaming_interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
    streaming_config.security_options.use_ephemeral_certs = True
    streaming_manager.streaming_config = streaming_config

    sensors_calib_json = streaming_manager.sensors_calibration()
    sensors_calib = device_calibration_from_json_string(sensors_calib_json)

    rgb_calib = sensors_calib.get_camera_calib("camera-rgb")
    rgb_calib
    rgb_linear_calib = get_linear_camera_calibration(
        image_width=1408,
        image_height=1408,
        focal_length=rgb_calib.get_focal_lengths()[0],
        label="camera-rgb",
        T_Device_Camera=rgb_calib.get_transform_device_camera(),
    )

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://{INTERNET_HOST}:10011")  # Change IP as needed

    # Mask can only be applied with project aria 1.5.7.
    # sensors_calib.set_devignetting_mask_folder_path(DEVIGNETTING_MASKS_PATH)
    # devignetting_mask = sensors_calib.load_devignetting_mask("camera-rgb")

    streaming_manager.start_streaming()

    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Rgb
    streaming_client.subscription_config = config

    class StreamingClientObserver:
        def __init__(self):
            self.rgb_image = None

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            self.rgb_image = image

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)

    streaming_client.subscribe()

    rgb_window = "Aria RGB"
    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 512, 512)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, 50, 50)

    with ctrl_c_handler() as ctrl_c:
        while not (quit_keypress() or ctrl_c):
            if observer.rgb_image is not None:

                rgb = cv2.cvtColor(observer.rgb_image, cv2.COLOR_BGR2RGB)

                # rgb = devignetting(src_image=rgb, devignetting_mask=devignetting_mask).astype(np.uint8)

                rgb = distort_by_calibration(
                    rgb, rgb_linear_calib, rgb_calib, InterpolationMethod.BILINEAR
                )
                rgb = np.rot90(rgb, k=-1)
                rgb = np.ascontiguousarray(rgb)

                cv2.imshow(rgb_window, rgb)

                _, buffer = cv2.imencode(".jpg", rgb)
                base64_data = base64.b64encode(buffer).decode("utf-8")
                data = {"rgb_image": base64_data, "timestamp": time.perf_counter_ns()}

                message = pickle.dumps(data)
                topic = b"rgb_image "
                message = topic + message

                socket.send(message)
                observer.rgb_image = None
            else:
                pass
    print("Stop listening to image data")
    streaming_client.unsubscribe()
    streaming_manager.stop_streaming()
    device_client.disconnect(device)


if __name__ == "__main__":
    main()
