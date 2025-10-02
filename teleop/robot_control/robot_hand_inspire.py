from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize  # dds
from inspire_sdkpy import inspire_dds, inspire_hand_defaut

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
import numpy as np
import threading
import time
import os
from multiprocessing import Process, Array

import logging_mp
logger_mp = logging_mp.get_logger(__name__)

inspire_tip_indices = [4, 9, 14, 19, 24]
Inspire_Num_Motors = 6

kTopicInspireCtrlLeft = "rt/inspire_hand/ctrl/l"
kTopicInspireCtrlRight = "rt/inspire_hand/ctrl/r"
kTopicInspireStateLeft = "rt/inspire_hand/state/l"
kTopicInspireStateRight = "rt/inspire_hand/state/r"


class Inspire_Controller:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock=None, dual_hand_state_array=None,
                 dual_hand_action_array=None, fps=100.0, Unit_Test=False, simulation_mode=False):
        logger_mp.info("Initialize Inspire_Controller...")
        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        self.dual_hand_data_lock = dual_hand_data_lock
        self.dual_hand_state_array = dual_hand_state_array
        self.dual_hand_action_array = dual_hand_action_array

        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.INSPIRE_HAND_Unit_Test)

        dds_interface = os.environ.get("XR_TELEOP_DDS_INTERFACE")
        if self.simulation_mode:
            ChannelFactoryInitialize(1, dds_interface)
        else:
            ChannelFactoryInitialize(0, dds_interface)

        self.LeftHandCmd_publisher = ChannelPublisher(kTopicInspireCtrlLeft, inspire_dds.inspire_hand_ctrl)
        self.LeftHandCmd_publisher.Init()
        self.RightHandCmd_publisher = ChannelPublisher(kTopicInspireCtrlRight, inspire_dds.inspire_hand_ctrl)
        self.RightHandCmd_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicInspireStateLeft, inspire_dds.inspire_hand_state)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicInspireStateRight, inspire_dds.inspire_hand_state)
        self.RightHandState_subscriber.Init()

        self.left_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)

        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        wait_count = 0
        while not (any(self.left_hand_state_array[:]) or any(self.right_hand_state_array[:])):
            time.sleep(0.01)
            wait_count += 1
            if wait_count % 100 == 0:
                logger_mp.warning("[Inspire_Controller] Waiting to subscribe to inspire hand state...")
            if wait_count > 500:
                logger_mp.warning("[Inspire_Controller] Timeout waiting for inspire hand state, continuing anyway.")
                break
        logger_mp.info("[Inspire_Controller] Hand state subscription ready.")

        hand_control_process = Process(target=self.control_process,
                                       args=(left_hand_array, right_hand_array,
                                             self.left_hand_state_array, self.right_hand_state_array))
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Inspire_Controller OK!\n")

    def _subscribe_hand_state(self):
        logger_mp.info("[Inspire_Controller] State subscriber started.")
        while True:
            left_state_msg = self.LeftHandState_subscriber.Read()
            if left_state_msg is not None and hasattr(left_state_msg, 'angle_act'):
                with self.left_hand_state_array.get_lock():
                    for i in range(min(Inspire_Num_Motors, len(left_state_msg.angle_act))):
                        self.left_hand_state_array[i] = left_state_msg.angle_act[i] / 1000.0

            right_state_msg = self.RightHandState_subscriber.Read()
            if right_state_msg is not None and hasattr(right_state_msg, 'angle_act'):
                with self.right_hand_state_array.get_lock():
                    for i in range(min(Inspire_Num_Motors, len(right_state_msg.angle_act))):
                        self.right_hand_state_array[i] = right_state_msg.angle_act[i] / 1000.0

            time.sleep(0.002)

    def _send_hand_command(self, left_angle_cmd_scaled, right_angle_cmd_scaled):
        left_cmd_msg = inspire_hand_defaut.get_inspire_hand_ctrl()
        left_cmd_msg.angle_set = left_angle_cmd_scaled
        left_cmd_msg.mode = 0b0001
        self.LeftHandCmd_publisher.Write(left_cmd_msg)

        right_cmd_msg = inspire_hand_defaut.get_inspire_hand_ctrl()
        right_cmd_msg.angle_set = right_angle_cmd_scaled
        right_cmd_msg.mode = 0b0001
        self.RightHandCmd_publisher.Write(right_cmd_msg)

    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array):
        logger_mp.info("[Inspire_Controller] Control process started.")
        running = True
        left_q_target_norm = np.ones(Inspire_Num_Motors)
        right_q_target_norm = np.ones(Inspire_Num_Motors)

        try:
            while running:
                start_time = time.time()

                with left_hand_array.get_lock():
                    left_hand_data = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()

                with left_hand_state_array.get_lock():
                    left_state_norm = np.array(left_hand_state_array[:])
                with right_hand_state_array.get_lock():
                    right_state_norm = np.array(right_hand_state_array[:])

                state_data = np.concatenate((left_state_norm, right_state_norm))

                human_valid = not np.all(right_hand_data == 0.0) and not np.all(
                    left_hand_data[4] == np.array([-1.13, 0.3, 0.15]))

                if human_valid:
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1, :]] - \
                        left_hand_data[self.hand_retargeting.left_indices[0, :]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1, :]] - \
                        right_hand_data[self.hand_retargeting.right_indices[0, :]]

                    raw_left_q = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[
                        self.hand_retargeting.left_dex_retargeting_to_hardware]
                    raw_right_q = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[
                        self.hand_retargeting.right_dex_retargeting_to_hardware]

                    def normalize(val, min_val, max_val):
                        return np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0)

                    for idx in range(Inspire_Num_Motors):
                        if idx <= 3:
                            left_q_target_norm[idx] = normalize(raw_left_q[idx], 0.0, 1.7)
                            right_q_target_norm[idx] = normalize(raw_right_q[idx], 0.0, 1.7)
                        elif idx == 4:
                            left_q_target_norm[idx] = normalize(raw_left_q[idx], 0.0, 0.5)
                            right_q_target_norm[idx] = normalize(raw_right_q[idx], 0.0, 0.5)
                        else:
                            left_q_target_norm[idx] = normalize(raw_left_q[idx], -0.1, 1.3)
                            right_q_target_norm[idx] = normalize(raw_right_q[idx], -0.1, 1.3)

                scaled_left_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in left_q_target_norm]
                scaled_right_cmd = [int(np.clip(val * 1000, 0, 1000)) for val in right_q_target_norm]

                if self.dual_hand_state_array is not None and self.dual_hand_action_array is not None and self.dual_hand_data_lock is not None:
                    with self.dual_hand_data_lock:
                        self.dual_hand_state_array[:] = state_data
                        self.dual_hand_action_array[:] = np.concatenate((left_q_target_norm, right_q_target_norm))

                self._send_hand_command(scaled_left_cmd, scaled_right_cmd)

                current_time = time.time()
                sleep_time = max(0, (1.0 / self.fps) - (current_time - start_time))
                if sleep_time > 0:
                    time.sleep(sleep_time)
        finally:
            logger_mp.info("Inspire_Controller control loop exited.")