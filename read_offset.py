# read_offset.py
#   @description: driver the UR5e robot

import time
import threading
import numpy as np
import rtde_receive
import rtde_control
from ur_ikfast import ur_kinematics
from interpolator import OnlineCubicInterpolator \
        as OnlineTrajectoryInterpolator


# global configuration for driver
CONFIG = {  "freq": 500, # explicit control freq
            "port": 30004, 
            # positions
            "center": [-0.085, -0.730, 0.070, 2.221, -2.221, 0.1], 
            "tools_offset": [0,0,-0.217,0,0,0], 
            # limitation
            "max_linear_velocity": 0.005, 
            "max_angular_velocity": 0.005,
            "joint_speed_limit": 0.2,
            # behaviour
            "offline": False, 
            "offline-print": False, 
            "interpolation": True, 
            # underlying parameters
            "interpolator_delay": 0.05, 
            "servoJ_dt": 0.02,  # the internal freq, always but not restricted to 
                                # be consist with 'freq'
            "servoJ_lookahead": 0.03, 
                    # lookahead_time – time [S], range [0.03,0.2] smoothens 
                    #   the trajectory with this lookahead time
                    # gain – proportional gain for following target position, 
                    #   range [100,2000]
            "servoJ_gain": 1000}


class UR_controller():
    def __init__(self, ip):
        self.config = CONFIG
        self.config["ip"] = ip
        self.ur_arm = ur_kinematics.URKinematics('ur5e')

        self.T_tool = UR_controller.vector2matrix(self.config["tools_offset"])
        self.T_real_ik = np.linalg.inv(UR_controller.vector2matrix([0,0,0,0,0,np.pi]))
        
        
        # Trying to connect the robot
        print("[   driver] Trying to connect UR5e")
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.config["ip"])
        self.rtde_c = rtde_control.RTDEControlInterface(self.config["ip"], 
                self.config["freq"],
                rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP,
                self.config["port"])
        self.rtde_r.waitPeriod(self.rtde_r.initPeriod())
        print("[   driver] Successfully connected UR5e")

    @staticmethod
    def vector2matrix(xyzrxryrz):
        x, y, z, rx, ry, rz = xyzrxryrz
        theta = np.linalg.norm([rx, ry, rz])
        if theta < 1e-16:
            R = np.eye(3)
        else:
            kx = rx / theta
            ky = ry / theta
            kz = rz / theta
            K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, -1] = [x, y, z]
        return T
 
    def robot_ik(self, pose_vector):
        if self.config["offline"]:
            q_guess = np.zeros(6)
        else:
            q_guess = self.rtde_r.getActualQ()


        pose_vector = pose_vector
        print(pose_vector)
        joint = self.rtde_c.getInverseKinematics(pose_vector, \
                max_position_error = self.config["max_linear_velocity"], \
                max_orientation_error = self.config["max_angular_velocity"])
        joint = np.array(joint)
        # 返回解，如果没有解则返回joint_alternate
        if joint is None or np.max(np.abs(q_guess[:5] - joint[:5])) > 1:
            print("Can not find stable solution!")
            print(f"current q: {q_guess} pose: {self.rtde_r.getActualTCPPose()}")
            print(f"want q: {joint} pose: {pose_vector}")
            return False, q_guess
        return True, joint # + np.array(self.config["system_q_offset"])

if __name__ == '__main__':
    IP = "192.168.0.10"
    ur5e = UR_controller(IP) 

    fuck = list()
    real_pose = ur5e.rtde_r.getActualTCPPose()
    real_joint = ur5e.rtde_r.getActualQ()
    succ, ik_joint = ur5e.robot_ik(real_pose)
    fuck.append(np.array(real_joint) - np.array(ik_joint))

    fuck = np.array(fuck)
    print(fuck)
    print(np.average(fuck, axis=0).tolist())
