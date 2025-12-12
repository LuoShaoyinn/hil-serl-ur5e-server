# driver.py
#   @description: driver the UR5e robot

import time
import threading
import numpy as np
import rtde_receive
import rtde_control
from threading import Lock
from scipy.spatial.transform import Rotation as R

# global configuration for driver
CONFIG = {  "ip": "192.168.0.10", 
            "freq": 500, # explicit control freq
            "port": 30004, 
            # positions
            "center": [-0.1060, -0.684, 0.0011, 2.221, 2.221, 0.0],
            "linear_factor": [1000, 1000, 1000],
            "linear_d_factor": 100.0,
            "angular_factor": 30,
            "angular_d_factor": 25.0,
            "force_damping_factor": 0.05, 
            # limitation
            "max_speed":  [1, 1, 1, 0.4, 0.4, 0.4], 
            "max_force":  [30, 30, 30], 
            "max_torque": [3, 3, 3], 
            # underlying parameters
            "inner_freq": 500, # implicit freq for robot
          }


class UR_controller():
    def __init__(self):
        self.config = CONFIG

        # global variables
        self.last_movel_time = time.time()
        self.target_pos = np.array(self.config["center"])[:3]
        self.target_ori = R.from_rotvec(np.array(self.config["center"])[3:])
        self.e_pos = np.zeros(3, dtype=np.float32)
        self.e_ori = np.zeros(3, dtype=np.float32)
        
        # start daemon thread
        self.daemon_status = "stop"
        self.daemon_command = "none"
        self.daemon_thread = threading.Thread(target = self.daemon)
        self.daemon_thread.start()
        self.lock = Lock()
        
        print("[main] Wait for the driver to initialize")
        while self.daemon_status != "running":
            time.sleep(0.2)
        print("[main] Controller running")


    def stop(self):
        print("[main] Wait for the driver to stop")
        self.daemon_command = "stop"
        while self.daemon_status == "running":
            time.sleep(0.2)

    #
    # Interfaces Implement
    #
    def get_TCP_pose(self): # x, y, z, rx, ry, rz in rotation vector
        try:
            self.tcp_pose = self.getActualTCPPose()
        except Exception:
            pass
        return self.tcp_pose

    def get_TCP_vel(self):
        try:
            self.lock.acquire()
            self.tcp_speed = self.rtde_r.getActualTCPSpeed()
            self.lock.release()
        except Exception:
            pass
        return self.tcp_speed
    
    def get_force(self):
        try:
            self.lock.acquire()
            self.tcp_force = self.rtde_r.getActualTCPForce()
            self.lock.release()
        except Exception:
            pass
        return self.tcp_force
    
    def get_torque(self):
        try:
            self.lock.acquire()
            self.joint_torque = self.rtde_c.getJointTorques()
            self.lock.release()
        except Exception:
            pass
        return self.joint_torque
    
    def get_joint_q(self):
        try:
            self.lock.acquire()
            self.actual_q = self.rtde_r.getActualQ()
            self.lock.release()
        except Exception:
            pass
        return self.actual_q
    
    def get_joint_dq(self):
        try:
            self.lock.acquire()
            self.joint_dq = self.rtde_r.getActualQd()
            self.lock.release()
        except Exception:
            pass
        return self.joint_dq
    
    def get_jacobian(self):
        return [[0] * 6] * 7
    
    def joint_reset(self):
        self.target_pos = np.array(self.config["center"])[:3]
        self.target_ori = R.from_rotvec(np.array(self.config["center"])[3:])

    def pose(self, pos, ori):
        self.daemon_command = "none"
        self.target_vel = 0.2
        self.target_pos = pos
        self.target_ori = ori

    def movel(self, pos, ori):
        self.target_pos = pos
        self.target_ori = ori
        self.daemon_command = "movel"

    
    def getActualTCPPose(self): # patch for uncertain rotate vector:
        self.lock.acquire()
        pose = np.array(self.rtde_r.getActualTCPPose())
        self.lock.release()
        return pose
    
    #
    # Daemon: method below are running in the daemon thread
    #
    def daemon(self):
        print("[daem] daemon started")

        # Trying to connect the robot
        self.daemon_status = "initializing"
        print("[daem] Trying to connect UR5e")
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.config["ip"])
        self.rtde_c = rtde_control.RTDEControlInterface(self.config["ip"], 
                self.config["inner_freq"],
                rtde_control.RTDEControlInterface.FLAG_USE_EXT_UR_CAP,
                self.config["port"])
        self.rtde_r.waitPeriod(self.rtde_r.initPeriod())
        self.rtde_c.zeroFtSensor()
        self.rtde_c.forceModeSetDamping(CONFIG["force_damping_factor"])
        self.rtde_c.forceModeSetGainScaling(1)

        # initlize parameters
        self.target_q = np.array(self.rtde_r.getActualQ())
        dt_ns = 1e9 / self.config["freq"]
        mono = time.monotonic_ns
        
        # begin main loop
        np.set_printoptions(
            precision=4,      # 保留4位小数
            suppress=True,    # 禁止科学计数法
            linewidth=200,    # 避免多行换行输出
            floatmode='fixed' # 强制使用定点表示法（非科学计数法）
        )
        self.daemon_status = "running"
        try:
            while True:
                if self.daemon_command == "movel":    
                    self.rtde_c.forceModeStop()
                    self.rtde_c.moveL(
                            np.concatenate([self.target_pos, \
                                self.target_ori.as_rotvec()]), \
                            speed = 0.2, acceleration = 1)
                    self.daemon_command = "none"

                next_time_ns = mono() + dt_ns
                 
                target_pos, target_ori_rotv = self.a_step()
                target_tcp_force = np.concatenate([target_pos, target_ori_rotv])

                # print(f"target: {target_tcp_force} actual: {actual_tcp_force}")
 
                self.rtde_c.forceMode(\
                        np.zeros(6, dtype=np.float32),\
                        np.ones(6, dtype=np.int32), \
                        target_tcp_force, \
                        2, 
                        np.asarray(CONFIG["max_speed"]))
                
                time.sleep(max(0, next_time_ns - mono()) / 1e9)
        except Exception as e:
            print(f"[daem] Exception got {e}")
        finally:
            self.rtde_r.disconnect()
            self.rtde_c.disconnect()
            print("[daem] Ur5e disconnected")
            print("[daem]  daemon exit !")
        self.daemon_status = "stop"

    
    def a_step(self):
        p_v = np.array(self.config["linear_factor"])
        d_v = np.array(self.config["linear_d_factor"])
        p_w = np.array(self.config["angular_factor"])
        d_w = np.array(self.config["angular_d_factor"])
        max_force = np.asarray(CONFIG["max_force"])
        max_torque = np.asarray(CONFIG["max_torque"])

        # compute error
        curr_pose = np.asarray(self.rtde_r.getActualTCPPose()) 
        e_pos = self.target_pos - curr_pose[:3]
        e_ori = (self.target_ori * R.from_rotvec(curr_pose[3:]).inv()).as_rotvec()

        # compute delta
        d_e_pos = e_pos - self.e_pos
        d_e_ori = e_ori - self.e_ori
        self.e_pos, self.e_ori = e_pos, e_ori
        
        # pd control
        force  = np.clip(p_v * e_pos + d_v * d_e_pos,  -max_force,  max_force)
        torque = np.clip(p_w * e_ori + d_w * d_e_ori, -max_torque, max_torque)
        
        return force, torque
 
