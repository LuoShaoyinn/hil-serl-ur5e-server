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
            "center": [-0.07381, -0.66531, -0.04133, 2.221, 2.221, 0.0],
            "linear_factor": [3.10, 3.1, 3.1],
            "angular_factor": 1.3,
            "linear_2_factor": [5.0, 5.0, 5.0],
            "angular_2_factor": 3.0,
            "linear_d_factor": 0.0,
            "angular_d_factor": 0.0,
            # limitation
            "tcp_torque_limit": [20, 20, 20, 2, 2, 2], 
            "angular_speed": 0.3,  # rad/s
            # underlying parameters
            "inner_freq": 500, # implicit freq for robot
          }


class UR_controller():
    def __init__(self):
        self.config = CONFIG

        # global variables
        self.target_vel = 0.2
        self.last_movel_time = time.time()
        self.target_pos = np.array(self.config["center"])[:3]
        self.target_ori = R.from_rotvec(np.array(self.config["center"])[3:])
        
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

    def movel(self, pos, ori):
        self.last_movel_time = time.time()

        self.target_vel = 0.2
        self.target_pos = pos
        self.target_ori = ori
    
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
        init_TCP_torque = np.asarray(self.rtde_r.getActualTCPForce())
        TCP_torque = np.zeros(6, dtype=np.float32)

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
                next_time_ns = mono() + dt_ns
                 
                speed = True
                target_pos, target_ori = self.a_step(dt_ns / 1e9, speed=speed)
                velocity = np.concatenate([target_pos, target_ori.as_rotvec()])
                # print(f" target={self.target_pos}, {self.target_ori.as_rotvec()} "
                #      + f"curr={np.array(self.rtde_r.getActualTCPPose())}, astep: {target_pose}")

                try:
                    TCP_torque = TCP_torque * 0.8 + (np.asarray(self.rtde_r.getActualTCPForce()) - init_TCP_torque) * 0.2
                except:
                    pass
                if np.any(abs(TCP_torque) > np.asarray(CONFIG["tcp_torque_limit"])) :
                    print(f"[!] protected ! {TCP_torque}")

                    v = velocity[:3]
                    w = velocity[3:]
                    F = TCP_torque[:3]
                    M = TCP_torque[3:]
                    v -= min(0.0, np.dot(v, F)) * F / np.dot(F, F)
                    w -= min(0.0, np.dot(w, M)) * M / np.dot(F, F)
                    velocity = np.concatenate([v, w])

                    print(f"velocity fixed to: {velocity}")


                if np.linalg.norm(velocity) < 2e-3:
                    velocity = np.zeros(6, dtype=np.float32)
                
                print(f"velocity: {velocity}")
                self.rtde_c.speedL(velocity, 1)
                
                # if resolv_succ:
                #    self.target_q = np.array(resolv_q)
                # current_q = np.array(self.rtde_r.getActualQ())
                # command_q = np.clip(self.target_q, \
                #               current_q - speed_q, \
                #               current_q + speed_q)
                # print(f" cmdq: {command_q} wantq: {self.target_q}")

                time.sleep(max(0, next_time_ns - mono()) / 1e9)
        except Exception as e:
            print(f"[daem] Exception got {e}")
        finally:
            self.rtde_r.disconnect()
            self.rtde_c.disconnect()
            print("[daem] Ur5e disconnected")
            print("[daem]  daemon exit !")
        self.daemon_status = "stop"

    
    def a_step(self, dt, speed):
        p_v = np.array(self.config["linear_factor"])
        p_w = np.array(self.config["angular_factor"])
        p2_v = np.array(self.config["linear_2_factor"])
        # p2_w = np.array(self.config["angular_2_factor"])
        d_v = np.array(self.config["linear_d_factor"])
        d_w = np.array(self.config["angular_d_factor"])
        if not speed:
            p_v, p_w = p_v * dt, p_w * dt
        v = np.clip(self.target_vel, 0, 0.2)
        w = np.array(self.config["angular_speed"])

        pose = np.array(self.rtde_r.getActualTCPPose())
        sped = np.array(self.rtde_r.getActualTCPSpeed())

        # limit the speed at linear axis
        curr_pos = pose[:3]
        sped_pos = sped[:3]
        delta_pos = (self.target_pos - curr_pos)
        # print(f"err={delta_pos} speed: {p_v * delta_pos}, {d_v * sped_pos}, {p2_v * delta_pos * np.linalg.norm(delta_pos)}")
        delta_pos = np.clip(p_v * delta_pos
                            + d_v * sped_pos
                            + p2_v * delta_pos * np.linalg.norm(delta_pos),
                            -v, v)
        
        # limit the speed
        curr_ori = R.from_rotvec(pose[3:])
        sped_ori = R.from_rotvec(sped[3:])
        sped_ori_rotv = sped_ori.as_rotvec()
        delta_ori = self.target_ori * curr_ori.inv()
        delta_ori_rotv = delta_ori.as_rotvec()
        delta_ori_rotv = np.clip(p_w * delta_ori_rotv
                                 + d_w * sped_ori_rotv,
                                 -w, w)
        delta_ori = R.from_rotvec(delta_ori_rotv)

        if speed:
            return delta_pos, delta_ori
        else:
            return delta_pos + curr_pos, delta_ori * curr_ori
 
