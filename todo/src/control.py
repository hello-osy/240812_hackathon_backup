#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from std_msgs.msg import Bool, Float64, Float32MultiArray
import signal
import time
import queue

### 1000x900
import numpy as np
import math
from jetracer.nvidia_racecar import NvidiaRacecar
nvidiaracecar = NvidiaRacecar()
nvidiaracecar.steering_offset = 0.0
nvidiaracecar.steering_gain = 1.0
nvidiaracecar.throttle_gain = 1.0
theta = 0
WIDTH = 1000
LEGTH = 900
W_MIN = 0.05 # 최소 조향
W_MAX = 0.3 # 최대 조향 
S_MIN = 0.0 # 최소 속도
S_MAX = 0.0 # 최대 속도
class CarController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.last_error = 0
        self.speed = 0
        self.angle = 0
        self.drive_mode =False
        self.start_time = 0
        self.size10queue = queue.Queue(3)
        
    def compute(self, error, dt):
        """PID 제어 계산을 수행하고 조정된 제어 값을 반환."""
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def start_green(self, msg, args):
        car_controller = args
        car_controller.drive_mode = True
        car_controller.start_time = time.time()

    def set_velocity(self, msg):
        self.speed = msg.data  # 현재 속도 업데이트

    def set_orientation(self, msg):
        self.angle = msg.data  # 현재 조향 업데이트

    def matching(self, x, input_min, input_max, output_min, output_max):
        return (x - input_min) * (output_max - output_min) / (input_max - input_min) + output_min # map() 함수 정의.

    def diff_queue(self, diff):
        d_queue = self.size10queue
        if diff is None:
            pass
        else:
            if d_queue.full():
                d_queue.get(block=False)
            d_queue.put(diff, False)

        # 큐가 비어있다면 0을 리턴
        if d_queue.empty():
            return 0

        # 큐의 모든 원소의 합과 개수를 계산하여 평균을 구함
        total_sum = 0
        total_count = 0
        temp_list = list(d_queue.queue)  # 큐의 원소를 리스트로 변환

        for item in temp_list:
            total_sum += item
            total_count += 1

        average = total_sum / total_count
        return average
            

    def car_position(self, list): # 차체 위치 에러
        x1, y1, x2, y2, x3, y3, x4, y4 = list
        center_line = (x1 + x3) / 2
        error = (WIDTH / 2) - center_line
        return error

    def steering_vanishing_point(self, x):
        standard_x = int(WIDTH / 2)
        diff = standard_x - x 
        return diff

    def steering_theta(self, w1, w2): # 차선 기울기 기준 조향
        if np.abs(w1) > np.abs(w2):  # 우회전
            if w1 * w2 < 0:  # 정방향 or 약간 틀어진 방향
                w1 = -w1
                angle = np.arctan(np.abs(math.tan(w1) - math.tan(w2)) / (1 + math.tan(w1) * math.tan(w2)))
                theta = self.matching(angle, 0, np.pi / 2, 0, 10)
            elif w1 * w2 > 0:  # 극한으로 틀어진 방향
                if w1 > w2:
                    theta = 0
                else:
                    theta = 0
            else:
                theta = 0
        elif np.abs(w1) < np.abs(w2):  # 좌회전
            if w1 * w2 < 0:  # 정방향 or 약간 틀어진 방향
                w1 = -w1
                angle = np.arctan(np.abs(math.tan(w1) - math.tan(w2)) / (1 + math.tan(w1) * math.tan(w2)))
                theta = self.matching(angle, 0, np.pi / 2, 0, -10)
            elif w1 * w2 > 0:  # 극한으로 틀어진 방향
                if w1 > w2:
                    theta = 0
                else:
                    theta = 0
            else:
                theta = 0
        else:
            theta = 0

        return theta

    def steering_roi_point(self, x, y, left_edge, right_edge):
        # 우회전 해야할때는 - 좌회전 해야 할때는 +로 리턴 - 부호가 반대임
        # print(x,y,left_edge, right_edge)
        X1, X2, Y1, Y2 = WIDTH*0.2, WIDTH*0.8, 1, 700
        if X1 <= x <= X2 and Y1 <= y <= Y2:
            if left_edge - 50 > 0 and right_edge- 900 < 0:
                return self.steering_vanishing_point(x)
            elif left_edge - 50 > 0:
                # print("far left : left_edge - 50 :" , left_edge)
                # print(self.steering_vanishing_point(x))
                return self.steering_vanishing_point(x) - (left_edge*2 - 50)
            elif right_edge- 700 < 0:
                # print("오른쪽으로 치우쳐짐 right_edge - 900 :", right_edge - 900)
                return self.steering_vanishing_point(x) - (right_edge*2 - 900)
            else:
                return self.steering_vanishing_point(x)
        else:
            return None
        # new_max 0.16 기본속도 0.3에서 잘감
    def normalize(self, value, min_val=0, max_val=210, new_min=0.01, new_max=0.2):
        """
        값을 주어진 범위에서 새 범위로 정규화합니다.
        
        Parameters:
            value (float): 정규화할 원래 값
            min_val (float): 원래 값의 최소값
            max_val (float): 원래 값의 최대값
            new_min (float): 새 범위의 최소값
            new_max (float): 새 범위의 최대값
            
        Returns:
            float: 새 범위로 정규화된 값
        """
        # 원래 값의 범위에서 새 범위로 선형 변환
        normalized_value = ((value - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
        return normalized_value
    
    def i_normalize(self, value, original_min, original_max, new_min, new_max):
        original_range = original_max - original_min
        new_range = new_max - new_min
        normalized_value = ((value - original_min) / original_range) * new_range + new_min
        return normalized_value
        
    def find_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        def line_params(x1, y1, x2, y2):
            """두 점을 통해 직선의 기울기와 절편을 구하는 함수"""
            if x2 - x1 == 0:  # 수직선의 경우
                return float('inf'), x1
            else:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                return m, b

        m1, b1 = line_params(x1, y1, x2, y2)
        m2, b2 = line_params(x3, y3, x4, y4)
        x_intersect = 0
        y_intersect = 0
        if m1 == m2:  # 평행한 경우
            return (0, 0, 0, 800)

        if m1 == float('inf'):  # 첫 번째 직선이 수직선인 경우
            x_intersect = b1
            y_intersect = m2 * x_intersect + b2
        elif m2 == float('inf'):  # 두 번째 직선이 수직선인 경우
            x_intersect = b2
            y_intersect = m1 * x_intersect + b1
        else:
            x_intersect = (b2 - b1) / (m1 - m2)
            y_intersect = m1 * x_intersect + b1


        left_edge = 0
        left_edge = (900 - b2) / m2        
        right_edge = 0
        right_edge = (900 - b1) / m1
        # print((x_intersect, y_intersect,))
        return (x_intersect, y_intersect, left_edge, right_edge )
    
    def steering_calcu(self, input_data): # 조향값 도출
        x3, y3, x4, y4, x1, y1, x2, y2 = input_data
        cross_x, cross_y, left_edge, right_edge = self.find_intersection(x3, y3, x4, y4, x1, y1, x2, y2)
        # 분모가 0인 경우를 예외 처리
        if x1 == x2 and x3 == x4:
            return -1, -1  # 두 점이 같은 x 좌표를 가지면 기울기가 무한대가 되므로 None을 반환
        else:
            if not np.isnan(cross_x) and not np.isnan(cross_y):
                steering_val = -(self.diff_queue(self.steering_roi_point(cross_x, cross_y,left_edge, right_edge)))
                normalized_steering_val = self.normalize(abs(steering_val), 0, 210, W_MIN, W_MAX)
                
                # 절댓값으로 정규화 했으므로 다시 음수를 조건에 따라 취함
                if steering_val > 0:
                    steering_angle = normalized_steering_val
                else:
                    steering_angle = -normalized_steering_val
                    
                if abs(steering_angle) < 0.01:
                    steering_speed = S_MAX
                else:
                    steering_speed = S_MIN

        return steering_angle, steering_speed

def motor_control(steering, throttle):
	nvidiaracecar.steering = steering
	nvidiaracecar.throttle = throttle

def lane_callback(msg, args):
    car_controller = args

    # 목표 지점과 현재 위치의 차이를 계산 (여기서는 단순히 가정) --> 좌표 오차를 각도 오차로 변환해야함
    # 목표 지점의 x 좌표를 거리 오차로 사용
    
    # 각도와 스피드 값 계산
    theta, speed = car_controller.steering_calcu(msg.data) # msg.data 넣어야 함. 그냥 msg에는 다른 정보도 들어있음.
    
    if theta == -1 and speed == -1: # 수직인 직선에 대한 값
        pass 
    else:
        theta = -theta
        theta += 0.27
        
        # print("speed",speed)
        # print("theta",theta)
        motor_control(theta, speed)

def main():
    rospy.init_node('control', anonymous=True)
    
    car_controller = CarController(kp=3, ki=0.8, kd=0.7)
    # 토픽 이름, 메시지 타입, 메시지 큐 크기

    # 토픽 이름, 메시지 타입, 콜백 함수

    rospy.Subscriber('/lane_detector', Float32MultiArray, lane_callback, (car_controller))
    # rospy.Subscriber('/green_light', Bool, car_controller.start_green, (car_controller))
    # rospy.Subscriber('/velocity', Float64, car_controller.set_velocity)
    # rospy.Subscriber('/orientation', Float64, car_controller.set_orientation)

    # Ctrl+C 누르면 종료할 수 있게 만듦.
    signal.signal(signal.SIGINT, signal_handler)

    rospy.spin()  # 콜백 함수를 호출하면서 계속 실행

def signal_handler(sig, frame):
    rospy.signal_shutdown('Shutting down')

if __name__ == '__main__':
    main()
