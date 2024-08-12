#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError
import signal
import time
import queue
import math
from jetracer.nvidia_racecar import NvidiaRacecar

# Initialize NvidiaRacecar
nvidiaracecar = NvidiaRacecar()
nvidiaracecar.steering_offset = 0.0
nvidiaracecar.steering_gain = 1.0
nvidiaracecar.throttle_gain = 1.0

# Constants
WIDTH = 1000
HEIGHT = 900
W_MIN = 0.05
W_MAX = 0.3
S_MIN = 0.2
S_MAX = 0.2

class RoadLaneDetector:
    def __init__(self):
        self.poly_bottom_width = 0.85
        self.poly_top_width = 0.07
        self.poly_height = 0.4
        self.img_center = None
        self.left_detect = False
        self.right_detect = False
        self.left_m = None
        self.right_m = None
        self.left_b = None
        self.right_b = None

    def filter_colors(self, img_frame):
        img_hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
        blurred_image = cv2.GaussianBlur(img_hsv, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 50, 150)
        lower_white_dark1 = np.array([149, 1,  254])
        upper_white_dark1 = np.array([151, 3, 255])
        lower_white_red = np.array([20, 20, 120])
        upper_white_red = np.array([150, 80, 255])
        white_mask = cv2.inRange(img_hsv, lower_white_dark1, upper_white_dark1)
        white_mask_red = cv2.inRange(img_hsv, lower_white_red, upper_white_red)
        white_mask = cv2.bitwise_or(white_mask, white_mask_red)
        edged_mask = cv2.bitwise_and(white_mask, edges)
        white_image = cv2.bitwise_and(img_frame, img_frame, mask=edged_mask)
        return white_image

    def limit_region(self, img_edges):
        height, width = img_edges.shape
        mask = np.zeros_like(img_edges)
        lower_left = (0, int(height * 0.84))
        upper_left = (int(width * 0.25), height // 5 * 3)
        upper_right = (int(width * 0.75), height // 5 * 3)
        lower_right = (width, int(height * 0.84))
        points = np.array([[lower_left, upper_left, upper_right, lower_right]], dtype=np.int32)
        cv2.fillPoly(mask, points, 255)
        region_limited_image = cv2.bitwise_and(img_edges, mask)
        return region_limited_image

    def hough_lines(self, img_mask):
        return cv2.HoughLinesP(img_mask, 1, np.pi / 180, 50, minLineLength=20, maxLineGap=30)

    def separate_lines(self, img_edges, lines):
        right_lines = []
        left_lines = []
        self.img_center = img_edges.shape[1] / 2

        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else np.inf
                if abs(slope) > 0.3:
                    if slope > 0 and x1 > self.img_center and x2 > self.img_center:
                        right_lines.append(line)
                        self.right_detect = True
                    elif slope < 0 and x1 < self.img_center and x2 < self.img_center:
                        left_lines.append(line)
                        self.left_detect = True

        return [right_lines, left_lines]

    def regression(self, separated_lines, img_input):
        output = [None] * 4
        right_points = []
        left_points = []

        if self.right_detect:
            for line in separated_lines[0]:
                for x1, y1, x2, y2 in line:
                    right_points.append((x1, y1))
                    right_points.append((x2, y2))

            if right_points:
                right_vx, right_vy, right_x, right_y = cv2.fitLine(np.array(right_points), cv2.DIST_L2, 0, 0.01, 0.01)
                self.right_m = right_vy / right_vx
                self.right_b = (right_x, right_y)

        if self.left_detect:
            for line in separated_lines[1]:
                for x1, y1, x2, y2 in line:
                    left_points.append((x1, y1))
                    left_points.append((x2, y2))

            if left_points:
                left_vx, left_vy, left_x, left_y = cv2.fitLine(np.array(left_points), cv2.DIST_L2, 0, 0.01, 0.01)
                self.left_m = left_vy / left_vx
                self.left_b = (left_x, left_y)

        y1 = img_input.shape[0]
        y2 = int(y1 * 0.6)

        if self.right_detect:
            right_x1 = int(((y1 - self.right_b[1]) / self.right_m) + self.right_b[0])
            right_x2 = int(((y2 - self.right_b[1]) / self.right_m) + self.right_b[0])
            output[0] = (right_x1, y1)
            output[1] = (right_x2, y2)

        if self.left_detect:
            left_x1 = int(((y1 - self.left_b[1]) / self.left_m) + self.left_b[0])
            left_x2 = int(((y2 - self.left_b[1]) / self.left_m) + self.left_b[0])
            output[2] = (left_x1, y1)
            output[3] = (left_x2, y2)

        return output

    def draw_line(self, img_input, lane):
        overlay = img_input.copy()
        cv2.addWeighted(overlay, 0.3, img_input, 0.7, 0, img_input)
        for i in range(0, len(lane),2):
            if i > 3:
                cv2.line(img_input, lane[i], lane[i+1], (255, 125, 0), 5)
            else:
                cv2.line(img_input, lane[i], lane[i+1], (0, 255, 255), 5)
        return img_input

class CarController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.last_error = 0
        self.speed = 0
        self.angle = 0
        self.drive_mode = False
        self.start_time = 0
        self.size10queue = queue.Queue(3)

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.last_error) / dt if dt > 0 else 0
        self.last_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

    def set_velocity(self, msg):
        self.speed = msg.data

    def set_orientation(self, msg):
        self.angle = msg.data

    def matching(self, x, input_min, input_max, output_min, output_max):
        return (x - input_min) * (output_max - output_min) / (input_max - input_min) + output_min

    def diff_queue(self, diff):
        d_queue = self.size10queue
        if diff is not None:
            if d_queue.full():
                d_queue.get(block=False)
            d_queue.put(diff, False)

        if d_queue.empty():
            return 0

        total_sum = 0
        total_count = 0
        temp_list = list(d_queue.queue)

        for item in temp_list:
            total_sum += item
            total_count += 1

        average = total_sum / total_count
        return average

    def car_position(self, list):
        x1, y1, x2, y2, x3, y3, x4, y4 = list
        center_line = (x1 + x3) / 2
        error = (WIDTH / 2) - center_line
        return error

    def steering_vanishing_point(self, x):
        standard_x = int(WIDTH / 2)
        diff = standard_x - x
        return diff

    def steering_theta(self, w1, w2):
        if np.abs(w1) > np.abs(w2):
            if w1 * w2 < 0:
                w1 = -w1
                angle = np.arctan(np.abs(math.tan(w1) - math.tan(w2)) / (1 + math.tan(w1) * math.tan(w2)))
                theta = self.matching(angle, 0, np.pi / 2, 0, 10)
            elif w1 * w2 > 0:
                if w1 > w2:
                    theta = 0
                else:
                    theta = 0
            else:
                theta = 0
        elif np.abs(w1) < np.abs(w2):
            if w1 * w2 < 0:
                w1 = -w1
                angle = np.arctan(np.abs(math.tan(w1) - math.tan(w2)) / (1 + math.tan(w1) * math.tan(w2)))
                theta = self.matching(angle, 0, np.pi / 2, 0, -10)
            elif w1 * w2 > 0:
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
        X1, X2, Y1, Y2 = WIDTH*0.2, WIDTH*0.8, 1, 700
        if X1 <= x <= X2 and Y1 <= y <= Y2:
            if left_edge - 50 > 0 and right_edge- 900 < 0:
                return self.steering_vanishing_point(x)
            elif left_edge - 50 > 0:
                return self.steering_vanishing_point(x) - (left_edge*2 - 50)
            elif right_edge- 700 < 0:
                return self.steering_vanishing_point(x) - (right_edge*2 - 900)
            else:
                return self.steering_vanishing_point(x)
        else:
            return None

    def normalize(self, value, min_val=0, max_val=210, new_min=0.01, new_max=0.2):
        normalized_value = ((value - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
        return normalized_value

    def find_intersection(self, x1, y1, x2, y2, x3, y3, x4, y4):
        def line_params(x1, y1, x2, y2):
            if x2 - x1 == 0:
                return float('inf'), x1
            else:
                m = (y2 - y1) / (x2 - x1)
                b = y1 - m * x1
                return m, b

        m1, b1 = line_params(x1, y1, x2, y2)
        m2, b2 = line_params(x3, y3, x4, y4)
        x_intersect = 0
        y_intersect = 0
        if m1 == m2:
            return (0, 0, 0, 800)

        if m1 == float('inf'):
            x_intersect = b1
            y_intersect = m2 * x_intersect + b2
        elif m2 == float('inf'):
            x_intersect = b2
            y_intersect = m1 * x_intersect + b1
        else:
            x_intersect = (b2 - b1) / (m1 - m2)
            y_intersect = m1 * x_intersect + b1

        left_edge = (900 - b2) / m2        
        right_edge = (900 - b1) / m1
        return (x_intersect, y_intersect, left_edge, right_edge)

    def steering_calcu(self, input_data):
        x3, y3, x4, y4, x1, y1, x2, y2 = input_data
        cross_x, cross_y, left_edge, right_edge = self.find_intersection(x3, y3, x4, y4, x1, y1, x2, y2)
        if x1 == x2 and x3 == x4:
            return -1, -1
        else:
            if not np.isnan(cross_x) and not np.isnan(cross_y):
                steering_val = -(self.diff_queue(self.steering_roi_point(cross_x, cross_y, left_edge, right_edge)))
                normalized_steering_val = self.normalize(abs(steering_val), 0, 210, W_MIN, W_MAX)
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

def find_intersection(line1, line2, line3, line4):
    if line1 == None or line3 == None:
        return(0, 0)
    x1, y1, x2, y2, x3, y3, x4, y4 = line1[0], line1[1], line2[0], line2[1], line3[0], line3[1], line4[0], line4[1]
    def det(a, b, c, d):
        return a * d - b * c

    denom = det(x1 - x2, y1 - y2, x3 - x4, y3 - y4)
    if denom == 0:
        return (0, 0)

    det1 = det(x1, y1, x2, y2)
    det2 = det(x3, y3, x4, y4)

    x_num = det(det1, x1 - x2, det2, x3 - x4)
    y_num = det(det1, y1 - y2, det2, y3 - y4)

    x = x_num / denom
    y = y_num / denom

    if x < 0 or y < 0:
        return (0, 0)

    return x, y

def draw_intersection(img, intersection):
    if intersection != (0, 0):
        cv2.circle(img, intersection, 5, (0, 0, 255), -1)
        cv2.putText(img, str(intersection[0]) + " " + str(intersection[1]), 
                    (intersection[0] + 10, intersection[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        cv2.putText(img, "No valid intersection", 
                    (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return img

def image_callback(msg, args):
    road_lane_detector, image_pub = args
    bridge = CvBridge()
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        img_filter = road_lane_detector.filter_colors(cv_image)
        img_gray = cv2.cvtColor(img_filter, cv2.COLOR_BGR2GRAY)
        img_edges = cv2.Canny(img_gray, 50, 150)
        img_mask = road_lane_detector.limit_region(img_edges)
        lines = road_lane_detector.hough_lines(img_mask)
        if lines is not None:
            separated_lines = road_lane_detector.separate_lines(img_mask, lines)
            lane = road_lane_detector.regression(separated_lines, cv_image)
            img_result = road_lane_detector.draw_line(cv_image, lane)
            intersection = tuple(map(int, find_intersection(*lane)))
            img_result = draw_intersection(img_result, intersection)
            array_msg = Float32MultiArray()
            publishing_lane_data = []
            if lane is not None and lane[0] is not None and lane[2] is not None:
                for cor in lane:
                    tmp = list(cor)
                    publishing_lane_data += tmp
                array_msg.data = publishing_lane_data
                image_pub.publish(array_msg)
            else:
                array_msg.data = [0,0,0,0,0,0,0,0]
                image_pub.publish(array_msg)
        else:
            array_msg = Float32MultiArray()
            array_msg.data = [0,0,0,0,0,0,0,0]
            image_pub.publish(array_msg)
            img_result = cv_image

        if cv2.waitKey(1) == 27:
            rospy.signal_shutdown("ESC pressed")

    except CvBridgeError as e:
        rospy.logerr("cv_bridge exception: %s", e)

def lane_callback(msg, args):
    car_controller = args
    theta, speed = car_controller.steering_calcu(msg.data)
    if theta == -1 and speed == -1:
        pass 
    else:
        theta = -theta
        theta += 0.27
        motor_control(theta, speed)

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1000,
    capture_height=900,
    display_width=1000,
    display_height=900,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def show_camera():
    rospy.init_node('camera_publisher', anonymous=False)
    image_pub = rospy.Publisher('/image', Image, queue_size=10)
    bridge = CvBridge()
    window_title = "CSI Camera"
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened(): 
        try:
            while not rospy.is_shutdown(): 
                ret_val, frame = video_capture.read() 
                if ret_val:
                    try:
                        ros_image = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                        image_pub.publish(ros_image)
                        cv2.imshow(window_title, frame)
                        keyCode = cv2.waitKey(10) & 0xFF
                        if keyCode == 27 or keyCode == ord('q'):
                            break
                    except CvBridgeError as e:
                        print(e)
                else:
                    print("Error: No frame received")
                    break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")

def main():
    rospy.init_node('new_control')
    road_lane_detector = RoadLaneDetector()
    bridge = CvBridge()
    first_msg = rospy.wait_for_message('/image', Image)
    cv_image = bridge.imgmsg_to_cv2(first_msg, "bgr8")
    image_pub = rospy.Publisher('/lane_detector', Float32MultiArray, queue_size=10)
    image_transport = rospy.Subscriber('/image', Image, image_callback, (road_lane_detector, image_pub))
    car_controller = CarController(kp=3, ki=0.8, kd=0.7)
    rospy.Subscriber('/lane_detector', Float32MultiArray, lane_callback, (car_controller))
    signal.signal(signal.SIGINT, signal_handler)
    rospy.spin()

def signal_handler(sig, frame):
    rospy.signal_shutdown('Shutting down')

if __name__ == '__main__':
    try:
        show_camera()
        main()
    except rospy.ROSInterruptException:
        pass

