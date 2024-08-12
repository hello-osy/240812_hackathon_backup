#! /usr/bin/env python2
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError

class RoadLaneDetector:
    def __init__(self):
        self.poly_bottom_width = 0.85 #관심영역 선택할 때 필요한 값
        self.poly_top_width = 0.07 #관심영역 선택할 때 필요한 값
        self.poly_height = 0.4 #관심영역 선택할 때 필요한 값
        self.img_center = None
        self.left_detect = False
        self.right_detect = False
        self.left_m = None
        self.right_m = None
        self.left_b = None
        self.right_b = None
        

        
    def filter_colors(self, img_frame):
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                hsv_image = param
                h, s, v = hsv_image[y, x]
                print("HSV :",h, s, v) # f#
                
        # cv2.imshow("CAMERA",img_frame)
        img_hsv = cv2.cvtColor(img_frame, cv2.COLOR_BGR2HSV)
        
        # hsv 마우스 올리면 값이 터미널에 출력
        # cv2.namedWindow('HSV Image')
        # cv2.setMouseCallback('HSV Image', on_mouse, img_hsv)
        # cv2.imshow("HSV Image",img_hsv)
        
        # # 블러처리 후 경계 추출
        blurred_image = cv2.GaussianBlur(img_hsv, (5, 5), 0)
        # cv2.imshow("blurred_image",blurred_image)
        edges = cv2.Canny(blurred_image, 50, 150)
        # cv2.imshow("edges",edges)
        
        #  차선을 감지하기 위한 흰색 범위 설정 (HSV)
        lower_white_dark1 = np.array([149, 1,  254])
        upper_white_dark1 = np.array([151, 3, 255])
    
        
        # in red zone
        lower_white_red = np.array([20, 20, 120])
        upper_white_red = np.array([150, 80, 255])
        
        # 마지막 직진 구간 시작 차선
        # lower_white_last = np.array([0, 0, 0])
        # upper_white_last = np.array([0, 0, 0])

        # 밝기가 변하는 부분 (HSV)
        # lower_white_change = np.array([107, 80, 80])
        # upper_white_change = np.array([120, 115, 154])
    

        white_mask = cv2.inRange(img_hsv, lower_white_dark1, upper_white_dark1)
        white_mask_red = cv2.inRange(img_hsv, lower_white_red, upper_white_red)
        # white_mask_last = cv2.inRange(img_hsv, lower_white_last, upper_white_last)
        
        
        # 여러 마스크를 결hap
        white_mask = cv2.bitwise_or(white_mask, white_mask_red)
        # white_mask = cv2.bitwise_or(white_mask, white_mask_last)
        # cv2.imshow("white_mask", white_mask)

        #펜스 제외 범위 합치기
        # edged_mask = cv2.bitwise_and(white_mask, exclude_mask1_inv)
        # 테두리와 둘다 같은 위치의 
        edged_mask = cv2.bitwise_and(white_mask, edges)
        # cv2.imshow("edged_mask", edged_mask)
        
        white_image = cv2.bitwise_and(img_frame, img_frame, mask=edged_mask)

        # 이거 나중에 지울 것
        # cv2.imshow("white_filtered", white_image)

        return white_image

    def limit_region(self, img_edges):
        height, width = img_edges.shape
        mask = np.zeros_like(img_edges)

        # # 밑 부분 네모 설정
        lower_left = (0, int(height * 0.90))
        upper_left = (0, int(height * 0.84))
        upper_right = (width, int(height * 0.84))
        lower_right = (width, int(height * 0.90))
        square = np.array([[lower_left, upper_left, upper_right, lower_right]], dtype=np.int32)
        
        # 밑 네모 부분 위에 사다리꼴정
        lower_left = (0, int(height * 0.84))
        upper_left = (int(width * 0.25), height // 5 * 3)
        upper_right = (int(width * 0.75), height // 5 * 3)
        lower_right = (width, int(height * 0.84))
        points = np.array([[lower_left, upper_left, upper_right, lower_right]], dtype=np.int32)
        
        # 중간 부분 지울 부분의 영역
        lower_left = (180, width)
        upper_left = (width/2-190, height // 5 * 3)
        upper_right = (width/2+130, height // 5 * 3)
        lower_right = (790, width)
        erase_points = np.array([[lower_left, upper_left, upper_right, lower_right]], dtype=np.int32)
        
        # cv2.fillPoly(mask, square, 255)
        cv2.fillPoly(mask, points, 255)
        cv2.fillPoly(mask, erase_points, 0)

        region_limited_image = cv2.bitwise_and(img_edges, mask)
        # 이거 나중에 지울 것
        # cv2.imshow("mask_region", mask)
        # cv2.imshow("region_limited", region_limited_image)
        return region_limited_image

    def hough_lines(self, img_mask):
        #입력 이미지, 거리 해상도, 각도 해상도, 직선으로 판단되기 위한 최소한의 투표 수, 검출된 직선의 최소 길이, 직선으로 간주할 최대 간격
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

def find_intersection(line1, line2, line3, line4):
    # x1, y1, x2, y2, x3, y3, x4, y4 = *line1, *line2, *line3, *line4
    if line1 == None or line3 ==None:
        return(0, 0)
    x1, y1, x2, y2, x3, y3, x4, y4 = line1[0], line1[1], line2[0], line2[1], line3[0], line3[1], line4[0], line4[1]
    def det(a, b, c, d):
        return a * d - b * c

    # Calculate the determinants
    denom = det(x1 - x2, y1 - y2, x3 - x4, y3 - y4)
    if denom == 0:
        return (0, 0)  # 선분들이 평행한 경우 교점이 없음

    det1 = det(x1, y1, x2, y2)
    det2 = det(x3, y3, x4, y4)

    x_num = det(det1, x1 - x2, det2, x3 - x4)
    y_num = det(det1, y1 - y2, det2, y3 - y4)

    x = x_num / denom
    y = y_num / denom

    # 화면상에서 x나 y가 0 미만이면 (0, 0)을 반환
    if x < 0 or y < 0:
        return (0, 0)

    return x, y

def draw_intersection(img, intersection):
    if intersection != (0, 0):
        cv2.circle(img, intersection, 5, (0, 0, 255), -1)  # 교점에 원을 그림
        cv2.putText(img, str(intersection[0]) + " " + str(intersection[1]),  # " f"
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
        # print(lines)
        if lines is not None:
            separated_lines = road_lane_detector.separate_lines(img_mask, lines)
            lane = road_lane_detector.regression(separated_lines, cv_image)

            # lane.extend(line1+line2+line3)
            # print(lane)
            img_result = road_lane_detector.draw_line(cv_image, lane)
            #print("lane", lane)
            intersection = tuple(map(int,find_intersection(*lane)))
            #print("intersection", intersection)
            img_result = draw_intersection(img_result, intersection)
            array_msg = Float32MultiArray()
            publishing_lane_data = []
            if lane != None and lane[0] != None and lane[2] != None:
                # print(lane)
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

        #창 이름, 표시할 이미지
        # cv2.imshow("result", img_result) 

        if cv2.waitKey(1) == 27:
            rospy.signal_shutdown("ESC pressed")

    except CvBridgeError as e:
        rospy.logerr("cv_bridge exception: %s", e)


def main():
    rospy.init_node('road_lane_detector')
    road_lane_detector = RoadLaneDetector()
    bridge = CvBridge() #CvBridge로 ROS 이미지 메시지와 OpenCV 이미지를 왔다갔다 할 수 있다.
    first_msg = rospy.wait_for_message('/image', Image)
    cv_image = bridge.imgmsg_to_cv2(first_msg, "bgr8")

    #창 이름, 표시할 이미지
    #이거 나중에 지울 것
    #cv2.imshow("cv_input", cv_image) 

    image_pub = rospy.Publisher('/lane_detector', Float32MultiArray, queue_size=10)
    
    #구독할 토픽, 구독할 메시지의 타입, 메시지 수신했을때 호출할 콜백 함수, 콜백 함수에 추가로 전달할 인수들
    image_transport = rospy.Subscriber('/image', Image, image_callback, (road_lane_detector, image_pub))

    #result라는 이름의 창 생성
    # cv2.namedWindow("result")
    
    rospy.spin() #현재 스레드에서 무한 루프를 실행해서 콜백함수가 호출될 수 있도록 대기함.


if __name__ == '__main__':
    main()
