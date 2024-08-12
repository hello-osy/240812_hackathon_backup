from jetracer.nvidia_racecar import NvidiaRacecar
print(1)
nvidiaracecar = NvidiaRacecar()
print(2)
nvidiaracecar.steering = 0.0
nvidiaracecar.throttle = 0.0
# nvidiaracecar.manual = 0.0
nvidiaracecar.steering_offset = -0.8
#
def motor_control(steering, throttle):
	print(3)
	nvidiaracecar.steering = steering
	nvidiaracecar.throttle = throttle
	print(4)
	print("nvidiaracecar.throttle", nvidiaracecar.throttle)
	print("throttle_gain", nvidiaracecar.throttle_gain)
	print("steering_offset", nvidiaracecar.steering_offset)
	print("steering_gain", nvidiaracecar.steering_gain)


motor_control(0.2, 0.0)
