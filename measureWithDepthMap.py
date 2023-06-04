import pyrealsense2 as rs
import numpy as np
import cv2
import math

point = (0, 0)
tickPoints = []

def distance(event, x, y, args, params):
    global point, tickPoints
    point = (x, y)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        tickPoints.append(x)
        tickPoints.append(y)

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# Create mouse event
cv2.namedWindow("Color frame")
cv2.setMouseCallback("Color frame", distance)

while True:
    frames = pipeline.wait_for_frames()
    depthFrame = frames.get_depth_frame()
    colorFrame = frames.get_color_frame()

    depth = np.asanyarray(depthFrame.get_data())
    color = np.asanyarray(colorFrame.get_data())
    
    cv2.circle(color, point, 4, (0, 0, 255))
    dist = depth[point[1], point[0]]
    
    colorIntrin = colorFrame.profile.as_video_stream_profile().intrinsics
    depthIntrin = depthFrame.profile.as_video_stream_profile().intrinsics
    
    depthColorFrame = rs.colorizer().colorize(depthFrame)
    depthColor = np.asanyarray(depthColorFrame.get_data())
    
    if len(tickPoints) == 4:
        uDist = depthFrame.get_distance(tickPoints[0], tickPoints[1])
        vDist = depthFrame.get_distance(tickPoints[2], tickPoints[3])
        
        point1 = rs.rs2_deproject_pixel_to_point(depthIntrin, [tickPoints[0], tickPoints[1]], uDist)
        point2 = rs.rs2_deproject_pixel_to_point(depthIntrin, [tickPoints[2], tickPoints[3]], vDist)
        measurment =  math.sqrt(math.pow(point1[0] - point2[0], 2) 
                                + math.pow(point1[1] - point2[1],2) 
                                + math.pow(point1[2] - point2[2], 2))
        
        print(measurment * 100)
        cv2.putText(color, "{}cm".format(measurment), (point[0], point[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
        tickPoints.clear()
        
    else:
        cv2.putText(color, "{}mm".format(dist), (point[0], point[1] - 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
    
    cv2.imshow("depth frame", depthColor)
    cv2.imshow("Color frame", color)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

pipeline.stop()
cv2.destroyAllWindows()
