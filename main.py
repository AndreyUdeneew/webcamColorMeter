import numpy as np
# import sheet as sheet
from matplotlib import pyplot as plt
import cv2
import io
import time
import xlwt

# Camera stream
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)
# Video stream (optional)
# cap = cv2.VideoCapture("videoplayback.mp4")

# Image crop
x, y, w, h = 700, 500, 100, 100
heartbeat_count = 128
colorRatio_count = 128
heartbeat_values = [0]*heartbeat_count
colorRatio = [0]*colorRatio_count
heartbeat_times = [time.time()]*heartbeat_count

# Matplotlib graph surface
fig = plt.figure()
ax = fig.add_subplot(111)

# book = xlwt.Workbook()
# sh = book.add_sheet(sheet)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    crop_img = img[y:y + h, x:x + w]

    # Update the data (4 heartrate)
    # heartbeat_values = heartbeat_values[1:] + [np.average(crop_img)]
    redIntense = np.average(crop_img[:, :, 0])
    greenIntense = np.average(crop_img[:, :, 1])
    colorRatio = colorRatio[1:] + [redIntense/greenIntense]
    heartbeat_times = heartbeat_times[1:] + [time.time()]

    # sh.write(colorRatio)
    #
    # book.save("C:\Users\user\Desktop\mest.xlsx")


    # Draw matplotlib graph to numpy array
    # ax.plot(heartbeat_times, heartbeat_values)
    ax.plot(heartbeat_times, colorRatio)
    fig.canvas.draw()
    plot_img_np = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    plot_img_np = plot_img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.cla()

    # Display the frames
    cv2.imshow('Crop', crop_img)
    cv2.imshow('Graph', plot_img_np)
    cv2.imshow('Red', crop_img[:, :, 0])
    cv2.imshow('Green', crop_img[:, :, 1])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()