import cv2
import mediapipe as mp
import time
from angle_calculations.event_scoringt import PoseAngles, EventAnalysis
import mimetypes
import variables
import numpy as np
import math
import skimage


# mp_pose  = mp.solutions.pose
mp_pose = mp.solutions.holistic

# image_pose = mp_pose.Pose(static_image_mode=True,model_complexity=2)
image_pose = mp_pose.Holistic(model_complexity=2)

video_pose = mp_pose.Holistic(model_complexity=2)

mp_draw = mp.solutions.drawing_utils

mimetypes.init()
# 2d distance


def dist2D(one, two):
    dx = one[0] - two[0]
    dy = one[1] - two[1]
    return math.sqrt(dx*dx + dy*dy)


def get_event_results(_pose,name):
    pose_estimations = PoseAngles(_pose)
    angles = pose_estimations.calculate()
    # write_csv = pose_estimations.create_csv_coords(name)
    # print(write_csv)
    ea = EventAnalysis(angles)
    event_results = ea.analyse()

    variables.var1Value = str(event_results[0])
    variables.var2Value = str(event_results[1])
    variables.var3Value = str(event_results[2])
    variables.spineFlexionAngle = str(angles['spine_flexion_extension'])
    variables.spineRotationAngle = str(angles['spine_axial_rotation'])
    variables.spineRotationStat = angles['spineRotationStat']
    variables.rightArmAngle = str(angles['arm_orientation_height'])
    variables.spineFlexionStat = angles['spine_flexion_stat']
    variables.eulerTest = str(angles['eulerTest'])
    variables.eulerTestStat = angles['eulerTestStat']
    variables.pelvic_tilt = angles['pelvic_tilt']
# NEW FUNCS


def slope(x_nos, y_nos, x_lhip, y_lhip):
    m = (y_lhip-y_nos)/(x_lhip-x_nos)
    return m


def get_percentage_diff(previous, current):
    try:
        percentage = abs(previous - current)/max(previous, current) * 100
    except ZeroDivisionError:
        percentage = float('inf')
    return percentage


def image_pose_estimation(name):
    img = cv2.imread(name)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = image_pose.process(imgRGB)
    pose1 = []
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            x_y_z = []
            h, w, c = img.shape
            x_y_z.append(lm.x)
            x_y_z.append(lm.y)
            x_y_z.append(lm.z)
            x_y_z.append(lm.visibility)
            pose1.append(x_y_z)
            # cv2.line(img, (x_y_z[0][0],x_y_z[0][1]), (x_y_z[16][0],x_y_z[16][1]), (255, 0, 0), 5)

            cx, cy = int(lm.x*w), int(lm.y*h)
            # print('ARRAY OF LANDMARKS',x_y_z)
            # print(cx)
            # print(type(cx))
            if id % 2 == 0:
                cv2.circle(img, (cx, cy), 2, (55, 0, 0), cv2.FILLED)
            else:
                cv2.circle(img, (cx, cy), 2, (55, 10, 25), cv2.FILLED)
        # print('POSEOBJ',[pose1])

    img = cv2.resize(img, (600, 800))
    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    bottom_end = (10, 700)
    fontScale = 2
    fontColor = (55, 55, 255)
    redCol = (215, 123,  0)
    thickness = 6
    lineType = 2
    # x_nos,y_nos =int(pose1[0][0]*600),int(pose1[0][1]*800)
    x_lhip, y_lhip = int(pose1[23][0]*600), int(pose1[23][1]*800)
    x_rhip, y_rhip = int(pose1[24][0]*600), int(pose1[24][1]*800)
    y_midhip=int((y_rhip+y_lhip)/2)
    lex1, ley1 = int(pose1[2][0]*600), int(pose1[2][1]*800)
    rex1, rey1 = int(pose1[5][0]*600), int(pose1[5][1]*800)
    mid_eye = int((lex1+rex1)/2)

    x_nos, y_nos = int(pose1[0][0]*600), int(pose1[0][1]*800)
    # x_lhip,y_lhip = int(pose1[31][0]*600),int(pose1[31][1]*800)
    # x_rhip,y_rhip = int(pose1[32][0]*600),int(pose1[32][1]*800)
    x_midhip = int((x_lhip+x_rhip)/2)
    x_lshould, y_lshould = int(pose1[11][0]*600), int(pose1[11][1]*800)
    x_rshould, y_rshould = int(pose1[12][0]*600), int(pose1[12][1]*800)
    mid_shouldx = int((x_lshould+x_rshould)/2)
    mid_shouldy = int((y_lshould+y_rshould)/2)
    y_centerum= int((y_midhip+mid_shouldy)/2)
    x_centerum= int((x_midhip+mid_shouldx))
    line_thickness = 2
    z_rhip=int(pose1[32][2]*(700))

    cv2.line(img, (mid_shouldx, mid_shouldy), (x_midhip, y_midhip),
             (0, 255, 0), thickness=line_thickness)
    cv2.line(img, (x_lshould, y_lshould), (x_rshould, y_rshould),
             (247, 180, 38), thickness=line_thickness)
    # cv2.circle(img, (z_rhip,y_rhip), 6, (255, 0, 255), cv2.FILLED)
    # cv2.circle(img, (60,666), 6, (255, 0, 255), cv2.FILLED)
    # cv2.circle(img, (30,3), 6, (255, 0, 255), cv2.FILLED)

    # M1=x_nos, y_nos, x_midhip, y_rhip
    M2 = (y_midhip-mid_shouldy)/(x_midhip-mid_shouldx) # vertical#####
    # M2 = (800-0)/(x_midhip-x_nos) # vertical

    # print('YMIDHIp',y_midhip)

    M1 = (y_rshould-y_lshould)/(x_rshould-x_lshould) #horizontal
    # M2=x_lshould, y_lshould, x_rshould, y_rshould
    # Store the tan value  of the angle
    angle = (M2 - M1) / (1 + M1 * M2)
    # angle = (M1 - M2) / (1 + M2 * M1)
    # angle = abs((M2 - M1) / (1 + M1 * M2))
    # Calculate tan inverse of the angle
    ret = math.atan(angle)
    # Convert the angle from
    # radian to degree
    val = (ret * 180) / math.pi
    # val =   90 - val
    
    # if pose1[23][2] > pose1[24][2]:
    #     val =  val
    # else:
    #     val = val

    rshould = np.around(pose1[12][2], decimals=6)
    lshould = np.around(pose1[11][2], decimals=6)
    print('rhsouldz', rshould)
    print('lhsouldz', lshould)

    should_z_diff = get_percentage_diff(rshould, lshould)
    print('PERC ENTAGE SHOULDER Z DIFF', get_percentage_diff(rshould, lshould))
    # if should_z_diff > -19:
    #     val = val + 90
    #     print('lshould higher than rshoulder')
    #     cv2.putText(img, 'lshould higher than rshoulder',
    #                 bottom_end,
    #                 font,
    #                 fontScale,  
    #                 redCol,
    #                 thickness,
    #                 lineType)
    # elif rshould < lshould:
    #     val = val
    #     cv2.putText(img, 'rshould higher than lshoulder',
    #                 bottom_end,
    #                 font,
    #                 fontScale,
    #                 redCol,
    #                 thickness,
    #                 lineType)
    #     print('rhould higher than lshoulder')

    # else:
    #     print('Not activated')
    #     pass

    # Print the result
    # val = val - 180
    texts = str(val)
    cta = ('CHEST TURN ANGLE', round(val, 2))

    cv2.putText(img, texts,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)

    line_thickness = 2
    # cv2.line(img, (x_nos, y_nos), (x_midhip, y_rhip), (0, 255, 0), thickness=line_thickness)
    # cv2.line(img, (x_lshould, y_lshould), (x_rshould, y_rshould), (247, 180, 38), thickness=line_thickness)

    cv2.imwrite('./static/im/im.jpg', img)

    get_event_results(pose1,name)

    time.sleep(1)


def video_pose_estimation(name):
    count = 1
    cap = cv2.VideoCapture(name)
    while count:
        frame_no = count*15
        cap.set(1, frame_no)
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = video_pose.process(imgRGB)
        pose1 = []
        # cordinates_xy=[]
        if results.pose_landmarks:

            for id, lm in enumerate(results.pose_landmarks.landmark):
                x_y_z = []
                # cx_cy=[]
                h, w, c = img.shape
                x_y_z.append(lm.x)
                x_y_z.append(lm.y)
                x_y_z.append(lm.z)
                x_y_z.append(lm.visibility)
                pose1.append(x_y_z)

                cx, cy = int(lm.x*w), int(lm.y*h)
                # cx_cy.append(cx)
                # cordinates_xy.append(cx_cy)
                if id % 2 == 0:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
                else:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        img = cv2.resize(img, (600, 800))
        # vertical line
        lex1, ley1 = int(pose1[2][0]*600), int(pose1[2][1]*800)
        rex1, rey1 = int(pose1[5][0]*600), int(pose1[5][1]*800)
        mid_eye = int((lex1+rex1)/2)

        x_lhip, y_lhip = int(pose1[31][0]*600), int(pose1[31][1]*800)
        x_rhip, y_rhip = int(pose1[32][0]*600), int(pose1[32][1]*800)
        x_midhip = int((x_lhip+x_rhip)/2)
        z_rhip=int(pose1[32][2]*(600*800/2))
        x_lshould, y_lshould = int(pose1[11][0]*600), int(pose1[11][1]*800)
        x_rshould, y_rshould = int(pose1[12][0]*600), int(pose1[12][1]*800)
        mid_shouldx = int((x_lshould+x_rshould)/2)
        mid_shouldy = int((y_lshould+y_rshould))
        line_thickness = 2
        cv2.line(img, (mid_eye, y_lshould), (x_midhip, y_rhip),
                 (0, 255, 0), thickness=line_thickness)
        cv2.line(img, (x_lshould, y_lshould), (x_rshould, y_rshould),
                 (247, 180, 38), thickness=line_thickness)
        cv2.circle(img, (z_rhip,y_rhip), 1, (155, 0, 55), cv2.FILLED)
        # cv2.line(img,(pose1[0][0:2],pose1[20][0:2],(255,0,0),5))
        cv2.imwrite('./static/im/im.jpg', img)

        get_event_results(pose1,name)
        # print(cordinates_xy)
        time.sleep(1)
        count += 1
