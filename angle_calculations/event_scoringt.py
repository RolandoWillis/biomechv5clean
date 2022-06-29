from xxlimited import foo
import numpy as np
import math
import pandas as pd
import os

# from pose import midpoint_hips, midpoint_shoulder 
import variables
import cv2

InOut = {}
OutIn = {}

def calculate_angle(a, b, c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c)
    
    v1 = a - b
    v2 = c - b
    
    v = np.dot(v1, v2)
    mag = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle =  np.arccos(v/mag)
    angle = (angle*180.0/np.pi)
    
    if angle > 180:
        return 360 - angle
    
    return angle


def new_calculate_angle(a,b,c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c)
    print('FLIPPED',a,b,c)
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle= np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle 


def get_percentage_diff(previous, current):
    try:
        percentage = abs(previous - current)/max(previous, current) * 100
    except ZeroDivisionError:
        percentage = float('inf')
    return percentage


def slope(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    return m

def dist2D(one, two):
    dx = one[0] - two[0]
    dy = one[1] - two[1]
    return math.sqrt(dx*dx + dy*dy)


class PosePoints:
    def __init__(self, pose):
        self.Nose = pose[0]
        self.L_Neck = pose[11]
        self.R_Neck = pose[12]
        self.R_Shoulder = pose[12]
        self.R_Elbow = pose[14]
        self.R_Wrist = pose[16]
        self.L_Shoulder = pose[11]
        self.L_Elbow = pose[13]
        self.L_Wrist = pose[15]
        self.R_Hip = pose[24]
        self.L_Hip = pose[23]
        self.R_Knee = pose[26]
        self.R_Ankle = pose[28]
        self.L_Knee = pose[25]
        self.L_Ankle = pose[27]
        self.R_Eye = pose[5]
        self.L_Eye = pose[2]
        self.R_Ear = pose[8]
        self.L_Ear = pose[7]
        self.L_Foot = pose[31]
        self.R_Foot = pose[32]
        self.R_Index = pose[20]
        self.L_Index = pose[19]
        self.L_Thumb = pose[21]
        self.R_Thumb = pose[22]
        self.L_Pinky = pose[17]
        self.R_Pinky = pose[18]
class PoseAngles:
    def __init__(self, _pose) -> None:
        self.pose = PosePoints(_pose)

        self.spine_flexion_stat = ''
        self.spineRotationStat = ''
        self.spine_axial_rotation = 0
        self.spine_flexion_extension = 0
        self.arm_orientation_height = 0
        self.eulerTest=0
        self.eulerTestStat=''
        self.pelvic_tilt=''

    def calculate(self):
        self.calculate_spine_axial_angle()
        self.calculate_spine_flexion_angle()
        self.calculate_arm_rotation_angle()
        self.calculate_euler_test_angle()
        self.calculate_pelvic_tilt()

        return {'spine_axial_rotation': self.spine_axial_rotation,
                'spineRotationStat':self.spineRotationStat, 
                'spine_flexion_extension': self.spine_flexion_extension,
                'arm_orientation_height': self.arm_orientation_height,
                'spine_flexion_stat': self.spine_flexion_stat,
                'eulerTest':self.eulerTest,
                'eulerTestStat':self.eulerTestStat,
                'pelvic_tilt':self.pelvic_tilt}

    def calculate_spine_axial_angle(self):
        pose = self.pose

       
        neck_diff = np.array(pose.L_Neck[0:3]) + np.array(pose.R_Neck[0:3])
        neck_diff = neck_diff/2

        eye_diff  = np.array(pose.L_Eye[0:3]) + np.array(pose.R_Eye[0:3])
        eye_diff  =eye_diff/2

        eyeneck_diff = np.array(neck_diff) + np.array(eye_diff) 
        eyeneck_diff = eyeneck_diff/2

        mid_foot = np.array(pose.L_Foot[0:3]) + np.array(pose.R_Foot[0:3])
        mid_foot = mid_foot/2

        shoulder_diff=np.array(pose.L_Shoulder[0:3]) + np.array(pose.R_Shoulder[0:3])
        shoulder_diff=shoulder_diff/2

        # midshould_neck = np.array(pose.L_Shoulder) + np.array(pose.R_Shoulder) + np.array(pose.L_Neck) + np.array(pose.R_Neck)
        # midshould_neck =  midshould_neck/
        
        midshould_neck = np.array(neck_diff) + np.array(shoulder_diff) 
        midshould_neck = midshould_neck/2

        lshoulder_elbow =np.array(pose.L_Shoulder[0:3]) + np.array(pose.L_Elbow[0:3]) 
        lshoulder_elbow = lshoulder_elbow/2

        hip_diff = np.array(pose.L_Hip[0:3]) + np.array(pose.R_Hip[0:3])
        hip_diff = hip_diff/2

        centerum= np.array(hip_diff) + np.array(shoulder_diff)
        centerum = centerum/2

      
        angle = calculate_angle(hip_diff,shoulder_diff,pose.L_Shoulder[0:3] )

        # angle = angle - 180
        # angle = angle -90
        # angle = abs(angle)
        if angle >= 160 and angle <= 180 : 
            angle =angle - 90 
        # if angle > 90:
        # if angle > 50:
        #     angle = angle + 90
        # #     angle = 90 -angle 
        # if pose.L_Shoulder
        self.spine_axial_rotation = angle
        
        # if 
        # self.spineRotationStat=

    
    

        # if chest_direction[2] > 0:

        # angle = -angle

    # def create_csv_coords(self,frame):
    #     pose = self.pose
    #     df= pd.DataFrame({'title':frame,'left_should_x':pose.L_Shoulder[0],'left_shoulder_y':pose.L_Shoulder[1],
    #     'right_should_x':pose.R_Shoulder[0],'right_shoulder_y':pose.R_Shoulder[1]
        
        
        
    #     },index=[0])
    #     filepath,ext=os.path.split(frame)
    #     df.to_csv(filepath+'.csv', sep='\t',mode='a',header=False)
    #     print('DATASET',df)




    def calculate_pelvic_tilt(self):
        pose= self.pose
        # print('RHIP Y',pose.R_Hip[2])
        # print('LHIP Y',pose.L_Hip[2])
        rhipy=np.around(pose.R_Hip[1],decimals=4)
        lhipy=np.around(pose.L_Hip[1],decimals=4)
        
        yhip_difference=get_percentage_diff(rhipy,lhipy)
        # print('PERCENTAGE HIPS',get_percentage_diff(rhipy,lhipy))
        if yhip_difference > 2:
            if lhipy > rhipy:
                self.pelvic_tilt='Left Pelvic Tilt'
            elif rhipy < lhipy:
                self.pelvic_tilt='Right Pelvic Tilt'
        else :
            self.pelvic_tilt ='Neutral Pelvic Tilt'

        # print('RHIP Y',np.array(rhipy-lhipy/lhipy))
        # print('LHIP Y',np.array(lhipy-rhipy/rhipy))
        # def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
        #     return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
        # print('ABSOLUTE DIFF HIPP Y',isclose(pose.R_Hip[1],pose.L_Hip[1]))
    def calculate_spine_flexion_angle(self):
        pose = self.pose

        # print ('z of HIP',pose.R_Hip[3], pose.L_Hip[3])

        neck_diff = np.array(pose.L_Neck) + np.array(pose.R_Neck)
        neck_diff = neck_diff/2

        eye_diff  = np.array(pose.L_Eye) + np.array(pose.R_Eye)
        eye_diff  =eye_diff/2

        eyeneck_diff = np.array(neck_diff) + np.array(eye_diff) 
        eyeneck_diff = eyeneck_diff/2

        mid_foot = np.array(pose.L_Foot) + np.array(pose.R_Foot)
        mid_foot = mid_foot/2

        shoulder_diff=np.array(pose.L_Shoulder[0:3]) + np.array(pose.R_Shoulder[0:3])
        shoulder_diff=shoulder_diff/2

        nosez=np.array(pose.Nose[0:3])
        shouldz=np.array(pose.L_Shoulder[0:3]) + np.array(pose.R_Shoulder[0:3])
        shouldz=shouldz/2
        hipz= np.array(pose.L_Hip[0:3]) + np.array(pose.R_Hip[0:3])
        hipz=hipz/2
        # midshould_neck = np.array(pose.L_Shoulder) + np.array(pose.R_Shoulder) + np.array(pose.L_Neck) + np.array(pose.R_Neck)
        # midshould_neck =  midshould_neck/
        footz=np.array(pose.L_Knee[0:3]) + np.array(pose.R_Knee[0:3])
        footz=footz/2
        
        midshould_neck = np.array(neck_diff[0:3]) + np.array(shoulder_diff[0:3]) 
        midshould_neck = midshould_neck/2

        hip_diff = np.array(pose.L_Hip[0:3]) + np.array(pose.R_Hip[0:3])
        hip_diff = hip_diff/2

        centerum= np.array(hip_diff[0:3]) + np.array(shoulder_diff[0:3])
        centerum = centerum/2
        y_midhip= hip_diff[1]
        x_midhip= hip_diff[0]
        x_nos=nosez[0]
        y_nos=nosez[1]
        new_shoulder= np.array(pose.R_Shoulder[0]) + np.array(pose.L_Shoulder[1])  +np.array(pose.L_Shoulder[2]) + np.array(pose.L_Shoulder[3])
        
        # angle = new_calculate_angle(footz,centerum,shoulder_diff)
        
        newhipdiff = np.array(pose.L_Hip[0:3]) + np.array(pose.R_Hip[0:3])
        newhipdiff = newhipdiff/2
        print('NEWHIPZZZ',newhipdiff)
        centerumz=centerum[1]
        centerumy=centerum[0]
        shoulder_diffz=shoulder_diff[1]
        shoulder_diffy=shoulder_diff[0]
        footzz=footz[1]
        footzy=footz[0]
        hipz=newhipdiff[1]
        hipy=newhipdiff[0]
        z_lshoulder=np.array(pose.L_Shoulder[2])
        y_lshould=np.array(pose.L_Shoulder[1])
        z_rshoulder=np.array(pose.R_Shoulder[2])
        y_rshould=np.array(pose.R_Shoulder[1])
        x_rshould=np.array(pose.R_Shoulder[0])
        x_lshould=np.array(pose.L_Shoulder[0])
        y_mid_shoulder=((y_rshould+y_lshould)/2)
        x_mid_should=((x_rshould+x_lshould)/2)
        print('LEFTSHOULD x',pose.L_Shoulder[0])
        print('RightShould x',pose.R_Shoulder[0])

        print('LeftShould y',pose.L_Shoulder[1])

        print('Rightshould y',pose.R_Shoulder[1])
        print('LeftShould z',pose.L_Shoulder[2])
        print('Rightshould z',pose.R_Shoulder[2])
        # print('leftwrist_x',pose.L_Wrist[0])
        # print('leftwrist_y',pose.L_Wrist[1])
        print('LeftHipx',pose.L_Hip[0])
        print('RightHipx',pose.R_Hip[0])
        print('LeftHipy',pose.L_Hip[1])
        print('RightHipy',pose.R_Hip[1])
        print('LeftHipz',pose.L_Hip[2])
        print('RightHipz',pose.R_Hip[2])

        nosey= np.array(pose.Nose[1])
        # M1 = (z_lshoulder)/(y_lshoulder) #horizontal
        # M2 = (footzz)/(nosey) #vertical


        M2 = (y_midhip-y_mid_shoulder)/(x_midhip-x_mid_should) # vertical#####
    
    
        M1 = (y_rshould-y_lshould)/(x_rshould-x_lshould) #horizontal
        # M1 = (centerumz-shoulder_diffz)/() 
        # M2 = (centerumz-shoulder_diffz)/(centerumy-shoulder_diffy)
        nangle = (M2 - M1) / (1 + M1 * M2)
        
        ret = math.atan(nangle)
        
        val = (ret * 180) / math.pi
        # val =  180 - val 
        print('NExxxW',val)
        angle = new_calculate_angle(footz,newhipdiff,centerum)
        angle = angle - 180

        # angle =  val - angle
        self.spine_flexion_extension = angle

        if (angle < 1):
            self.spine_flexion_stat = 'Extension'

        elif angle >= 12 and angle <=25 :
            self.spine_flexion_stat = 'Light Flexion'
        
        elif angle >= 25 and angle <= 45:
            self.spine_flexion_stat = 'Moderate Flexion'
        elif angle >=45 :
            self.spine_flexion_stat = 'Flexion Out of Range'

    def calculate_euler_test_angle(self):

           #TEST FOR PINKY HIDDEN
        # if pose.L_Pinky[2] < pose.L_Index[2]:
        #     print('pinky is showing')
        # elif pose.L_Pinky[2] > pose.L_Index[2]:
        #     print( 'hidden')

        # x_diff_index_pink = pose.L_Index[0] -pose.L_Pinky[0] /
        # print ('X DIFFERENCE BETWEEN INDEX / PINKY', x_diff_index_pink) 
        
        # y_diff_index_pink = pose.L_Index[1] -pose.L_Pinky[1]
        # print ('Y DIFFERENCE BETWEEN INDEX / PINKY', y_diff_index_pink) 
        
        # leftpinky_to_thumb=dist2D(pose.L_Thumb,pose.L_Pinky)
        pose = self.pose
        neck_diff = np.array(pose.L_Neck[0:3]) + np.array(pose.R_Neck[0:3])
        neck_diff = neck_diff/2

        wrist_diff = np.array(pose.L_Wrist) + np.array(pose.R_Wrist)
        wrist_diff = wrist_diff/2

        indexfinger_diff = np.array(pose.L_Index) + np.array(pose.R_Index)
        indexfinger_diff = indexfinger_diff/2

        print('indexfinger',pose.L_Index)
        print('pinkyfinger',pose.L_Pinky)

        mid_foot = np.array(pose.L_Foot) + np.array(pose.R_Foot)
        mid_foot = mid_foot/2
        shoulder_diff=np.array(pose.L_Shoulder[0:3]) + np.array(pose.R_Shoulder[0:3])
        shoulder_diff=shoulder_diff/2

        hip_diff = np.array(pose.L_Hip[0:3]) + np.array(pose.R_Hip[0:3])
        hip_diff = hip_diff/2
        centerum= np.array(hip_diff) + np.array(shoulder_diff)
        centerum = centerum/2
        print('HIPDIFF',hip_diff)
     
        midthumb=np.array(pose.L_Thumb) + np.array(pose.R_Thumb)
        midthumb=midthumb/2
        z_rhip=np.array(pose.R_Hip[2])
        z_lhip=np.array(pose.L_Hip[2])
        z_midhip=int((z_lhip+z_rhip)/2)
        y_rhip=np.array(pose.R_Hip[1])
        y_lhip=np.array(pose.L_Hip[1])
        y_midhip=int((y_lhip+y_rhip)/2)
        z_lshoulder=np.array(pose.L_Shoulder[2])
        y_lshould=np.array(pose.L_Shoulder[1])
        z_rshoulder=np.array(pose.R_Shoulder[2])
        y_rshould=np.array(pose.R_Shoulder[1])
        x_rshould=np.array(pose.R_Shoulder[0])
        x_lshould=np.array(pose.L_Shoulder[0])
        y_mid_shoulder=((y_rshould+y_lshould)/2)
        z_mid_shoulder=((z_rshoulder+z_lshoulder)/2)
        y_nose=np.array(pose.Nose[1])
        rfooty=np.array(pose.R_Foot[1])
        lfooty=np.array(pose.L_Foot[1])
        # y_lankle=np.array(pose.L_Ankle[1])
        # y_rankle=np.array(pose.R_Ankle[1])
        # left_ank_hip=int((pose.L_Ankle+))
        print('RFOOTY',rfooty)
        print('LFOOTY',lfooty)
        print('YNOSE',y_nose)

        # taangle = calculate_angle(hip_diff,shoulder_diff,)
        

        M2 = (y_midhip-y_mid_shoulder)/(z_midhip-z_mid_shoulder) # vertical#####

        M1 = (y_rshould-y_lshould)/(z_rshoulder-z_lshoulder) #horizontal
       
        angle = (M2 - M1) / (1 + M1 * M2)
 
        ret = math.atan(angle)
       
        val = (ret * 180) / math.pi
        # print('tangle',taangle)
        self.eulerTest = val
        
        if (angle < 0):
            self.eulerTestStat = 'Neutral Pelvis'

        elif angle >= 0 and angle <= 25:
            self.eulerTestStat = 'Strong Pelvis Turn'
        else:
            self.eulerTestStat = 'Weak Pelvis Turn'

    
    def calculate_arm_rotation_angle(self):
        pose = self.pose

       
        shoulder = np.array( pose.L_Shoulder[0:2]) + np.array( pose.R_Shoulder[0:2])
        shoulder = shoulder/2

        # shoulder = pose.L_Shoulder
        elbow = pose.L_Elbow[0:2]
        wrist = pose.L_Wrist[0:2]

        #lead arm orientation-vertical - no.8
        vertice= pose.L_Shoulder[0],800
        horizon= 0,pose.L_Shoulder[1]
        angle = calculate_angle( vertice,pose.L_Shoulder[0:2], pose.L_Wrist[0:2]) # arm orientation lead vertical
        # angle = calculate_angle( horizon,pose.L_Shoulder[0:2], pose.L_Wrist[0:2]) # arm orientation lead depth

        # angle = angle - 90
        self.arm_orientation_height = angle    

class EventAnalysis:
    def __init__(self, angles) -> None:
        self.angles = angles
        self.event_count = 3

    def event1(self):
        return (self.angles['spine_axial_rotation'] >= 0 and self.angles['spine_axial_rotation'] <= 0) or \
            (self.angles['spine_flexion_extension'] >= 30 and self.angles['spine_flexion_extension'] <= 40)

    def event2(self):
        return (self.angles['spine_axial_rotation'] >= 70 and self.angles['spine_axial_rotation'] <= 125) or \
            (self.angles['spine_flexion_extension'] >= -15 and self.angles['spine_flexion_extension'] <= 15) 
            # (self.angles['arm_orientation_height'] >= 25 and self.angles['arm_orientation_height'] <= 40)

    def event3(self):
        return (self.angles['spine_axial_rotation'] >= 15 and self.angles['spine_axial_rotation'] <= 40) or \
            (self.angles['spine_flexion_extension'] >= 30 and self.angles['spine_flexion_extension'] <= 40) or \
            (self.angles['arm_orientation_height'] >= 15 and self.angles['arm_orientation_height'] <= 30) 

    def analyse(self):
        event_results = []

        for i in range(self.event_count):
            event_results.append(getattr(self, f'event{i+1}')())

        return event_results