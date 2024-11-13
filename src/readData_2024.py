import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import find_peaks

# Define motion names
name_motion = ['Walking', 'Jogging', 'Crouch']
name_grf = ['Walking_FP', 'Jogging_FP', 'Crouch_FP']

# Set the motion index
index = 0  # Example: index=0 for Walking, index=1 for Jogging, index=2 for Crouch

# Read data
data_trc = pd.read_csv(f"{name_motion[index]}.csv")
data_grf = pd.read_csv(f"{name_grf[index]}.csv")


# Downsample ground reaction data to match the trajectory data length
data_grf_s = data_grf.iloc[::10, :].reset_index(drop=True)

# Conversion factor from mm to meters
to_meters = 1 / 1000

#constants for momentum calculations
bodyWeight = 72.8
footWeight = 0.0145 * bodyWeight
gravity = 9.81

# Extract and convert relevant columns from marker trajectory data
RTOE_x = data_trc['RTOO_Y'] * to_meters
RTOE_y = data_trc['RTOO_Z'] * to_meters
LTOE_x = data_trc['LTOO_Y'] * to_meters
LTOE_y = data_trc['LTOO_Z'] * to_meters

RANKLE_x = data_trc['RAJC_Y'] * to_meters
RANKLE_y = data_trc['RAJC_Z'] * to_meters
LANKLE_x = data_trc['LAJC_Y'] * to_meters
LANKLE_y = data_trc['LAJC_Z'] * to_meters

RKNEE_x = data_trc['RKJC_Y'] * to_meters
RKNEE_y = data_trc['RKJC_Z'] * to_meters
LKNEE_x = data_trc['LKJC_Y'] * to_meters
LKNEE_y = data_trc['LKJC_Z'] * to_meters

RHIP_x = data_trc['RHJC_Y'] * to_meters
RHIP_y = data_trc['RHJC_Z'] * to_meters
LHIP_x = data_trc['LHJC_Y'] * to_meters
LHIP_y = data_trc['LHJC_Z'] * to_meters

PELO_x = data_trc['PELO_Y'] * to_meters
PELO_y = data_trc['PELO_Z'] * to_meters
PELP_x = data_trc['PELP_Y'] * to_meters
PELP_y = data_trc['PELP_Z'] * to_meters

TRXO_x = data_trc['TRXO_Y'] * to_meters
TRXO_y = data_trc['TRXO_Z'] * to_meters
TRXP_x = data_trc['TRXP_Y'] * to_meters
TRXP_y = data_trc['TRXP_Z'] * to_meters

FP1_force_x = data_grf_s['FP1_Force_Y']
FP1_force_y = data_grf_s['FP1_Force_Z']
FP1_COP_x = data_grf_s['FP1_COP_Y'] * to_meters
FP1_COP_y = data_grf_s['FP1_COP_Z'] * to_meters

FP2_force_x = data_grf_s['FP2_Force_Y']
FP2_force_y = data_grf_s['FP2_Force_Z']
FP2_COP_x = data_grf_s['FP2_COP_Y'] * to_meters
FP2_COP_y = data_grf_s['FP2_COP_Z'] * to_meters

leftFootVectorLenght = LANKLE_x[41] - LTOE_x[41]
rightFootVectorLeght = (RANKLE_x[41] - RTOE_x[41], RANKLE_y[41] - RTOE_y[41])
rightFootLength = np.sqrt(rightFootVectorLeght[0]**2 + rightFootVectorLeght[1]**2)

#arrays to be populated during frame analysis
rightAnkleAngles = []
leftAnkleAngles = []

rightAnkleRadiants = []
leftAnkleRadiants = []

rightKneeAngles = []
leftKneeAngles = []

rightKneeRadiants = []
leftKneeRadiants = []


rightHipAngle = []
leftHipAngle = []

rightHipRadiants = []
leftHipRadiants = []

pelvisAngleR = []
pelvisAngleL = []

rightPelvisRadiants = []
leftPelvisRadiants = []


trunkAngleR = []
trunkAngleL = []

rightTrunkRadiants = []
leftTrunkRadiants = []


rightFootGXCoord= []
rightFootGYCoord= []

leftFootGXCoord = []
leftFootGYCoord = []

rightFootGRFX= []
rightFootGRFY= []

leftFootGRFX= []
leftFootGRFY= []

rightFootXCOP= []
rightFootYCOP= []

leftFootXCOP= []
leftFootYCOP= []

absoluteAnglesRightAnkle = []

horizontal_vector = np.array([1, 0])


def calcAngleInRadiantsBetweenVectors(vec1, vec2):
    # Convert input vectors to numpy arrays
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    
    # Calculate the angle in radians and then convert to degrees
    angle_radians = np.arccos(dot_product / (magnitude_vec1 * magnitude_vec2))
    
    return angle_radians

def calcHipAngle(vec1,vec2):
      # Convert input vectors to numpy arrays
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Calculate the dot product and magnitudes of the vectors
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    
    # Calculate the angle in radians using arccos of the normalized dot product
    angle_radians = np.arccos(dot_product / (magnitude_vec1 * magnitude_vec2))
    
    # Calculate the cross product to determine the angle's sign in 2D
    cross_product = np.cross(vec1, vec2)
    
    # Convert to degrees and adjust for full 0-360 range
    angle_degrees = np.degrees(angle_radians)
    
    # If cross product is negative, adjust angle to reflect 360-degree range
    if cross_product < 0:
        angle_degrees = 360 - angle_degrees
    
    return angle_degrees

def calculateFA(listOfPositions, listOfForces, weight, isY):
    timeStamp = 0.01
    listOfPositionsVelocityOnG = np.gradient(listOfPositions,timeStamp)
    listOfPositionsAccellerationOnG = np.gradient(listOfPositionsVelocityOnG, timeStamp)
    accelerationArray = weight * listOfPositionsAccellerationOnG
    pureAccellerationValue = listOfPositionsAccellerationOnG
    if isY != True:
        return accelerationArray + listOfForces
    else:
        return accelerationArray - listOfForces

def plotAngleForJoint(right_angles,left_angles, joint):
    timestamps = np.arange(len(right_angles))
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, right_angles, marker='o', linestyle='-', color='g')
    plt.plot(timestamps, left_angles, marker='o', linestyle='-', color='r')

    plt.title("Angle Variation Over Time for Right and Left" + joint)
    plt.xlabel("Time" + (" (seconds)" if timestamps is not None else " (Index)"))
    plt.ylabel("Angle (degrees)")
    plt.grid(True)
    plt.show()

def plot (value1):
    timestamps = np.arange(len(value1))
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, value1, marker='o', linestyle='-', color='g')
    plt.grid(True)
    plt.show()

def plot4(value1,value2 = [],value3 = [] ,value4 = []):
    timestamps = np.arange(len(value1))
    # Plotting
    plt.figure(figsize=(10, 5))
    if value1.any():
        plt.plot(timestamps, value1, marker='o', linestyle='-', color='g')
    if value2.any():
        plt.plot(timestamps, value2, marker='o', linestyle='-', color='b')
    if value3.any() :
        plt.plot(timestamps, value3, marker='o', linestyle='-', color='r')
    if value4.any() :
        plt.plot(timestamps, value4, marker='o', linestyle='-', color='y')    

    plt.grid(True)
    plt.show()



def plotAngleForAbsolute(right_angles,left_angles, joint):
    timestamps = np.arange(len(right_angles))
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, right_angles, marker='o', linestyle='-', color='g')
    plt.plot(timestamps, left_angles, marker='o', linestyle='-', color='r')

    plt.title("Absolute " + joint + " angle over time")
    plt.xlabel("Time" + (" (seconds)" if timestamps is not None else " (Index)"))
    plt.ylabel("Angle (degrees)")
    plt.grid(True)
    plt.show()

def calculateTheMa(GRFx,GRFy, COPx,COPy, Gx,Gy, FAx, FAy, Ay, Ax, Igf, Of):
    COPy = np.array(COPy)
    Gy = np.array(Gy)
    GRFx = np.array(GRFx)
    GRFy = np.array(GRFy)
    COPx = np.array(COPx)
    Gx = np.array(Gx)
    FAx = np.array(FAx)
    FAy = np.array(FAy)
    Ay = np.array(Ay)
    Ax = np.array(Ax)
    # Ma = []
    # for i in range(len(COPx)):
    #     if COPx[i] <= Gx[i]:
    #         Ma.append(np.multiply(Igf, Of) - GRFx[i] * (COPy[i] - Gy[i]) - GRFy[i] * (COPx[i] - Gx[i]) + FAx[i] * (Ay[i] - Gy[i]) + FAy[i] * (Gx[i] - Ax[i]))
    #     else:
    #         Ma.append(np.multiply(Igf, Of) - GRFx[i] * (COPy[i] - Gy[i]) - GRFy[i] * (COPx[i] - Gx[i]) + FAx[i] * (Ay[i] - Gy[i]) + FAy[i] * (Gx[i] - Ax[i]))
    
    FX = FAx * (Ay - Gy)
    FY = FAy * (Gx - Ax)
    GFX =  GRFx * (COPy - Gy) 
    GFY = GRFy * (COPx - Gx )

    Ma = np.multiply(Igf, Of) - FX + FY - GFX - GFY
    
    plot4(FX,FY,GFX,GFY)

    return Ma 


for i in range(46, 151):
    
    #Vectors
    vr_right_Knee = np.array([RKNEE_x[i] - RANKLE_x[i], RKNEE_y[i] - RANKLE_y[i]])
    vr_right_Ankle = np.array([RANKLE_x[i] - RTOE_x[i], RANKLE_y[i] - RTOE_y[i]])
    vr_right_Hip = np.array([RHIP_x[i] - RKNEE_x[i], RHIP_y[i] - RKNEE_y[i]])
    pelvic_vector = np.array([PELO_x[i] - PELP_x[i], PELO_y[i] - PELP_y[i]])    
    trunk_vector = np.array([TRXO_x[i] - TRXP_x[i], TRXO_y[i] - TRXP_y[i]])    
    
    #Centers
    rightFootGX = (RTOE_x[i] + RANKLE_x[i]) / 2
    rightFootGY = (RTOE_y[i] + RANKLE_y[i]) / 2

    #GroundForces
    RightGRFx = FP2_force_x[i]
    RightGRFy = FP2_force_y[i]
    RightCopGRFx = FP2_COP_x[i]
    RightCopGRFy = FP2_COP_y[i]
    
    
    # Calculate angle between vectors and add to list
    rightAnkleAngles.append(np.degrees(calcAngleInRadiantsBetweenVectors(vr_right_Knee, vr_right_Ankle)) - 80)
    rightAnkleRadiants.append(calcAngleInRadiantsBetweenVectors(vr_right_Knee, vr_right_Ankle))

    rightKneeAngles.append(np.degrees(calcAngleInRadiantsBetweenVectors(vr_right_Hip,vr_right_Knee)))
    rightKneeRadiants.append(calcAngleInRadiantsBetweenVectors(vr_right_Hip,vr_right_Knee))

    rightHipAngle.append(calcHipAngle(pelvic_vector, vr_right_Hip)-180)
    rightHipRadiants.append(calcAngleInRadiantsBetweenVectors(pelvic_vector,vr_right_Hip))

    pelvisAngleR.append(np.degrees(calcAngleInRadiantsBetweenVectors(pelvic_vector, horizontal_vector)) - 90)
    rightPelvisRadiants.append(calcAngleInRadiantsBetweenVectors(pelvic_vector, horizontal_vector))

    trunkAngleR.append(np.degrees(calcAngleInRadiantsBetweenVectors(trunk_vector, horizontal_vector)) - 90)
    rightTrunkRadiants.append(calcAngleInRadiantsBetweenVectors(trunk_vector, horizontal_vector))

    #Coordinates for the centers of gravitiy
    rightFootGXCoord.append(rightFootGX)
    rightFootGYCoord.append(rightFootGY)
    
    #Forces Array
    rightFootGRFX.append(RightGRFx)
    rightFootGRFY.append(RightGRFy)

    #COP
    rightFootXCOP.append(RightCopGRFx)
    rightFootYCOP.append(RightCopGRFy)


for j in range(98, 203):
    
    #Vectors
    vr_left_Knee = np.array([LKNEE_x[j] - LANKLE_x[j], LKNEE_y[j] - LANKLE_y[j]])
    vr_left_Ankle = np.array([LANKLE_x[j] - LTOE_x[j], LANKLE_y[j] - LTOE_y[j]])
    vr_left_Hip = np.array([LHIP_x[j] - LKNEE_x[j], LHIP_y[j] - LKNEE_y[j]])
    pelvic_vector = np.array([PELO_x[j] - PELP_x[j], PELO_y[j] - PELP_y[j]])
    trunk_vector = np.array([TRXO_x[j] - TRXP_x[j], TRXO_y[j] - TRXP_y[j]])    

    #Centers
    leftFootGX = (LTOE_x[j] + LANKLE_x[j]) / 2
    leftFootGY = (LTOE_y[j] + LANKLE_y[j]) / 2
    
    #GroundForces
    LeftGRFx = FP1_force_x[j]
    LeftGRFy = FP1_force_y[j]
    LeftCopGRFx = FP1_COP_x[j]
    LeftCopGRFy = FP1_COP_y[j]

    # Calculate angle between vectors and radiants between vectors saving them in arrays
    leftAnkleAngles.append(np.degrees(calcAngleInRadiantsBetweenVectors(vr_left_Knee, vr_left_Ankle)) - 85)
    leftAnkleRadiants.append(calcAngleInRadiantsBetweenVectors(vr_left_Knee, vr_left_Ankle))

    leftKneeAngles.append(np.degrees(calcAngleInRadiantsBetweenVectors(vr_left_Hip,vr_left_Knee)))
    leftKneeRadiants.append(calcAngleInRadiantsBetweenVectors(vr_left_Hip,vr_left_Knee))

    
    leftHipAngle.append(calcHipAngle( pelvic_vector,vr_left_Hip)-180)
    leftHipRadiants.append(calcAngleInRadiantsBetweenVectors(pelvic_vector,vr_left_Hip))

    pelvisAngleL.append(np.degrees(calcAngleInRadiantsBetweenVectors(pelvic_vector, horizontal_vector))- 90)
    leftPelvisRadiants.append(calcAngleInRadiantsBetweenVectors(pelvic_vector, horizontal_vector))

    trunkAngleL.append(np.degrees(calcAngleInRadiantsBetweenVectors(trunk_vector, horizontal_vector)) - 90)
    leftTrunkRadiants.append(calcAngleInRadiantsBetweenVectors(trunk_vector, horizontal_vector))

    #Coordinates for the centers of gravitiy        
    leftFootGXCoord.append(leftFootGX)
    leftFootGYCoord.append(leftFootGY)

    #Forces Array
    leftFootGRFX.append(LeftGRFx)
    leftFootGRFY.append(LeftGRFy)

    #COP
    leftFootXCOP.append(LeftCopGRFx)
    leftFootYCOP.append(LeftCopGRFy)

    #AbsoluteAngles
#Calculate the forces

rightFootSumFx = calculateFA(rightFootGXCoord,rightFootGRFX, footWeight, False)
leftFootSumFx = calculateFA(leftFootGXCoord,leftFootGRFX, footWeight, False)

rightFootSumFy = calculateFA(rightFootGYCoord,rightFootGRFY,footWeight, True)
leftFootSumFy = calculateFA(leftFootGYCoord,leftFootGRFY, footWeight,True)


timestamps = np.arange(len(rightFootSumFx))
rightFootIgf  = ((0.450 * rightFootLength) ** 2) * footWeight
subset_RANKLE_x = RANKLE_x[46:151]  # End index is exclusive, so use 152
subset_RANKLE_y = RANKLE_y[46:151]
acclerationOfAnkle = np.gradient(rightAnkleRadiants,1)
velocityOfAnkle = np.gradient(acclerationOfAnkle,1)

rightAnkleMoment = calculateTheMa(rightFootGRFX,rightFootGRFY,rightFootXCOP,rightFootYCOP,rightFootGXCoord,rightFootGYCoord, rightFootSumFx,rightFootSumFy,subset_RANKLE_x,subset_RANKLE_y,rightFootIgf,velocityOfAnkle)

timestamps = np.arange(len(rightAnkleMoment))

# # Plotting
# plt.figure(figsize=(10, 5))
# plt.plot(timestamps, rightAnkleMoment, marker='o', linestyle='-', color='g')
# plt.title("")
# plt.xlabel("Time" + (" (seconds)" if timestamps is not None else " (Index)"))
# plt.ylabel("")
# plt.grid(True)
# plt.show()

plot(rightAnkleMoment)

# # Plotting
# plt.figure(figsize=(10, 5))
# plt.plot(timestamps, rightFootGXCoord, marker='o', linestyle='-', color='g')
# plt.plot(timestamps, rightFootXCOP, marker='o', linestyle='-', color='b')
# plt.title("")
# plt.xlabel("Time" + (" (seconds)" if timestamps is not None else " (Index)"))
# plt.ylabel("")
# plt.grid(True)
# plt.show()

# timestamps = np.arange(len(rightAnkleAngles))

# # Plot the calculated angles
# plotAngleForJoint(rightAnkleAngles,leftAnkleAngles, " Ankle")

# plotAngleForJoint(rightKneeAngles,leftKneeAngles, " Knee")

# plotAngleForJoint(rightHipAngle,leftHipAngle, " Hip")

# plotAngleForAbsolute(pelvisAngleR,pelvisAngleL, "Pelvis")

# plotAngleForAbsolute(trunkAngleR,trunkAngleL, "Trunk")




# # Animation setup
# run_animation = False
# color_body = "#0077be"
# num_frames = len(data_trc)

# if run_animation:
#     fig, ax = plt.subplots(figsize=(12, 7))
#     fig.patch.set_facecolor('white')

#     def update(frame):
#         ax.clear()
#         i = frame * 5  # Increment frame by 5 to skip some frames for speed
#         if i >= num_frames:
#             return

#         # Plot body segments
#         ax.plot([RANKLE_x[i], RTOE_x[i]], [RANKLE_y[i], RTOE_y[i]], color=color_body, linewidth=2)
#         ax.plot([RANKLE_x[i], RKNEE_x[i]], [RANKLE_y[i], RKNEE_y[i]], color='g', linewidth=2)
#         ax.plot([LANKLE_x[i], LTOE_x[i]], [LANKLE_y[i], LTOE_y[i]], color=color_body, linewidth=2)
#         ax.plot([LANKLE_x[i], LKNEE_x[i]], [LANKLE_y[i], LKNEE_y[i]], color='r', linewidth=2)
#         ax.plot([LHIP_x[i], LKNEE_x[i]], [LHIP_y[i], LKNEE_y[i]], color=color_body, linewidth=2)
#         ax.plot([RHIP_x[i], RKNEE_x[i]], [RHIP_y[i], RKNEE_y[i]], color=color_body, linewidth=2)
#         ax.plot([PELO_x[i], PELP_x[i]], [PELO_y[i], PELP_y[i]], color=color_body, linewidth=2)
#         ax.plot([TRXO_x[i], TRXP_x[i]], [TRXO_y[i], TRXP_y[i]], color=color_body, linewidth=2)

#         # Add text annotations
#         ax.text(RANKLE_x[i], RANKLE_y[i], 'Rankle', fontsize=8)
#         ax.text(LANKLE_x[i], LANKLE_y[i], 'Lankle', fontsize=8)
#         ax.text(RKNEE_x[i], RKNEE_y[i], 'Rknee', fontsize=8)
#         ax.text(LKNEE_x[i], LKNEE_y[i], 'Lknee', fontsize=8)

#         # Ground reaction force visualization
#         if i >= 1:
#             if FP1_COP_x[i] != FP1_COP_x[i - 1]:
#                 ax.plot(FP1_COP_x[i], FP1_COP_y[i], 'ok', linewidth=2)
#                 ax.quiver(FP1_COP_x[i], FP1_COP_y[i], FP1_force_x[i] / 1000, FP1_force_y[i] / 1000,
#                           angles='xy', scale_units='xy', color='r', linewidth=2)
#             if FP2_COP_x[i] != FP2_COP_x[i - 1]:
#                 ax.plot(FP2_COP_x[i], FP2_COP_y[i], 'ok', linewidth=2)
#                 ax.quiver(FP2_COP_x[i], FP2_COP_y[i], FP2_force_x[i] / 1000, FP2_force_y[i] / 1000,
#                           angles='xy', scale_units='xy', color='r', linewidth=2)

#         # Set plot limits and labels
#         ax.set_xlim(-3.5, 3.5)
#         ax.set_ylim(-0.25, 3)
#         ax.set_xlabel("horizontal distance [m]", fontsize=20)
#         ax.set_ylabel("vertical distance [m]", fontsize=20)

#     # Create animation
#     anim = FuncAnimation(fig, update, frames=num_frames // 5, interval=100)
#     plt.show()
