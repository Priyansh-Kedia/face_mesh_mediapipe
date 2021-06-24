import math

def radians(a1, a2, b1, b2):
  return math.atan2(b2 - a2, b1 - a1)

def cal(mesh): #10, 33, 152, 263
  roll = math.degrees(radians(mesh[1][0], mesh[1][1], mesh[3][0], mesh[3][1]))
  yaw =math.degrees( radians(mesh[1][0], mesh[1][2], mesh[3][0], mesh[3][2]))
  pitch= math.degrees(radians(mesh[0][1], mesh[0][2], mesh[2][1], mesh[2][2]))
  # print(f"yaw {yaw}, roll {roll}, pitch {pitch}")
  # print(f"yaw {(yaw)}, roll {(roll)}, pitch {(pitch)}")
  return {"yaw": yaw, "roll":roll, "pitch":pitch}