import numpy as np
import pandas as pd

landmark_names = [
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky', 'right_pinky',
        'left_index', 'right_index',
        'left_thumb', 'right_thumb',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]

def landmark_2_array(landmarks):
  """
    Convert a list of landmark objects to a 3D numpy array of shape
    (time_steps, key_points, 4).

    Args:
    - landmarks: A list of landmark objects.

    Returns:
    - markers_array: A 3D numpy array containing the x, y, z coordinates
    and visibility of each key point for each frame.
    - visibility_array: A 2D numpy array containing the visibility of
    each key point for each frame.
  """
  time_steps = len(landmarks)
  key_points = len(landmarks[0].pose_world_landmarks.landmark)
  output = np.zeros((time_steps,key_points,4))
  for i in range(time_steps):
    world_landmarks_i = landmarks[i].pose_world_landmarks.landmark
    for j in range(key_points):
      world_landmarks_i_j = world_landmarks_i[j]
      output[i,j,2] = world_landmarks_i_j.x
      output[i,j,1] = world_landmarks_i_j.y
      output[i,j,0] = world_landmarks_i_j.z
      output[i,j,3] = world_landmarks_i_j.visibility

  # flip y axis for compatability with openSim
  y_axis = -output[:,:,1]
  # add minimum so ground will be 0
  output[:,:,1] = y_axis - np.min(y_axis)   
  #output[:,:,1] = y_axis
  visibility_array = output[:,:,-1]
  markers_array = output[:,:,0:3]
  return markers_array,visibility_array

def landmarks_2_table(landmarks,landmark_names = landmark_names):
  """
    Convert a list of landmark objects to two pandas dataframes containing the
    x, y, and z coordinates of each key point for each frame, and the
    visibility of each key point for each frame.

    Args:
    - landmarks: A list of landmark objects.
    - landmark_names: A list of landmark names in the same order as the
    key points in the landmark objects.

    Returns:
    - marker_df: A pandas dataframe containing the x, y, and z coordinates
    of each key point for each frame.
    - visibility_df: A pandas dataframe containing the visibility of each
    key point for each frame.
  """
  table_columns = []
  for i in range(len(landmark_names)):
    name= landmark_names[i]
    table_columns.append(name+"_x")
    table_columns.append(name+"_y")
    table_columns.append(name+"_z")

  markers_array,visibility_array = landmark_2_array(landmarks)
  # Flatten third dim (x,y,z) to the main table
  markers_array = markers_array.reshape((markers_array.shape[0],markers_array.shape[1]*3))


  marker_df = pd.DataFrame(markers_array,columns = table_columns)
  visibility_df = pd.DataFrame(visibility_array,columns = landmark_names)

  return marker_df,visibility_df
