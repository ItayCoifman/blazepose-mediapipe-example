import torch
import cv2
from tqdm import tqdm
from Video import Video

def anotate_frame(img, boxes_ltwh, boxes_conf=None, boxes_class_ind=None, class_dict=None, conf_threshold=0, objects_index=None, device=None,bbox_color =(0, 0, 255)):
  """
  This function annotates a given frame with bounding boxes.

  Arguments:

  img: A numpy ndarray representing an image.
  boxes_ltwh: A numpy ndarray of shape (n, 4) representing the (left, top, width, height) of the boxes in the image.
  boxes_conf: A numpy ndarray of shape (n,) representing the confidence scores for the boxes in the image.
  boxes_class_ind: A numpy ndarray of shape (n,) representing the class indices for the boxes in the image.
  class_dict: (optional) A dictionary mapping class indices to class names.
  conf_threshold: (optional) A float value representing the minimum confidence score required to annotate a box.
  Returns:
  An annotated image as a numpy ndarray.
  """
  ###################TODO###################
  #Change this functions to support cuda
  ##########################################
  img = img.copy()
  '''
  if boxes_conf is not None:
    boxes_conf = torch.as_tensor(boxes_conf).to(device)
  if boxes_class_ind is not None:
    boxes_class_ind = torch.as_tensor(boxes_class_ind).to(device)
  if objects_index is not None:
    objects_index = torch.as_tensor(objects_index).to(device)
  '''
  if isinstance(boxes_ltwh, torch.Tensor):
    boxes_ltwh = boxes_ltwh.cpu().detach().numpy()
  boxes_ltwh = boxes_ltwh.astype(int)
     
  if boxes_class_ind is not None:
    if isinstance(boxes_class_ind, torch.Tensor):
      boxes_class_ind = boxes_class_ind.cpu().detach()
      boxes_class_ind = boxes_class_ind.numpy()

  if boxes_conf is not None:
    if isinstance(boxes_conf, torch.Tensor):
      boxes_conf = boxes_conf.cpu().detach()
      boxes_conf = boxes_conf.numpy()
    
  for i in range(len(boxes_ltwh)):
      if boxes_conf is not None and boxes_conf[i] < conf_threshold:
          continue
      box_ltwh = boxes_ltwh[i]
      x1, y1, w, h = box_ltwh
      cv2.rectangle(img, (x1, y1), (x1+w, y1+h), bbox_color, 2)
      if boxes_class_ind is not None:
          box_class_ind = boxes_class_ind[i]
          class_name = int(box_class_ind) if class_dict is None else class_dict[int(box_class_ind)]
          cv2.putText(img, str(class_name), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)    
      if objects_index is not None:
          object_id = f"ID-{objects_index[i]}"
          cv2.putText(img, object_id, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

  return img

def anotate_video(video,predictions,output_path = None, modelPred_2_pred_func = None, conf_threshold = 0,class_dict = None):
  '''
    Annotate the input video with object detection results using the provided model predictions.

    Parameters:
    video (Video): A Video object representing the input video.
    predictions (list): A list of model predictions for each frame in the video.
    output_path (str, optional): The path where the annotated video will be saved. If not provided, the video will be saved in a default location.
    modelPred_2_pred_func (function, optional): A function that converts the model predictions to the required format (boxes_ltwh, boxes_conf, boxes_class_ind). The default is `modelPred_2_pred`.
    conf_threshold (float, optional): The minimum confidence threshold to display the object detection result. The default is 0.
    class_dict (dict, optional): A dictionary mapping class indices to class names. If not provided, class indices will be used as class names.

    Returns:
    Video: A Video object representing the annotated video.

  '''
  #############TODO##########
  #init video class from path
  ###########################
  vidcap = cv2.VideoCapture(video.path)
  if output_path is None:
     #############TODO##########
    pass
  #out_dir = os.path.dirname(output_path) + '/'
  #tmp_output_path = out_dir + "tmp_" + video.name
  video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MP4V"), video.fps, (video.width, video.height))
  frame_index = 0
  print(f"antotating-{video.name}")
  for frame_index in tqdm(range(video.nFrames)):
        it_worked, img = vidcap.read()
        if not it_worked:
            break
        frame_pred = predictions[frame_index]
        if modelPred_2_pred_func is not None:
          boxes_ltwh,boxes_conf,boxes_class_ind = modelPred_2_pred_func(frame_pred)
        else:
          boxes_ltwh,boxes_conf,boxes_class_ind = frame_pred
        #img = anotate_frame(img,fram_pred)
        img = anotate_frame(img,boxes_ltwh,boxes_conf,boxes_class_ind,class_dict = class_dict , conf_threshold = 0)
        #write frame to ouputvideo
        video_writer.write(img)

  # Close the output video
  video_writer.release()
  # conver video to format wich can be 
  #output_path = Convert_video(tmp_output_path,output_path,delete_input = True)
  # Return the processed video and the model outputs
  vid = Video(output_path, video.fps, video.nFrames, video.width, video.height)
  print(f"\n anotated video saved at: \n {output_path}")
  return vid


  KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  """Draws the keypoint predictions on image.

  Args:
    image: A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)
  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  if output_image_height is not None:
    output_image_width = int(output_image_height / height * width)
    image_from_plot = cv2.resize(
        image_from_plot, dsize=(output_image_width, output_image_height),
         interpolation=cv2.INTER_CUBIC)
  return image_from_plot

def to_gif(images, fps):
  """Converts image sequence (4D numpy array) to gif."""
  imageio.mimsave('./animation.gif', images, fps=fps)
  return embed.embed_file('./animation.gif')

def progress(value, max=100):
  return HTML("""
      <progress
          value='{value}'
          max='{max}',
          style='width: 100%'
      >
          {value}
      </progress>
  """.format(value=value, max=max))



  def plot_images(images, labels, n_cols=4, figure_size=(12, 3)):
    """
    Plot a set of images with corresponding labels as titles.

    Args:
        images: A numpy array of images with shape (num_images, image_height, image_width, num_channels).
        labels: A numpy array of labels with shape (num_images,).
        n_cols: The number of columns in the grid of subplots. Default is 4.
        figure_size: The size of the figure to create. Default is (12, 3).
    """
    # Create a figure with subplots arranged in a grid
    num_images = len(images)
    num_rows = (num_images + n_cols - 1) // n_cols
    fig, axes = plt.subplots(nrows=num_rows, ncols=n_cols, figsize=figure_size)
    fig.subplots_adjust(hspace=0.5)

    # Convert the image tensor to integer
    images = images.astype(int)

    # Iterate over the images and corresponding labels and plot them
    for i, (image, label) in enumerate(zip(images, labels)):
        row = i // n_cols
        col = i % n_cols
        if num_rows>1:
          ax = axes[row][col]
        else:
          ax = axes[col]
        ax.imshow(image)
        ax.set_title(f"Label: {label}")
        ax.axis("off")

    # Show the plot
    plt.show()


def sample_from_dataset(dataset, n =1):
    """
    Sample `n` images from each class in the dataset and return a list of sampled image tensors.

    Args:
        dataset: A `tf.data.Dataset` object containing image and label data.
        n: An integer specifying the number of times to sample images from each class.

    Returns:
        A tuple containing a list of `n` sampled image tensors from each class in the dataset,
        and a numpy array of class names.
    """
    # Sample image from each class and return a list of images corresponding to each class
    class_names = np.array(dataset.class_names)
    n_classes = len(class_names)
    sampled_images = []
    labels = []
    
    # Group the dataset by label and sample one random image from each group
    for i in range(n_classes):
        class_data = dataset.unbatch().filter(lambda x, y: y == i)
        class_size = len(list(class_data))
        for j in range(n):
            rand_index = random.randint(0, class_size-1)
            sampled_image, _ = next(iter(class_data.skip(rand_index).take(1)))
            sampled_images.append(sampled_image.numpy().astype(int))
            labels.append(class_names[i])

    return np.array(sampled_images), np.array(labels)
    



        