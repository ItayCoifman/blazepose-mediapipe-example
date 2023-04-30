# BlazePose Video Pose Estimation

This repository contains Python code that uses the BlazePose model from the Mediapipe library to estimate the pose of a person in a video file. The code can detect 33 body keypoints, including face, hands, and feet, and returns a processed video object and the output from the model for each frame.

## Usage

To use the code, simply run the `video_pose_estimation` function in the `blazepose_simple_use_case_example.ipynb` notebook. This function takes a video file path as input and returns a processed video object, as well as two data frames containing the x, y, and z coordinates of each landmark for each frame of the processed video and the visibility score for each landmark for each frame of the processed video.

## Example

You can find an example of how to use this code in the `blazepose_simple_use_case_example.ipynb` notebook. This notebook demonstrates how to upload a video file to Google Colab, apply the pose estimation model to each frame of the video, and view the results.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ItayCoifman/blazepose-mediapipe-example/blob/main/Blazepose_simple_use_case_example.ipynb)

## Installation

### Option 1: Local Installation

1. Clone this repository: `git clone https://github.com/ItayCoifman/blazepose-mediapipe-example.git`
2. Navigate to the repository: `cd blazepose-mediapipe-example`
3. Create a new virtual environment (optional): `python -m venv venv`
4. Activate the virtual environment (optional): `source venv/bin/activate`
5. Install the required packages: `pip install -r requirements.txt`
6. Run the example script: `python example.py`

### Option 2: Google Colab Installation

1. Open a new Google Colab notebook: https://colab.research.google.com/
2. Run the following command to install the required packages: 

   ```
   !pip install mediapipe
   ```

3. Copy and paste the relevant contents of `Blazepose_simple_use_case_example` into the notebook.
4. Upload your video file to the notebook (either through the Colab UI or by mounting Google Drive).
5. Run the cells in the notebook to process the video and view the results.

You can install these dependencies using pip:

```
pip install opencv-python mediapipe pandas
```


## Contributing

If you'd like to contribute to this project, feel free to submit a pull request or open an issue. We welcome contributions from the community!

## License

This code is licensed under the MIT License. See the `LICENSE` file for details.





