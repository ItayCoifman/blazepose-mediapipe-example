from matplotlib import pyplot as plt
import os
import imageio


def plot_frame(frame_data,
               figsize=(10, 10),
               elevation: int = 10,
               azimuth: int = 10,
               connections=None,
               up="Y"):
    """
    This function creates a 3D scatter plot of the frame data provided.
    The plot is created using matplotlib and can be customized using the input parameters.

    Args:
    frame_data: numpy array containing the x, y, z coordinates of the data points to be plotted
    figsize: tuple of width and height of the figure (default=(10, 10))
    elevation: angle of elevation of the plot (default=10)
    azimuth: angle of azimuth of the plot (default=10)
    connections: list of tuples representing the indices of the points to be connected in the plot (default=None)

    Returns:
    None
    """
    # Re order the data to fit matplotlib 3D axis
    x, y, z = frame_data[:, 0], frame_data[:, 1], frame_data[:, 2]
    fig = plt.figure(figsize=figsize, dpi=100, frameon=False)
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elevation, azim=azimuth)

    if up == "Y":
        temp = z
        z = y
        y = temp
        plt.ylabel("Z")
        ax.set_zlabel("Y")

    else:
        #
        pass

    # add lims
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    ax.set_zlim(0, 2)
    # add cordinates axis
    plt.xlabel("X")
    ax.scatter(x, y, z)

    ## add conections:
    if connections:
        # Draws the connections if the start and end landmarks are both visible.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            ax.plot3D(
                xs=[x[start_idx], x[end_idx]],
                ys=[y[start_idx], y[end_idx]],
                zs=[z[start_idx], z[end_idx]],
                color='b',
            )
    return fig


def create_frame(frame_data,
                 path,
                 connections=None,
                 figsize=(10, 10),
                 elevation: int = 10,
                 azimuth: int = 10):
    """
    This function creates a 3D scatter plot of the frame data provided and saves it to a specified file path.
    It uses the plot_frame function to create the figure and adds the functionality to save it to a file.

    Args:
    frame_data: numpy array containing the x, y, z coordinates of the data points to be plotted
    path: string representing the file path and name where the plot will be saved
    connections: list of tuples representing the indices of the points to be connected in the plot (default=None)
    figsize: tuple of width and height of the figure (default=(10, 10))
    elevation: angle of elevation of the plot (default=10)
    azimuth: angle of azimuth of the plot (default=10)

    Returns:
    None
    """
    fig = plot_frame(frame_data, figsize, elevation, azimuth, connections=connections)

    plt.savefig(path,
                transparent=False,
                facecolor='white'
                )

    plt.close()
    return path


def create_gif(markers_array, output_path='./example.gif',
               fps=5,
               connections=None,
               roatate=False,
               ):
    """
    This function creates a gif animation from a series of frames of marker data. It creates a separate image file for each frame using the create_frame function and then combines them into a gif animation.

    Args:
    markers_array: numpy array containing the marker data for each frame. The shape of the array should be (number of frames, number of markers, 3) where the last dimension represent x,y,z coordinates of the markers.
    output_path: string representing the file path and name where the gif will be saved (default='./example.gif')
    fps: frames per second of the gif animation (default=5)
    connections: list of tuples representing the indices of the markers to be connected in each frame (default=None)
    rotate: boolean flag indicating whether the animation should rotate the view of the plot or not (default=False)

    Returns:
    output_path: the path to the generated gif
    """
    frames = []
    tmp_path = "./tmp/"
    if not (os.path.exists(tmp_path)):
        # os.remove(output_path)
        os.mkdir(tmp_path)

    n_frames = len(markers_array)
    for i in range(n_frames):
        if roatate:
            azim = int((i / n_frames) * 360)
        else:
            azim = 10

        frame = markers_array[i, :, :]
        temp_image_path = f"./tmp/{i}.png"
        create_frame(frame, temp_image_path, azimuth=azim, connections=connections)
        image = imageio.imread(temp_image_path)
        ########
        # fig, canvas = plot_frame(frame)
        # image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        frames.append(image)
        # remove file
        os.remove(temp_image_path)

    imageio.mimsave(output_path,  # output gif
                    frames,  # array of input frames
                    fps=fps)  # optional: frames per second
    os.rmdir(tmp_path)
    return output_path