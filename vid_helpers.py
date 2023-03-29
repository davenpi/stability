"""
This file contains the helper functions needed to analyze images and extract
the shape of the front of the snake.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import skimage
from skimage.util import invert


def plot_img(img: np.ndarray, size: int = 6) -> None:
    """
    Plot an image.

    Parameters
    ----------
    img : np.ndarray
        Image array.
    figsize : int
        Size of figure. Default 8x8.

    Returns
    -------
    """
    plt.figure(figsize=(size, size))
    skimage.io.imshow(img, cmap="gray")
    plt.show()


def load_and_plot(img_path: str) -> np.ndarray:
    """
    Load the image and plot it. I'm plotting the inverse so that the snake has
    high pixel values and the background has low pixel values.

    Parameters
    ----------
    img_path : str
        Path to image.

    Returns
    -------
    img : np.ndarray
        Inverted image array.
    """
    img = skimage.io.imread(img_path)
    img = invert(img)
    img = make_truly_bw(img)
    plot_img(img)
    return img


def load_img(img_path: str) -> np.ndarray:
    """
    Load the image and make it black and white.
    """
    img = skimage.io.imread(img_path)
    img = invert(img)
    img = make_truly_bw(img)
    return img


def make_truly_bw(img: np.ndarray) -> np.ndarray:
    """
    Force all pixels to be black or white.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image array.

    Returns
    -------
    img : np.ndarray
        Image containing only black and white pixel values.
    """
    img = 255 * (img > 100)
    return img


def extract_front(cropped: np.ndarray) -> np.ndarray:
    """
    Extract the front of the snake from the image and return the pixel values
    describing the extracted shape.

    Parameters
    ----------
    cropped : np.ndarray
        A cropped image of the snake.

    Returns
    -------
    front : np.ndarray
        A 2D array holding the x,y pixel values of the front of the snake.
    """
    rows = []
    columns = []
    for i in range(cropped.shape[0]):
        if np.any(cropped[i]):
            row_idx = i
            rows.append(row_idx)
            column_idx = np.where(cropped[i])[0][0]
            columns.append(column_idx)
    rows = np.array(rows)
    columns = np.array(columns)
    front = np.array(list(zip(columns, rows)), dtype=np.float32)
    return front


def extract_bottom(cropped: np.ndarray):
    """
    Exctract a line describing the bottom of the snake.

    Extract the back of the snake from the image and return the pixel values
    describing the extracted shape.

    Parameters
    ----------
    cropped : np.ndarray
        A cropped image of the snake.

    Returns
    -------
    back : np.ndarray
        A 2D array holding the x, y pixel values of the bottom of the snake.
    """
    rows = []
    columns = []
    for i in range(cropped.shape[1]):
        if np.any(cropped[:, i]):
            col_idx = i
            columns.append(col_idx)
            row_idx = np.where(cropped[:, i])[0][-1]
            rows.append(row_idx)
    rows = np.array(rows)
    columns = np.array(columns)
    bottom = np.array(list(zip(columns, rows)), dtype=np.float32)
    return bottom


def leaning_down(front):
    """
    Is the snake leaning down according to the front outline.
    """
    top_y = front[0][1]
    bottom_y = front[-1][1]
    front_says_down = top_y < bottom_y
    return front_says_down


def leaning_down(front, bottom):
    """
    Is the snake leaning down according to the front outline.
    """
    leaning_back = leaning_backwards(bottom=bottom)
    tilted_up = tilted_high(front=front)
    if leaning_back or tilted_up:
        down = False
    else:
        top_x = front[0][0]
        bottom_x = front[-1][0]
        x_says_down = top_x > bottom_x
        down = x_says_down
    return down


def tilted_high(front, tilt_px=250):
    """
    Is the snake reaching up high? Measure the distance between the lowest
    and highest point of the snake and check if it is above a (hand chosen)
    threshold.
    """
    highest_point = front[:, 1].min()
    lowest_point = front[:, 1].max()
    diff = lowest_point - highest_point
    tilted = diff > tilt_px
    return tilted


def leaning_backwards(bottom: np.ndarray):
    """
    Tell if the snake is leaning backwards i.e if some part of it's upper
    body is behind the point of departure.
    """
    bot_rows = bottom[:, 1]
    row_diffs = np.diff(bot_rows)
    leaning_back = np.any(row_diffs < -5)
    return leaning_back


def leaning_fill_gap(front: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """
    Check if the snake is leaning backwards and hybridize appropriately.

    If the snake is bent back with some of it's back hanging
    over the base, don't just use the front as the outline but
    supplement it with some of the bits from the back. We just need to take
    care to not include any of the back in the outline. That is what the
    'break' statement does in the for loop. We break out of the loop if we
    encounter a place where the snake is leaning backwards. Currently I am
    checking if it is leaning backwards if there is a 5 px difference in
    consecutive y-values (i.e scanning from left to right the y value on the
    right is 5 pixels higher than the one it is next to.)

    Parameters
    ----------
    front : np.ndarray
        2D array containing the outline of the front of the snake.
    bottom : np.ndarray
        2D array containing the outline of the bottom of the snake.

    Returns
    -------
    hybrid : np.ndarray
        2D array containing an outline which combines the front and bottom
        outlines.
    """
    dec_bot = []  # selects the points where i know im on the bottom
    for i in range(bottom.shape[0]):
        diff = bottom[i + 1, 1] - bottom[i, 1]
        if diff < -5:
            dec_bot.append(bottom[i])
            break
        else:
            dec_bot.append(bottom[i])

    # selects the points where we don't have any front coverage
    spec_bot = np.array([tup for tup in dec_bot if tup[0] > front[-10:, 0].max()])
    hybrid = np.vstack((front, spec_bot))
    return hybrid


def hybridize(front: np.ndarray, bottom: np.ndarray, reaching_num: int) -> np.ndarray:
    """
    Smartly combine the images of the front and the bottom.

    If the snake is leaning down, use the bottom outline. Otherwise
    combine the front and the bottom for a full view.
    """

    front_says_down = leaning_down(front, bottom)
    leaning = leaning_backwards(bottom)
    reaching_high = tilted_high(front, tilt_px=reaching_num)

    # First deal with downward leaning snake
    if front_says_down or not reaching_high:
        hybrid = bottom
        print("using bottom")
        return hybrid
    elif leaning:
        try:
            hybrid = leaning_fill_gap(front=front, bottom=bottom)
            return hybrid
        except:
            hybrid = front
            print("Hit exception and using front.")
            return hybrid
    # otherwise use simple hybrid algorithm
    else:
        bot_cols = bottom[:, 0]
        # find where the bottom columns go beyond the maximum column (x value) from the front outline
        extra_cols = list(np.where(bot_cols > np.amax(front[:, 0]))[0])
        extra_bit = bottom[extra_cols]
        hybrid = np.vstack((front, extra_bit))
        print("Using hybrid")
    return hybrid


def plot_outline(cropped: np.ndarray, outline: np.ndarray, size: int = 6) -> None:
    """
    Plot an outline of the shape of the snake.

    Parameters
    ----------
    cropped : np.ndarray
        Cropped image of the snake.
    outline : np.ndarray
        A 2D array giving pixel values for an outline.

    Returns
    -------
    """
    plt.figure(figsize=(size, size))
    plt.imshow(cropped, cmap="gray")
    plt.scatter(outline[:, 0], outline[:, 1], s=2, color="red")
    plt.show()


def get_height_length_ratio(
    distance_param: np.ndarray, interpolated_line: np.ndarray, px_per_cm: float
):
    head_y = interpolated_line[0, 1]
    base_y = interpolated_line[-1, 1]
    height = np.abs(head_y - base_y) / px_per_cm
    length = distance_param[-1]
    ratio = height / length
    return ratio


def get_distance_parameter(outline: np.ndarray, px_per_cm: float) -> np.ndarray:
    """
    Use the outline to construct a distance parameter.

    Parameters
    ----------
    outline : np.ndarray
        Outline shape (front, back, or center) of the snake.
    px_per_cm : float
        Magic number which set the distance scale (in cm) in the image. This
        number comes from knowing a physical distance in the image and seeing
        how many pixels that distance takes up.

    Returns
    -------
    distance : np.ndarray
        This array contains the distance between the outline points (in cm).
    """
    distance = np.cumsum(np.sqrt(np.sum(np.diff(outline, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0)
    distance = distance / px_per_cm
    return distance


def get_interp_points(distance, num_points: int = 50):
    """Get the points I will do the interpolation on."""
    interp_points = np.linspace(distance.min(), distance.max(), num_points)
    return interp_points


def get_interpolation(distance: np.ndarray, outline: np.ndarray, num_points: int = 50):
    """Get the interpolation of the line."""
    interpolator = interp1d(distance, outline, kind="cubic", axis=0)
    interp_points = get_interp_points(distance, num_points)
    interpolated_line = interpolator(interp_points)
    return interpolated_line, interpolator


def plot_interpolated_line(
    cropped: np.ndarray,
    outline: np.ndarray,
    interpolated_line: np.ndarray,
    size: int = 8,
):
    """
    Plot the interpolation line over top of the snake image.

    Paramters
    ---------
    cropped : np.ndarray
        Cropped image of the snake.
    outline : np.ndarray
        A 2D array with the outline (front, back, or center) outline of the
        snake. The columns correspond to x and y pixel values.
    interpolated_line : np.ndarray
        A 2D array with the interpolation x and y pixel values for the outline.

    Returns
    -------
    """
    plt.figure(figsize=(size, size))
    plt.imshow(cropped, cmap="gray")
    plt.scatter(outline[:, 0], outline[:, 1], s=2, color="red")
    plt.plot(interpolated_line[:, 0], interpolated_line[:, 1], linewidth=2)
    plt.title("Overlay of interpolated line")
    plt.show()


def check_parameterization(
    interpolated_line: np.ndarray, interpolation_points: np.ndarray
) -> bool:
    """
    Check that the parameterization is "proper."

    A proper parameterization means the derivatives dx/dt and dy/dt exist
    everywhere and are not zero simultaneously.

    Parameters
    ----------
    interpolated_line : np.ndarray
        A 2D array containing the x, y pixel values of the interpolation line.
    interpolation_points : np.ndarray
        Points at which the interpolation took place. These are values of the
        parameter we interpolate from.

    Returns
    -------
    proper : bool
        True if the parameterization is "proper" as described above. False otherwise.
    """
    x = interpolated_line[:, 0]
    y = interpolated_line[:, 1]
    t = interpolation_points
    spacing = np.diff(t)[0]
    dx_dt = np.gradient(x, spacing)
    dy_dt = np.gradient(y, spacing)
    deriv_mag = np.sqrt(dx_dt**2 + dy_dt**2)
    no_nans_x = np.all(~np.isnan(dx_dt))
    no_nans_y = np.all(~np.isnan(dy_dt))
    no_nans = no_nans_x and no_nans_y
    non_zero = np.all(deriv_mag > 0)
    proper = no_nans and non_zero
    return proper


def compute_plot_curvature(
    interpolated_line: np.ndarray,
    interpolation_points: np.ndarray,
    px_per_cm: float = 36.4,
) -> np.ndarray:
    """
    Compute and plot the curvature.

    We compute and plot the curvature of the fit line after some smoothing.

    Parameters
    ----------
    interpolated_line : np.ndarray
        A 2D array containing the x, y pixel values of the interpolation.
    interpolation_points : np.ndarray
        Array containing the points we will do the interpolation at.

    Returns
    -------
    smooth_kappa : np.ndarray
        Array containing the curvature at several points along the snake.
    """
    x = interpolated_line[:, 0] / px_per_cm
    y = interpolated_line[:, 1] / px_per_cm
    t = interpolation_points
    spacing = np.diff(t)[0]
    dx_dt = np.gradient(x, spacing)
    dy_dt = np.gradient(y, spacing)
    d2x_dt = np.gradient(dx_dt, spacing)
    d2y_dt = np.gradient(dy_dt, spacing)
    kappa = np.abs(dx_dt * d2y_dt - d2x_dt * dy_dt) / (
        np.power(dx_dt**2 + dy_dt**2, 3 / 2)
    )
    smooth_interp = interp1d(interpolation_points, kappa, kind="cubic")
    smooth_points = np.linspace(
        interpolation_points.min(), interpolation_points.max(), 100
    )
    smooth_kappa = smooth_interp(smooth_points)
    # plt.plot(interpolation_points, kappa, label="regular")
    plt.plot(smooth_points, smooth_kappa)
    plt.xlabel("Distance along snake (cm)", fontsize=15)
    plt.ylabel(r"Curvature $\kappa$ ($cm^{-1}$)", fontsize=15)
    # plt.legend()
    plt.title("Curvature along the snake.")
    plt.show()
    return smooth_kappa, smooth_points


def compute_curvature(
    interpolated_line: np.ndarray,
    interpolation_points: np.ndarray,
    px_per_cm: float = 36.4,
) -> np.ndarray:
    """
    Compute and plot the curvature.

    We compute and plot the curvature of the fit line after some smoothing.

    Parameters
    ----------
    interpolated_line : np.ndarray
        A 2D array containing the x, y pixel values of the interpolation.
    interpolation_points : np.ndarray
        Array containing the points we will do the interpolation at.
    px_per_cm : float
        Number of pixels per centimeter in the video.

    Returns
    -------
    smooth_kappa : np.ndarray
        Array containing the curvature at several points along the snake.
    """
    x = interpolated_line[:, 0] / px_per_cm
    y = interpolated_line[:, 1] / px_per_cm
    t = interpolation_points
    spacing = np.diff(t)[0]
    dx_dt = np.gradient(x, spacing)
    dy_dt = np.gradient(y, spacing)
    d2x_dt = np.gradient(dx_dt, spacing)
    d2y_dt = np.gradient(dy_dt, spacing)
    kappa = (dx_dt * d2y_dt - d2x_dt * dy_dt) / (
        np.power(dx_dt**2 + dy_dt**2, 3 / 2)
    )  # move to computing signed curvature
    smooth_interp = interp1d(
        interpolation_points, kappa, kind="cubic", fill_value=89, bounds_error=False
    )
    smooth_points = np.linspace(
        interpolation_points.min(),
        interpolation_points.max(),
        100,
    )
    smooth_kappa = smooth_interp(smooth_points)
    return smooth_kappa, smooth_points, smooth_interp


def find_unique(arr1: np.ndarray, arr2: np.ndarray) -> list:
    """
    Find the values in a that are not in b and return a list of the values.

    Parameters
    ----------
    arr1 : np.ndarray

    arr2 : np.ndarray

    Returns
    -------
    unique_to_arr1 : list
        A list of the unique values in arr1 that are not in arr2.
    """
    set1 = set(arr1)
    set2 = set(arr2)
    diff = set1 - set2
    unique_to_arr1 = list(diff)
    return unique_to_arr1


def get_idx_of_vals(arr: np.ndarray, vals: list) -> list:
    """
    Get the indices of specific values of an array and return them in a list.

    Parameters
    ----------
    arr : np.ndarray

    vals : list
        List of values in arr I want to know the index of.

    Returns
    -------
    idxs : list
        List of indexes where the values are.
    """
    idxs = []
    for val in vals:
        idx = np.where(arr == val)[0][0]
        idxs.append(idx)
    return idxs


def stack_and_sort_2d(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """
    Stack two 2d arrays and sort the values by the first column.

    Break ties in the first column by the value in the second column.

    Parameters
    ----------
    arr1 : np.ndarray
        Shape (n, 2) array.
    arr2 : np.ndarray
        Shape (m, 2) array.

    Returns
    -------
    stacked_sorted : np.ndarray
        Shape (n+m, 2) array sorted by value in the first column. If there are
        equal column values, then sort by the value in the second column.
    """
    stacked = np.vstack((arr1, arr2))
    # sorted = np.sort(stacked, axis=0)
    return stacked


def find_unique_and_stack(front: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """
    Find the points in the bottom that are not in the front and add them in.

    The bottom and front outlines will not have the same points. Here I am
    adding in the missed bottom points to the front outline.

    Parameters
    ----------
    front : np.ndarray
        Shape (n, 2) array.
    bottom : np.ndarray
        Shape (m, 2) array.

    Returns
    -------
    hybrid : np.ndarray
        Shape (n + x, 2) array where x is the number of columns in the bottom
        that are not included in the front.
    """
    unique_to_bottom = find_unique(bottom[:, 0], front[:, 0])
    unique_idxs = get_idx_of_vals(arr=bottom, vals=unique_to_bottom)
    unique_arr = bottom[unique_idxs]
    hybrid = stack_and_sort_2d(front, unique_arr)
    return hybrid


def new_hybridize(
    front: np.ndarray, bottom: np.ndarray, reaching_num: float
) -> np.ndarray:
    """
    Smartly combine the images of the front and the bottom.

    If the snake is leaning down, use the bottom outline. Otherwise
    combine the front and the bottom for a full view.
    """

    front_says_down = leaning_down(front, bottom)
    leaning = leaning_backwards(bottom)
    reaching_high = tilted_high(front, tilt_px=reaching_num)

    # First deal with downward leaning snake
    if front_says_down or not reaching_high:
        hybrid = bottom
        print("using bottom")
        return hybrid
    elif leaning:
        try:
            hybrid = leaning_fill_gap(front=front, bottom=bottom)
            print("Filling gap")
            return hybrid
        except:
            print("there was a problem with the hybridization")
            hybrid = front
            return hybrid
    else:
        # the columns where there is nothing from the front, add in the bottom
        hybrid = find_unique_and_stack(front=front, bottom=bottom)
        # x, y = hybrid[:, 0], hybrid[:, 1]
        # xnew = np.linspace(x.min(), x.max(), num = 500)
        # ynew = np.interp(xnew, x, y)
        # processed_hybrid = np.array(list(zip(xnew, ynew))) # hybridizing so the new array
        # does not contain repeat values on both dimensions.
        print("Using front and bottom")
    return hybrid