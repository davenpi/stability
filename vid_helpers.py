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


def make_truly_bw(img):
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
    front = np.array(list(zip(columns, rows)))
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
        A 2D array holding the x, y pixel values of the back of the snake.
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
    bottom = np.array(list(zip(columns, rows)))
    return bottom

def hybridize(front: np.ndarray, bottom: np.ndarray)-> np.ndarray:
    """
    Smartly combine the images of the front and the bottom.
    """
    sorted_front = np.sort(front, axis=0)
    tip_y = sorted_front[0][1]
    base_y = sorted_front[-1][1]
    # First deal with downward leaning snake
    if tip_y > base_y:
        hybrid = bottom
    # otherwise use simple hybrid algorithm
    else:
        bot_cols = bottom[:, 0]
        extra_cols = list(np.where(bot_cols > np.amax(front[:, 0]))[0])
        extra_bit = bottom[extra_cols]
        hybrid = np.vstack((front, extra_bit))
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


def get_distance_parameter(outline, px_per_cm=36.4):
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
    ------
    distance : np.ndarray
        This array contains the distance between the outline points (in cm).
    """
    distance = np.cumsum(np.sqrt(np.sum(np.diff(outline, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0)
    distance = distance / px_per_cm
    return distance


def get_interp_points(distance, num_points: int = 15):
    """Get the points I will do the interpolation on."""
    interp_points = np.linspace(distance.min(), distance.max(), num_points)
    return interp_points


def get_interpolation(distance: np.ndarray, outline: np.ndarray, num_points: int = 15):
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
    return smooth_kappa, smooth_points
