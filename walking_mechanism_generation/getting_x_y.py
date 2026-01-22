import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- PROVIDED FUNCTIONS ----------------

def digitize_seq(nums, minlim, maxlim, bin_size=64):
    bins = np.linspace(minlim, maxlim, bin_size-1)
    nums_indices = np.digitize(nums, bins)
    return nums_indices


def get_image_from_point_cloud(points, xylim, im_size, inverted=True, label=None):
    mat = np.zeros((im_size, im_size, 1), dtype=np.uint8)
    x = digitize_seq(points[:, 0], -xylim, xylim, im_size)
    if inverted:
        y = digitize_seq(points[:, 1] * -1, -xylim, xylim, im_size)
        mat[y, x, 0] = 1
    else:
        y = digitize_seq(points[:, 1], -xylim, xylim, im_size)
        mat[x, y, 0] = 1
    return mat


def rotate_curve(curve, phi):
    infunction_scale = 100
    x = curve[:, 0] * infunction_scale
    y = curve[:, 1] * infunction_scale
    x_rot = x * np.cos(phi) - y * np.sin(phi)
    y_rot = x * np.sin(phi) + y * np.cos(phi)
    return np.column_stack((x_rot, y_rot)) / infunction_scale


def center_data(X):
    m = np.mean(X, axis=0)
    return X - m, np.matrix([[1, 0, -m[0]],
                             [0, 1, -m[1]],
                             [0, 0, 1]])


def scale_data(X, scaling=0):
    if scaling == 0:
        denom = np.sqrt(np.var(X[:, 0]) + np.var(X[:, 1]))
        scaled = X / denom
        S = np.matrix([[1/denom, 0, 0],
                       [0, 1/denom, 0],
                       [0, 0, 1]])
    else:
        max_dist = np.max(np.linalg.norm(X, axis=1))
        scaled = X * scaling / max_dist
        S = np.matrix([[scaling/max_dist, 0, 0],
                       [0, scaling/max_dist, 0],
                       [0, 0, 1]])
    return scaled, S

# ---------------- LOAD IMAGE & EXTRACT CURVE ----------------

image_id = 4
path = f"motiongen/{image_id}.png"

img = cv2.imread(path)
if img is None:
    raise FileNotFoundError(path)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(
    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)

curve = max(contours, key=cv2.contourArea)
points = curve.reshape(-1, 2).astype(float)

# ---------------- NORMALIZATION PIPELINE ----------------
# 1. Translate
points_centered, T = center_data(points)

# 2. Scale
points_scaled, S = scale_data(points_centered, scaling=0)

# 3. Rotate
phi = 0.0  # set rotation angle here (radians)
points_normalized = rotate_curve(points_scaled, phi)

# ---------------- DIGITIZE INTO IMAGE ----------------

im_size = 64
xylim = 2.0

generated = get_image_from_point_cloud(
    points_normalized,
    xylim=xylim,
    im_size=im_size,
    inverted=True
)

# ---------------- VISUAL CHECK (NO TEXT) ----------------

# fig, axs = plt.subplots(1, 2, figsize=(8, 4))

# axs[0].imshow(img_rgb)
# axs[0].axis("off")

# axs[1].imshow(generated[:, :, 0], cmap="gray")
# axs[1].axis("off")

# plt.tight_layout()
# plt.show()

# ---------------- SAVE AS JPG ----------------

output_path = f"motiongen/generated_{image_id}.jpg"

# Convert {0,1} → {0,255} for proper image saving
out_img = (generated[:, :, 0] * 255).astype(np.uint8)

cv2.imwrite(output_path, out_img)
