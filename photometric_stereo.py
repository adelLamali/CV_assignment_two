import numpy as np
import cv2
import glob

#Load images:
image_paths = sorted(glob.glob("images/*.png"))

images = []

for path in image_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    images.append(img.astype(np.float32))

images = np.stack(images, axis=0)

#Load light directions:
L = np.loadtxt("light_directions.txt")

#Load mask:
mask = cv2.imread("images/img01.png", 0) > 0

#Step 2 Collect Intensity Vector
#For each pixel:
h, w = mask.shape
num_images = images.shape[0]

albedo = np.zeros((h, w))
normals = np.zeros((h, w, 3))

#Step 3 Build Light Matrix

#Step 4 Solve Least Squares

L_pinv = np.linalg.pinv(L)

for y in range(h):
    for x in range(w):

        if not mask[y, x]:
            continue

        I = images[:, y, x]

        B = L_pinv @ I

#Step 5 Compute Albedo and Normals
rho = np.linalg.norm(B)

if rho > 1e-5:
    N = B / rho
else:
    rho = 0
    N = np.array([0, 0, 1])

albedo[y, x] = rho
normals[y, x] = N

#Step 6 Save Albedo and Normal Maps

albedo_img = cv2.normalize(
    albedo,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

cv2.imwrite("albedo.png", albedo_img)

#Normal Map

normal_img = ((normals + 1) / 2 * 255).astype(np.uint8)

cv2.imwrite("normal_map.png", normal_img)
