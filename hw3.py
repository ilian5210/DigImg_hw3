import cv2
import numpy as np

def gaussian_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0) 

def sobel_gradients(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)   # dx dy
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)   
    magnitude = np.sqrt(grad_x**2 + grad_y**2)  # 梯度幅值
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
    return magnitude, angle

def non_max_suppression(magnitude, angle):
    angle = angle % 180  
    suppressed = np.zeros_like(magnitude)   # 和 magnitude 一樣大小的 0 矩陣
    for i in range(1, magnitude.shape[0] - 1):  #  rows 邊緣不處理
        for j in range(1, magnitude.shape[1] - 1):  
            q = 255
            r = 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):  # 0度  水平相鄰
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:  # 45度  斜向相鄰 (左下右上)
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:  # 90度 垂直相鄰
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:  # 135度 斜向相鄰 (左上右下)
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]
            if magnitude[i, j] >= q and magnitude[i, j] >= r:  
                suppressed[i, j] = magnitude[i, j]  # 保留
    return suppressed

def double_threshold(image, low_threshold, high_threshold): 
    strong = 255
    weak = 75
    res = np.zeros_like(image)
    strong_i, strong_j = np.where(image >= high_threshold)  # 大於高閾值
    weak_i, weak_j = np.where((image >= low_threshold) & (image < high_threshold))  # 介於低閾值和高閾值之間
    res[strong_i, strong_j] = strong  # 設為強邊緣
    res[weak_i, weak_j] = weak  # 設為弱邊緣
    return res, weak, strong

def edge_tracking(image, weak, strong=255):
    for i in range(1, image.shape[0] - 1):  # 邊緣不處理
        for j in range(1, image.shape[1] - 1):  
            if image[i, j] == weak:
                if ((image[i + 1, j - 1] == strong) or (image[i + 1, j] == strong) or (image[i + 1, j + 1] == strong)  # 8個方向有無強邊緣
                        or (image[i, j - 1] == strong) or (image[i, j + 1] == strong)
                        or (image[i - 1, j - 1] == strong) or (image[i - 1, j] == strong) or (image[i - 1, j + 1] == strong)):
                    image[i, j] = strong  # 設為強邊緣
                else:
                    image[i, j] = 0  # 非邊緣
    return image

def hough_transform_line(edge_image, rho_res=1, theta_res=np.pi/180, threshold=100):
    height, width = edge_image.shape
    diag_len = int(np.sqrt(height**2 + width**2))  # 最大可能的 rho 值
    rhos = np.arange(-diag_len, diag_len, rho_res)
    thetas = np.arange(0, np.pi, theta_res)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    y_idxs, x_idxs = np.nonzero(edge_image)  # 获取边缘点的坐标
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta)) + diag_len
            accumulator[rho, t_idx] += 1
    lines = []
    for r_idx in range(accumulator.shape[0]):
        for t_idx in range(accumulator.shape[1]):
            if accumulator[r_idx, t_idx] >= threshold:
                rho = rhos[r_idx]
                theta = thetas[t_idx]
                lines.append((rho, theta))
    return lines

# 讀取影像
image = cv2.imread('imgs/img1.jpg', 0)
max_value = np.max(image)

# 高斯模糊
blurred_image = gaussian_blur(image)

# 計算梯度幅值 M(x, y) 和角度
magnitude, angle = sobel_gradients(blurred_image)

# 非極大值抑制
suppressed_image = non_max_suppression(magnitude, angle)

# 雙閾值處理
thresholded_image, weak, strong = double_threshold(suppressed_image, max_value * 0.2, max_value * 0.6)

# 邊緣連接
edges = edge_tracking(thresholded_image, weak, strong)

# 顯示結果
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('imgs/edges.jpg', edges)


