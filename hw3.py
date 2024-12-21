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
    # 獲取影像的形狀
    rows, cols = image.shape
    
    # 創建一個堆疊來存儲需要處理的像素
    stack = []
    
    # 將所有強邊緣像素推入堆疊
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if image[i, j] == strong:
                stack.append((i, j))
    
    # 處理堆疊中的像素
    while stack:
        i, j = stack.pop()
        
        # 檢查 8 個方向
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i + di, j + dj
                if image[ni, nj] == weak:
                    image[ni, nj] = strong
                    stack.append((ni, nj))
    
    # 將所有剩餘的弱邊緣設為非邊緣
    image[image == weak] = 0
    
    return image

def hough_transform_line(edge_image, rho_res=1, theta_res=np.pi/180, threshold=100):
    height, width = edge_image.shape
    diag_len = int(np.sqrt(height**2 + width**2))  # 對角線長度
    rhos = np.arange(-diag_len, diag_len, rho_res) # rho 的範圍 (-diag_len, diag_len)
    thetas = np.arange(0, np.pi, theta_res)  # theta 的範圍 (0, π)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)  #[len(rhos), len(thetas)]的矩陣
    y_idxs, x_idxs = np.nonzero(edge_image)  # 強邊緣點的座標
    for i in range(len(x_idxs)):  # 強邊緣點
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx, theta in enumerate(thetas):  # 所有 theta enumerate() 函數用於將一個可遍歷的數據對象(如列表、元組或字符串)組合為一個索引序列，同時列出數據和數據下標，一般用在 for 循環當中。
            rho = int(x * np.cos(theta) + y * np.sin(theta)) + diag_len  # rho = x cos θ + y sin θ  
            accumulator[rho, t_idx] += 1 #投票
    lines = []
    threshold = np.max(accumulator) * 0.4  # 閾值
    for r_idx in range(accumulator.shape[0]):  # 所有 rho
        for t_idx in range(accumulator.shape[1]):  # 所有 theta
            if accumulator[r_idx, t_idx] >= threshold: # 投票數大於閾值
                rho = rhos[r_idx] 
                theta = thetas[t_idx] 
                lines.append((rho, theta)) # 線段
    return lines

def draw_lines(image, lines): 
    height, width = image.shape
    length = int(np.sqrt(height**2 + width**2))  # 對角線長度
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + length * (-b))
        y1 = int(y0 + length * a)
        x2 = int(x0 - length * (-b))
        y2 = int(y0 - length * a)
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 1)

def img_handeling(image, low_threshold, high_threshold, img_name):
    blurred_image = gaussian_blur(image)
    magnitude, angle = sobel_gradients(blurred_image)
    suppressed_image = non_max_suppression(magnitude, angle)
    thresholded_image, weak, strong = double_threshold(suppressed_image, low_threshold, high_threshold)
    edges = edge_tracking(thresholded_image, weak, strong)

    lines = hough_transform_line(edges)
    output_image = edges.copy()
    draw_lines(output_image, lines)

    cv2.imshow(f'{img_name} Edges', edges)
    cv2.imshow(f'{img_name} Detected Lines', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(f'imgs/{img_name}_edges.jpg', edges)
    cv2.imwrite(f'imgs/{img_name}_detected_lines.jpg', output_image)


    
# img2
# img_handeling(cv2.imread('imgs/img2.jpg', 0), 25, 75, 'img2')

#img5
img_handeling(cv2.imread('imgs/img5.jpg', 0), 30, 90, 'img5')