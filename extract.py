import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import csv
import easyocr

# === OCR Initialization ===
reader = easyocr.Reader(['en'], gpu=False)

# === Load Image ===
def load_local_image(path):
    return cv2.imread(path)

# === Detect Black Circular Marks ===
def detect_black_circular_patches(img, min_radius=18, max_radius=40, fill_ratio=0.8):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    inverted = cv2.bitwise_not(binary)
    blurred = cv2.GaussianBlur(inverted, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=2 * min_radius,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    positions = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0, :]:
            mask = np.zeros_like(binary)
            cv2.circle(mask, (x, y), r, 255, -1)
            circle_pixels = binary[mask == 255]
            black_pixels = np.sum(circle_pixels == 0)
            total_pixels = len(circle_pixels)
            if total_pixels == 0:
                continue
            if black_pixels / total_pixels >= fill_ratio:
                positions.append((int(x), int(y)))
    return positions, binary

# === Draw Circles on Binary Image ===
def draw_circles_on_binary(binary_img, positions, radius=10):
    img_copy = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for (x, y) in positions:
        cv2.circle(img_copy, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(img_copy, (x, y), 2, (0, 0, 255), -1)
    return img_copy

# === Display Image with Matplotlib ===
def show_image_with_matplotlib(image, title="Image"):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

# === Detect Left Margin using OCR ===
def find_left_text_margin_with_easyocr(img):
    h, w = img.shape[:2]
    img = img[0:int(h * 0.2), 0:int(w * 0.2)]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)

    x_coords = []
    for (bbox, text, confidence) in results:
        if confidence > 0.3:
            (top_left, _, _, _) = bbox
            x_coords.append(int(top_left[0]))

    if not x_coords:
        return None
    return min(x_coords)

# === Draw Vertical Line at x ===
def draw_vertical_line(img, x, color=(255, 0, 255), thickness=2):
    img_copy = img.copy()
    cv2.line(img_copy, (x, 0), (x, img_copy.shape[0]), color, thickness)
    return img_copy

# === Process All Images in Directory ===
def process_directory(directory_path):
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory_path, ext)))

    mark_data = []

    for image_path in sorted(image_files):
        image = load_local_image(image_path)
        positions, binary = detect_black_circular_patches(image)
        count = len(positions)
        binary_with_circles = draw_circles_on_binary(binary, positions)

        left_margin_x = find_left_text_margin_with_easyocr(image)
        if left_margin_x:
            marked_img_with_margin = draw_vertical_line(binary_with_circles, left_margin_x)
        else:
            marked_img_with_margin = binary_with_circles

        mark_data.append((os.path.basename(image_path), count, positions, left_margin_x, marked_img_with_margin, image))

    return mark_data

# === USAGE ===
directory_path = "../process-data/sheets_images/normalization_posttest_lab_renamed"
output_dir = "./marked_binary"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "mark_positions.csv")

mark_data = process_directory(directory_path)

# === Save CSV and Display Images ===
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image", "marked", "count", "left_margin_x"])

    for filename, count, positions, left_margin_x, marked_img_with_margin, image in mark_data:
        # Save the image with circles and left margin line
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, marked_img_with_margin)

        # Save to CSV with left margin x value
        writer.writerow([filename, str(positions), count, left_margin_x if left_margin_x is not None else "N/A"])

        # Display if count not 6
        # if count != 6:
        #     print(f"{filename} -> {count} marks")
        #     if left_margin_x:
        #         show_image_with_matplotlib(marked_img_with_margin, title=f"{filename} | Left margin: x={left_margin_x}")
        #     else:
        #         show_image_with_matplotlib(marked_img_with_margin, title=f"{filename} | No margin detected")
