import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import csv

def load_local_image(path):
    return cv2.imread(path)

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
                positions.append((int(x), int(y)))  # Convert to Python int
    return positions, binary

def draw_circles_on_binary(binary_img, positions, radius=10):
    img_copy = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    for (x, y) in positions:
        cv2.circle(img_copy, (x, y), radius, (0, 255, 0), 2)
        cv2.circle(img_copy, (x, y), 2, (0, 0, 255), -1)
    return img_copy

def show_image_with_matplotlib(image, title="Image"):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(image_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

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
        mark_data.append((os.path.basename(image_path), count, binary_with_circles, positions))

    return mark_data

# === USAGE ===
directory_path = "../process-data/sheets_images/normalization_posttest_lab_renamed"
output_dir = "./marked_binary"
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "mark_positions.csv")

mark_data = process_directory(directory_path)

# Save all marked binary images and CSV data
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image", "marked", "count"])
    for filename, count, binary_with_circles, positions in mark_data:
        # Save image
        output_path = os.path.join(output_dir, filename)
        # cv2.imwrite(output_path, binary_with_circles)

        # Write to CSV with clean coordinates
        writer.writerow([filename, str(positions), count])

# Show images where number of marks is NOT 6
print("\nImages where number of marks is not 6:")
for filename, count, binary_with_circles, _ in mark_data:
    if count != 6:
        print(f"{filename} -> {count} marks")
        # show_image_with_matplotlib(binary_with_circles, title=f"{filename} ({count} marks)")
