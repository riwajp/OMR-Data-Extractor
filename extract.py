import cv2
import numpy as np
import os
import glob

def load_local_image(path):
    return cv2.imread(path)

def detect_black_circular_patches(img, min_radius=15, max_radius=40, fill_ratio=0.6):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to black and white (binary)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Invert image (so black becomes white)
    inverted = cv2.bitwise_not(binary)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(inverted, (9, 9), 2)
    
    # Detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=2*min_radius,
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
                positions.append((x, y))

    return positions, binary  # Also return binary for display

def draw_detected_circles(img, positions, radius=10):
    for (x, y) in positions:
        cv2.circle(img, (x, y), radius, (128), 5)  # Use gray circle
    return img

def process_directory(directory_path):
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory_path, ext)))

    for image_path in sorted(image_files):
        print(f"\nProcessing: {os.path.basename(image_path)}")
        image = load_local_image(image_path)
        positions, binary = detect_black_circular_patches(image)
        print("Detected circular black spots at:", positions)

        # Draw and show
        image_with_circles = draw_detected_circles(binary.copy(), positions)
        resized = cv2.resize(image_with_circles, (800, 800))
        cv2.namedWindow("Detected Black Circles (B&W)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Detected Black Circles (B&W)", 800, 800)
        cv2.imshow("Detected Black Circles (B&W)", resized)

        key = cv2.waitKey(0)
        if key == 27:  # ESC key to break
            break

    cv2.destroyAllWindows()

# === USAGE ===
directory_path = "../process-data/sheets_images/normalization_posttest_lab_renamed"  # Replace with your directory
process_directory(directory_path)
