import cv2
import numpy as np

def load_local_image(path):
    return cv2.imread(path)

def detect_black_circular_patches(img, min_radius=20, max_radius=40):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert image (so black patches become white)
    inverted = cv2.bitwise_not(gray)

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(inverted, (9, 9), 2)

    # Detect circles using Hough Transform
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
        for circle in circles[0, :]:
            x, y, r = circle
            positions.append((x, y))

    return positions

def draw_detected_circles(img, positions, radius=5):
    for (x, y) in positions:
        cv2.circle(img, (x, y), radius, (0, 255, 0), 2)
    return img

# === USAGE ===
image_path = "../process-data/sheets_images/er_posttest_lab_renamed/er_posttest_lab_1.png"  # Replace with the path to your local image
image = load_local_image(image_path)
positions = detect_black_circular_patches(image)

print("Detected circular black spots at:", positions)

# Optional: show image with detected circles
image_with_circles = draw_detected_circles(image, positions)
# Resize image to fit window (e.g., 800x800) for zoomed-out view
resized = cv2.resize(image_with_circles, (800, 800))

# Create a named window with custom size
cv2.namedWindow("Detected Black Circles", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detected Black Circles", 800, 800)
cv2.imshow("Detected Black Circles", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
