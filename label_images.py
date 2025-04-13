import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import csv
import easyocr

def getLabeledImages(src_path, out_path):
    reader = easyocr.Reader(['en'], gpu=False)

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
                    positions.append((int(x), int(y)))
        return positions, binary

    def draw_circles_on_binary(binary_img, positions, radius=10):
        img_copy = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
        for (x, y) in positions:
            cv2.circle(img_copy, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(img_copy, (x, y), 2, (0, 0, 255), -1)
        return img_copy

    def find_left_text_margin_with_easyocr(img):
        h, w = img.shape[:2]
        img = img[0:int(h * 0.2), 0:int(w * 0.2)]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray)
        x_coords = [int(bbox[0][0]) for (bbox, text, confidence) in results if confidence > 0.3]
        return min(x_coords) if x_coords else None

    def find_second_margin_green(img, left_margin_x, search_range=200, pixel_threshold=10):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        start_x = left_margin_x
        end_x = min(width, start_x + search_range)

        for y in range(height - 1, -1, -1):
            line = gray[y, start_x:end_x]
            black_pixels = np.where(line < 50)[0]
            if len(black_pixels) > pixel_threshold:
                return int(start_x + black_pixels[0] - 20)
        return None

    def draw_two_margins(img, red_x=None, green_x=None):
        img_copy = img.copy()
        if red_x:
            cv2.line(img_copy, (red_x, 0), (red_x, img_copy.shape[0]), (0, 0, 255), 2)
        if green_x:
            cv2.line(img_copy, (green_x, 0), (green_x, img_copy.shape[0]), (0, 255, 0), 2)
        return img_copy

    def process_directory(directory_path, output_dir, csv_writer):
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory_path, ext)))

        batch = []

        for i, image_path in enumerate(sorted(image_files), 1):
            image = load_local_image(image_path)
            positions, binary = detect_black_circular_patches(image)
            count = len(positions)
            binary_with_circles = draw_circles_on_binary(binary, positions)

            left_margin_x = find_left_text_margin_with_easyocr(image)
            green_margin_x = None

            if left_margin_x is not None:
                green_margin_x = find_second_margin_green(image, left_margin_x)
                marked_img = draw_two_margins(binary_with_circles, red_x=left_margin_x, green_x=green_margin_x)
            else:
                marked_img = binary_with_circles

            filename = os.path.basename(image_path)
            batch.append((
                filename,
                str(positions),
                count,
                left_margin_x if left_margin_x is not None else "N/A",
                green_margin_x if green_margin_x is not None else "N/A",
                marked_img
            ))

            if i % 10 == 0 or i == len(image_files):
                # Write to CSV and save images
                for filename, pos_str, count, red_x, green_x, img in batch:
                    csv_writer.writerow([filename, pos_str, count, red_x, green_x])
                    cv2.imwrite(os.path.join(output_dir, filename), img)
                batch.clear()  # Clear memory

    # Main run
    os.makedirs(out_path, exist_ok=True)
    csv_path = os.path.join(out_path, "labeled-data.csv")

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image", "marked", "count", "left_margin_x", "green_margin_x"])
        process_directory(src_path, out_path, writer)
