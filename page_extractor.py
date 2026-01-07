import cv2
import numpy as np
import argparse
import os
import logging

# --- LOGGING SETUP FUNCTION ---
def setup_logging(logging_on, image_file_path, log_file_path):
    """
    Configures the logging system based on user-provided arguments.
    """
    if not logging_on:
        # If logging is OFF, return a logger that does nothing (NullHandler)
        logging.getLogger().addHandler(logging.NullHandler())
        return

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 1. Determine the log file path
    if log_file_path:
        # Case 2: Append to user-specified log file
        final_log_path = log_file_path
        file_mode = 'a' # 'a' for append
    else:
        # Case 1: Create log file named after the input image
        base, _ = os.path.splitext(image_file_path)
        final_log_path = base + '.log'
        file_mode = 'w' # 'w' for overwrite/new file

    # Check if a FileHandler already exists to prevent duplicate output
    if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        # Create file handler, respecting the file mode (append or write)
        file_handler = logging.FileHandler(final_log_path, mode=file_mode)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Also keep a basic stream handler for console feedback
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logger.addHandler(stream_handler)

    print(f"Logging messages will be saved to: {final_log_path}")

# --- HELPER FUNCTIONS (Unchanged) ---

def order_points(pts):
    """Orders the 4 points consistently for perspective transform (TL, TR, BR, BL)."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Applies the perspective warp to the image."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_A), int(width_B))

    height_A = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_B = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_A), int(height_B))

    dst = np.array([
        [0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped

# --- MAIN FUNCTION (Uses logging instead of print) ---

def extract_page(image_path, use_postprocess):
    logging.info(f"Attempting to process image: {image_path}")

    # --- 1. Load and Pre-Process Image ---
    img = cv2.imread(image_path)

    if img is None:
        logging.error(f"Failed to load image at {image_path}. File might be corrupted or path incorrect.")
        return

    original_img = img.copy()

    center_x_orig = original_img.shape[1] // 2
    center_y_orig = original_img.shape[0] // 2

    height = img.shape[0]
    if height > 500:
        ratio = height / 500.0
        img = cv2.resize(img, (int(img.shape[1] / ratio), 500))
    else:
        ratio = 1.0

    center_point_resized = (int(center_x_orig / ratio), int(center_y_orig / ratio))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # --- 2. Find Page Contour with Center Constraint ---
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    page_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            if cv2.pointPolygonTest(c, center_point_resized, False) >= 0:
                page_contour = approx
                break

    if page_contour is None:
        logging.error(f"Contour Error: Could not find a valid 4-point contour containing the center for {image_path}.")
        return

    # --- 3. Perspective Transformation ---
    pts = page_contour.reshape(4, 2) * ratio
    warped = four_point_transform(original_img, pts)

    # --- 4. Optional Post-Processing and Save ---
    if use_postprocess:
        logging.info("Applying B&W post-processing (Adaptive Thresholding)...")
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        output_image = cv2.adaptiveThreshold(warped_gray, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        save_suffix = '_extracted_bw.png'
    else:
        logging.info("Skipping post-processing. Saving color image.")
        output_image = warped
        save_suffix = '_extracted_color.png'

    base, ext = os.path.splitext(image_path)
    output_path = base + save_suffix

    cv2.imwrite(output_path, output_image)
    logging.info(f"SUCCESS: Saved extracted page to: {output_path}")

# --- MAIN EXECUTION BLOCK (Modified) ---

if __name__ == '__main__':
    metadata = """
Metadata:
  Author: Dorde Stojicevic
  License: GPL-2.0
"""

    examples = """
Examples:
  1. Process a single file with logging ON (log file: my_screenshot.log):
     python page_extractor.py my_screenshot.png --logging ON

  2. Batch process files and append all results to a single log file:
     for file in *.png; do python page_extractor.py "$file" --logging ON --logfile batch_run.log; done

  3. Process a single file with no logging (and color output):
     python page_extractor.py another_shot.jpg --logging OFF --postprocess OFF
"""

    # Combine metadata and examples into the epilog
    full_epilog = metadata + "\n" + examples

    parser = argparse.ArgumentParser(
        description="""
        A robust document scanner script using OpenCV to extract a book page
        from a screenshot (or photo) and correct its perspective.
        It assumes the book page is the largest 4-sided object covering the center of the image.
        """,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=full_epilog
    )

    parser.add_argument("image_file", help="The path to the input screenshot/image file (e.g., photo_of_book.jpg).")

    parser.add_argument("--postprocess",
                        choices=['ON', 'OFF'],
                        default='ON',
                        help="""
                        Controls the final image enhancement:
                        ON: (Default) Converts the extracted page to B&W using adaptive thresholding (best for text/OCR).
                        OFF: Saves the extracted page as a raw color image.
                        """)

    parser.add_argument("--logging",
                        choices=['ON', 'OFF'],
                        default='OFF',
                        help="Turn logging ON or OFF (Default: OFF).")

    parser.add_argument("--logfile",
                        type=str,
                        default=None,
                        help="""
                        (Optional) Specify an external log file path.
                        If provided, the script appends to this file.
                        If omitted, a new log file named after the input image is created/overwritten.
                        """)

    args = parser.parse_args()

    # Setup Logging first, as it needs to capture messages from extract_page
    logging_flag = (args.logging.upper() == 'ON')
    if logging_flag:
        setup_logging(logging_flag, args.image_file, args.logfile)

    postprocess_flag = (args.postprocess.upper() == 'ON')

    extract_page(args.image_file, postprocess_flag)

    print(f"\n--- Script Finished ---")
