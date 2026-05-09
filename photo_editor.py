import cv2 as cv
import numpy as np

def apply_photo_settings(original, brightness, contrast, saturation, grayscale, sharpen):
    edited = original.copy()

    # Brightness and contrast
    # Formula: new_pixel = old_pixel * contrast + brightness
    edited = edited.astype(np.float32)
    edited = edited * contrast + brightness
    edited = np.clip(edited, 0, 255)
    edited = edited.astype(np.uint8)

    # Saturation adjustment
    hsv = cv.cvtColor(edited, cv.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv = hsv.astype(np.uint8)
    edited = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # GrayScale
    if grayscale == 1:
        gray = cv.cvtColor(edited, cv.COLOR_BGR2GRAY)
        edited = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

    # Sharpen 
    # Using the Laplacian-style sharpening kernel
    if sharpen == 1:
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        edited = cv.filter2D(edited, -1, kernel)
    return edited


def open_photo_editor(image, save_path="winner_photo.jpg"):
    if image is None:
        return None, False 
    
    window_name = "Photo Editor"

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    cv.createTrackbar("Brightness", window_name, 100, 200, lambda x: None)
    cv.createTrackbar("Contrast", window_name, 100, 200, lambda x: None)
    cv.createTrackbar("Saturation", window_name, 100, 200, lambda x: None)
    cv.createTrackbar("Grayscale", window_name, 0, 1, lambda x: None)
    cv.createTrackbar("Sharpen", window_name, 0, 1, lambda x: None)

    final_image = image.copy()

    while True:
        brightness_bar = cv.getTrackbarPos("Brightness", window_name)
        contrast_bar = cv.getTrackbarPos("Contrast", window_name)
        saturation_bar = cv.getTrackbarPos("Saturation", window_name)
        grayscale = cv.getTrackbarPos("Grayscale", window_name)
        sharpen = cv.getTrackbarPos("Sharpen", window_name)

        brightness = brightness_bar - 100
        contrast = max(0.1, contrast_bar / 100)
        saturation = max(0.0, saturation_bar / 100)

        final_image = apply_photo_settings(image, brightness, contrast, saturation, grayscale, sharpen)

        preview = final_image.copy()

    
        cv.putText(preview, "Adjust settings with trackbars", (20, 35), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv.putText(preview, "S: Save, ESC: Cancel, R: Restart", (20, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv.imshow(window_name, preview)

        key = cv.waitKey(30) & 0xFF

        # Saving the photo in the curren directory
        if key == ord("s"):
            cv.imwrite(save_path, final_image)
            cv.destroyWindow(window_name)
            return save_path
        # Press ESC if you don't want to perform photo editing 
        if key == 27:
            cv.destroyWindow(window_name)
            return None
        
        # R restarts game
        if key == ord("r"):
            cv.destroyWindow(window_name)
            return None, True