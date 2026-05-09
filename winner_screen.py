import cv2 as cv
import numpy as np

def show_winner_screen(photo_path, username):

    photo = cv.imread(photo_path)

    if photo is None:
        return
    
    screen_h = 720
    screen_w = 1280

    screen = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

    # Reize winner photo
    photo = cv.resize(photo, (350, 350))

    x = screen_w // 2 - 175
    y = 170 
    
    screen[y: y+350, x:x +350] = photo

    cv.putText(screen, "Congratulations!", (screen_w // 2 - 230, 90), cv.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)

    cv.putText(screen, f"you are the kign of rock paper scissors!!!", (screen_w // 2 - 430, 580), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    cv.putText(screen, "Press R to restart game", (screen_w // 2 - 190, 660), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    window_name = "Winner Screen"
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    while True:
        cv.imshow(window_name, screen)

        key = cv.waitKey(30) & 0xFF

        if key == ord("r"):
            cv.destrouWindow(window_name)
            return