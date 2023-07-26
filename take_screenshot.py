import numpy as np
import cv2
import mediapipe as mp
import pygame.mixer

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

pygame.mixer.init()
pygame.mixer.music.load("L.mp3")
is_music_playing = False

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # Accessing the landmarks by their position
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # Array to hold True or False if finger is folded
            finger_fold_status = []
            for tip in finger_tips:
                # Getting the landmark tip position and drawing blue circle
                x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
                cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

                # Writing condition to check if finger is folded, i.e., checking if finger tip position is smaller than finger starting position (inner landmark) for the index finger
                # If finger is folded, changing color to green
                if lm_list[tip].x < lm_list[tip - 3].x:
                    cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            print(finger_fold_status)

            # Checking if all fingers are folded
            if all(finger_fold_status):
                # Play or Pause music
                if is_music_playing:
                    pygame.mixer.music.pause()
                    is_music_playing = False
                else:
                    pygame.mixer.music.unpause()
                    is_music_playing = True

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))

    cv2.imshow("hand tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break