import cv2
import mediapipe as mp
import pyautogui
import os
import time
import math
import webbrowser


# ============================================================
# 1. MEDIAPIPE SETUP
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task"
)

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions


options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = HandLandmarker.create_from_options(options)


# ============================================================
# 2. SCREEN
# ============================================================

screen_width, screen_height = pyautogui.size()


# ============================================================
# 3. CAMERA
# ============================================================

cap = cv2.VideoCapture(0)


# ============================================================
# 4. SETTINGS
# ============================================================

# Increase this if mouse is still slow
MOUSE_SENSITIVITY = 1.8

# Pinch threshold
PINCH_THRESHOLD = 0.75

# How long pinch must be held before we consider it a click
PINCH_HOLD_TIME = 0.15

# Screenshot gesture must remain stable for this many seconds
FIST_CONFIRM_TIME = 0.15

# YouTube song
BRAVE_SONG_URL = "https://www.youtube.com/shorts/CFKE6xV55BU"


# ============================================================
# 5. VARIABLES
# ============================================================

frame_timestamp = 0

last_action_time = 0

# Mouse smoothing
previous_mouse_x = None
previous_mouse_y = None

# Pinch
pinch_active = False
pinch_start_time = None

# Screenshot
palm_detected_time = None
fist_start_time = None
screenshot_taken = False

# Brave song gesture state
brave_gesture_active = False
brave_gesture_start_time = None
BRAVE_CONFIRM_TIME = 0.5


# ============================================================
# 6. FUNCTIONS
# ============================================================

def distance(point1, point2):
    """
    Calculate distance between two landmarks.
    """

    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


def palm_size(hand):
    """
    Estimate hand size.

    We use wrist (0) and middle finger MCP (9)
    to estimate how large the hand appears.
    """

    return distance(
        hand[0],
        hand[9]
    )


def is_pinch(hand):
    """
    Detect thumb + index pinch.

    Uses hand-size normalization so the
    detection works better at different
    distances from the camera.
    """

    thumb_tip = hand[4]
    index_tip = hand[8]

    pinch_distance = distance(
        thumb_tip,
        index_tip
    )

    hand_size = palm_size(hand)

    if hand_size == 0:
        return False

    ratio = pinch_distance / hand_size

    return ratio < PINCH_THRESHOLD


def fingers_up(hand):
    """
    Returns:

    [index, middle, ring, little]

    True = finger up
    False = finger folded
    """

    fingers = []

    # Index
    fingers.append(
        hand[8].y < hand[6].y
    )

    # Middle
    fingers.append(
        hand[12].y < hand[10].y
    )

    # Ring
    fingers.append(
        hand[16].y < hand[14].y
    )

    # Little
    fingers.append(
        hand[20].y < hand[18].y
    )

    return fingers


def is_open_palm(hand):

    return fingers_up(hand) == [
        True,
        True,
        True,
        True
    ]


def is_fist(hand):

    return fingers_up(hand) == [
        False,
        False,
        False,
        False
    ]


def can_perform_action(cooldown=0.5):

    global last_action_time

    current_time = time.time()

    if current_time - last_action_time >= cooldown:

        last_action_time = current_time

        return True

    return False


def hand_label(result, hand_index):
    """Return MediaPipe's Left/Right label for a detected hand."""
    try:
        return result.handednesses[hand_index][0].category_name
    except (IndexError, AttributeError):
        return None


# ============================================================
# 7. MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Detect hands
    result = detector.detect_for_video(
        mp_image,
        frame_timestamp
    )

    frame_timestamp += 1


    hands = result.hand_landmarks

    gesture_text = "No Gesture"


    # ========================================================
    # DRAW LANDMARKS
    # ========================================================

    if hands:

        for hand in hands:

            points = []

            for landmark in hand:

                x = int(
                    landmark.x *
                    frame.shape[1]
                )

                y = int(
                    landmark.y *
                    frame.shape[0]
                )

                points.append(
                    (x, y)
                )

                cv2.circle(
                    frame,
                    (x, y),
                    4,
                    (0, 255, 0),
                    -1
                )


            connections = [

                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),

                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),

                (5, 9),
                (9, 10),
                (10, 11),
                (11, 12),

                (9, 13),
                (13, 14),
                (14, 15),
                (15, 16),

                (13, 17),
                (17, 18),
                (18, 19),
                (19, 20),

                (0, 17)
            ]


            for start, end in connections:

                cv2.line(
                    frame,
                    points[start],
                    points[end],
                    (255, 0, 0),
                    2
                )


    # ========================================================
    # TWO HAND GESTURES
    # ========================================================

    if len(hands) == 2:

        fingers1 = fingers_up(hands[0])
        fingers2 = fingers_up(hands[1])

        # The camera image is mirrored, so determine the user's
        # left/right hand from its position in the camera window.
        # Left side of the window = user's LEFT hand.
        # Right side of the window = user's RIGHT hand.
        center_x1 = sum(lm.x for lm in hands[0]) / len(hands[0])
        center_x2 = sum(lm.x for lm in hands[1]) / len(hands[1])

        if center_x1 < center_x2:
            left_fingers = fingers1
            right_fingers = fingers2
        else:
            left_fingers = fingers2
            right_fingers = fingers1

        # ----------------------------------------------------
        # LEFT INDEX + RIGHT THREE FINGERS → PLAY BRAVE
        # ----------------------------------------------------

        # Three fingers = index + middle + ring up,
        # little finger down.
        if (
            left_fingers == [True, False, False, False]
            and
            right_fingers == [True, True, True, False]
        ):

            gesture_text = "PLAY BRAVE"

            # Open YouTube only once while the gesture is held.
            # Require the gesture to stay stable for 0.5 second.
            # Then open the song only ONCE.
            if not brave_gesture_active:
                if brave_gesture_start_time is None:
                    brave_gesture_start_time = time.time()

                if time.time() - brave_gesture_start_time >= BRAVE_CONFIRM_TIME:
                    webbrowser.open(BRAVE_SONG_URL)
                    brave_gesture_active = True

        # ----------------------------------------------------
        # BOTH OPEN PALMS → SHOW DESKTOP
        # ----------------------------------------------------

        elif (
            fingers1 == [True, True, True, True]
            and
            fingers2 == [True, True, True, True]
        ):

            gesture_text = "SHOW DESKTOP"

            if can_perform_action(1.5):
                pyautogui.hotkey("win", "d")

        # ----------------------------------------------------
        # BOTH INDEX FINGERS → VOLUME UP
        # ----------------------------------------------------

        elif (
            fingers1 == [True, False, False, False]
            and
            fingers2 == [True, False, False, False]
        ):

            gesture_text = "VOLUME UP"

            if can_perform_action(0.4):
                pyautogui.press("volumeup")

        # ----------------------------------------------------
        # BOTH INDEX + MIDDLE → VOLUME DOWN
        # ----------------------------------------------------

        elif (
            fingers1 == [True, True, False, False]
            and
            fingers2 == [True, True, False, False]
        ):

            gesture_text = "VOLUME DOWN"

            if can_perform_action(0.4):
                pyautogui.press("volumedown")

        else:
            # The Brave gesture ended, so it can be triggered again.
            brave_gesture_active = False
            brave_gesture_start_time = None

    # ========================================================
    # ONE HAND
    # ========================================================

    elif len(hands) == 1:

        hand = hands[0]

        fingers = fingers_up(hand)


        # ====================================================
        # PINCH → LEFT CLICK
        # ====================================================

        pinch = is_pinch(hand)


        if pinch:

            gesture_text = "PINCH"

            if not pinch_active:

                pinch_active = True

                pinch_start_time = time.time()


        else:

            if pinch_active:

                pinch_duration = (
                    time.time() -
                    pinch_start_time
                )


                # Click when pinch is released
                if (
                    pinch_duration >=
                    PINCH_HOLD_TIME
                ):

                    if can_perform_action(0.5):

                        pyautogui.click()


                        gesture_text = (
                            "LEFT CLICK"
                        )


                pinch_active = False

                pinch_start_time = None


        # ====================================================
        # INDEX ONLY → MOUSE
        # ====================================================

        if fingers == [
            True,
            False,
            False,
            False
        ]:

            gesture_text = "MOUSE"


            index_tip = hand[8]


            # Camera coordinates
            camera_x = index_tip.x
            camera_y = index_tip.y


            # Add sensitivity
            target_x = (
                camera_x *
                screen_width
            )

            target_y = (
                camera_y *
                screen_height
            )


            # Smooth movement
            if previous_mouse_x is None:

                previous_mouse_x = target_x
                previous_mouse_y = target_y


            smooth_x = (
                previous_mouse_x * 0.6
                +
                target_x * 0.4
            )

            smooth_y = (
                previous_mouse_y * 0.6
                +
                target_y * 0.4
            )


            # Sensitivity
            center_x = screen_width / 2
            center_y = screen_height / 2


            mouse_x = (
                center_x
                +
                (smooth_x - center_x)
                * MOUSE_SENSITIVITY
            )

            mouse_y = (
                center_y
                +
                (smooth_y - center_y)
                * MOUSE_SENSITIVITY
            )


            # Keep mouse inside screen
            mouse_x = max(
                0,
                min(
                    screen_width - 1,
                    mouse_x
                )
            )

            mouse_y = max(
                0,
                min(
                    screen_height - 1,
                    mouse_y
                )
            )


            pyautogui.moveTo(
                int(mouse_x),
                int(mouse_y),
                duration=0.01
            )


            previous_mouse_x = smooth_x
            previous_mouse_y = smooth_y


        # ====================================================
        # OPEN PALM → SCROLL
        # ====================================================

        elif is_open_palm(hand):

            gesture_text = "OPEN PALM"


            palm_y = hand[9].y


            # Remember palm
            if palm_detected_time is None:

                palm_detected_time = time.time()


            # Store previous palm position
            if not hasattr(
                is_open_palm,
                "previous_y"
            ):

                is_open_palm.previous_y = palm_y


            movement = (
                palm_y -
                is_open_palm.previous_y
            )


            # ------------------------------------------------
            # UP
            # ------------------------------------------------

            if movement < -0.001:

                gesture_text = "SCROLL UP"

                if can_perform_action(0.15):

                    pyautogui.scroll(10)


            # ------------------------------------------------
            # DOWN
            # ------------------------------------------------

            elif movement > 0.001:

                gesture_text = "SCROLL DOWN"

                if can_perform_action(0.2):

                    pyautogui.scroll(-10)


            is_open_palm.previous_y = palm_y


        # ====================================================
        # FIST → SCREENSHOT
        # ====================================================

        elif is_fist(hand):
            gesture_text = "FIST"

            if fist_start_time is None:
                fist_start_time = time.time()

            if time.time() - fist_start_time >= 0.15:

                if not screenshot_taken:

                    screenshot_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "gesture_screenshot.png"
                    )

                    pyautogui.screenshot(screenshot_path)

                    screenshot_taken = True
                    gesture_text = "SCREENSHOT TAKEN"

        # ====================================================
        # OTHER GESTURES
        # ====================================================

        else:

            # Reset mouse smoothing
            previous_mouse_x = None
            previous_mouse_y = None


    # ========================================================
    # RESET STATES
    # ========================================================

    # If no hands
    if len(hands) == 0:

        fist_start_time = None
        screenshot_taken = False
        brave_gesture_active = False
        brave_gesture_start_time = None

        previous_mouse_x = None
        previous_mouse_y = None


    # If two hands
    elif len(hands) == 2:

        # Reset single-hand screenshot state
        fist_start_time = None


    # ========================================================
    # DISPLAY GESTURE
    # ========================================================

    cv2.putText(
        frame,
        gesture_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "Press Q to exit",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # Show camera
    cv2.imshow(
        "AI Hand Gesture Controller",
        frame
    )


    # Exit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

detector.close()

cv2.destroyAllWindows()