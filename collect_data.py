 
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

# ==========================================
# 1. DATASET CONFIGURATION
# ==========================================

OUTPUT_DIR = "data/collected"
CSV_FILE = os.path.join(OUTPUT_DIR, "gestures.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add your own custom signs here
CUSTOM_SIGNS = [
    
    "YES",
    "NO",
    
]

SAMPLES_PER_SIGN = 10

# 126 features + label
FEATURE_COLUMNS = [
    f"f{i}" for i in range(126)
]

COLUMNS = FEATURE_COLUMNS + ["label"]

# ==========================================
# 2. MEDIAPIPE INITIALIZATION
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "hand_landmarker.task not found!"
    )

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# 3. EXTRACT 126 LANDMARK FEATURES
# ==========================================

def extract_features(hand_landmarks):

    hands = []

    for hand in hand_landmarks:

        coords = np.array([
            [lm.x, lm.y, lm.z]
            for lm in hand
        ], dtype=np.float32)

        center_x = float(np.mean(coords[:, 0]))

        hands.append((center_x, coords))

    # Sort hands from left to right
    hands.sort(key=lambda item: item[0])

    hands = [item[1] for item in hands[:2]]

    # Two hands x 21 landmarks x 3 coordinates
    features = np.zeros(
        (2, 21, 3),
        dtype=np.float32
    )

    for i, hand in enumerate(hands):
        features[i] = hand

    if len(hands) == 0:
        return None

    # Normalize around wrist center
    wrist_center = np.mean(
        features[:len(hands), 0, :],
        axis=0
    )

    features[:len(hands)] -= wrist_center

    # Scale normalization
    scale = np.max(
        np.abs(features[:len(hands), :, :2])
    )

    if scale > 1e-6:
        features[:len(hands)] /= scale

    return features.flatten()


# ==========================================
# 4. DRAW HAND LANDMARKS
# ==========================================

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def draw_landmarks(frame, hand_landmarks):

    height, width, _ = frame.shape

    for hand in hand_landmarks:

        points = []

        for lm in hand:

            x = int(lm.x * width)
            y = int(lm.y * height)

            points.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        for start, end in CONNECTIONS:

            cv2.line(
                frame,
                points[start],
                points[end],
                (255, 0, 0),
                2
            )


# ==========================================
# 5. CHECK EXISTING SAMPLES
# ==========================================

def get_sample_count(sign):

    if not os.path.exists(CSV_FILE):
        return 0

    if os.path.getsize(CSV_FILE) == 0:
        return 0

    df = pd.read_csv(CSV_FILE)

    if "label" not in df.columns:
        raise ValueError("CSV does not contain label column.")

    if list(df.columns) != COLUMNS:
        raise ValueError(
            "Existing CSV has a different feature format. "
            "Use a separate file or back up the old CSV."
        )

    return int((df["label"] == sign).sum())


# ==========================================
# 6. START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam not found!")

print("Webcam started successfully!")
print("Press S to save one sample.")
print("Press Q to stop collection.")

# ==========================================
# 7. COLLECT CUSTOM SIGN DATA
# ==========================================

try:

    with HandLandmarker.create_from_options(options) as landmarker:

        frame_number = 0

        for sign in CUSTOM_SIGNS:

            count = get_sample_count(sign)

            if count >= SAMPLES_PER_SIGN:

                print(f"{sign} already completed.")
                continue

            print("\n================================")
            print("COLLECTING SIGN:", sign)
            print("================================")
            print("Perform the sign.")
            print("Press S to save each sample.")
            print("Press Q to stop.")

            while count < SAMPLES_PER_SIGN:

                success, frame = cap.read()

                if not success:
                    print("Unable to read webcam frame.")
                    break

                # Mirror webcam
                frame = cv2.flip(frame, 1)

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = frame_number * 33

                results = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                frame_number += 1

                features = None

                if results.hand_landmarks:

                    draw_landmarks(
                        frame,
                        results.hand_landmarks
                    )

                    features = extract_features(
                        results.hand_landmarks
                    )

                    status = "HAND DETECTED - PRESS S"

                else:

                    status = "SHOW YOUR HANDS"

                # Display sign name
                cv2.putText(
                    frame,
                    f"SIGN: {sign}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # Display sample count
                cv2.putText(
                    frame,
                    f"SAMPLES: {count}/{SAMPLES_PER_SIGN}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    status,
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Q = STOP",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.imshow(
                    "Custom ISL Sign Collection",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                # Stop collection
                if key == ord('q'):
                    raise KeyboardInterrupt

                # Save sample
                if key == ord('s'):

                    if features is None:
                        print("No hand detected. Try again.")
                        continue

                    row = list(features)
                    row.append(sign)

                    df = pd.DataFrame(
                        [row],
                        columns=COLUMNS
                    )

                    df.to_csv(
                        CSV_FILE,
                        mode="a",
                        header=(
                            not os.path.exists(CSV_FILE)
                            or os.path.getsize(CSV_FILE) == 0
                        ),
                        index=False
                    )

                    count += 1

                    print(
                        f"{sign}: "
                        f"{count}/{SAMPLES_PER_SIGN} saved"
                    )

            print(f"{sign} collection completed.")

except KeyboardInterrupt:

    print("\nCollection stopped by user.")

finally:

    cap.release()
    cv2.destroyAllWindows()

print("\n================================")
print("DATA COLLECTION FINISHED")
print("================================")

print("Saved dataset:", CSV_FILE)
