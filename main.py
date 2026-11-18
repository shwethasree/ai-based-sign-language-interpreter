 
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

from sentence_builder import add_sign, get_sentence, clear_sentence
from speech import speak_sentence


# ==========================================
# 1. MODEL CONFIGURATION
# ==========================================

MODEL_PATH = "models/isl_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"

CONFIDENCE_THRESHOLD = 0.65
EXPECTED_FEATURES = 126


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Trained model not found! Run train_model.py first."
    )

if not os.path.exists(HAND_MODEL_PATH):
    raise FileNotFoundError(
        "hand_landmarker.task not found!"
    )


model = joblib.load(MODEL_PATH)

print("Trained model loaded successfully!")

if model.n_features_in_ != EXPECTED_FEATURES:
    raise ValueError(
        f"Model expects {model.n_features_in_} features, "
        f"but this program provides {EXPECTED_FEATURES}."
    )


# ==========================================
# 2. MEDIAPIPE INITIALIZATION
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# 3. EXTRACT 126 FEATURES
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

    hands.sort(key=lambda item: item[0])

    hands = [item[1] for item in hands[:2]]

    if len(hands) == 0:
        return None

    features = np.zeros(
        (2, 21, 3),
        dtype=np.float32
    )

    for i, hand in enumerate(hands):
        features[i] = hand

    wrist_center = np.mean(
        features[:len(hands), 0, :],
        axis=0
    )

    features[:len(hands)] -= wrist_center

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
# 5. START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam not found!")


print("Camera started successfully!")
print("Show your sign in front of the camera.")

print("\nKEYBOARD CONTROLS")
print("A = Add recognized sign")
print("C = Clear sentence")
print("S = Speak sentence once per press")
print("Q = Exit webcam")


# ==========================================
# 6. LIVE SIGN RECOGNITION
# ==========================================

try:

    with HandLandmarker.create_from_options(options) as landmarker:

        frame_number = 0

        while True:

            success, frame = cap.read()

            if not success:
                print("Unable to read webcam frame.")
                break

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

            predicted_sign = "SHOW YOUR HAND"
            confidence = 0.0


            # ==================================
            # 7. PREDICT SIGN
            # ==================================

            if results.hand_landmarks:

                draw_landmarks(
                    frame,
                    results.hand_landmarks
                )

                features = extract_features(
                    results.hand_landmarks
                )

                if features is not None:

                    input_data = np.array(
                        features
                    ).reshape(1, -1)

                    prediction = model.predict(
                        input_data
                    )[0]

                    if hasattr(model, "predict_proba"):

                        probabilities = model.predict_proba(
                            input_data
                        )[0]

                        confidence = float(
                            np.max(probabilities)
                        )

                        if confidence >= CONFIDENCE_THRESHOLD:
                            predicted_sign = prediction
                        else:
                            predicted_sign = "UNKNOWN"

                    else:
                        predicted_sign = prediction


            # ==================================
            # 8. DISPLAY PREDICTION
            # ==================================

            cv2.putText(
                frame,
                "PREDICTED SIGN:",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                str(predicted_sign),
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence * 100:.1f}%",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


            # ==================================
            # 9. DISPLAY FORMED SENTENCE
            # ==================================

            sentence = get_sentence()

            cv2.putText(
                frame,
                "SENTENCE:",
                (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                sentence[:45],
                (20, 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )


            # ==================================
            # 10. DISPLAY KEYBOARD INSTRUCTIONS
            # ==================================

            cv2.putText(
                frame,
                "A: ADD SIGN",
                (20, 330),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "C: CLEAR SENTENCE",
                (20, 365),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "S: SPEAK SENTENCE",
                (20, 400),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Q: EXIT",
                (20, 435),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            # ==================================
            # 11. SHOW WEBCAM
            # ==================================

            cv2.imshow(
                "AI Indian Sign Language Interpreter",
                frame
            )


            # ==================================
            # 12. KEYBOARD CONTROLS
            # ==================================

            key = cv2.waitKey(1) & 0xFF

            # A = Add recognized sign
            if key == ord('a'):

                if predicted_sign not in [
                    "UNKNOWN",
                    "SHOW YOUR HAND"
                ]:

                    add_sign(predicted_sign)

                    print("Added:", predicted_sign)
                    print("Current sentence:", get_sentence())

                else:
                    print("No valid sign detected!")


            # C = Clear sentence
            elif key == ord('c'):

                clear_sentence()

                print("Sentence cleared!")


            # S = Speak sentence exactly once per press
            elif key == ord('s'):

                sentence = get_sentence().strip()

                if sentence:

                    print("Speech request added:", sentence)

                    speak_sentence(sentence)

                else:
                    print("No sentence available to speak!")


            # Q = Exit
            elif key == ord('q'):
                break


finally:

    cap.release()
    cv2.destroyAllWindows()

    print("Camera closed successfully!")
