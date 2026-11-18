 
import pyttsx3
import threading
import queue

# Speech request queue
speech_queue = queue.Queue()


def speech_worker():

    # Initialize engine inside the worker thread
    engine = pyttsx3.init()

    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)

    while True:

        text = speech_queue.get()

        try:
            if text.strip():
                print("Speaking:", text)

                engine.say(text)
                engine.runAndWait()

        except Exception as e:
            print("Speech error:", e)

        finally:
            speech_queue.task_done()


# Start only ONE background speech thread
worker_thread = threading.Thread(
    target=speech_worker,
    daemon=True
)

worker_thread.start()


def speak_sentence(text):

    if text and text.strip():

        # Add exactly one speech request
        speech_queue.put(text.strip())
