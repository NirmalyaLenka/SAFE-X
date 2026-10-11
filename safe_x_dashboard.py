
import json
import socket
import threading
import queue
from pathlib import Path
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# Existing SAFE-X configuration
ROOT = Path(r"C:\Users\Nirmalya Lenka\Documents\SAFE-X")
WORKERS_FILE = ROOT / "data" / "workers.json"
MODEL_PATH = Path(
    r"C:\Users\Nirmalya Lenka\runs\detect\results\safe_x_baseline\weights\best.pt"
)

RFID_HOST = "0.0.0.0"
RFID_PORT = 8765
PHONE_URL = "http://10.213.204.242:8080/video"

st.set_page_config(
    page_title="SAFE-X Dashboard",
    page_icon="🦺",
    layout="wide"
)

# Keep shared resources across Streamlit reruns.
@st.cache_resource
def load_model(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"YOLO model not found: {path}")
    return YOLO(path)

@st.cache_resource
def get_camera_manager():
    return {
        "source": None,
        "capture": None
    }

@st.cache_resource
def get_rfid_queue():
    return queue.Queue()


@st.cache_resource
def start_rfid_server():
    events = get_rfid_queue()

    def server():
        try:
            server_socket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM
            )
            server_socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
            )
            server_socket.bind((RFID_HOST, RFID_PORT))
            server_socket.listen(5)

            while True:
                conn, address = server_socket.accept()
                with conn:
                    buffer = ""
                    while True:
                        data = conn.recv(1024)
                        if not data:
                            break
                        
                        buffer += data.decode(
                            "utf-8", errors="ignore"
                        )
                        print("RFID RAW DATA:", repr(data), flush=True)

                        while "\n" in buffer or "\r" in buffer:
                            positions = [
                                p for p in (
                                    buffer.find("\n"),
                                    buffer.find("\r")
                                ) if p >= 0
                            ]
                            cut = min(positions)
                            uid = buffer[:cut].strip()
                            buffer = buffer[cut + 1:]

                            if uid:
                                events.put({
                                    "UID": uid,
                                    "Device IP": address[0],
                                    "Time": datetime.now().strftime(
                                        "%Y-%m-%d %H:%M:%S"
                                    )
                                })

                    if buffer.strip():
                        events.put({
                            "UID": buffer.strip(),
                            "Device IP": address[0],
                            "Time": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        })

        except Exception as exc:
            events.put({
                "UID": "RFID SERVER ERROR: " + str(exc),
                "Device IP": "",
                "Time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            })

    threading.Thread(
        target=server, daemon=True, name="SAFE-X-RFID"
    ).start()
    return True


def normalize_uid(uid):
    return "".join(
        char for char in str(uid).upper()
        if char in "0123456789ABCDEF"
    )



def load_workers():
    if not WORKERS_FILE.exists():
        return {}, f"Missing file: {WORKERS_FILE}"

    try:
        raw = json.loads(
            WORKERS_FILE.read_text(encoding="utf-8-sig")
        )
    except Exception as exc:
        return {}, f"Cannot read workers.json: {exc}"

    # Support the actual format:
    # {"workers": {"DE:B8:08:05": {"name": "Nirmalya"}}}
    if isinstance(raw, dict) and isinstance(raw.get("workers"), dict):
        records = raw["workers"].items()

    # Also support {"workers": [{...}, {...}]}
    elif isinstance(raw, dict) and isinstance(raw.get("workers"), list):
        records = [
            (
                next((
                    item.get(key)
                    for key in (
                        "uid", "rfid_uid", "card_uid", "rfid",
                        "card", "card_id", "tag_uid"
                    )
                    if item.get(key)
                ), None),
                item
            )
            for item in raw["workers"]
            if isinstance(item, dict)
        ]

    # Support a top-level UID-to-worker dictionary
    elif isinstance(raw, dict):
        records = raw.items()

    # Support a top-level list of worker records
    elif isinstance(raw, list):
        records = [
            (
                next((
                    item.get(key)
                    for key in (
                        "uid", "rfid_uid", "card_uid", "rfid",
                        "card", "card_id", "tag_uid"
                    )
                    if item.get(key)
                ), None),
                item
            )
            for item in raw
            if isinstance(item, dict)
        ]
    else:
        return {}, "Unsupported workers.json structure"

    workers = {}

    for uid, worker in records:
        if not uid or not isinstance(worker, dict):
            continue

        normalized = normalize_uid(uid)
        if normalized:
            workers[normalized] = worker

    return workers, None


def worker_name(worker):
    for key in ("name", "worker_name", "full_name", "employee_name"):
        if worker.get(key):
            return str(worker[key])
    return "Registered worker"
# Refresh the dashboard every 1 second to process new RFID scans
st_autorefresh(interval=1000, key="safe_x_refresh")

st.title("SAFE-X")
st.caption(
    "RFID Worker Verification | AI-Based PPE Monitoring"
)

workers, worker_error = load_workers()

try:
    start_rfid_server()
    rfid_server_message = (
        "RFID listener started or already initialized in this app."
    )
except Exception as exc:
    rfid_server_message = f"RFID initialization error: {exc}"

if worker_error:
    st.error(worker_error)

with st.sidebar:
    st.header("Monitoring settings")

    camera_source = st.selectbox(
        "Select camera",
        ["Laptop Webcam", "Mobile Camera"]
    )

    phone_url = st.text_input(
        "Mobile camera stream URL",
        value=PHONE_URL
    )

    confidence = st.slider(
        "YOLO confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05
    )

    start_detection = st.toggle(
        "Enable PPE detection",
        value=False
    )

    st.divider()
    st.write("RFID TCP port:", RFID_PORT)
    st.caption(rfid_server_message)

events = get_rfid_queue()

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []


while True:
    try:
        event = events.get_nowait()
    except queue.Empty:
        break

    normalized = normalize_uid(event["UID"])

    print("SCANNED UID:", repr(event["UID"]), flush=True)
    print("NORMALIZED UID:", repr(normalized), flush=True)
    print("REGISTERED UID KEYS:", list(workers.keys()), flush=True)

    worker = workers.get(normalized)

    event["Worker"] = (
        worker_name(worker) if worker else "UNKNOWN CARD"
    )
    event["Identity"] = (
        "REGISTERED" if worker else "DENY"
    )

    st.session_state.scan_history.insert(0, event)
    st.session_state.scan_history = (
        st.session_state.scan_history[:100]
    )
    st.session_state.last_scan = event

last_scan = st.session_state.get("last_scan", {})

m1, m2, m3 = st.columns(3)
m1.metric("Registered workers", len(workers))
m2.metric(
    "Last RFID scan",
    last_scan.get("Time", "None")
)
m3.metric(
    "Last scanned worker",
    last_scan.get("Worker", "None")
)

camera_column, worker_column = st.columns([1.7, 1])

with camera_column:
    st.subheader("Live PPE monitoring")
    frame_placeholder = st.empty()
    detection_placeholder = st.empty()

    if start_detection:
        try:
            model = load_model(str(MODEL_PATH))
            manager = get_camera_manager()

            desired_source = (
                "Laptop Webcam"
                if camera_source == "Laptop Webcam"
                else phone_url
            )

            if (
                manager["capture"] is None
                or manager["source"] != desired_source
                or not manager["capture"].isOpened()
            ):
                old_camera = manager["capture"]

                if old_camera is not None:
                    old_camera.release()

                if camera_source == "Laptop Webcam":
                    camera = cv2.VideoCapture(0)
                else:
                    camera = cv2.VideoCapture(phone_url)

                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                manager["capture"] = camera
                manager["source"] = desired_source

            camera = manager["capture"]

            if not camera.isOpened():
                st.error("Cannot open the selected camera.")
            else:
                ok, frame = camera.read()

                if not ok:
                    st.warning(
                        "Waiting for a camera frame. Check the camera connection."
                    )
                else:
                    result = model.predict(
                        frame,
                        conf=confidence,
                        verbose=False,
                        imgsz=640
                    )[0]

                    annotated = result.plot()

                    frame_placeholder.image(
                        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )

                    names = model.names
                    detections = []

                    if result.boxes is not None:
                        for box in result.boxes:
                            index = int(box.cls[0].item())
                            detections.append(str(names[index]))

                    detection_placeholder.write(
                        "**Detected classes:** "
                        + (
                            ", ".join(detections)
                            if detections
                            else "No detections"
                        )
                    )

        except Exception as exc:
            st.error(f"Detection error: {exc}")

    else:
        st.info(
            "Enable PPE detection in the sidebar to start camera monitoring."
        )

with worker_column:
    st.subheader("Latest RFID scan")

    if last_scan:
        st.write("**Worker:**", last_scan["Worker"])
        st.write("**UID:**", last_scan["UID"])
        st.write("**Time:**", last_scan["Time"])
        st.write("**Device IP:**", last_scan["Device IP"])

        if last_scan["Identity"] == "DENY":
            st.error("Unknown RFID card — access denied.")
        else:
            st.success("RFID identity registered.")
    else:
        st.info("Waiting for an RFID scan.")
st.divider()
st.subheader("RFID scan history")

if st.session_state.scan_history:
    st.dataframe(
        pd.DataFrame(st.session_state.scan_history),
        use_container_width=True,
        hide_index=True
    )
else:
    st.caption("No RFID scans recorded in this session.")

st.caption(
    "SAFE-X is a prototype. Do not use AI detections alone "
    "to make real-world industrial access decisions."
)