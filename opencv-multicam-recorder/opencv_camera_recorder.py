#!/usr/bin/python3
import time
from datetime import datetime
import cv2
import numpy as np
import threading
import json
import sys
from cv_image_utils import put_text_multiline
from cv_image_utils import cat_vert
from cv_image_utils import cat_horiz
from cv_image_utils import get_camera_name


# TODO: Autodetect devices
# TODO: Get/Display frame stats
# TODO: Improve recording/playback memory management. Free memory after save frames.
# TODO: Replace replay buffer in memory by reading frames from file
# TODO: Option to show camera id (device) and name overlay


class MultiStreamVideoGUI:
    STATE_DISPLAY = 0
    STATE_RECORD = 1
    STATE_RECORD_PAUSED = 2
    STATE_REPLAY = 3
    STATE_REPLAY_PAUSED = 4

    help_text = """ 
    esc      (exit program)\n
    r        (start recording)\n
    s        (save recorded clip and reset)\n
    d        (discard current clip and reset)\n
    p        (enter exit replay mode)\n
    space    (pause or play the replay video)\n
    ->       (pause and advance one frame)\n
    <-       (pause and one frame back)\n
    tab      (select stream)\n
    q        (rotate selected stream 90deg)\n
    h        (toggle this message)"""

    exit_key = 27  # esc      (exit program)
    record_key = 114  # r        (start recording)
    save_key = 115  # s        (save recorded clip and reset)
    erase_key = 100  # d        (discard recorded clip and reset)
    replay_key = 112  # p        (enter exit replay mode)
    pause_key = 32  # space    (pause or play the replay video)
    next_frame_key = 101  # e        (pause and advance one frame)
    prev_frame_key = 119  # w        (pause and one frame back)
    select_stream_key = 9  # tab      (select stream)
    rotate_stream_key = 113  # q        (rotate selected stream 90deg)
    display_stopwatch = 116  # t        (toggle stopwatch)
    display_help_key = 104  # h        (toggle this message)

    def __init__(self, config):
        self.caps = MultiStreamVideoCapturer(config)
        self.caps.start()
        self.writers = MultiStreamVideoWriter(config["fps"],
                                              config["data_folder"])
        self.win_name = config["win_name"]
        self.img_sep = config["img_sep"]
        self.scale_mosaic = config["scale_mosaic"]
        self.target_fps = config["fps"]
        self.data_folder = config["data_folder"]

    def run(self):
        # Init recording variables
        state = self.STATE_DISPLAY

        # Frames are only for replay, store only the shown frame
        frames = list()
        nframe = 0
        help_on = False
        timer_on = True
        selected_stream = -1
        times = dict()

        # Open CV Window
        cv2.namedWindow(self.win_name)

        t_stopwatch = time.time()

        key = 0
        while key != self.exit_key:
            t_start = time.time()

            # If displaying or recording capture frames
            if state == self.STATE_DISPLAY or state == self.STATE_RECORD:
                # Capture a frame from each stream
                t_ini = time.time()
                cap_frames = self.caps.read()
                if cap_frames[0] is None:
                    continue
                times["cap"] = time.time() - t_ini

                # Highlight selected stream
                t_ini = time.time()
                if state == self.STATE_DISPLAY:
                    if selected_stream != -1:
                        cap_frames[selected_stream] = cv2.rectangle(cap_frames[selected_stream], (0, 0),
                                                                    (cap_frames[selected_stream].shape[1],
                                                                     cap_frames[selected_stream].shape[0]),
                                                                    color=(0, 0, 255), thickness=5)
                times["draw_selected_rect"] = time.time() - t_ini

                t_ini = time.time()
                show_frame = np.copy(cap_frames[0])
                times["copy_show_frame"] = time.time() - t_ini

                # Compose the mosaic to show if there are more than one streams active
                t_ini = time.time()
                if len(cap_frames) > 1:
                    for f in cap_frames[1:]:
                        # TODO: Use the resolution of the screen to make the mosaic. Here the horizontal
                        #       screen resolution is set manually.
                        if show_frame.shape[1] + f.shape[1] < 1920:
                            show_frame = cat_horiz(show_frame, f, self.img_sep, self.scale_mosaic)
                        else:
                            show_frame = cat_vert(show_frame, f, self.img_sep, self.scale_mosaic)
                cap_frames.append(show_frame)
                times["compose_mosaic"] = time.time() - t_ini

            if state == self.STATE_RECORD:
                t_ini = time.time()
                self.writers.write(cap_frames)
                times["write_frames"] = time.time() - t_ini
                t_ini = time.time()
                frames.append(np.copy(cap_frames[-1]))
                nframe += 1
                times["copy_frames_for_playback_mode"] = time.time() - t_ini

            elif state == self.STATE_REPLAY:
                t_ini = time.time()
                if nframe >= len(frames):
                    nframe = 0
                show_frame = np.copy(frames[nframe])
                nframe += 1
                times["copy_frame_for_playback_mode"] = time.time() - t_ini

            elif state == self.STATE_REPLAY_PAUSED:
                if nframe >= len(frames):
                    nframe = 0
                show_frame = np.copy(frames[nframe])

            if help_on:
                t_ini = time.time()
                show_frame = put_text_multiline(show_frame, self.help_text, (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (255, 128, 255), 1, line_advance=10)
                times["draw_help_msg"] = time.time() - t_ini
            else:
                t_ini = time.time()
                state_text, state_color = self.get_state_text_color(state, self.writers.sequence_n,
                                                                    self.target_fps, nframe, len(frames))

                state_text += f"\n{time.time() - t_stopwatch:5.2f}s"

                show_frame = put_text_multiline(show_frame, state_text, (10, 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2, line_advance=30)
                times["draw_state_msg"] = time.time() - t_ini

            t_ini = time.time()
            if key == self.record_key:
                if state == self.STATE_REPLAY_PAUSED:
                    state = self.STATE_RECORD_PAUSED
                elif state == self.STATE_DISPLAY or state == self.STATE_REPLAY:
                    state = self.STATE_RECORD

            if key == self.replay_key:
                if (state == self.STATE_DISPLAY or state == self.STATE_RECORD) and len(frames) > 0:
                    if state != self.STATE_DISPLAY:
                        t_stopwatch = time.time()
                    state = self.STATE_REPLAY

                elif state == self.STATE_RECORD_PAUSED:
                    state = self.STATE_REPLAY_PAUSED
                    t_stopwatch = time.time()

                elif state == self.STATE_REPLAY or state == self.STATE_REPLAY_PAUSED:
                    state = self.STATE_DISPLAY

            if key == self.pause_key:
                if state == self.STATE_REPLAY:
                    state = self.STATE_REPLAY_PAUSED

                elif state == self.STATE_REPLAY_PAUSED:
                    state = self.STATE_REPLAY

                elif state == self.STATE_RECORD:
                    state = self.STATE_RECORD_PAUSED

                elif state == self.STATE_RECORD_PAUSED:
                    state = self.STATE_RECORD

            if key == self.save_key:
                self.writers.reset()
                state = self.STATE_DISPLAY
                frames.clear()
                frames = list()
                nframe = 0

            if key == self.erase_key:
                self.writers.reset(overwrite=True)
                state = self.STATE_DISPLAY
                frames.clear()
                frames = list()
                nframe = 0

            if key == self.rotate_stream_key and selected_stream != -1:
                self.caps.rotate(selected_stream)

            if key == self.select_stream_key:
                selected_stream += 1
                if selected_stream >= len(cap_frames) - 1:
                    selected_stream = -1

            if key == self.display_help_key:
                help_on = not help_on

            if key == self.prev_frame_key and state == self.STATE_REPLAY_PAUSED:
                nframe -= 1
                nframe = max(0, nframe)

            if key == self.next_frame_key and state == self.STATE_REPLAY_PAUSED:
                nframe += 1
                nframe = min(len(frames) - 1, nframe)

            times["process_keys"] = time.time() - t_ini

            if help_on:
                debug_str = "\n".join([f"{key}: {val*1000:5.3f}ms" for (key, val) in times.items()])
                show_frame = put_text_multiline(show_frame, debug_str, (20, 240),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, state_color, 1, line_advance=12)

            t_elapsed_ms = int((time.time() - t_start) * 1000)
            wait_ms = int((1/self.target_fps) * 1000) - t_elapsed_ms
            show_frame = put_text_multiline(show_frame,
                                            f"{1000.0/(t_elapsed_ms + max(1,wait_ms)):5.1f}FPS \n "
                                            f"wait_ms:{wait_ms}\n "
                                            f"proc_ms:{t_elapsed_ms}", (10, 80),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, state_color, 1, line_advance=12)

            t_ini = time.time()
            cv2.imshow(self.win_name, show_frame)
            times["imshow"] = time.time() - t_ini

            key = cv2.waitKey(max(1,wait_ms))

        self.caps.release()
        self.writers.reset()

    def get_state_text_color(self, state, seq, fps, nframe, ntotalframes):
        if state == self.STATE_REPLAY_PAUSED:
            state_text = f"mode::replay::paused frame:{nframe}/{ntotalframes - 1}"
            state_color = (255, 0, 0)
        elif state == self.STATE_REPLAY:
            state_text = f"mode::replay frame:{nframe}/{ntotalframes - 1}"
            state_color = (255, 0, 0)
        elif state == self.STATE_RECORD_PAUSED:
            state_text = f"mode::record::paused frame:{nframe} fps:{fps}"
            state_color = (0, 0, 255)
        elif state == self.STATE_RECORD:
            state_text = f"mode::record frame:{nframe} fps:{fps}"
            state_color = (0, 0, 255)
        elif state == self.STATE_DISPLAY:
            state_text = "mode::display"
            state_color = (0, 255, 0)
        else:
            raise ValueError(f"Invalid state {state}")

        return state_text, state_color


class MultiStreamVideoCapturer(threading.Thread):
    def __init__(self, config):
        super().__init__()
        self.video_caps = list()
        self.streams_rotation = list()
        self.streams_timestamp = list()
        self.target_fps = config["fps"]
        self.streams_cfg = config["streams"]
        for stream_cfg in config["streams"]:
            # Fill in default capture options
            if "auto_exposure" not in stream_cfg.keys():
                stream_cfg["auto_exposure"] = 3
            if "exposure" not in stream_cfg.keys():
                stream_cfg["exposure"] = 100
            if "rotation" not in stream_cfg.keys():
                stream_cfg["rotation"] = -1
            if "timestamp" not in stream_cfg.keys():
                stream_cfg["timestamp"] = False

            cap = cv2.VideoCapture(stream_cfg["id"], cv2.CAP_V4L2)
            print(f"Opening cam: {stream_cfg['id']}. Backend: {cv2.CAP_V4L2}")
            if not cap.set(cv2.CAP_PROP_FRAME_WIDTH, stream_cfg["width"]):
                print(f"|-> Unable to set frame width to {stream_cfg['width']} on device: {stream_cfg['id']}")
            if not cap.set(cv2.CAP_PROP_FRAME_HEIGHT, stream_cfg["height"]):
                print(f"|-> Unable to set frame height to {stream_cfg['height']} on device: {stream_cfg['id']}")
            if not cap.set(cv2.CAP_PROP_BUFFERSIZE, stream_cfg["buffer_size"]):
                print(f"|-> Unable to set buffer_size to {stream_cfg['buffer_size']} on device: {stream_cfg['id']}")
            if not cap.set(cv2.CAP_PROP_FPS, self.target_fps):
                print(f"|-> Unable to set FPS to {self.target_fps} on device: {stream_cfg['id']}")
            if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, stream_cfg['auto_exposure']):
                print(f"|-> Unable to set AUTO_EXPOSURE to {stream_cfg['auto_exposure']} on device: {stream_cfg['id']}")
            if not cap.set(cv2.CAP_PROP_EXPOSURE, stream_cfg['exposure']):
                print(f"|-> Unable to set EXPOSURE to {stream_cfg['exposure']} on device: {stream_cfg['id']}")

            if cap.isOpened():
                self.video_caps.append(cap)
                self.streams_rotation.append(stream_cfg["rotation"])
                self.streams_timestamp.append(stream_cfg["timestamp"])
                print(f"Opened cam: {stream_cfg['id']}. Backend: {cap.getBackendName()}")
                print(f"|-> name: {get_camera_name(stream_cfg['id'])}")
                print(f"|-> config: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x"
                      f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}@{cap.get(cv2.CAP_PROP_FPS)}")
            else:
                print(f"|-> ERROR: Unable to open camera {stream_cfg['id']} with "
                      f"{stream_cfg['width']}w x {stream_cfg['height']}h@{config['fps']}fps")

        self.nstreams = len(self.video_caps)
        self.streams_fps = [0.0] * self.nstreams
        self.seq = 0
        self.nframe = 0
        self.mutex = threading.Lock()
        self.frames = [None] * self.nstreams
        self.running = False

    def start(self):
        self.running = True
        super().start()

    def stop(self):
        self.running = False

    def rotate(self, idx):
        self.streams_rotation[idx] += 1
        if self.streams_rotation[idx] > 2:
            self.streams_rotation[idx] = -1

    def read(self):
        cap_frames = [None] * self.nstreams
        self.mutex.acquire()
        for i, frame in enumerate(self.frames):
            if frame is not None:
                cap_frames[i] = np.copy(frame)
        self.mutex.release()
        return cap_frames

    def run(self):
        while self.running:
            cap_frames = list()
            for i, cap in enumerate(self.video_caps):
                cap.grab()

            for i, cap in enumerate(self.video_caps):
                if cap.isOpened():
                    ret, frame = cap.retrieve()
                    if not ret:
                        print(f"Failed to acquire frame from device: "
                              f"{self.streams_cfg[i]['id']}")
                        continue
                    # Rotate if necessary
                    if self.streams_rotation[i] != -1:
                        frame = cv2.rotate(frame, self.streams_rotation[i])
                    if self.streams_timestamp[i]:
                        frame = put_text_multiline(frame,
                                                   datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                                                   (5, frame.shape[0]-10),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1, line_advance=10)
                    cap_frames.append(frame)
            self.mutex.acquire()
            self.frames = cap_frames
            self.mutex.release()
            time.sleep(0.001)

    def release(self):
        self.running = False
        for cap in self.video_caps:
            cap.release()


class MultiStreamVideoWriter:
    def __init__(self, fps, data_folder=""):
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writers = None
        self.sequence_n = 0
        self.target_fps = fps
        self.data_folder = data_folder

    def __del__(self):
        if self.video_writers is not None:
            for writer in self.video_writers:
                if writer is not None:
                    writer.release()

    def reset(self, overwrite=False):
        if self.video_writers is not None:
            for writer in self.video_writers:
                if writer is not None:
                    writer.release()
        if not overwrite:
            self.sequence_n += 1
        self.video_writers = None

    def write(self, frames):
        if self.video_writers is None:
            self.video_writers = [None] * len(frames)

        for i in range(len(frames)):
            if self.video_writers[i] is None:
                date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                filename = f"{self.data_folder}/{date_str}_seq_{self.sequence_n}_cam_{i}.mp4"
                self.video_writers[i] = \
                    cv2.VideoWriter(filename,
                                    self.fourcc,
                                    self.target_fps,
                                    (frames[i].shape[1], frames[i].shape[0]))
            self.video_writers[i].write(frames[i])


if __name__ == "__main__":
    config_file = "../config/tork_config.json"
    data_folder = "../videos"

    if len(sys.argv) == 2:
        filename = sys.argv[1]

    if len(sys.argv) == 3:
        data_folder = sys.argv[2]

    print(f"Config file: {config_file}")
    print(f"Data folder: {data_folder}")

    with open(config_file) as fp:
        config = json.load(fp)

    config["data_folder"] = data_folder

    gui = MultiStreamVideoGUI(config)
    gui.run()
