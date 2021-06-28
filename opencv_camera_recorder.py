import time
from datetime import datetime
import cv2
import numpy as np
import threading


# TODO: Display timer
# TODO: Display FPS
# TODO: Control record and playback speed relation to real time
# TODO: Timestamp frames
# TODO: Timestamp saved clips
# TODO: Autodetect devices
# TODO: Profiling metrics


def cat_horiz(image1, image2, img_sep=5, scale=False):
    if scale:
        high = np.maximum(image1.shape[0], image2.shape[0])
        low = np.minimum(image1.shape[0], image2.shape[0])
        scale_fac = float(high) / float(low)
        if image1.shape[0] < image2.shape[0]:
            im_size = (int(image1.shape[1] * scale_fac), int(image1.shape[0] * scale_fac))
            image1 = cv2.resize(image1, dsize=im_size)
        elif image1.shape[0] > image2.shape[0]:
            im_size = (int(image2.shape[1] * scale_fac), int(image2.shape[0] * scale_fac))
            image2 = cv2.resize(image2, dsize=im_size)

    cat_shape = [np.maximum(image1.shape[0], image2.shape[0]),
                 image1.shape[1] + image2.shape[1] + img_sep,
                 image2.shape[2]]

    frames = np.zeros(cat_shape, dtype=image1.dtype)
    frames[0:image1.shape[0], 0:image1.shape[1]] = image1
    frames[0:image2.shape[0], image1.shape[1] + img_sep:] = image2
    return frames


def cat_vert(image1, image2, img_sep=5, scale=False):
    if scale:
        high = np.maximum(image1.shape[1], image2.shape[1])
        low = np.minimum(image1.shape[1], image2.shape[1])
        scale_fac = float(high) / float(low)
        if image1.shape[0] < image2.shape[0]:
            im_size = (int(image1.shape[1] * scale_fac), int(image1.shape[0] * scale_fac))
            image1 = cv2.resize(image1, dsize=im_size)
        elif image1.shape[0] > image2.shape[0]:
            im_size = (int(image2.shape[1] * scale_fac), int(image2.shape[0] * scale_fac))
            image2 = cv2.resize(image2, dsize=im_size)

    cat_shape = [image1.shape[0] + image2.shape[0] + img_sep,
                 np.maximum(image1.shape[1], image2.shape[1]),
                 image2.shape[2]]

    frames = np.zeros(cat_shape, dtype=image1.dtype)
    frames[0:image1.shape[0], 0:image1.shape[1]] = image1
    frames[image1.shape[0] + img_sep:, 0:image2.shape[1]] = image2
    return frames


def put_text_multiline(img, text, pos=(20, 20), font=cv2.FONT_HERSHEY_SIMPLEX,
                       scale=0.5, color=(255, 128, 255), thickness=2, line_advance=20):
    posl = [pos[0], pos[1]]
    for l in text.split("\n"):
        img = cv2.putText(img, l, (posl[0], posl[1]), font, scale, color, thickness)
        posl[1] += line_advance
    return img


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
    display_help_key = 104  # h        (toggle this message)

    def __init__(self, config):
        self.caps = MultiStreamVideoCapturer(config)
        self.caps.start()
        self.writers = MultiStreamVideoWriter(config["fps"])
        self.win_name = config["win_name"]
        self.img_sep = config["img_sep"]
        self.scale_mosaic = config["scale_mosaic"]
        self.target_fps = config["fps"]

    def run(self):
        # Init recording variables
        state = self.STATE_DISPLAY
        frames = [[] for i in range(self.caps.nstreams + 1)]
        nframe = 0
        help_on = False
        selected_stream = -1
        times = dict()

        # Open CV Window
        cv2.namedWindow(self.win_name)

        key = 0
        while key != self.exit_key:
            t_start = time.time()

            # If displaying or recording capture frames
            if state == self.STATE_DISPLAY or state == self.STATE_RECORD:
                # Capture a frame from each stream
                cap_t = time.time()
                cap_frames = self.caps.read()
                if cap_frames[0] is None:
                    continue
                times["cap"] = time.time() - cap_t

                # Highlight selected stream
                if state == self.STATE_DISPLAY:
                    if selected_stream != -1:
                        cap_frames[selected_stream] = cv2.rectangle(cap_frames[selected_stream], (0, 0),
                                                                    (cap_frames[selected_stream].shape[1],
                                                                     cap_frames[selected_stream].shape[0]),
                                                                    color=(0, 0, 255), thickness=5)
                show_frame = np.copy(cap_frames[0])

                # Compose the mosaic to show if there are more than one streams active
                if len(cap_frames) > 1:
                    for f in cap_frames[1:]:
                        if show_frame.shape[0] + f.shape[0] > show_frame.shape[1] + f.shape[1]:
                            show_frame = cat_horiz(show_frame, f, self.img_sep, self.scale_mosaic)
                        else:
                            show_frame = cat_vert(show_frame, f, self.img_sep, self.scale_mosaic)
                cap_frames.append(show_frame)

            if state == self.STATE_RECORD:
                self.writers.write(cap_frames)
                for i in range(len(cap_frames)):
                    frames[i].append(np.copy(cap_frames[i]))
                nframe += 1

            elif state == self.STATE_REPLAY:
                if nframe >= len(frames[-1]):
                    nframe = 0
                show_frame = np.copy(frames[-1][nframe])
                nframe += 1

            elif state == self.STATE_REPLAY_PAUSED:
                if nframe >= len(frames[-1]):
                    nframe = 0
                show_frame = np.copy(frames[-1][nframe])

            if help_on:
                show_frame = put_text_multiline(show_frame, self.help_text, (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (255, 128, 255), 2)

            ntotal_frames = 0
            if len(frames[-1]) > 0:
                ntotal_frames = len(frames[-1])
            state_text, state_color = self.get_state_text_color(state, self.writers.sequence_n,
                                                                self.target_fps, nframe, ntotal_frames)

            show_frame = put_text_multiline(show_frame, state_text, (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2)
            cv2.imshow(self.win_name, show_frame)

            if key == self.record_key:
                if state == self.STATE_REPLAY_PAUSED:
                    state = self.STATE_RECORD_PAUSED
                elif state == self.STATE_DISPLAY or state == self.STATE_REPLAY:
                    state = self.STATE_RECORD

            if key == self.replay_key:
                if (state == self.STATE_DISPLAY or state == self.STATE_RECORD) and len(frames) > 0:
                    state = self.STATE_REPLAY

                elif state == self.STATE_RECORD_PAUSED:
                    state = self.STATE_REPLAY_PAUSED

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
                frames = [[] for i in range(self.caps.nstreams + 1)]
                nframe = 0

            if key == self.erase_key:
                self.writers.reset(overwrite=True)
                state = self.STATE_DISPLAY
                frames = [[] for i in range(self.caps.nstreams + 1)]
                nframe = 0

            if key == self.rotate_stream_key and selected_stream != -1:
                self.caps.rotate(selected_stream)

            if key == self.select_stream_key:
                selected_stream += 1
                if selected_stream >= len(frames) - 1:
                    selected_stream = -1

            if key == self.display_help_key:
                help_on = not help_on

            if key == self.prev_frame_key and state == self.STATE_REPLAY_PAUSED:
                nframe -= 1
                nframe = max(0, nframe)

            if key == self.next_frame_key and state == self.STATE_REPLAY_PAUSED:
                nframe += 1
                nframe = min(len(frames[-1]) - 1, nframe)

            if key != -1:
                print(f"Key press: {key}")

            t_elapsed_ms = int((time.time() - t_start) * 1000)
            key = cv2.waitKey(int((1/self.target_fps) * 1000) - t_elapsed_ms)

        self.writers.reset()
        self.caps.release()

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
        self.target_fps = config["fps"]
        for stream_cfg in config["streams"]:
            cap = cv2.VideoCapture(stream_cfg["id"], cv2.CAP_V4L2)
            cap.set(3, stream_cfg["width"])
            cap.set(4, stream_cfg["height"])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, stream_cfg["buffer_size"])
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # TODO: Check video is properly opened
            if cap.isOpened():
                self.video_caps.append(cap)
                print(f"Opened cam: {stream_cfg['id']}. Backend: {cap.getBackendName()}")
            else:
                print(f"ERROR: Unable to open camera {stream_cfg['id']} with "
                      f"{stream_cfg['width']}w x {stream_cfg['height']}h@{config['fps']}fps")

        self.nstreams = len(self.video_caps)
        self.streams_rotation = [-1] * self.nstreams
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
                ret, frame = cap.retrieve()
                assert ret
                # Rotate if necessary
                if self.streams_rotation[i] != -1:
                    frame = cv2.rotate(frame, self.streams_rotation[i])
                cap_frames.append(frame)
            self.mutex.acquire()
            self.frames = cap_frames
            self.mutex.release()
            time.sleep(0.001)

    def release(self):
        for cap in self.video_caps:
            cap.release()
        self.running = False


class MultiStreamVideoWriter:
    def __init__(self, fps):
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writers = None
        self.sequence_n = 0
        self.target_fps = fps

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
                self.video_writers[i] = cv2.VideoWriter(f"{date_str}_seq_{self.sequence_n}_cam_{i}.mp4", self.fourcc,
                                                        self.target_fps, (frames[i].shape[1], frames[i].shape[0]))
            self.video_writers[i].write(frames[i])


if __name__ == "__main__":
    cameras = [{"name": "c1", "id": "/dev/video0", "width": 640, "height": 360, "buffer_size": 2},
               {"name": "c2", "id": "/dev/video2", "width": 800, "height": 448, "buffer_size": 2}]

    config = {"win_name": "Multi Camera Capturer :: javier.felip.leon@gmail.com",
              "img_sep": 5, "streams": cameras, "scale_mosaic": False, "fps": 30}

    gui = MultiStreamVideoGUI(config)
    gui.run()
