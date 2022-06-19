# opencv-multicam-recorder
Simple multicamera recorder based on OpenCV. Produces separated
videos for each stream and a mosaic version of the video. The interactive
GUI lets you configure the orientation of the streams, record, pause and
playback.


### Installation
Device detection requires python package v4lctl, the version in pip
is outdated and you need to install at least the version 0.1b1 from
the source repo.
```
git clone git@github.com:jfelip/v4l2ctl.git
cd v4l2ctl
git checkout origin/Release_v0.1b1
pip install .
```


### Usage
Control keys
```text
    esc      (exit program)
    r        (start recording)
    s        (save recorded clip and reset)
    d        (discard current clip and reset)
    p        (enter exit replay mode)
    space    (pause or play the replay video)
    ->       (pause and advance one frame)
    <-       (pause and one frame back)
    tab      (select stream)
    q        (rotate selected stream 90deg)
    h        (toggle this message)
```

Configure the desired cameras to open on the config dictionary and start.
```python
    cameras = [{"name": "c1", "id": "/dev/video0", "width": 640, "height": 360, "buffer_size": 2},
               {"name": "c2", "id": "/dev/video1", "width": 800, "height": 448, "buffer_size": 2}]
```

Use the V4LCameraDetector.py script to detect the cameras that are
available on your system, their supported resolutions and frame rates.
Configure the capturer .json configuration with the desired camera
resolutions and frame rates.

Todo List
 * Autodetect cameras
 * Overlay camera data on each frame
   * Resolution, frame rate, timestamp
 * Add audio recording capabilities
 * Add interactive auto exposure and exposure controls
 * General config file
   * Display types
   * Overlay text size and colors