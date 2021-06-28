# opencv-multicam-recorder

Simple multicamera recorder based on OpenCV. Produces separated 
videos for each stream and a mosaic version of the video. The interactive
GUI lets you configure the orientation of the streams, record, pause and
playback.

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

Camera ids and devices can be checked with v4l2-ctl
```shell
v4l2-ctl --list-devices
```
