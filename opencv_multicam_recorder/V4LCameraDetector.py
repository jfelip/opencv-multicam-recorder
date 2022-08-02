import v4l2ctl


class V4L2CameraDetector:
    def __init__(self, ncameras=50):
        for id in range(ncameras):
            try:
                dev = v4l2ctl.V4l2Device(id)
            except FileNotFoundError as e:
                continue

            # Filter out devices w/o video capturing capabilities
            if v4l2ctl.V4l2Capabilities.VIDEO_CAPTURE not in dev.capabilities:
                continue

            print(f"Device({id}): {dev.name}")
            print(f"Formats:")
            for f in dev.formats:
                print(f"|--> {f}")
                for sz in f.sizes():
                    intervals = list()
                    for interval in sz.intervals():
                        intervals.append(interval.interval.denominator)
                    print(f"   |--> {sz.width}x{sz.height} @ {intervals}")


if __name__ == "__main__":
    V4L2CameraDetector()
