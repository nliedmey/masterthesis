from pyk4a import PyK4A, PyK4ARecord, Config, ImageFormat
import cv2
import numpy as np
import helpers


class KinectControlWrapper:

    # Display video streams
    def display_video(self, color=True, depth=True, ir=True):
        k4a = PyK4A()
        k4a.start()
        while 1:
            capture = k4a.get_capture()
            if np.any(capture):
                if color:
                    cv2.imshow("k4a color", capture.color[:, :, :3])
                if depth:
                    cv2.imshow("k4a depth", helpers.colorize(capture.depth, (None, 5000), cv2.COLORMAP_HSV))
                if ir:
                    cv2.imshow("k4a ir", capture.ir)
                key = cv2.waitKey(10)
                if key != -1:
                    cv2.destroyAllWindows()
                    break
        k4a.stop()
        return None

    def record_sequence(self):
        print("Starting device")
        config = Config(color_format=ImageFormat.COLOR_MJPG)
        device = PyK4A(config=config, device_id=0)
        device.start()

        print(f"Open record file")
        record = PyK4ARecord(device=device, config=config, path="storage\output_pyk4a.mkv")
        record.create()
        try:
            print("Recording... Press CTRL-C to stop recording.")
            while True:
                capture = device.get_capture()
                record.write_capture(capture)
        except KeyboardInterrupt:
            print("CTRL-C pressed. Exiting.")

        record.flush()
        record.close()
        print(f"{record.captures_count} frames written.")
        return None
