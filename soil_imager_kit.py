# SPDX-FileCopyrightText: 2024 Holger Schaefer
# SPDX-License-Identifier: MIT

"""
`soil_imager_kit`
===================================
The Soil imager kit contains functions to operate a 
low-cost high resolution imaging device for hyphae in soil.
The device moves a digital microscope camera (DMC)
within an observable volume of 70 mm (horizontal) \u00d7 
210 mm (vertical) \u00d7 1.5 mm (focus depth).

* Author: Holger Schaefer

Implementation Notes
-------------------

Dependencies:
* OpenCV <https://pypi.org/project/opencv-python/>
* Adafruit Blinka <https://github.com/adafruit/Adafruit_Blinka>
* Adafruit CircuitPython MotorKit <https://github.com/adafruit/
                                      Adafruit_CircuitPython_MotorKit>
* pandas <https://pypi.org/project/pandas/>
"""

__version__ = "0.1.0"

import sys
import os
import time
import math
import tty
import termios
import multiprocessing
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple
from typing import Union
import cv2
import pandas as pd
import board
import busio
from adafruit_motorkit import MotorKit
from adafruit_motor import stepper


@dataclass(frozen = True)
class ImgConf:
    """Configuration of the image dimensions and overlaps.

    Attributes:
        DIM (tuple): Width, height, and depth of a single DMC image (in mm).
        OLAP_MIN (tuple): Minimum overlap between neighboring DMC images
            along the X, Z, and F axis (in mm).   
    """
    DIM: tuple = (0.83, 0.62, 0.025)
    OLAP_MIN: tuple = (0.10, 0.05, 0.0)


@dataclass(frozen = True)
class MotConf:
    """Configuration of the stepper motors of the X, Z, and F axis.

    Attributes:
        AXIS_ID (tuple): Axis IDs.
        NUM_MOTORS (tuple): Number of stepper motors per axis.
        DIST_STEP (tuple): Distance traveled per full step (in mm).
        MICROSTEPS (tuple): Number of microsteps per full step.
        DIR_POS (tuple): Direction parameter in the CircuitPython MotorKit 
            resulting in a positive (forward) move of the DMC.
        DIR_NEG (tuple): Direction parameter in the CircuitPython MotorKit 
            resulting in a negative (backward) move of the DMC.
        DECIMALS (tuple): Number of decimals the DMC position is shown with.
        TIME_INT (tuple): Time interval (in seconds) between microsteps.
        POS_MIN (tuple): Minimum position of the observable volume (in mm).
        POS_MAX (tuple): Maximum position of the observable volume (in mm).
    
    Note:
        A positive (forward) move of the DMC is either a move to the right 
        on the soil profile (X axis), down on the soil profile (Z axis), 
        or towards the soil profile (F axis).
    """

    AXIS_ID: tuple = (0, 1, 2)
    NUM_MOTORS: tuple = (1, 2, 1)
    DIST_STEP: tuple = (0.08, 0.04, 0.00125)
    MICROSTEPS: tuple = (8, 4, 4)
    DIR_POS: tuple = (stepper.FORWARD, stepper.BACKWARD, stepper.BACKWARD)
    DIR_NEG: tuple = (stepper.BACKWARD, stepper.FORWARD, stepper.FORWARD)
    DECIMALS: tuple = (3, 3, 7)
    TIME_INT: tuple = (0.001, 0.001, 0.005)
    POS_MIN: tuple = (0, 0, -0.5)
    POS_MAX: tuple = (70.0, 210.0, 1.5)


@dataclass(frozen = True)
class AuFocConf:
    """Configuration of the DMC's autofocus.

    Attributes:
        TEST_INT (float): Interval size at which the image sharpness 
            is measured and compared (in mm).
        SHARP_DEV (float): Maximum deviation in the image sharpness allowed 
            for a second measurement at the same XZF position (in %).
        TRIES (float): Number of times the autofocus procedure is attempted.
    """

    TEST_INT: float = 0.0125
    SHARP_DEV: float = 20
    TRIES: int = 3


class PositionError(Exception):
    """Error raised when the DMC is moved beyond the 
    observable volume of 70 mm (horizontal) \u00d7 
    210 mm (vertical) \u00d7 1.5 mm (focus depth). """



class CameraError(Exception):
    """Error raised when the DMC cannot be turned on."""
    pass


class AutofocusError(Exception):
    """Error raised when the autofocus failed."""
    pass


def _live_view(cap: cv2.VideoCapture,
               saving_event: cv2.VideoCapture,
               sharpness_event: multiprocessing.Event,
               end_event: multiprocessing.Event,
               name_queue: multiprocessing.Queue
               ) -> None:
    """Live view process for the DMC.
        
        Args:
            cap (cv2.VideoCapture): Video capture object of the DMC.
            saving_event (cv2.VideoCapture): Multiprocess event object
                for image saving.
            sharpness_event (multiprocessing.Event): Multiprocessing event 
                object for image sharpness measurement.
            end_event (multiprocessing.Event): Multiprocessing event object
                for finishing the live view process.
            name_queue (multiprocessing.Queue): Multiprocessing queue object
                for image file names.
        
        Note:
            This process had to be outside of the Imager class to work.
        """

    while True:
        time.sleep(0.01)
        # Frame capturing
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        cv2.imshow("Live View", frame)
        # Image saving event
        if saving_event.is_set():
            saving_event.clear()
            nameforimg = name_queue.get()
            cv2.imwrite(nameforimg, frame)
            print("Image saved as:", nameforimg)
        # Sharpness measurement event
        if sharpness_event.is_set():
            sharpness_event.clear()
            print("Image sharpness is:",
                  round(cv2.Laplacian(frame, cv2.CV_64F).var(), 2))
        # Live view termination event
        if end_event.is_set():
            cap.release()
            cv2.destroyWindow("Live View")
            break
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


class Imager():
    """Class representing the soil imager."""

    def __init__(self) -> None:
        """Initialize imager object."""

        self.session_name = "{}_{}".format(
            "session", datetime.now().strftime("%Y%m%d%H%M%S"))
        self.xzf_pos = [0.0, 0.0, 0.0]
        self.cycle_counter = 0
        self.manual_img_counter = 0
        self.log_dict = {}
        self.terminal_raw_mode = False
        self.dmc_running = False
        self.error = None
        self.error_count = 0
        self._setup_motors()
        self._calc_pos_dist()

    # Functions for user interaction
    def move_dmc(self,
             pos_x: Union[float, int, None] = None,
             pos_z: Union[float, int, None] = None,
             pos_f: Union[float, int, None] = None) -> None:
        """Move the DMC to a specified absolute position on either axis.
        
        Args:
            pos_x (:obj:`float` | :obj:`int`, optional): Horizontal position 
                along the X axis (in mm). Valid range is 0 - 70. 
                Defaults to None.
            pos_z (:obj:`float` | :obj:`int`, optional): Vertical position 
                along the Z axis (in mm). Valid range is 0 - 210. 
                Default is None.
            pos_f (:obj:`float` | :obj:`int`, optional): Focus depth position 
                along the F axis (in mm). Valid range is -0.5 - 1.5. 
                Default is None.
        
        Note: 
            The imaging device has to be turned on with the DMC in the 
            top-left corner for the position ranges to be valid. 
            Failing to do so may cause damage to the device.  
            An error will be raised when the DMC is moved beyond the 
            observable volume of 70 mm (horizontal) \u00d7 210 mm 
            (vertical) \u00d7 1.5 mm (focus depth). 

        Example: 
            >>> imager1.move_dmc(pos_x = 1, pos_z = 0.5)
        """

        pos = [pos_x, pos_z, pos_f]
        for axis in range(3):
            if pos[axis] is not None:
                pos[axis] = self._check_pos(axis, pos[axis])
                self._axis_move(axis, pos[axis] - self.xzf_pos[axis])

    def set_dmc(self, img_size: int, img_type: str = "jpg") -> None:
        """Set the image size and type. 

        Args:
            img_size (int): Image size in pixels. Accepted values are 
                640 (640 \u00d7 480 pixels), 1280 (1280 \u00d7 960 pixels), and
                1600 (1600 \u00d7 1200 pixels).
            img_type (:obj:`float`, optional): File type images are stored in.
                Accepted values are "jpg", "png", "tif", etc. 
                Defaults to "jpg".
        
        Note: 
            Any file type accepted by the function *imwrite()* of the OpenCV
            module may be used for image storage. For details, see 
            [here](https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html).
            Regardless of the file type for image storage, images are taken in
            the MJPEG fromat.

        Example: 
            >>> imager1.set_dmc(img_size = 1280, img_type = "png")
        """

        print("------------------------")
        print("Setting up the DMC")
        if self.dmc_running:
            if img_size == self.img_size:
                self.img_type = img_type
                return
            self.cap.release()
        self._dmc_on()
        self._change_dmc_res(img_size)
        self.img_size = img_size
        self.img_type = img_type

    def adjust_focus(self,
                   dist_x: Union[float, int] = 1,
                   dist_z: Union[float, int] = 1,
                   dist_f: Union[float, int] = 0.0125) -> None:
        """Adjust the DMC's focus depth to the observation box surface
        with keyboard controls during a live view of the soil profile.
        To do this, move back and forth along the F axis while checking
        the image sharpness. Set the F axis position to zero when the image 
        sharpness is at its maximum.
        
        Args:
            dist_x (:obj:`float` | :obj:`int`, optional): Distance traveled 
                along the X axis with one keystroke (in mm). Defaults to 1.
            dist_z (:obj:`float` | :obj:`int`, optional): Distance traveled 
                along the Z axis with one keystroke (in mm). Defaults to 1.
            dist_f (:obj:`float` | :obj:`int`, optional): Distance traveled 
                along the F axis with one keystroke (in mm). 
                Defaults to 0.0125.

        Note:
            The image sharpness is measured as the variation in the 
            Laplacian operator. See e.g.
            [Pech-Pacheco et al.](https://doi.org/10.1109/ICPR.2000.903548)
            (2000). The adjust_focus function may also be used for manual 
            soil profile imaging. An error will be raised when the
            DMC is moved beyond the observable volume of 
            70 mm (horizontal) \u00d7 210 mm (vertical) \u00d7 1.5 mm 
            (focus depth).
            
        Example:
            >>> imager1.adjust_focus(dist_f = 0.0125)
        """

        print("------------------------")
        print("Starting the manual focus adjustment")
        dist = (float(dist_x), float(dist_z), float(dist_f))
        # Initiate live view with keyboard controls
        self._init_live_view()
        self._show_key_guide(dist)
        self._switch_terminal_mode()
        self._read_key_input(dist)
        # Terminate live view
        self._switch_terminal_mode()
        self.live_view_proc.terminate()
        # DMC has to be restarted after being used in the live view process
        self.cap.release()
        self.dmc_running = False
        self.set_dmc(self.img_size, self.img_type)

    def image_soil(self,
                   width: Union[float, int],
                   height: Union[float, int],
                   depth: Union[float, int]) -> None:
        """Conduct automatic imaging of the soil profile. Abort with ctrl + c.
        
        Args:
            width (:obj:`float` | :obj:`int`): Width of imaged volume (in mm). 
                Valid range is 0 - 70.
            height (:obj:`float` | :obj:`int`): Height of imaged volume 
                (in mm). Valid range is 0 - 210.
            depth (:obj:`float` | :obj:`int`): Depth of imaged volume (in mm). 
                Valid range is 0 - 1.5.

        Raises:
            KeyboardInterrupt: If imaging is interrupted with ctrl + c.

        Note:
            If the automatic imaging is interrupted, the DMC will be homed
            and the image session ended. An error will be raised when the
            DMC is moved beyond the observable volume of 
            70 mm (horizontal) \u00d7 210 mm (vertical) \u00d7 1.5 mm 
            (focus depth).

        Example:
            >>> imager1.image_soil(width = 10, height = 10, depth = 0.1)
        """

        print("------------------------")
        print("Starting the automatic soil imaging")
        vol = [float(width), float(height), float(depth)]
        print("Volume:", vol[0], "mm \u00d7",
              vol[1], "mm \u00d7", vol[2], "mm")
        self.move_dmc(pos_f = 0)
        vol = self._check_vol(vol)
        self.cycle_counter += 1
        self.start_time_cycle = time.time()
        self.pos_time = time.time()
        self._calc_n_imgs(vol)
        self._dmc_wiggle()
        try:
            self._imaging_cycle()
        except KeyboardInterrupt as e:
            print(e)
            self.end_session()
        if self.n_imgs[2] > 1:
            self._auto_focus()
        self._save_cycle_info(vol)
        time.sleep(10)

    def end_session(self) -> None:
        """End the imaging session and home the DMC.
        
        Note: 
            This homes the DMC, i.e. returns it to position 
            zero on each axes.
        
        Example: 
            >>> imager1.end_session()
        """

        print("------------------------")
        print("Ending the imaging session")
        # End manual focusing
        if self.terminal_raw_mode:
            self._switch_terminal_mode()
        if "self.live_view_proc" in locals():
            if self.live_view_proc.is_alive():
                self.live_view_proc.terminate()
                time.sleep(10)
        # Home and turn off the DMC
        if self.xzf_pos != [0.0, 0.0, 0.0]:
            self.move_dmc(0, 0, 0)
        if self.dmc_running:
            self.cap.release()
        sys.exit()

    # Functions for motor and DMC control
    def _setup_motors(self) -> None:
        """Set up the stepper motors using the CircuitPython MotorKit."""

        mboard1_x = MotorKit(i2c = busio.I2C(board.SCL, board.SDA),
                             address= 0x61,
                             steppers_microsteps = MotConf.MICROSTEPS[0])
        mboard1_f = MotorKit(i2c = busio.I2C(board.SCL, board.SDA),
                            address= 0x61,
                            steppers_microsteps = MotConf.MICROSTEPS[2])
        mboard2 = MotorKit(i2c = busio.I2C(board.SCL, board.SDA),
                            steppers_microsteps = MotConf.MICROSTEPS[1])
        self.steppers = ([mboard1_x.stepper1],
                         [mboard2.stepper2, mboard2.stepper1],
                         [mboard1_f.stepper2])

    def _axis_move(self, axis: int,
                   dist: Union[float, int]) -> None:
        """Move along one axis using the Adafruit CircuitPython MotorKit.
        
        Args:
            axis (int): Axis to travel along. Accepted values are 0 (x axis), 
                1 (z axis), and 2 (f axis).
            dist (:obj:`float` | :obj:`int`): Distance to travel (in mm).
        """

        if dist == 0:
            return
        n_steps = int(abs(dist/MotConf.DIST_STEP[axis]))
        for step in range(n_steps):
            for mstep in range(MotConf.MICROSTEPS[axis]):
                if dist >= 0:
                    for motor in self.steppers[axis]:
                        motor.onestep(direction = MotConf.DIR_POS[axis],
                                    style = stepper.MICROSTEP)
                        time.sleep(MotConf.TIME_INT[axis])
                else:
                    for motor in self.steppers[axis]:
                        motor.onestep(direction = MotConf.DIR_NEG[axis],
                                        style = stepper.MICROSTEP)
                        time.sleep(MotConf.TIME_INT[axis])
            self.xzf_pos[axis] = round(self.xzf_pos[axis]
                                    + (dist/abs(dist))*MotConf.DIST_STEP[axis],
                                    MotConf.DECIMALS[axis])
        for motor in self.steppers[axis]:
            motor.release()# Release to prevent overheating
        self._print_pos()

    def _print_pos(self) -> None:
        """Display current absolute position of the DMC."""

        print("X", self.xzf_pos[0], "mm, Z", self.xzf_pos[1],
              "mm, F", self.xzf_pos[2], "mm")        

    def _dmc_on(self) -> None:
        """Turn on the DMC.
        
        Raises:
            CameraError: When the DMC cannot be turned on.
        """

        self.cap = cv2.VideoCapture(0)
        try:
            if self.cap.isOpened():
                self.dmc_running = True
            else:
                raise CameraError
        except CameraError as e:
            print("Camera cannot be turned on.")
            print(e)
            self.end_session()
        time.sleep(5)

    def _change_dmc_res(self, img_size: int) -> None:
        """Change the imaging resolution of the DMC.
        
        Args:
            img_size (int): Image size in pixels.
        """

        if img_size == 640:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        elif img_size == 1280:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        elif img_size == 1600:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
        print("Image size set to:")
        print(int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "\u00d7",
              int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "pixels")

    def _img_cap(self) -> cv2.Mat:
        """Display the image currently captured by the DMC. 
        
        Returns:
            cv2.Mat: Image captured from the DMC
        
        Note: 
            Since the DMC has a video buffer, eight frames have to be read.
        """

        for i in range(8):
            ret, frame = self.cap.read()
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            cv2.imshow("Captured Frame",frame)
            cv2.waitKey(1)
        return frame

    def _calc_pos_dist(self):
        """Calculate distance to travel along each axis 
        between neighboring images during automatic imaging (in mm)."""

        self.pos_dist = []
        for ax in range(3):
            pos_dist = ((ImgConf.DIM[ax] - ImgConf.OLAP_MIN[ax])
                        //MotConf.DIST_STEP[ax]*MotConf.DIST_STEP[ax])
            self.pos_dist.append(pos_dist)

    def _meas_sharp(self, img: cv2.Mat) -> float:
        """Estimate image sharpness based on the variance of the 
        Laplacian operator.
        
        Args:
            img (cv2.Mat): Input image.

        Returns:
            float: Variance of the Laplacian operator.
        """

        lapla = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F)
        lapla_var = round(lapla.var(), 2)
        return lapla_var

    def _check_pos(self, axis: int, pos: Union[float, int]) -> None:
        """Check, if the position is within the observable volume of 
        70 mm (horizontal) \u00d7 210 mm (vertical) \u00d7 1.5 mm 
        (focus depth).
        
        Args:
            axis (int): Axis to travel along. Accepted values are 0 (x axis), 
                1 (z axis), and 2 (f axis).
            pos (:obj:`float` | :obj:`int`): Position along axis (in mm).
        
        Raises:
            PositionError: When the DMC is moved beyond the observable volume.
        """

        axis_names = ["X", "Z", "F"]
        try:
            if pos < MotConf.POS_MIN[axis]:
                pos_checked = MotConf.POS_MIN[axis]
                error_axis = axis_names[axis]
                raise PositionError
            if pos > MotConf.POS_MAX[axis]:
                pos_checked = MotConf.POS_MAX[axis]
                error_axis = axis_names[axis]
                raise PositionError
            else:
                return pos
        except PositionError:
            print("Limiting", error_axis, "to observable volume.")
            return pos_checked

    # Functions for manual focus
    def _init_live_view(self) -> None:
        """Initialize the DMC's live view for the manual focusing."""

        self.saving_event = multiprocessing.Event()
        self.sharpness_event = multiprocessing.Event()
        self.end_event = multiprocessing.Event()
        self.name_queue = multiprocessing.Queue(maxsize = 1)
        self.live_view_proc = multiprocessing.Process(
            target = _live_view,
            args = (self.cap, self.saving_event,
                    self.sharpness_event,
                    self.end_event,
                    self.name_queue))
        if not self.live_view_proc.is_alive():
            self.live_view_proc.start()

    def _show_key_guide(self, dist: Tuple[float, float, float]) -> None:
        """Show the key guide for the manual focusing.
                
        Args:
            dist (tuple): Distances to travel along the X, Z, and F axis
            with one keystroke (in mm).
        """

        note = """
Adjust the DMC's focus depth to the observation box surface 
manually. To do this, move back and forth along the F axis (keys e, r)
while checking the image sharpness (key g, or visually). Set the F axis 
position to zero (key m) when the image sharpness is at its maximum.
The DMC may be moved horizontally and vertically to find a better 
XZ location for the focus adjustment. Do not move the DMC beyond 
the observable volume of 70 mm (horizontal) \u00d7 210 mm 
(vertical) \u00d7 1.5 mm (focus depth).
              """
        print(note)
        print("------------------------")
        print("Control keys:")
        print("a,d: Move", dist[0], "mm left, right on the soil profile")
        print("w,s: Move", dist[1], "mm up, down on the soil profile")
        print("e,r: Move", dist[2], "mm away from or towards the soil profile")
        print("f: Save image")
        print("g: Assess image sharpness (Laplacian operator)")
        print("b: Set current X axis position as zero")
        print("n: Set current Z axis position as zero")
        print("m: Set current F axis position as zero")
        print("l: Show key control guide")
        print("q: End manual focusing and continue with imaging session")
        print("x: Abort imaging session")
        print("------------------------")

    def _switch_terminal_mode(self) -> None:
        """Switch the terminal input mode between raw mode and cooked mode.
        
        Note: 
            For terminal inputs to be processed instantly during the live 
            view, the terminal mode is changed to raw mode.
        """

        if self.terminal_raw_mode:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.filedescript)
            self.terminal_raw_mode = False
        else:
            self.filedescript = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin)
            self.terminal_raw_mode = True

    def _read_key_input(self, dist: Tuple[float, float, float]) -> None:
        """Read key inputs by the user during the live view.
        
        Args:
            dist (tuple): Distances to travel along the X, Z, and F axis
            with one keystroke (in mm).
        """

        key_input = 0
        while 1:
            key_input = sys.stdin.read(1)[0]
            if key_input == "a":
                print(dist[0], "mm left")
                self._axis_move(0, -dist[0])
            if key_input == "d":
                print(dist[0], "mm right")
                self._axis_move(0, dist[0])
            if key_input == "w":
                print(dist[1], "mm up")
                self._axis_move(1, -dist[1])
            if key_input == "s":
                print(dist[1], "mm down")
                self._axis_move(1, dist[1])
            if key_input == "e":
                print(dist[2], "mm backward")
                self._axis_move(2, -dist[2])
            if key_input == "r":
                print(dist[2], "mm forward")
                self._axis_move(2, dist[2])
            if key_input == "f":
                self._save_man_img()
            if key_input == "g":
                self.sharpness_event.set()
            if key_input == "b":
                self.xzf_pos[0] = 0.0
                self._print_pos()
            if key_input == "n":
                self.xzf_pos[1] = 0.0
                self._print_pos()
            if key_input == "m":
                self.xzf_pos[2] = 0.0
                self._print_pos()
            if key_input == "l":
                print("Reprint key control guide")
                self._show_key_guide(dist)
            if key_input == "q":
                print("Ending manual focusing")
                self.end_event.set()
                time.sleep(2)# Needed
                return
            if key_input == "x":
                print("Aborting the imaging session")
                self.end_event.set()
                time.sleep(2)# Needed
                self.end_session()

    def _save_man_img(self) -> None:
        """Save image during live view."""

        self.manual_img_counter += 1
        if self.manual_img_counter == 1:
            self.folder_manu = os.path.join(self.session_name, "manual")
            os.makedirs(self.folder_manu,exist_ok=True)
        file_name = "{}_{}{}{}_{}.{}".format(
            "m"+"{0:0>3}".format(self.manual_img_counter),
            "X"+"{0:0>5}".format(round(self.xzf_pos[0]*1000)),
            "Z"+"{0:0>5}".format(round(self.xzf_pos[1]*1000)),
            "F"+"{0:0>5}".format(round(self.xzf_pos[2]*1000)),
            "t"+datetime.now().strftime("%Y%m%d%H%M%S"),
            self.img_type
            )
        file_path = os.path.join(self.folder_manu, file_name)
        self.name_queue.put(file_path)
        self.saving_event.set()

    # Functions for automatic imaging
    def _check_vol(self,
                   vol: Tuple[float, float, float]
                   ) -> Tuple[float, float, float]:
        """Check, if volume to be imaged is within the observable volume of 
        70 mm (horizontal) \u00d7 210 mm (vertical) \u00d7 1.5 mm 
        (focus depth).
        
        Args:
            vol (tuple): Volume to be imaged (in mm).
        
        Raises:
            PositionError: When the DMC is moved beyond the observable volume.
        """

        dim_names = ["Width", "Height", "Depth"]
        try:
            for axis in range(3):
                if vol[axis] <= 0:
                    error_dim = dim_names[axis]
                    print(error_dim, "not valid.")
                    raise PositionError
                pos_final = self.xzf_pos[axis] + vol[axis]
                pos_checked = self._check_pos(axis, pos_final)
                vol[axis] = vol[axis] - (pos_final - pos_checked)
                if pos_final != pos_checked:
                    error_dim = dim_names[axis]
                    print(error_dim, "reduced to", vol[axis], "mm")
            return vol
        except PositionError:
            self.end_session()

    def _calc_n_imgs(self, vol: Tuple[float, float, float]) -> None:
        """Calculate the number of images to be taken along the X, Z, and 
            F axis.
        
        Args:
            vol (tuple): Width, height, and depth of imaged volume (in mm).
        """

        self.n_imgs = []
        for ax in range(2):
            self.n_imgs.append(math.ceil(vol[ax]/self.pos_dist[ax]))
        # When focus on glass surface, only about half of the DMC's
        # depth of field is outside the observation box
        self.n_imgs.append(math.ceil(
            (vol[2] + self.pos_dist[2]/2)
            /self.pos_dist[2]))

    def _dmc_wiggle(self) -> None:
        """Wiggle the DMC in the beginning of an imaging cycle."""

        self._axis_move(2, -self.pos_dist[2])
        self._axis_move(2, self.pos_dist[2])
        for ax in range(1,-1,-1):
            if self.xzf_pos[ax] >= MotConf.DIST_STEP[ax]:
                self._axis_move(ax, -MotConf.DIST_STEP[ax])
                self._axis_move(ax, MotConf.DIST_STEP[ax])

    def _imaging_cycle(self) -> None:
        """Take images row by row in a snake pattern 
        (right, down, left, down, ...)."""

        start_pos = self.xzf_pos.copy()
        # Image
        for depth in range(self.n_imgs[2]):
            self._create_img_folder()
            self.pos_no = 0
            for row in range(self.n_imgs[1]):
                for num in range(self.n_imgs[0]):
                    self.pos_no += 1
                    self._save_auto_img()
                    self._show_img_info(depth)
                    self._log_img_info()
                    if num < (self.n_imgs[0] - 1):
                        # Move right or left
                        if row % 2 == 0:
                            self._axis_move(0, self.pos_dist[0])
                        else:
                            self._axis_move(0, -self.pos_dist[0])
                # Move down
                if row < (self.n_imgs[1] - 1):
                    self._axis_move(1, self.pos_dist[1])
                    self._save_log_file()
            # Move outward
            if depth < (self.n_imgs[2] - 1):
                self._axis_move(2, self.pos_dist[2])
            # Move back to starting positions
            self.move_dmc(pos_x = start_pos[0])
            self.move_dmc(pos_z = start_pos[1])
        self.move_dmc(pos_f = start_pos[2])

    def _create_img_folder(self):
        """Create an image folder for the current imaging cycle."""

        self.folder_auto = os.path.join(
            self.session_name,"{}{}".format(
                "cyc"+"{0:0>3}".format(self.cycle_counter),
                "foc"+"{0:0>5}".format(round(self.xzf_pos[2]*1000))
                )
            )
        os.makedirs(self.folder_auto, exist_ok = True)

    def _save_auto_img(self) -> None:
        """Save image taken during automatic imaging."""

        frame = self._img_cap()
        self.sharp = self._meas_sharp(frame)
        file_name = "{}_{}.{}".format("tile",
                                      "{0:0>5}".format(self.pos_no),
                                      self.img_type)
        cv2.imwrite(os.path.join(self.folder_auto, file_name), frame)

    def _show_img_info(self, depth: int) -> None:
        """Display data for current imaging position.
        
        Args:
            depth (int): Current value of the focus depth iterator.

        """

        print("Cycle: ", self.cycle_counter,
              " | F position: ", depth + 1, 
              "/", self.n_imgs[2],
              " | XZ position: ", self.pos_no, 
              "/", self.n_imgs[0]*self.n_imgs[1],
              " | Image sharpness: ", self.sharp, 
              " | Time: ", round(time.time() - self.pos_time, 1)," s", 
              sep = "")
        self.pos_time = time.time()

    def _log_img_info(self) -> None:
        "Log data for current imaging position."

        self.log_dict.setdefault("datetime", []).append(
            datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            )
        self.log_dict.setdefault("cycle_no", []).append(self.cycle_counter)
        self.log_dict.setdefault("f_depth", []).append(
            round(self.xzf_pos[2], 5)
            )
        self.log_dict.setdefault("xz_pos_no", []).append(self.pos_no)
        self.log_dict.setdefault("sharp", []).append(self.sharp)

    def _save_log_file(self) -> None:
        """Write the data logged during automatic imaging to a CSV file."""

        log_df = pd.DataFrame.from_dict(self.log_dict)
        file_name = f"log_{self.session_name[8:]}.csv"
        file_path = os.path.join(self.session_name, file_name)
        log_df.to_csv(file_path, sep = ",", index = False)

    def _save_cycle_info(self, vol: Tuple[float, float, float]) -> None:
        """Write the imaging cycle data to a CSV file.
        
        Args:
            vol (tuple): Width, height, and depth of imaged volume (in mm).
        """

        cycle_df = pd.DataFrame(
            {"cycle_no":[self.cycle_counter],
             "time":[round(time.time() - self.start_time_cycle,1)],
             "pos_x":[self.xzf_pos[0]], "pos_z":[self.xzf_pos[1]],
             "vol_w":[vol[0]], "vol_h":[vol[1]], "vol_d":[vol[2]], 
             "n_imgs_x":[self.n_imgs[0]], 
             "n_imgs_z":[self.n_imgs[1]], 
             "n_imgs_f":[self.n_imgs[2]],
             "img_w":[ImgConf.DIM[0]], 
             "img_h":[ImgConf.DIM[1]], 
             "img_d":[ImgConf.DIM[2]],
             "olap_x":[ImgConf.DIM[0] - self.pos_dist[0]],
             "olap_z":[ImgConf.DIM[1] - self.pos_dist[1]], 
             "olap_f":[ImgConf.DIM[2] - self.pos_dist[2]],
             }
            )

        file_name = f"cycles_{self.session_name[8:]}.csv"
        file_path = os.path.join(self.session_name, file_name)
        if self.cycle_counter > 1:
            cycle_df_0 = pd.read_csv(file_path, delimiter= ",")
            cycle_df = pd.concat([cycle_df_0, cycle_df], ignore_index = True)
        cycle_df.to_csv(file_path, sep = ",", index = False)

    def _auto_focus(self) -> None:
        """Adjust the DMC's focus depth automatically. This is 
        to correct for potential (minor) deviations of F axis position 
        zero from the observation box surface. F axis position zero is 
        reset to the focus depth with the highest image sharpness in 
        front or behind the current position.

        Note: 
            The image sharpness is measured as the variation in the 
            Laplacian operator. See e.g.
            [Pech-Pacheco et al.](https://doi.org/10.1109/ICPR.2000.903548)
            (2000). The autofocus procedure is repeated twice, if it 
            fails. After the third failure, the DMC is homed and the 
            imaging session ended.
        """

        print("------------------------")
        print("Autofocusing")
        sharp_list = []
        pos_f_list = []
        dev_max = (self.n_imgs[2] - 1)*self.pos_dist[2]
        intervals = round(dev_max/AuFocConf.TEST_INT)*2 + 1
        # Measure the image sharpness at the defined focus depth intervals
        self.move_dmc(pos_f = -(dev_max + AuFocConf.TEST_INT))
        for interval in range(intervals):
            self._axis_move(2, AuFocConf.TEST_INT)
            time.sleep(0.3)
            pos_f_list.append(self.xzf_pos[2])
            img_sharp = self._meas_sharp(self._img_cap())
            sharp_list.append(img_sharp)
            print("Image sharpness:", img_sharp)
        # Move to the focus depth with the highest image sharpness
        sharp_peak = max(sharp_list)
        pos_max_sharp = pos_f_list[sharp_list.index(sharp_peak)]
        print("Focus depth with maximum image sharpness:", pos_max_sharp, "mm")
        self.move_dmc(pos_f = pos_max_sharp)
        self._comp_sharpness(sharp_peak)

    def _comp_sharpness(self, sharp_prev: float) -> None:
        """Check, if the given image sharpness is similar to the one
        measured currently at the same XZF position.
        
        Args:
            sharp_prev (float): Image sharpness measured before.
        
        Raises:
            AutofocusError: When the image sharpness is not similar.
        """

        try:
            sharp_test = self._meas_sharp(self._img_cap())
            print("Confirming maximum image sharpness at",
                  self.xzf_pos[2], "mm:", sharp_test)
            if (sharp_test <= sharp_prev
                    - sharp_prev*AuFocConf.SHARP_DEV/100
                    or sharp_test >= sharp_prev
                    + sharp_prev*AuFocConf.SHARP_DEV/100):
                self.error_count += 1
                raise AutofocusError
            self.xzf_pos[2] = 0.0
            print("Autofocus successful.")
            self._print_pos()
        except AutofocusError as e:
            print("Autofocus failed.")
            print(e)
            if self.error_count == AuFocConf.TRIES:
                print("Continuing imaging with original focus.")
                return
            print("Retrying...")
            self._auto_focus()
