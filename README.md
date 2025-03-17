#  *Hyphascope*: a low-cost high-resolution imaging device for hyphae in soil

## Description

<img align="right" src="https://github.com/h-schaefer/hyphascope/blob/main/hyphascope.png" width="300" /> 

This repository hosts the *Python* code needed to operate a low-cost high-resolution imaging device for hyphae in soil called *Hyphascope*. Repeated imaging of a soil profile enables researchers to observe and quantify changes in the amount, distribution, and morphology of hyphae.

Using a digital microscope camera (DMC; 600× magnification), the imaging device takes detailed images within an observable volume of 70 (horizontal) × 210 (vertical) × 1.5 mm (focus depth). DMC image sizes of 640 × 480, 1280 × 960, and 1600 × 1200 pixels correspond to imaging resolutions of 1.30 μm px<sup>-1</sup> (19600 dpi), 0.65 μm px<sup>-1</sup> (39200 dpi), and 0.52 μm px<sup>-1</sup> (49000 dpi), respectively.

For more information on the imaging device, see this article published in PLOS ONE:  
[Schaefer H (2025). Assembly and application of a low-cost high-resolution imaging device for hyphae in soil. PLoS ONE 20(1): e0318083.](https://doi.org/10.1371/journal.pone.0318083)

For a detailed description of the assembly and application of the imaging device, see this protocol on protocols.io:  
[Schaefer H (2024). Hyphascope: Do-it-yourself assembly and application of an imaging device for hyphae in soil. protocols.io.](https://dx.doi.org/10.17504/protocols.io.bp2l6xo3zlqe/v1)

[<img align="right" src="https://github.com/h-schaefer/hyphascope/blob/main/certification-mark-JP000021-wide.png" width="200" />](https://certification.oshwa.org/jp000021.html)  

Furthermore, all [design files used to assemble the imaging device](https://doi.org/10.5281/zenodo.10689905) and several [sets of images yielded with the imaging device](https://doi.org/10.5281/zenodo.10730414) are available from the data repository Zenodo.

## Dependencies

The code uses the *Python* base modules *sys*, *os*, *datetime*, *time*, *math*, *tty*, *termios*, *dataclasses*, *typing*, and *multiprocessing* as well as the below third-party packages: 

- [*OpenCV*](https://pypi.org/project/opencv-python/)
- [*Adafruit Blinka*](https://github.com/adafruit/Adafruit_Blinka/)
- [*Adafruit CircuitPython MotorKit*](https://github.com/adafruit/Adafruit_CircuitPython_MotorKit/)
- [*pandas*](https://pypi.org/project/pandas/)

Furthermore, the window manager [*screen*](https://www.gnu.org/software/screen/) is used during soil imaging.

## Installation

After assembling *Hyphascope*, follow the section *Installation of the software* in  [the protocol on protocols.io](https://dx.doi.org/10.17504/protocols.io.bp2l6xo3zlqe/v1) to install all software dependencies on the imaging device. Then, download the file *imaging_session.py* to a host computer that can reach the imaging device via the Secure Shell Protocol (SSH).

## Usage

- Step 1. Open the file *imaging_session.py* in a code editor.

- Step 2. Compose an imaging session from the five functions below: 
    - *Imager.move_dmc()*: Move the DMC to a specified position along three axes. 
    - *Imager.set_dmc()*: Set the image size and type to be stored. 
    - *Imager.adjust_focus()*: Manually adjust the focus of the DMC.
    - *Imager.image_soil()*: Perform the automated imaging of the soil profile within a given volume.
    - *Imager.end_session()*: End the imaging session.

- Step 3. Save and close the file *imaging_session.py*.

- Step 4. Transfer the file to the imaging device and carry out the imaging session following the section *Application of the device* in [the protocol on protocols.io](https://www.protocols.io/).

- Step 5. Transfer the soil profile images to the host computer and process them following the same section.

## Example

```ruby
# Initialize the soil imager
imager1 = soil_imager_kit.Imager()
# Move the DMC to position 1 mm on the X axis and 0.5 mm on the Z axis
imager1.move_dmc(pos_x = 1, pos_z = 0.5)
# Set the image size and type to be stored
imager1.set_dmc(img_size = 640, img_type = "png")
# Adjust the focus depth manually to the observation box surface
imager1.adjust_focus()
# Conduct automated imaging of a 10 × 10 × 0.05 mm volume
imager1.move_dmc(15, 1)
imager1.image_soil(width = 10, height = 10, depth = 0.05)
# Repeat the imaging with a different resolution
imager1.set_dmc(1280, "png")
imager1.image_soil(10, 10, 0.05)
# End the imaging session
imager1.end_session()
```

## Credits

H. Schaefer, Forestry and Forest Products Research Institute (FFPRI), Japan
