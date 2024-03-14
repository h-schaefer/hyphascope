import soil_imager_kit

if __name__ == "__main__":
    # Initialize the soil imager
    imager1 = soil_imager_kit.Imager()
    # Move the DMC to position 1 mm on the X axis and 0.5 mm on the Z axis
    imager1.move_dmc(pos_x = 1, pos_z = 0.5)
    # Set the image size and type to be stored
    imager1.set_dmc(img_size = 640, img_type = "png")
    # Adjust the focus depth manually to the observation box surface
    imager1.adjust_focus()
    # Conduct automatic imaging of a 10 × 10 × 0.05 mm volume
    imager1.move_dmc(15, 1)
    imager1.image_soil(width = 10, height = 10, depth = 0.05)
    # Repeat the imaging with a different resolution
    imager1.set_dmc(1280, "png")
    imager1.image_soil(10, 10, 0.05)
    # End the imaging session
    imager1.end_session()



