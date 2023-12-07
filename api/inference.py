import numpy as np
from PIL import Image
import imageio

from demo.animate import MagicAnimate


def read_video(video_path):
    reader = imageio.get_reader(video_path)
    # You might want to process the video frames here
    return video_path


def read_image(image_path, size=512):
    image = Image.open(image_path)
    return np.array(image.resize((size, size)))


# Instantiate the MagicAnimate object once
animator = MagicAnimate()

def generate_vid(reference_image, motion_sequence, seed, steps, guidance_scale):
    global animator  # Use the global instance

    animation = animator(reference_image, motion_sequence, seed, steps, guidance_scale)

    # Assuming the output is in a format that can be saved with imageio
    # imageio.mimsave('output_animation.mp4', animation)
    return animation


# Set your values here
reference_image_path = (
    "/home/bilal/magic-animate/inputs/applications/source_image/multi1_source.png"
)
motion_sequence_path = (
    "/home/bilal/magic-animate/inputs/applications/driving/densepose/multi_dancing.mp4"
)
seed = 1
steps = 25
guidance_scale = 7.5

if __name__ == "__main__":
    # Read the image and video
    reference_image = read_image(reference_image_path)
    motion_sequence = read_video(motion_sequence_path)

    # Run the main function
    generate_vid(reference_image, motion_sequence, seed, steps, guidance_scale)
