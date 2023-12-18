from pydantic import BaseModel
from fastapi import FastAPI
from api.inference import read_video, read_image, generate_vid

pose_mapper = {
    "dancing_1": "/home/bilal/magic-animate/inputs/applications/driving/densepose/demo4.mp4",
    "dancing_2": "/home/bilal/magic-animate/inputs/applications/driving/densepose/dancing2.mp4",
    "jogging": "/home/bilal/magic-animate/inputs/applications/driving/densepose/running2.mp4",
    "running": "/home/bilal/magic-animate/inputs/applications/driving/densepose/running.mp4",
    "multi-dancing": "/home/bilal/magic-animate/inputs/applications/driving/densepose/multi_dancing.mp4",
    "boxing": "/home/bilal/magic-animate/inputs/applications/driving/densepose/boxing.mp4",
    "dancing_3": "/home/bilal/magic-animate/inputs/applications/driving/densepose/tiktok_dance.mp4",
    "punching": "/home/bilal/magic-animate/inputs/applications/driving/densepose/punching.mp4"
}

seed = 1
steps = 25
guidance_scale = 7.5


class AnimateImageRequest(BaseModel):
    image_path: str
    text: str = "running"


app = FastAPI()


@app.post("/video_gen_dance_byte/")
async def animate_image(request: AnimateImageRequest):
    """
    Takes an image path and optional text as input
    and returns video path
    """
    video_path = pose_mapper[request.text]
    image_path = request.image_path

    # Read the image and video
    image = read_image(image_path)
    video = read_video(video_path)

    generated_vid_path = generate_vid(image, video, seed, steps, guidance_scale)

    return generated_vid_path
