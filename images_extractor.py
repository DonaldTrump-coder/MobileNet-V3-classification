import imageio
import os
import shutil

video_name = "data\\snow\\snow.mp4"
times =15 # save every n frames

def extractor(video_name, times):
    video_folder = os.path.dirname(video_name)
    video_base_name = os.path.splitext(os.path.basename(video_name))[0]
    output = os.path.join(video_folder, video_base_name)
    if os.path.exists(output):
        shutil.rmtree(output)
    sparse = os.path.join(output, "sparse", "0")
    output = os.path.join(output, "images")
    os.makedirs(output, exist_ok=True)
    os.makedirs(sparse, exist_ok=True)
    reader = imageio.get_reader(video_name)
    for i, frame in enumerate(reader):
        if i % times == 0:
            frame_filename = os.path.join(output, f'frame_{i:04d}.jpg')
            imageio.imwrite(frame_filename, frame)

    print(f'Finishes frames extraction. The frames are saved in {output}.')

if __name__ == "__main__":
    extractor(video_name, times)