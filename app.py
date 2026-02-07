import streamlit as st
import cv2
import os
import tempfile
import re
from fpdf import FPDF
from PIL import Image
import yt_dlp
from skimage.metrics import structural_similarity as ssim

# ----------------------------------
# STREAMLIT CONFIG
# ----------------------------------
st.set_page_config(
    page_title="YouTube Video → PDF",
    layout="centered"
)

st.title("📹➡️📄 YouTube Video to PDF")
st.write("Extract unique frames from a YouTube video or playlist and export as a PDF.")

# ----------------------------------
# HELPER FUNCTIONS
# ----------------------------------

def download_video(url, filename):
    ydl_opts = {
        "outtmpl": filename,
        "format": "best"
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return filename


def get_video_id(url):
    patterns = [
        r"shorts\/(\w+)",
        r"youtu\.be\/([\w\-_]+)",
        r"v=([\w\-_]+)",
        r"live\/(\w+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_playlist_videos(playlist_url):
    ydl_opts = {
        "ignoreerrors": True,
        "extract_flat": True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)
        return [entry["url"] for entry in info["entries"] if entry]


def extract_unique_frames(video_file, output_folder, n=3, ssim_threshold=0.8):
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    last_frame = None
    saved_frame = None
    frame_number = 0
    last_saved_frame_number = -1
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % n == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 72))

            if last_frame is not None:
                similarity = ssim(
                    gray, last_frame,
                    data_range=gray.max() - gray.min()
                )

                if similarity < ssim_threshold:
                    if saved_frame is not None and frame_number - last_saved_frame_number > fps:
                        path = os.path.join(
                            output_folder,
                            f"frame{frame_number:04d}_{frame_number // fps}.png"
                        )
                        cv2.imwrite(path, saved_frame)
                        timestamps.append((frame_number, frame_number // fps))

                    saved_frame = frame
                    last_saved_frame_number = frame_number
                else:
                    saved_frame = frame
            else:
                path = os.path.join(
                    output_folder,
                    f"frame{frame_number:04d}_{frame_number // fps}.png"
                )
                cv2.imwrite(path, frame)
                timestamps.append((frame_number, frame_number // fps))
                last_saved_frame_number = frame_number

            last_frame = gray

        frame_number += 1

    cap.release()
    return timestamps


def convert_frames_to_pdf(input_folder, output_file, timestamps):
    frame_files = sorted(
        os.listdir(input_folder),
        key=lambda x: int(x.split("_")[0].replace("frame", ""))
    )

    pdf = FPDF("L")
    pdf.set_auto_page_break(False)

    for frame_file, (_, ts) in zip(frame_files, timestamps):
        path = os.path.join(input_folder, frame_file)
        image = Image.open(path)

        pdf.add_page()
        pdf.image(path, x=0, y=0, w=pdf.w, h=pdf.h)

        timestamp = f"{ts//3600:02d}:{(ts%3600)//60:02d}:{ts%60:02d}"

        region = image.crop((5, 5, 65, 20)).convert("L")
        mean_pixel = region.resize((1, 1)).getpixel((0, 0))

        pdf.set_text_color(255, 255, 255 if mean_pixel < 64 else 0)
        pdf.set_xy(5, 5)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 0, timestamp)

    pdf.output(output_file)


def get_video_title(url):
    with yt_dlp.YoutubeDL({"skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info["title"]
        return re.sub(r'[\\/*?:"<>|]', "-", title).strip(".")


# ----------------------------------
# STREAMLIT UI
# ----------------------------------

url = st.text_input("🔗 Enter YouTube Video or Playlist URL")

if st.button("🎬 Generate PDF"):
    if not url:
        st.warning("Please enter a URL")
        st.stop()

    with st.spinner("Processing video..."):
        video_id = get_video_id(url)

        if video_id:
            urls = [url]
        else:
            urls = get_playlist_videos(url)

        for video_url in urls:
            video_title = get_video_title(video_url)
            output_pdf = f"{video_title}.pdf"

            with tempfile.TemporaryDirectory() as temp_dir:
                video_path = download_video(video_url, "video.mp4")
                timestamps = extract_unique_frames(video_path, temp_dir)
                convert_frames_to_pdf(temp_dir, output_pdf, timestamps)

            os.remove(video_path)

            with open(output_pdf, "rb") as f:
                st.success(f"✅ PDF ready: {output_pdf}")
                st.download_button(
                    label="⬇️ Download PDF",
                    data=f,
                    file_name=output_pdf,
                    mime="application/pdf"
                )
