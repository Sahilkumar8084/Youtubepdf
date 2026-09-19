import sys
import time
from PIL import ImageFile

from helper import cleanup_temp_files, get_playlist_videos, get_video_id, get_video_info, process_single_video
sys.modules['ImageFile'] = ImageFile
import cv2
import os
import tempfile
import re
import uuid
from fpdf import FPDF
from PIL import Image
import yt_dlp
from skimage.metrics import structural_similarity as ssim
import streamlit as st

            
            
def main():
    st.title("YouTube Video to PDF Frame Extractor")
    st.write("Extract unique frames from YouTube videos and create a PDF with timestamps")

    cleanup_temp_files()

    url = st.text_input("Enter the YouTube video or playlist URL:")

    if url:
        video_id_preview = get_video_id(url)
        if video_id_preview:
            with st.spinner("Fetching video info..."):
                info = get_video_info(url)
            if info:
                preview_col1, preview_col2 = st.columns([1, 2])
                with preview_col1:
                    if info.get('thumbnail'):
                        st.image(info['thumbnail'], use_container_width=True)
                with preview_col2:
                    st.markdown(f"**{info['title']}**")
                    if info.get('uploader'):
                        st.caption(f"by {info['uploader']}")
                    if info.get('duration'):
                        mins, secs = divmod(int(info['duration']), 60)
                        st.caption(f"Duration: {mins}:{secs:02d}")
            else:
                st.warning("Couldn't fetch a preview for this URL - double-check it. You can still try processing.")
        else:
            st.info("Playlist URL detected - previews aren't shown for playlists.")

    col1, col2 = st.columns(2)
    with col1:
        quality_choice = st.selectbox(
            "Speed vs quality",
            ["Fast (480p, recommended)", "Best quality (original resolution)"],
        )
        quality = 'fast' if quality_choice.startswith('Fast') else 'quality'
    with col2:
        frame_skip = st.slider(
            "Frame sampling interval (higher = faster, may skip quick changes)",
            min_value=1, max_value=10, value=3,
        )

    if not url:
        st.info("Please enter a YouTube URL to begin")
        return

    if st.button("Process Video/Playlist"):
        video_id = get_video_id(url)

        if video_id:
            output_pdf = process_single_video(url, quality=quality, frame_skip=frame_skip)

            if output_pdf and os.path.exists(output_pdf):
                st.success("PDF created successfully! ✅")
                with open(output_pdf, "rb") as f:
                    st.download_button(
                        label="📥 Download PDF",
                        data=f,
                        file_name=output_pdf,
                        mime="application/pdf"
                    )
                try:
                    os.remove(output_pdf)
                except:
                    pass

        else:
            st.info("Detected playlist. Extracting videos...")
            video_urls = get_playlist_videos(url)

            if not video_urls:
                st.error("No videos found in playlist or invalid playlist URL")
                return

            st.info(f"Found {len(video_urls)} videos in playlist")

            for idx, video_url in enumerate(video_urls, 1):
                st.write(f"Processing video {idx}/{len(video_urls)}")
                output_pdf = process_single_video(video_url, quality=quality, frame_skip=frame_skip)

                if output_pdf and os.path.exists(output_pdf):
                    with open(output_pdf, "rb") as f:
                        st.download_button(
                            label=f"📥 Download {os.path.basename(output_pdf)}",
                            data=f,
                            file_name=output_pdf,
                            mime="application/pdf",
                            key=f"download_{idx}"
                        )
                    try:
                        os.remove(output_pdf)
                    except Exception as e:
                        st.write("output_pdf Removing failed")
                        st.write(f"Error: {e}")
                        

            st.success("All videos processed! ✅")

if __name__ == "__main__":
    main()
