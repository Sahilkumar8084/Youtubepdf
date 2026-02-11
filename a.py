import sys
from PIL import ImageFile
sys.modules['ImageFile'] = ImageFile
import cv2
import os
import tempfile
import re
import uuid
from fpdf import FPDF
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import streamlit as st
import requests
import json

def download_video_via_api(url, max_retries=3):
    """Download video using cobalt.tools API (free, no auth needed)"""
    unique_id = str(uuid.uuid4())[:8]
    filename = f"video_{unique_id}.mp4"
    
    # Delete any existing file
    if os.path.exists(filename):
        os.remove(filename)
    
    # Cobalt API endpoint
    api_url = "https://api.cobalt.tools/api/json"
    
    for attempt in range(max_retries):
        try:
            st.info(f"Downloading video via API (attempt {attempt + 1}/{max_retries})...")
            
            # Request download link from Cobalt
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json"
            }
            
            payload = {
                "url": url,
                "vCodec": "h264",
                "vQuality": "720",
                "aFormat": "mp3",
                "isAudioOnly": False,
                "filenamePattern": "basic"
            }
            
            response = requests.post(api_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("status") == "redirect" or data.get("status") == "stream":
                    download_url = data.get("url")
                    
                    if download_url:
                        # Download the video file
                        st.info("Downloading video file...")
                        video_response = requests.get(download_url, stream=True, timeout=120)
                        
                        if video_response.status_code == 200:
                            with open(filename, 'wb') as f:
                                for chunk in video_response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            
                            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                                st.success("Video downloaded successfully!")
                                return filename
                        else:
                            st.warning(f"Failed to download video file. Status: {video_response.status_code}")
                    else:
                        st.warning("No download URL in API response")
                elif data.get("status") == "error":
                    st.warning(f"API Error: {data.get('text', 'Unknown error')}")
                else:
                    st.warning(f"Unexpected API status: {data.get('status')}")
            else:
                st.warning(f"API request failed with status {response.status_code}")
            
        except Exception as e:
            st.warning(f"Download attempt {attempt + 1}/{max_retries} failed: {str(e)[:200]}")
            
            if os.path.exists(filename):
                os.remove(filename)
            
            if attempt < max_retries - 1:
                continue
            else:
                st.error(f"Failed to download video after {max_retries} attempts.")
                return None
    
    return None

def get_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    # Match YouTube Shorts URLs
    video_id_match = re.search(r"shorts\/(\w+)", url)
    if video_id_match:
        return video_id_match.group(1)
    
    # Match youtube.be shortened URLs
    video_id_match = re.search(r"youtu\.be\/([\w\-_]+)(\?.*)?", url)
    if video_id_match:
        return video_id_match.group(1)
    
    # Match regular YouTube URLs
    video_id_match = re.search(r"v=([\w\-_]+)", url)
    if video_id_match:
        return video_id_match.group(1)
    
    # Match YouTube live stream URLs
    video_id_match = re.search(r"live\/(\w+)", url)
    if video_id_match:
        return video_id_match.group(1)
    
    return None

def extract_unique_frames(video_file, output_folder, n=3, ssim_threshold=0.8):
    """Extract unique frames from video using SSIM comparison"""
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        st.error(f"Failed to open video file: {video_file}")
        return []
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default fallback
    
    last_frame = None
    saved_frame = None
    frame_number = 0
    last_saved_frame_number = -1
    timestamps = []
    
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress
        if total_frames > 0:
            progress_bar.progress(min(frame_number / total_frames, 1.0))
        
        if frame_number % n == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, (128, 72))
            
            if last_frame is not None:
                similarity = ssim(gray_frame, last_frame, data_range=gray_frame.max() - gray_frame.min())
                
                if similarity < ssim_threshold:
                    if saved_frame is not None and frame_number - last_saved_frame_number > fps:
                        timestamp_seconds = frame_number // fps
                        frame_path = os.path.join(output_folder, f'frame{frame_number:04d}_{timestamp_seconds}.png')
                        cv2.imwrite(frame_path, saved_frame)
                        timestamps.append((frame_number, timestamp_seconds))
                    
                    saved_frame = frame
                    last_saved_frame_number = frame_number
                else:
                    saved_frame = frame
            else:
                # First frame
                timestamp_seconds = frame_number // fps
                frame_path = os.path.join(output_folder, f'frame{frame_number:04d}_{timestamp_seconds}.png')
                cv2.imwrite(frame_path, frame)
                timestamps.append((frame_number, timestamp_seconds))
                saved_frame = frame
                last_saved_frame_number = frame_number
            
            last_frame = gray_frame
        
        frame_number += 1
    
    # Save the last saved frame if it exists
    if saved_frame is not None and last_saved_frame_number < frame_number - fps:
        timestamp_seconds = frame_number // fps
        frame_path = os.path.join(output_folder, f'frame{frame_number:04d}_{timestamp_seconds}.png')
        cv2.imwrite(frame_path, saved_frame)
        timestamps.append((frame_number, timestamp_seconds))
    
    progress_bar.progress(1.0)
    cap.release()
    return timestamps

def convert_frames_to_pdf(input_folder, output_file, timestamps):
    """Convert extracted frames to PDF with timestamps"""
    frame_files = sorted(os.listdir(input_folder), 
                        key=lambda x: int(x.split('_')[0].replace('frame', '')))
    
    if not frame_files:
        st.warning("No frames found to convert to PDF")
        return False
    
    pdf = FPDF("L")
    pdf.set_auto_page_break(0)
    
    progress_bar = st.progress(0)
    
    for idx, (frame_file, (frame_number, timestamp_seconds)) in enumerate(zip(frame_files, timestamps)):
        progress_bar.progress((idx + 1) / len(frame_files))
        
        frame_path = os.path.join(input_folder, frame_file)
        
        if not os.path.exists(frame_path):
            continue
        
        try:
            image = Image.open(frame_path)
            pdf.add_page()
            pdf.image(frame_path, x=0, y=0, w=pdf.w, h=pdf.h)
            
            # Format timestamp
            timestamp = f"{timestamp_seconds // 3600:02d}:{(timestamp_seconds % 3600) // 60:02d}:{timestamp_seconds % 60:02d}"
            
            # Determine text color based on background
            x, y, width, height = 5, 5, 60, 15
            region = image.crop((x, y, x + width, y + height)).convert("L")
            mean_pixel_value = region.resize((1, 1)).getpixel((0, 0))
            
            if mean_pixel_value < 64:
                pdf.set_text_color(255, 255, 255)
            else:
                pdf.set_text_color(0, 0, 0)
            
            pdf.set_xy(x, y)
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 0, timestamp)
        except Exception as e:
            st.warning(f"Error processing frame {frame_file}: {e}")
            continue
    
    try:
        pdf.output(output_file)
        return True
    except Exception as e:
        st.error(f"Error creating PDF: {e}")
        return False

def cleanup_temp_files():
    """Clean up any leftover temporary video files"""
    try:
        for file in os.listdir('.'):
            if file.startswith('video_') and file.endswith('.mp4'):
                try:
                    os.remove(file)
                except:
                    pass
    except:
        pass

def process_single_video(url):
    """Process a single video URL"""
    st.info(f"Processing video...")
    
    # Clean up any old temp files first
    cleanup_temp_files()
    
    video_file = download_video_via_api(url)
    if not video_file or not os.path.exists(video_file):
        st.error("❌ Failed to download video.")
        st.info("💡 **Alternative options:**")
        st.markdown("1. Try a different video")
        st.markdown("2. Download the video manually and use a local tool")
        st.markdown("3. Use a browser extension for frame extraction")
        return None
    
    try:
        output_pdf_name = f"youtube_frames_{uuid.uuid4().hex[:8]}.pdf"
        
        with tempfile.TemporaryDirectory() as temp_folder:
            st.info("📸 Extracting unique frames...")
            timestamps = extract_unique_frames(video_file, temp_folder)
            
            if not timestamps:
                st.warning("No unique frames extracted from video")
                return None
            
            st.info(f"📄 Creating PDF with {len(timestamps)} frames...")
            success = convert_frames_to_pdf(temp_folder, output_pdf_name, timestamps)
            
            if not success:
                return None
        
        # Clean up video file
        if os.path.exists(video_file):
            try:
                os.remove(video_file)
            except:
                pass
        
        return output_pdf_name
    except Exception as e:
        st.error(f"Error processing video: {e}")
        if video_file and os.path.exists(video_file):
            try:
                os.remove(video_file)
            except:
                pass
        return None

def main():
    st.title("🎬 YouTube Video to PDF Frame Extractor")
    st.write("Extract unique frames from YouTube videos and create a PDF with timestamps")
    
    # Clean up old files on start
    cleanup_temp_files()
    
    # Warning about current limitations
    st.warning("⚠️ **Note:** Due to YouTube's recent restrictions, some videos may fail to download. We're using a third-party API service which may have limitations.")
    
    # Input URL
    url = st.text_input("Enter the YouTube video URL:", placeholder="https://www.youtube.com/watch?v=...")
    
    if not url:
        st.info("👆 Please enter a YouTube URL to begin")
        st.markdown("### How to use:")
        st.markdown("1. Paste a YouTube video URL (single video only)")
        st.markdown("2. Click 'Process Video'")
        st.markdown("3. Download the generated PDF with timestamped frames")
        return
    
    if st.button("Process Video", type="primary"):
        video_id = get_video_id(url)
        
        if video_id:  # Single video
            output_pdf = process_single_video(url)
            
            if output_pdf and os.path.exists(output_pdf):
                st.success("✅ PDF created successfully!")
                st.balloons()
                with open(output_pdf, "rb") as f:
                    st.download_button(
                        label="📥 Download PDF",
                        data=f,
                        file_name=output_pdf,
                        mime="application/pdf"
                    )
                # Clean up
                try:
                    os.remove(output_pdf)
                except:
                    pass
        else:
            st.error("Invalid YouTube URL or playlists are not supported at this time.")

if __name__ == "__main__":
    main()
