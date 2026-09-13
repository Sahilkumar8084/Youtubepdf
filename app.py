import sys
import time
from PIL import ImageFile
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

# def download_video(url, max_retries=3):
#     """Download video with proper cleanup and unique filenames to avoid resume errors"""
#     # Generate unique filename to avoid conflicts
#     unique_id = str(uuid.uuid4())[:8]
#     filename = f"video_{unique_id}.mp4"
    
#     # Delete any existing partial downloads
#     if os.path.exists(filename):
#         os.remove(filename)
#     if os.path.exists(f"{filename}.part"):
#         os.remove(f"{filename}.part")
    
#     ydl_opts = {
#         'outtmpl': filename,
#         'format': 'best[ext=mp4]/best',
#         'quiet': True,
#         'no_warnings': False,
#         'ignoreerrors': False,
#         'noprogress': True,
#         'no_color': True,
#         'overwrites': True,  # Overwrite existing files
#         'continuedl': False,  # Don't continue partial downloads
#     }
    
#     for attempt in range(max_retries):
#         try:
#             # Clean up before each attempt
#             if os.path.exists(filename):
#                 os.remove(filename)
#             if os.path.exists(f"{filename}.part"):
#                 os.remove(f"{filename}.part")
            
#             with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#                 ydl.download([url])
                
#             if os.path.exists(filename):
#                 return filename
#             else:
#                 st.warning(f"Download completed but file not found. Attempt {attempt + 1}/{max_retries}")
                
#         except Exception as e:
#             st.warning(f"Download attempt {attempt + 1}/{max_retries} failed: {str(e)[:100]}")
#             # Clean up failed downloads
#             if os.path.exists(filename):
#                 os.remove(filename)
#             if os.path.exists(f"{filename}.part"):
#                 os.remove(f"{filename}.part")
            
#             if attempt < max_retries - 1:
#                 continue
#             else:
#                 st.error(f"Failed to download video after {max_retries} attempts.")
#                 return None
    
#     return None


# def download_video(url, max_retries=3):
#     """Download video with proper cleanup, unique filenames, and YouTube
#     anti-bot workarounds (rotating player clients + browser-like headers)."""
#     unique_id = str(uuid.uuid4())[:8]
#     filename = f"video_{unique_id}.mp4"

#     if os.path.exists(filename):
#         os.remove(filename)
#     if os.path.exists(f"{filename}.part"):
#         os.remove(f"{filename}.part")

#     # YouTube's bot-detection frequently blocks the default 'web' client.
#     # Rotate through clients that are less likely to be challenged.
#     client_strategies = [
#         ['android', 'web'],
#         ['ios', 'android'],
#         ['web'],
#     ]

#     common_headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
#                       '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
#         'Accept-Language': 'en-us,en;q=0.5',
#     }

#     last_error = None

#     for attempt in range(max_retries):
#         if os.path.exists(filename):
#             os.remove(filename)
#         if os.path.exists(f"{filename}.part"):
#             os.remove(f"{filename}.part")

#         clients = client_strategies[attempt % len(client_strategies)]

#         ydl_opts = {
#             'outtmpl': filename,
#             'format': 'best[ext=mp4]/best',
#             'quiet': True,
#             'no_warnings': False,
#             'ignoreerrors': False,
#             'noprogress': True,
#             'no_color': True,
#             'overwrites': True,
#             'continuedl': False,
#             'nocheckcertificate': True,
#             'geo_bypass': True,
#             'extractor_args': {
#                 'youtube': {
#                     'player_client': clients,
#                 }
#             },
#             'http_headers': common_headers,
#         }

#         try:
#             with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#                 ydl.download([url])

#             if os.path.exists(filename):
#                 return filename
#             else:
#                 last_error = "Download reported success but the output file is missing."
#                 st.warning(f"Attempt {attempt + 1}/{max_retries}: {last_error}")

#         except Exception as e:
#             last_error = str(e)
#             st.warning(
#                 f"Download attempt {attempt + 1}/{max_retries} failed "
#                 f"(player_client={clients}): {last_error}"
#             )
#             if os.path.exists(filename):
#                 os.remove(filename)
#             if os.path.exists(f"{filename}.part"):
#                 os.remove(f"{filename}.part")

#     st.error(
#         f"Failed to download video after {max_retries} attempts.\n\n"
#         f"Last error: {last_error}\n\n"
#         "If this mentions 'Sign in to confirm you're not a bot' or an HTTP 403, "
#         "your yt-dlp version is most likely too old for YouTube's current checks - "
#         "run `pip install -U yt-dlp` in your venv and try again."
#     )
#     return None


def download_video(url, max_retries=3, quality='fast'):
    """Download video with proper cleanup, unique filenames, YouTube
    anti-bot workarounds, a live progress bar, and a speed/quality switch."""
    unique_id = str(uuid.uuid4())[:8]
    filename = f"video_{unique_id}.mp4"

    if os.path.exists(filename):
        os.remove(filename)
    if os.path.exists(f"{filename}.part"):
        os.remove(f"{filename}.part")

    # 'fast' downloads a capped 480p stream - much smaller/quicker to fetch
    # and decode later, since frames only need to be compared at 128x72
    # anyway. 'quality' keeps the original resolution.
    format_map = {
        'fast': 'best[height<=480][ext=mp4]/best[height<=480]/best[ext=mp4]/best',
        'quality': 'best[ext=mp4]/best',
    }
    video_format = format_map.get(quality, format_map['fast'])

    client_strategies = [
        ['android', 'web'],
        ['ios', 'android'],
        ['web'],
    ]

    common_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept-Language': 'en-us,en;q=0.5',
    }

    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_hook(d):
        if d.get('status') == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate')
            downloaded = d.get('downloaded_bytes', 0)
            speed = d.get('speed')
            eta = d.get('eta')
            total_mb = None
            if total:
                progress_bar.progress(min(downloaded / total, 1.0))
                total_mb = total / (1024 * 1024)
            downloaded_mb = downloaded / (1024 * 1024)
            speed_txt = f"{speed / (1024 * 1024):.1f} MB/s" if speed else "..."
            eta_txt = f"{eta}s" if eta is not None else "..."
            size_txt = f"{downloaded_mb:.1f}MB" + (f" / {total_mb:.1f}MB" if total_mb else "")
            status_text.text(f"⬇️ Downloading: {size_txt} | Speed: {speed_txt} | ETA: {eta_txt}")
        elif d.get('status') == 'finished':
            progress_bar.progress(1.0)
            status_text.text("Download complete, preparing video...")

    last_error = None

    for attempt in range(max_retries):
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists(f"{filename}.part"):
            os.remove(f"{filename}.part")

        clients = client_strategies[attempt % len(client_strategies)]

        ydl_opts = {
            'outtmpl': filename,
            'format': video_format,
            'quiet': True,
            'no_warnings': False,
            'ignoreerrors': False,
            'noprogress': True,
            'no_color': True,
            'overwrites': True,
            'continuedl': False,
            'nocheckcertificate': True,
            'geo_bypass': True,
            'extractor_args': {
                'youtube': {
                    'player_client': clients,
                }
            },
            'http_headers': common_headers,
            'progress_hooks': [progress_hook],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            if os.path.exists(filename):
                progress_bar.empty()
                status_text.empty()
                return filename
            else:
                last_error = "Download reported success but the output file is missing."
                st.warning(f"Attempt {attempt + 1}/{max_retries}: {last_error}")

        except Exception as e:
            last_error = str(e)
            st.warning(
                f"Download attempt {attempt + 1}/{max_retries} failed "
                f"(player_client={clients}): {last_error}"
            )
            if os.path.exists(filename):
                os.remove(filename)
            if os.path.exists(f"{filename}.part"):
                os.remove(f"{filename}.part")

    progress_bar.empty()
    status_text.empty()
    st.error(
        f"Failed to download video after {max_retries} attempts.\n\n"
        f"Last error: {last_error}\n\n"
        "If this mentions 'Sign in to confirm you're not a bot' or an HTTP 403, "
        "your yt-dlp version is most likely too old for YouTube's current checks - "
        "run `pip install -U yt-dlp` in your venv and try again."
    )
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

def get_playlist_videos(playlist_url):
    """Extract all video URLs from a playlist"""
    ydl_opts = {
        'ignoreerrors': True,
        'playlistend': 1000,
        'extract_flat': True,
        'quiet': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_info = ydl.extract_info(playlist_url, download=False)
            if playlist_info and 'entries' in playlist_info:
                return [f"https://www.youtube.com/watch?v={entry['id']}" 
                        for entry in playlist_info['entries'] 
                        if entry and 'id' in entry]
            else:
                return []
    except Exception as e:
        st.error(f"Error extracting playlist: {e}")
        return []

# def extract_unique_frames(video_file, output_folder, n=3, ssim_threshold=0.8):
#     """Extract unique frames from video using SSIM comparison"""
#     cap = cv2.VideoCapture(video_file)
    
#     if not cap.isOpened():
#         st.error(f"Failed to open video file: {video_file}")
#         return []
    
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     if fps == 0:
#         fps = 30  # Default fallback
    
#     last_frame = None
#     saved_frame = None
#     frame_number = 0
#     last_saved_frame_number = -1
#     timestamps = []
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         if frame_number % n == 0:
#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             gray_frame = cv2.resize(gray_frame, (128, 72))
            
#             if last_frame is not None:
#                 similarity = ssim(gray_frame, last_frame, data_range=gray_frame.max() - gray_frame.min())
                
#                 if similarity < ssim_threshold:
#                     if saved_frame is not None and frame_number - last_saved_frame_number > fps:
#                         timestamp_seconds = frame_number // fps
#                         frame_path = os.path.join(output_folder, f'frame{frame_number:04d}_{timestamp_seconds}.png')
#                         cv2.imwrite(frame_path, saved_frame)
#                         timestamps.append((frame_number, timestamp_seconds))
                    
#                     saved_frame = frame
#                     last_saved_frame_number = frame_number
#                 else:
#                     saved_frame = frame
#             else:
#                 # First frame
#                 timestamp_seconds = frame_number // fps
#                 frame_path = os.path.join(output_folder, f'frame{frame_number:04d}_{timestamp_seconds}.png')
#                 cv2.imwrite(frame_path, frame)
#                 timestamps.append((frame_number, timestamp_seconds))
#                 saved_frame = frame
#                 last_saved_frame_number = frame_number
            
#             last_frame = gray_frame
        
#         frame_number += 1
    
#     # Save the last saved frame if it exists
#     if saved_frame is not None and last_saved_frame_number < frame_number - fps:
#         timestamp_seconds = frame_number // fps
#         frame_path = os.path.join(output_folder, f'frame{frame_number:04d}_{timestamp_seconds}.png')
#         cv2.imwrite(frame_path, saved_frame)
#         timestamps.append((frame_number, timestamp_seconds))
    
#     cap.release()
#     return timestamps

def extract_unique_frames(video_file, output_folder, n=3, ssim_threshold=0.8):
    """Extract unique frames from video using SSIM comparison, with a live
    progress bar/ETA and a grab()-only fast path for skipped frames."""
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        st.error(f"Failed to open video file: {video_file}")
        return []

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()
    update_every = max(1, (total_frames // 100) if total_frames > 0 else 30)

    last_frame = None
    saved_frame = None
    frame_number = 0
    last_saved_frame_number = -1
    timestamps = []

    while cap.isOpened():
        if frame_number % n == 0:
            ret, frame = cap.read()
        else:
            # Skip decoding/copying frames we don't need - just advance
            # the internal position. Much cheaper than a full read().
            ret = cap.grab()
            frame = None
        if not ret:
            break

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
                timestamp_seconds = frame_number // fps
                frame_path = os.path.join(output_folder, f'frame{frame_number:04d}_{timestamp_seconds}.png')
                cv2.imwrite(frame_path, frame)
                timestamps.append((frame_number, timestamp_seconds))
                saved_frame = frame
                last_saved_frame_number = frame_number

            last_frame = gray_frame

        frame_number += 1

        if frame_number % update_every == 0:
            elapsed = time.time() - start_time
            if total_frames > 0:
                pct = min(frame_number / total_frames, 1.0)
                rate = frame_number / elapsed if elapsed > 0 else 0
                remaining = (total_frames - frame_number) / rate if rate > 0 else 0
                progress_bar.progress(pct)
                status_text.text(
                    f"🔍 Analyzing frames: {frame_number}/{total_frames} ({pct * 100:.0f}%) "
                    f"| {len(timestamps)} unique so far | ETA: {remaining:.0f}s"
                )
            else:
                status_text.text(
                    f"🔍 Analyzing frames: {frame_number} processed "
                    f"| {len(timestamps)} unique so far | Elapsed: {elapsed:.0f}s"
                )

    if saved_frame is not None and last_saved_frame_number < frame_number - fps:
        timestamp_seconds = frame_number // fps
        frame_path = os.path.join(output_folder, f'frame{frame_number:04d}_{timestamp_seconds}.png')
        cv2.imwrite(frame_path, saved_frame)
        timestamps.append((frame_number, timestamp_seconds))

    cap.release()
    progress_bar.progress(1.0)
    status_text.text(f"✅ Frame analysis done in {time.time() - start_time:.0f}s - {len(timestamps)} unique frames found")
    return timestamps


# def convert_frames_to_pdf(input_folder, output_file, timestamps):
#     """Convert extracted frames to PDF with timestamps"""
#     frame_files = sorted(os.listdir(input_folder), 
#                         key=lambda x: int(x.split('_')[0].replace('frame', '')))
    
#     if not frame_files:
#         st.warning("No frames found to convert to PDF")
#         return False
    
#     pdf = FPDF("L")
#     pdf.set_auto_page_break(0)
    
#     for frame_file, (frame_number, timestamp_seconds) in zip(frame_files, timestamps):
#         frame_path = os.path.join(input_folder, frame_file)
        
#         if not os.path.exists(frame_path):
#             continue
        
#         try:
#             image = Image.open(frame_path)
#             pdf.add_page()
#             pdf.image(frame_path, x=0, y=0, w=pdf.w, h=pdf.h)
            
#             # Format timestamp
#             timestamp = f"{timestamp_seconds // 3600:02d}:{(timestamp_seconds % 3600) // 60:02d}:{timestamp_seconds % 60:02d}"
            
#             # Determine text color based on background
#             x, y, width, height = 5, 5, 60, 15
#             region = image.crop((x, y, x + width, y + height)).convert("L")
#             mean_pixel_value = region.resize((1, 1)).getpixel((0, 0))
            
#             if mean_pixel_value < 64:
#                 pdf.set_text_color(255, 255, 255)
#             else:
#                 pdf.set_text_color(0, 0, 0)
            
#             pdf.set_xy(x, y)
#             pdf.set_font("Arial", size=12)
#             pdf.cell(0, 0, timestamp)
#         except Exception as e:
#             st.warning(f"Error processing frame {frame_file}: {e}")
#             continue
    
#     try:
#         pdf.output(output_file)
#         return True
#     except Exception as e:
#         st.error(f"Error creating PDF: {e}")
#         return False
def convert_frames_to_pdf(input_folder, output_file, timestamps):
    """Convert extracted frames to PDF with timestamps"""
    frame_files = sorted(os.listdir(input_folder),
                        key=lambda x: int(x.split('_')[0].replace('frame', '')))

    if not frame_files:
        st.warning("No frames found to convert to PDF")
        return False

    pdf = FPDF("L")
    pdf.set_auto_page_break(0)

    total = len(frame_files)
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    for idx, (frame_file, (frame_number, timestamp_seconds)) in enumerate(zip(frame_files, timestamps), 1):
        frame_path = os.path.join(input_folder, frame_file)

        if not os.path.exists(frame_path):
            continue

        try:
            image = Image.open(frame_path)
            pdf.add_page()
            pdf.image(frame_path, x=0, y=0, w=pdf.w, h=pdf.h)

            timestamp = f"{timestamp_seconds // 3600:02d}:{(timestamp_seconds % 3600) // 60:02d}:{timestamp_seconds % 60:02d}"

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

        if idx % max(1, total // 50) == 0 or idx == total:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0
            remaining = (total - idx) / rate if rate > 0 else 0
            progress_bar.progress(idx / total)
            status_text.text(f"📄 Building PDF: page {idx}/{total} | ETA: {remaining:.0f}s")

    progress_bar.progress(1.0)
    status_text.text(f"✅ PDF pages built in {time.time() - start_time:.0f}s")

    try:
        pdf.output(output_file)
        return True
    except Exception as e:
        st.error(f"Error creating PDF: {e}")
        return False
    
def get_video_title(url):
    """Get video title from YouTube URL"""
    ydl_opts = {
        'skip_download': True,
        'quiet': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            video_info = ydl.extract_info(url, download=False)
            title = video_info.get('title', 'video')
            # Sanitize filename
            title = title.replace('/', '-').replace('\\', '-').replace(':', '-')
            title = title.replace('*', '-').replace('?', '-').replace('<', '-')
            title = title.replace('>', '-').replace('|', '-').replace('"', '-')
            title = title.strip('.')
            return title[:100]  # Limit length
    except Exception as e:
        st.warning(f"Could not get video title: {e}")
        return "video"

def cleanup_temp_files(pattern="video_*.mp4"):
    """Clean up any leftover temporary video files"""
    try:
        for file in os.listdir('.'):
            if file.startswith('video_') and (file.endswith('.mp4') or file.endswith('.mp4.part')):
                try:
                    os.remove(file)
                except:
                    pass
    except:
        pass

# def process_single_video(url):
#     """Process a single video URL"""
#     st.info(f"Processing video...")
    
#     # Clean up any old temp files first
#     cleanup_temp_files()
    
#     video_file = download_video(url)
#     if not video_file or not os.path.exists(video_file):
#         st.error("Failed to download video. Please check the URL and try again.")
#         return None
    
#     try:
#         video_title = get_video_title(url)
#         output_pdf_name = f"{video_title}.pdf"
        
#         with tempfile.TemporaryDirectory() as temp_folder:
#             st.info("Extracting unique frames...")
#             timestamps = extract_unique_frames(video_file, temp_folder)
            
#             if not timestamps:
#                 st.warning("No unique frames extracted from video")
#                 return None
            
#             st.info(f"Extracted {len(timestamps)} unique frames. Creating PDF...")
#             success = convert_frames_to_pdf(temp_folder, output_pdf_name, timestamps)
            
#             if not success:
#                 return None
        
#         # Clean up video file
#         if os.path.exists(video_file):
#             try:
#                 os.remove(video_file)
#             except:
#                 pass
        
#         return output_pdf_name
#     except Exception as e:
#         st.error(f"Error processing video: {e}")
#         if os.path.exists(video_file):
#             try:
#                 os.remove(video_file)
#             except:
#                 pass
#         return None
def process_single_video(url, quality='fast', frame_skip=3):
    """Process a single video URL"""
    overall_start = time.time()
    st.info(f"Processing video...")

    cleanup_temp_files()

    video_file = download_video(url, quality=quality)
    if not video_file or not os.path.exists(video_file):
        st.error("Failed to download video. Please check the URL and try again.")
        return None

    try:
        video_title = get_video_title(url)
        output_pdf_name = f"{video_title}.pdf"

        with tempfile.TemporaryDirectory() as temp_folder:
            timestamps = extract_unique_frames(video_file, temp_folder, n=frame_skip)

            if not timestamps:
                st.warning("No unique frames extracted from video")
                return None

            success = convert_frames_to_pdf(temp_folder, output_pdf_name, timestamps)

            if not success:
                return None

        if os.path.exists(video_file):
            try:
                os.remove(video_file)
            except:
                pass

        st.info(f"Total time: {time.time() - overall_start:.0f}s")
        return output_pdf_name
    except Exception as e:
        st.error(f"Error processing video: {e}")
        if os.path.exists(video_file):
            try:
                os.remove(video_file)
            except:
                pass
        return None
    
    
    
# def main():
#     st.title("YouTube Video to PDF Frame Extractor")
#     st.write("Extract unique frames from YouTube videos and create a PDF with timestamps")
    
#     # Clean up old files on start
#     cleanup_temp_files()
    
#     # Input URL
#     url = st.text_input("Enter the YouTube video or playlist URL:")
    
#     if not url:
#         st.info("Please enter a YouTube URL to begin")
#         return
    
#     if st.button("Process Video/Playlist"):
#         video_id = get_video_id(url)
        
#         if video_id:  # Single video
#             output_pdf = process_single_video(url)
            
#             if output_pdf and os.path.exists(output_pdf):
#                 st.success("PDF created successfully! ✅")
#                 with open(output_pdf, "rb") as f:
#                     st.download_button(
#                         label="📥 Download PDF",
#                         data=f,
#                         file_name=output_pdf,
#                         mime="application/pdf"
#                     )
#                 # Clean up
#                 try:
#                     os.remove(output_pdf)
#                 except:
#                     pass
            
#         else:  # Playlist
#             st.info("Detected playlist. Extracting videos...")
#             video_urls = get_playlist_videos(url)
            
#             if not video_urls:
#                 st.error("No videos found in playlist or invalid playlist URL")
#                 return
            
#             st.info(f"Found {len(video_urls)} videos in playlist")
            
#             for idx, video_url in enumerate(video_urls, 1):
#                 st.write(f"Processing video {idx}/{len(video_urls)}")
#                 output_pdf = process_single_video(video_url)
                
#                 if output_pdf and os.path.exists(output_pdf):
#                     with open(output_pdf, "rb") as f:
#                         st.download_button(
#                             label=f"📥 Download {os.path.basename(output_pdf)}",
#                             data=f,
#                             file_name=output_pdf,
#                             mime="application/pdf",
#                             key=f"download_{idx}"
#                         )
#                     # Clean up
#                     try:
#                         os.remove(output_pdf)
#                     except:
#                         pass
            
#             st.success("All videos processed! ✅")
            
            
            
            
def main():
    st.title("YouTube Video to PDF Frame Extractor")
    st.write("Extract unique frames from YouTube videos and create a PDF with timestamps")

    cleanup_temp_files()

    url = st.text_input("Enter the YouTube video or playlist URL:")

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