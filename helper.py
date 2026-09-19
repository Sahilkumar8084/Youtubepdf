import sys
import time
import requests
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


@st.cache_data(show_spinner=False, ttl=3600)
def get_video_info(url):
    """Fast preview info (title + thumbnail + channel) via YouTube's public
    oEmbed endpoint - a single small JSON request, no format/signature
    resolution like a full yt-dlp extraction needs. Cached per URL so
    re-running the Streamlit script (e.g. moving a slider) never refetches.
    Note: oEmbed doesn't provide duration, so that field is always None."""
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=4,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return {
            'title': data.get('title', 'Unknown title'),
            'thumbnail': data.get('thumbnail_url'),
            'uploader': data.get('author_name'),
            'duration': None,
        }
    except Exception:
        return None


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
