from pathlib import Path

from video_asr_summary import process_video

result = process_video(Path("/Users/rlan/Downloads/nvidia-and-gold.mp4"), language="zh-CN")
print(result["summary"]["summary"])
