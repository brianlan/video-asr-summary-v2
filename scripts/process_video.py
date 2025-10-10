from pathlib import Path

from video_asr_summary import process_video

result = process_video(
	Path("/Users/rlan/Downloads/suzhou.mp4"),
	language="zh",
	audio_format="mp3",
	audio_bitrate="48k",
)
# print(result["summary"]["summary"])
print(result)
