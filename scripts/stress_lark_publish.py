#!/usr/bin/env python3
"""
Stress test script for Lark publishing.
Runs consecutive subprocess calls to process_video.py with --publish-to-lark.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Stress test for Lark publishing")
    parser.add_argument(
        "--runs", type=int, default=10, help="Number of consecutive runs (default: 10)"
    )
    parser.add_argument(
        "--video-url",
        type=str,
        default="https://www.bilibili.com/video/BV1mgA6zUEqN/",
        help="Video URL to process",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python interpreter path (default: sys.executable)",
    )
    args = parser.parse_args()

    python_path = args.python or sys.executable
    evidence_dir = Path(".sisyphus/evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failure_count = 0

    for i in range(1, args.runs + 1):
        log_file = evidence_dir / f"stress-run-{i}.log"

        # Build the command
        cmd = [
            python_path,
            "scripts/process_video.py",
            args.video_url,
            "--publish-to-lark",
            "--enable-transcript-correction",
            "--debug",
            "--image-context-frame-interval-seconds",
            "5",
            "--enable-image-context",
        ]

        # Set up environment: inherit current env but modify proxy settings
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        # Unset HTTP(S)_PROXY
        env.pop("HTTP_PROXY", None)
        env.pop("HTTPS_PROXY", None)
        env.pop("http_proxy", None)
        env.pop("https_proxy", None)
        # Set NO_PROXY
        env["NO_PROXY"] = "127.0.0.1,localhost"

        print(f"Run {i}/{args.runs}: Executing...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                cwd=os.getcwd(),
            )

            # Write combined stdout/stderr to log file
            combined_output = result.stdout + "\n" + result.stderr
            log_file.write_text(combined_output)

            if result.returncode == 0:
                print(f"Run {i}: SUCCESS (exit code 0)")
                success_count += 1
            else:
                print(f"Run {i}: FAILED (exit code {result.returncode})")
                failure_count += 1

        except Exception as e:
            error_msg = f"Exception occurred: {e}"
            print(f"Run {i}: ERROR - {error_msg}")
            log_file.write_text(error_msg)
            failure_count += 1

    print(f"\n=== Summary ===")
    print(f"Total runs: {args.runs}")
    print(f"Successes: {success_count}")
    print(f"Failures: {failure_count}")

    if failure_count > 0:
        print(f"\nFAILED: {failure_count} run(s) failed")
        sys.exit(1)
    else:
        print("\nSUCCESS: All runs completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
