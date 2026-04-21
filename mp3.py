import os
import subprocess

input_dir = r"C:\Users\Francisco\Videos\4K Video Downloader+"
output_dir = os.path.join(input_dir, "mp3")

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(input_dir):
    if file.endswith(".m4a"):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file.replace(".m4a", ".mp3"))

        subprocess.run([
            "ffmpeg",
            "-i", input_path,
            "-codec:a", "libmp3lame",
            "-qscale:a", "2",
            output_path
        ])

        print(f"Convertido: {file}")

print("Conversão concluída!")