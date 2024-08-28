import yt_dlp

def download_audio(youtube_url, output_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

if __name__ == "__main__":
    youtube_url = input("Enter the YouTube video URL: ")
    output_path = "/home/han/Desktop/seamless/clip.mp3"
    download_audio(youtube_url, output_path)
    print(f"MP3 file saved at: {output_path}")
