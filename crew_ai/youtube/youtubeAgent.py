import os
import sys
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# YouTube API key from environment variables
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def search_youtube_videos(query, max_results=20):
    """
    Fetches video link, title, views, likes, comments, channel name, and subscribers for YouTube videos matching the query.
    """
    try:
        # Build the YouTube API client
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        # Search for videos
        search_response = youtube.search().list(
            q=query,
            part="id,snippet",
            maxResults=max_results,
            type="video"
        ).execute()

        results = []
        for item in search_response.get("items", []):
            video_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            channel_id = item["snippet"]["channelId"]
            channel_name = item["snippet"]["channelTitle"]

            # Fetch video statistics
            video_response = youtube.videos().list(
                part="statistics",
                id=video_id
            ).execute()

            # Fetch channel statistics
            channel_response = youtube.channels().list(
                part="statistics",
                id=channel_id
            ).execute()

            if video_response["items"] and channel_response["items"]:
                stats = video_response["items"][0]["statistics"]
                channel_stats = channel_response["items"][0]["statistics"]

                views = stats.get("viewCount", "0")
                likes = stats.get("likeCount", "0")
                comments = stats.get("commentCount", "0")
                subscribers = channel_stats.get("subscriberCount", "0")
                video_url = f"https://www.youtube.com/watch?v={video_id}"

                results.append({
                    "title": title,
                    "views": views,
                    "likes": likes,
                    "comments": comments,
                    "url": video_url,
                    "channel_name": channel_name,
                    "subscribers": subscribers
                })

        return results

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    # Get the query from command line arguments or use a default value
    query = sys.argv[1] if len(sys.argv) > 1 else "Learn Python programming"

    # Fetch results
    video_results = search_youtube_videos(query)

    # Print results
    if isinstance(video_results, str):
        print(video_results)  # Print error message if any
    else:
        for idx, video in enumerate(video_results):
            print(f"{idx + 1}. Title: {video['title']}")
            print(f"   Views: {video['views']}")
            print(f"   Likes: {video['likes']}")
            print(f"   Comments: {video['comments']}")
            print(f"   URL: {video['url']}")
            print(f"   Channel: {video['channel_name']}")
            print(f"   Subscribers: {video['subscribers']}")
            print()
