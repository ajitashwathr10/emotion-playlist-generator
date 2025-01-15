import customtkinter as ctk 
from textblob import TextBlob
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import random
import webbrowser
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv

class Spotify:
    def __init__(self, client_id: str, client_secret: str):
        self.client_credentials = SpotifyClientCredentials(
            client_id = client_id,
            client_secret = client_secret
        )
        self.sp = spotipy.Spotify(client_credentials_manager = self.client_credentials)
        self.emotion_mapping = {
            'joy': {
                'seed_genres': ['pop', 'dance', 'happy'],
                'attributes': {
                    'target_valence': 0.8,
                    'target_energy': 0.8,
                    'target_tempo': 120
                }
            },
            'sadness': {
                'seed_genres': ['sad', 'blues', 'indie'],
                'attributes': {
                    'target_valence': 0.2,
                    'target_energy': 0.3,
                    'target_tempo': 70
                }
            },
            'anger': {
                'seed_genres': ['ambient', 'classical', 'chill'],
                'attributes': {
                    'target_valence': 0.5,
                    'target_energy': 0.3,
                    'target_tempo': 90
                }
            }
        }
    
    def get_recommendations(self, emotion: str, limit: int = 5) -> List[Dict]:
        emotion_props = self.emotion_mapping[emotion]
        try: 
            recommendations = self.sp.recommendations(
                seed_genres = emotion_props['seed_genres'],
                limit = limit,
                **emotion_props['attributes']
            )
            songs = []
            for track in recommendations['tracks']:
                songs.append({
                    'name': track['name'],
                    'artist': track['artist'][0]['name'],
                    'album': track['album']['name'],
                    'url': track['external_urls']['spotify'],
                    'preview_url': track['preview_url'],
                    'duration': track['duration_ms'] // 1000
                })
            return songs
        except Exception as e:
            print(f"Error while getting recommendations: {e}")
            return []
        
class Emotion:
    def analyze_emotion(self, text: str) -> str:
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        if polarity > 0.3:
            return 'joy'
        elif polarity < -0.3:
            if subjectivity > 0.7:
                return 'anger'
            else:
                return 'sadness'
        else:
            return 'calm'

class Playlist(ctk.CTk):
    def __init__(self, spotify_manager: Spotify):
        super().__init__()
        self.spotify_manager = spotify_manager
        self.emotion_analyzer = Emotion()
        self.title("Emotion Spotify Playlist Generator")
        self.geometry("800x900")
        self.grid_columnconfigure(0, weight = 1)
        self.grid_rowconfigure(3, weight = 1)
        self.create_widgets()

    def create_widgets(self):
        self.title_label = ctk.CTkLabel(
            self, 
            text = "Emotion Playlist Generator",
            font = ctk.CTkFont(size = 24, weight = "bold")
        )
        self.title_label.grid(row = 0, column = 0, padx = 20, pady = (20, 10))
        self.instruction_label = ctk.CTkLabel(
            self, 
            text = "Share your feelings, and I'll create a Spotify playlist for you: ",
            font = ctk.CTkFont(size = 16)
        )
        self.instruction_label.grid(row = 1, column = 0, padx = 20, pady = (0, 10))
        self.text_input = ctk.CTkTextbox(
            self,
            height = 100,
            font = ctk.CTkFont(size = 14)
        )
        self.text_input.grid(row = 2, column = 0, padx = 20, pady = (0, 10), sticky = "ew")
        self.generate_button = ctk.CTkButton(
            self,
            text = "Generate Playlist",
            command = self.generate_playlist,
            font = ctk.CTkFont(size = 14)
        )
        self.generate_button.grid(row = 3, column = 0, padx = 20, pady = (0, 10))
        self.results_frame = ctk.CTkFrame(self)
        self.results_frame.grid(row = 4, column = 0, padx = 20, pady = (0, 20), sticky = "nsew")
        self.results_frame.grid_columnconfigure(0, weight = 1)
        self.emotion_label = ctk.CTkLabel(
            self.results_frame,
            text = "",
            font = ctk.CTkFont(size = 18, weight = "bold")
        )
        self.emotion_label.grid(row = 0, column = 0, padx = 20, pady = (20, 10))
        self.genres_label = ctk.CTkLabel(
            self.results_frame,
            text = "",
            font = ctk.CTkFont(size = 14)
        )
        self.genres_label.grid(row = 1, column = 0, padx = 20, pady = (0, 10))
        self.songs_frame = ctk.CTkScrollableFrame(
            self.results_frame,
            height = 400
        )
        self.songs_frame.grid(row = 2, column = 0, padx = 20, pady = (0, 20), sticky = "nsew")
        self.songs_frame.grid_columnconfigure(0, weight = 1)
        self.emotion_colors = {
            'joy': '#FFD700',
            'sadness': '#4169E1',
            'anger': '#DC143C',
            'calm': '#98FB98',
        }

    def create_song_frame(self, song: Dict, index: int) -> None:
        song_frame = ctk.CTkFrame(self.songs_frame)
        song_frame.grid(row = index, column = 0, padx = 10, pady = 5, sticky = "ew")
        song_frame.grid_columnconfigure(1, weight = 1)

        number_label = ctk.CTkLabel(
            song_frame,
            text = f"{index + 1}.",
            font = ctk.CTkFont(size = 14, weight = "bold"),
            width = 30
        )
        number_label.grid(row = 0, column = 0, padx = (10, 5), pady = 10)

        duration_mins = song['duration'] // 60
        duration_secs = song['duration'] % 60
        info_text = f"{song['name']}\n{song['artist']} • {song['album']}\n{duration_mins}:{duration_secs:02d}"
        info_label = ctk.CTkLabel(
            song_frame,
            text = info_text,
            font = ctk.CTkFont(size=14),
            justify = "left"
        )
        info_label.grid(row = 0, column = 1, padx = 5, pady = 10, sticky = "w")
        spotify_button = ctk.CTkButton(
            song_frame,
            text = "Open in Spotify",
            command = lambda url=song['url']: webbrowser.open(url),
            width = 120,
            font = ctk.CTkFont(size=12)
        )
        spotify_button.grid(row = 0, column = 2, padx = 10, pady = 10)

    def generate_playlist(self):
        for widget in self.songs_frame.winfo_children():
            widget.destroy()
        text = self.text_input.get("1.0", "end-1c")
        
        if len(text.split()) < 3:
            self.show_error("Please provide more context about your feelings for better analysis.")
            return
        emotion = self.emotion_analyzer.analyze_emotion(text)
        songs = self.spotify_manager.get_recommendations(emotion)
        self.emotion_label.configure(
            text = f"Detected Emotion: {emotion.title()}",
            text_color = self.emotion_colors.get(emotion)
        )
        genres = self.spotify_manager.emotion_mapping[emotion]['seed_genres']
        self.genres_label.configure(
            text = f"Recommended Genres: {', '.join(genres)}"
        )
        for i, song in enumerate(songs):
            self.create_song_frame(song, i)

    def show_error(self, message):
        error_window = ctk.CTkToplevel(self)
        error_window.title("Error")
        error_window.geometry("300x150")
        error_label = ctk.CTkLabel(
            error_window,
            text = message,
            wraplength = 250
        )
        error_label.pack(pady = 20)
        ok_button = ctk.CTkButton(
            error_window,
            text = "OK",
            command = error_window.destroy
        )
        ok_button.pack(pady = 20)

def main():
    load_dotenv('.env')
    CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
    CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
    spotify_manager = Spotify(CLIENT_ID, CLIENT_SECRET)
    app = Playlist(spotify_manager)
    app.mainloop()

if __name__ == "__main__":
    main()
