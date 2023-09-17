import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json

with open("./env.json", "tr", encoding="utf-8") as F:
    env = json.load(F)

def playlistGeneration(genres):  # Ex.) genres = ["hip-hop", "rainy-day"]
    if len(genres) > 5:
        genres = genres[:5]
    client_credentials_manager = SpotifyClientCredentials(client_id=env["CLIENT_ID"], client_secret=env["SECRET_KEY"])
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    results = sp.recommendations(seed_genres=genres, limit=20)

    songs = results['tracks']

    for idx, album in enumerate(songs):
        print(album['name'], "-", album['artists'][0]['name'])