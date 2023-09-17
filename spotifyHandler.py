import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import json

def getGenres(emoNum):
    genres = {
        0: ['acoustic', 'disco', 'funk', 'pop', 'reggae'],
        1: ['blues', 'singer-songwriter', 'emo', 'acoustic', 'indie-pop'],
        2: ['ambient', 'chill', 'bossanova', 'jazz', 'acoustic'],
        3: ['heavy-metal', 'hardcore', 'punk-rock', 'grindcore', 'rock'],
        4: []
    }
    return genres[emoNum]

def playlistGeneration(genres, amountSongs):  # Ex.) genres = ["hip-hop", "rainy-day"]
    with open("./env.json", "tr", encoding="utf-8") as F:
        env = json.load(F)
    if len(genres) > 5:
        genres = genres[:5]
    if amountSongs < 1:
        amountSongs = 1
    elif amountSongs > 100:
        amountSongs = 100
    client_credentials_manager = SpotifyClientCredentials(client_id=env["CLIENT_ID"], client_secret=env["SECRET_KEY"])
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    results = sp.recommendations(seed_genres=genres, limit=amountSongs)

    songs = results['tracks']

    songList = []

    for song in songs:
        songList.append(" ".join([song['name'], "-", song['artists'][0]['name']]))
    
    return songList