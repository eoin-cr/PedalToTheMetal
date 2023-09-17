import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

def playlistGeneration(genres): # Ex.) genres = ["hip-hop", "rainy-day"]
	client_credentials_manager = SpotifyClientCredentials(client_id=os.getenv("CLIENT_ID"), client_secret=os.getenv("SECRET_KEY"))
	sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
	
	results = sp.recommendations(seed_genres=genres, limit=20)
	
	res = results['tracks']
	
	for item in res:
		print(item['name'], "-", item['artists'][0]['name'])

