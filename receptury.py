import pandas as pd
## receptures spotify
recipe_spotify_names=["Spotify Streams",
                      "Spotify Playlist Count",
                      "Spotify Popularity",
                      "Spotify Popularity"]
recipe_spotify_col=pd.Index(recipe_spotify_names)

## receptures youtube
recipe_youtube_names=["YouTube Views",
                      "YouTube Likes",]
recipe_youtube_col=pd.Index(recipe_youtube_names)

## receptures Tiktok
recipe_tiktok_names=["TikTok Posts",
                     "TikTok Likes",
                     "TikTok Views"]
recipe_tiktok_col=pd.Index(recipe_tiktok_names)



# to be finished 
recipe_all = recipe_spotify_col.union(recipe_youtube_col).union(recipe_tiktok_col)



"""
Track                            0
Album Name                       0
Artist                           0
Release Date                     0
ISRC                             0
All Time Rank                    0
Track Score                      0
Spotify Streams                108
Spotify Playlist Count          65
Spotify Playlist Reach          67
Spotify Popularity             799
YouTube Views                  303
YouTube Likes                  310
TikTok Posts                  1168
TikTok Likes                   975
TikTok Views                   976
YouTube Playlist Reach        1004
Apple Music Playlist Count     556
AirPlay Spins                  493
Deezer Playlist Count          916
Deezer Playlist Reach          923
Amazon Playlist Count         1050
Explicit Track                   0

"""
