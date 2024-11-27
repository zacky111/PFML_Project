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

# all features summed
recipe_all = recipe_spotify_col.union(recipe_youtube_col).union(recipe_tiktok_col)


