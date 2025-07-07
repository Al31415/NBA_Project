from nba_api.stats.endpoints import LeagueGameFinder, PlayByPlayV2
import pandas as pd
import time
from requests.exceptions import ReadTimeout, Timeout
import backoff
from pathlib import Path

SEASON = '2024-25'
SEASON_TYPE = 'Regular Season'
TIMEOUT = 60
SLEEP_TIME = 2
OUTPUT_PATH = Path("data/nba_regular_season_playbyplay_data.csv")

@backoff.on_exception(backoff.expo, (ReadTimeout, Timeout), max_tries=5)
def make_api_call(func, *args, **kwargs):
    return func(*args, **kwargs)

def get_season_games():
    print(f"Retrieving {SEASON_TYPE} game list...")
    gamefinder = LeagueGameFinder(
        season_nullable=SEASON,
        season_type_nullable=SEASON_TYPE, 
        timeout=TIMEOUT
    )
    games_df = make_api_call(gamefinder.get_data_frames)[0]
    print("Sample of retrieved games:")
    print(games_df.head())
    return games_df['GAME_ID'].unique().tolist()

def get_play_by_play(game_id):
    print(f"Fetching play-by-play data for game {game_id}...")
    try:
        pbp = PlayByPlayV2(game_id=game_id, timeout=TIMEOUT)
        data = make_api_call(pbp.get_data_frames)[0]
        time.sleep(SLEEP_TIME)
        return data
    except (ReadTimeout, Timeout) as e:
        print(f"Timeout error for game {game_id}. Skipping...")
        return None

def main():
    try:
        game_ids = get_season_games()
        print(f"\nFound {len(game_ids)} {SEASON_TYPE} games in season {SEASON}.")

        playbyplay_data = []
        for game_id in game_ids:
            if data := get_play_by_play(game_id):
                playbyplay_data.append(data)

        if playbyplay_data:
            combined_pbp_df = pd.concat(playbyplay_data, ignore_index=True)
            print("\nCombined play-by-play data sample:")
            print(combined_pbp_df.head())

            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            combined_pbp_df.to_csv(OUTPUT_PATH, index=False)
            print(f"\nData saved to {OUTPUT_PATH}")
        else:
            print("\nNo play-by-play data was successfully retrieved.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
