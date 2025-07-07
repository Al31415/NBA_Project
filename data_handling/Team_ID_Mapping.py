from time import sleep
import requests.exceptions
from tqdm import tqdm
import csv
from pathlib import Path
from nba_api.stats.endpoints import boxscoresummaryv2
from functools import lru_cache
import pandas as pd

# Load play-by-play data
df = pd.read_csv("data/nba_regular_season_playbyplay_data.csv")

@lru_cache(maxsize=None)
def getTeamIDs(game_id: int) -> tuple[int, int]:
    """
    Return (home_team_id, visitor_team_id) from BoxScoreSummaryV2.
    Falls back to the old text-field heuristic if the NBA API call
    times out or rate-limits.
    """
    try:
        if len(str(game_id)) == 8:
            game_id = '00' + str(game_id)
        summary = boxscoresummaryv2.BoxScoreSummaryV2(game_id=str(game_id))
        header = summary.game_summary.get_data_frame().iloc[0]
        home_tid = int(header["HOME_TEAM_ID"])
        vis_tid = int(header["VISITOR_TEAM_ID"])
        return home_tid, vis_tid
    except Exception as e:
        print('Exception:', e)
        gdf = df[df.GAME_ID == game_id]  # fallback uses local rows
        home_id = gdf.loc[gdf.HOMEDESCRIPTION.notna(),
                         "PLAYER1_TEAM_ID"].mode().iat[0]
        vis_id = gdf.loc[gdf.VISITORDESCRIPTION.notna(),
                        "PLAYER1_TEAM_ID"].mode().iat[0]
        print(f"[WARN] NBA API failed for {game_id}: {e} – using heuristic")
        return int(home_id), int(vis_id)

def save_progress(team_ids: list, success_csv: Path):
    """Helper function to save current progress to CSV"""
    with open(success_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID'])
        writer.writeheader()
        writer.writerows(team_ids)

def get_team_ids_with_retry(game_id: int, current_delay: int) -> tuple[int | None, int | None, Exception | None]:
    """Helper function to get team IDs with exponential backoff"""
    try:
        sleep(current_delay)  # Wait before making request
        home_id, visitor_id = getTeamIDs(game_id)
        return home_id, visitor_id, None
    except (requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
        return None, None, e
    except Exception as e:
        print(f"Unexpected error for game {game_id}: {e}")
        return None, None, e

def main():
    # Get unique game IDs
    game_ids = df.GAME_ID.unique()
    team_ids = []

    # Configure timeout handling
    initial_delay = 2  # Start with 2 second delay
    max_delay = 60     # Maximum delay between retries
    max_retries = 5    # Maximum number of retry attempts
    backoff_factor = 2 # Multiply delay by this after each timeout

    # CSV file path for saving successful results
    success_csv = Path("team_ids.csv")

    # Load any previously saved results
    if success_csv.exists():
        with open(success_csv, 'r') as f:
            reader = csv.DictReader(f)
            team_ids = list(reader)
            # Convert string IDs back to int
            for row in team_ids:
                row['GAME_ID'] = int(row['GAME_ID'])
                row['HOME_TEAM_ID'] = int(row['HOME_TEAM_ID'])
                row['VISITOR_TEAM_ID'] = int(row['VISITOR_TEAM_ID'])
        
        # Get remaining games to process
        processed_games = set(row['GAME_ID'] for row in team_ids)
        game_ids = [gid for gid in game_ids if gid not in processed_games]
        print(f"Loaded {len(team_ids)} previously processed games")
        print(f"Remaining games to process: {len(game_ids)}")

    # Process remaining games
    for game_id in tqdm(game_ids, desc="Processing games"):
        current_delay = initial_delay
        success = False
        
        for attempt in range(max_retries):
            home_id, visitor_id, error = get_team_ids_with_retry(game_id, current_delay)
            
            if error is None:  # Success
                team_ids.append({
                    'GAME_ID': game_id,
                    'HOME_TEAM_ID': home_id,
                    'VISITOR_TEAM_ID': visitor_id
                })
                # Save progress after each successful game
                save_progress(team_ids, success_csv)
                success = True
                break
            
            # If timeout occurred, increase delay for next attempt
            if isinstance(error, (requests.exceptions.Timeout, requests.exceptions.ReadTimeout)):
                current_delay = min(current_delay * backoff_factor, max_delay)
                print(f"Timeout for game {game_id}, attempt {attempt+1}/{max_retries}. "
                      f"Retrying in {current_delay} seconds...")
            else:
                break  # Don't retry on non-timeout errors
        
        if not success:
            print(f"Failed to get team IDs for game {game_id} after all attempts")

    # Convert final results to dataframe
    team_ids_df = pd.DataFrame(team_ids)
    print(f"\nFinal results: Successfully retrieved {len(team_ids)} out of {len(df.GAME_ID.unique())} games")
    print(team_ids_df)

if __name__ == "__main__":
    main()
