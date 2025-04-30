import pandas as pd 
import os
import numpy as np

""" 
Data obtained from https://www.kaggle.com/datasets/guillemservera/tennis
"""

DATA_PATH = os.path.join("data", "kaggle_challenge")
RAW_DATA_PATH = os.path.join(DATA_PATH, "raw")
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed")
matches = []
for file in os.listdir(RAW_DATA_PATH):
    if "matches" in file:
        matches.append(pd.read_csv(os.path.join(RAW_DATA_PATH, file)))
matches = pd.concat(matches)
matches = matches[~matches["tourney_name"].str.contains("Davis Cup")]
matches[["year", "tournament_id"]] = matches["tourney_id"].str.split("-", expand=True)
matches["date"] = pd.to_datetime(matches["tourney_date"], format="%Y%m%d")
matches["winner_birth_date"] = matches["date"] - pd.to_timedelta(matches["winner_age"] * 365.25, unit="D")
matches["loser_birth_date"] = matches["date"] - pd.to_timedelta(matches["loser_age"] * 365.25, unit="D").copy()
matches["loser_id"] = matches["loser_id"].replace({212021 :211776})
player_ids = list(set(matches["winner_id"].unique()) | set(matches["loser_id"].unique()))


players = pd.read_csv(os.path.join(RAW_DATA_PATH, "atp_players.csv"))
ranking = pd.concat([
    pd.read_csv(os.path.join(RAW_DATA_PATH, "atp_rankings_10s.csv")),
    pd.read_csv(os.path.join(RAW_DATA_PATH, "atp_rankings_20s.csv")),
])

# tournaments table
tournaments = matches.groupby("tournament_id")[["surface", "draw_size", "tourney_level"]].first().reset_index()

# players_table
winners = matches.groupby("winner_id")["winner_birth_date"].median()
losers = matches.groupby("loser_id")["loser_birth_date"].median()

birth_dates = pd.concat([winners, losers], axis=1)
birth_dates["birth_date"] = birth_dates.mean(axis=1, skipna=True)
players = players.merge(birth_dates[["birth_date"]], left_on="player_id", right_index=True, how="left")
players = players[players["player_id"].isin(player_ids)]
players = players.drop(columns=["name_first", "name_last", "wikidata_id", "dob"]).reset_index(drop=True)

# ranking table
ranking["ranking_date"] = pd.to_datetime(ranking["ranking_date"], format="%Y%m%d")
ranking = ranking[ranking["player"].isin(player_ids)].rename(columns={"player": "player_id"})
ranking = ranking.drop(columns=["points"])
ranking = ranking[ranking["ranking_date"] >= pd.to_datetime("2022-01-01")]


# matches table
matches = matches[
    [
        "winner_id",
        "loser_id",
        "tournament_id",
        "year",
        "date",
        "winner_seed",
        "loser_seed",
        "score",
        "best_of",
        "minutes",
        "w_ace",
        "w_df",
        "w_svpt",
        "w_1stIn",
        "w_1stWon",
        "w_2ndWon",
        "w_SvGms",
        "w_bpSaved",
        "w_bpFaced",
        "l_ace",
        "l_df",
        "l_svpt",
        "l_1stIn",
        "l_1stWon",
        "l_2ndWon",
        "l_SvGms",
        "l_bpSaved",
        "l_bpFaced",
    ]
]


test_tournaments = ["560", "6242", "0422"]
test_start_date = pd.to_datetime("2023-08-14")

test_matches = matches[
    (matches["tournament_id"].isin(test_tournaments) & (matches["year"] ==  "2023"))
].copy()

test_matches["player_1"] = np.where(np.random.rand(len(test_matches)) > 0.5, test_matches["winner_id"], test_matches["loser_id"])
test_matches["player_2"] = np.where(test_matches["player_1"] == test_matches["winner_id"], test_matches["loser_id"], test_matches["winner_id"])
test_matches["player_1_seed"] = np.where(test_matches["player_1"] == test_matches["winner_id"], test_matches["winner_seed"], test_matches["loser_seed"])
test_matches["player_2_seed"] = np.where(test_matches["player_2"] == test_matches["winner_id"], test_matches["winner_seed"], test_matches["loser_seed"])
test_matches["player_1_won"] = np.where(test_matches["player_1"] == test_matches["winner_id"], 1, 0)

test_matches = test_matches[[
    "player_1",
    "player_2",
    "tournament_id",
    "year",
    "player_1_seed",
    "player_2_seed",
    "best_of",
    "player_1_won"
]]
train_matches = matches[matches["date"]< test_start_date].copy()
train_matches["player_1"] = np.where(np.random.rand(len(train_matches)) > 0.5, train_matches["winner_id"], train_matches["loser_id"])
train_matches["player_2"] = np.where(train_matches["player_1"] == train_matches["winner_id"], train_matches["loser_id"], train_matches["winner_id"])
train_matches["player_1_seed"] = np.where(train_matches["player_1"] == train_matches["winner_id"], train_matches["winner_seed"], train_matches["loser_seed"])
train_matches["player_2_seed"] = np.where(train_matches["player_2"] == train_matches["winner_id"], train_matches["winner_seed"], train_matches["loser_seed"])
train_matches["player_1_won"] = np.where(train_matches["player_1"] == train_matches["winner_id"], 1, 0)

ranking = ranking[ranking["ranking_date"] < test_start_date].copy()

# save processed files
train_matches.to_parquet(os.path.join(PROCESSED_DATA_PATH, "train_matches.parquet"))
test_matches.to_parquet(os.path.join(PROCESSED_DATA_PATH, "test_matches.parquet"))
players.to_parquet(os.path.join(PROCESSED_DATA_PATH, "players.parquet"))
ranking.to_parquet(os.path.join(PROCESSED_DATA_PATH, "rankings.parquet"))
tournaments.to_parquet(os.path.join(PROCESSED_DATA_PATH, "tournaments.parquet"))
