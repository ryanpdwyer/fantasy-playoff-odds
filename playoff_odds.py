import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import requests
from collections import OrderedDict, defaultdict, Counter

def groupby(d):
    res = defaultdict(list)
    for key, val in sorted(d.items()):
        res[val].append(key)
    return res

def get_opponents(d):
    x = np.array(list(groupby(d).values())) - 1 # Subtract 1 so we have indices
    out = np.zeros(x.size, dtype=int)
    for i in range(x.shape[0]):
        out[x[i, 0]] = x[i, 1]
        out[x[i, 1]] = x[i, 0]
    return out 

def get_id(url_or_id):
    possible_ids = [x for x in url_or_id.split("/") if x.isdigit()]
    return possible_ids[0]

@st.cache
def get_league(id):
    r = requests.get("https://api.sleeper.app/v1/league/"+id)
    return r.json()

@st.cache
def get_users(id):
    return requests.get("https://api.sleeper.app/v1/league/{}/users".format(id)).json()

@st.cache
def get_rosters(id):
    return requests.get("https://api.sleeper.app/v1/league/{}/rosters".format(id)).json()

@st.cache
def get_matchups(id, week):
    return requests.get("https://api.sleeper.app/v1/league/{}/matchups/{}".format(id, week)).json()

def filter_rosters(rosters):
    return {r['roster_id']: dict(wins=r['settings']["wins"], losses=r['settings']['losses'], ties=r['settings']['ties'],
            pts=(int(r['settings']['fpts'])+int(r['settings']['fpts_decimal'])/100),
            division=r['settings']['division']) for r in rosters}

def intx(x):
    return int(x) if x != '' else x

def rotisserie(pts):
    rotis = np.argsort(pts)
    n_games, n_teams = pts.shape
    rr = np.zeros_like(pts)
    for i in range(n_games):
        rr[i, rotis[i]] = np.arange(n_teams)
    
    return rr

@st.cache
def simulate_remaining_weeks(games_left, n_teams, N, pts_regress, stdev, future_opponents):
    pts_unplayed = np.zeros((games_left, n_teams, N))
    wins_unplayed = np.zeros((games_left, n_teams, N), dtype=bool)

    for i in range(games_left):
        pts_unplayed[i] = rng.normal(scale=stdev,
                        loc=pts_regress.reshape((-1, 1)), size=(n_teams, N))
        wins_unplayed[i] = pts_unplayed[i] > pts_unplayed[i][future_opponents[i]]
    
    return pts_unplayed, wins_unplayed

@st.cache
def getSeedsArray(n_teams, N, overall_pts, overall_wins):
    seeds = np.zeros((N, n_teams))
    for i, (pts_, wins_) in enumerate(zip(overall_pts, overall_wins)):
        inds_ = np.lexsort((pts_, wins_))[::-1]
        seeds[i, inds_] = np.arange(n_teams)+1
    
    return seeds

def makeSeeds(seeds, teams, n_playoff_teams):
    seedCounts = {team: Counter(s) for team, s in zip(teams, seeds.T)}
    dfSeeds = pd.DataFrame.from_dict(seedCounts, orient='index')
    dfSeedsPercent = dfSeeds.fillna(0) / len(seeds) * 100
    dfSeedsPercent['avgSeed'] = dfSeedsPercent[np.arange(1, len(teams)+1)].values @ np.arange(1, len(teams)+1) / 100
    dfSeedsPercent.sort_values('avgSeed', ascending=True, inplace=True)
    dfS = dfSeedsPercent[np.arange(1, n_playoff_teams+1)]
    dfS['Any'] = dfS.values.sum(axis=1)
    inds_out = ['Any']
    inds_out.extend(np.arange(1, n_playoff_teams+1))
    return dfS.loc[:, inds_out]


st.title("Fantasy Football Playoff Odds")
st.write("Enter your Sleeper league url or ID below:")
url = st.text_input("Sleeper URL or ID")
season_weeks = intx(st.text_input("Regular season weeks"))
st.write("TO DO: Add options to choose exactly how playoff teams are chosen and seeded here")


if url != "" and season_weeks != '':
    id = get_id(url)
    rjson = get_league(id)
    users = get_users(id)
    rosters = get_rosters(id)

    owners = OrderedDict((user['user_id'], user['display_name']) for user in users)
    # teams = list(owners.values())
    roster_owner = OrderedDict((r['roster_id'], r['owner_id']) for r in rosters)
    roster_display = OrderedDict((key, owners[val]) for key, val in roster_owner.items())
    teams_canonical = np.array(list(roster_display.values()))
    n_teams = len(teams_canonical)

    n_playoff_teams = rjson['settings']['playoff_teams']
    st.header("{} League".format(rjson['name']))
    st.write("Teams: {}".format(rjson['settings']['num_teams']))
    st.write("Playoff teams: {}".format(rjson['settings']['playoff_teams']))



    rr = filter_rosters(rosters)
    rr_df = pd.DataFrame.from_dict({owners[roster_owner[key]]: val for key, val in rr.items()},
                    orient='index')
    st.write("Current Standings")
    st.dataframe(rr_df)

    games = (rr_df['wins'].iloc[0] + rr_df['losses'].iloc[0] + rr_df['ties'].iloc[0])
    week = games + 1
    # Contains wins, losses, points, almost everything
    # we need on a week-by-week basis

    # rr
    ptsAvg = rr_df['pts'].mean() / games # Average pts / week...
    pts_regress = rr_df['pts'].values/games * 0.5 + ptsAvg * 0.5

    matchups = [get_matchups(id, w) for w in range(1, season_weeks+1)]
    
    # Using roster_id as the canonical ordering
    pts_dict = [{x['roster_id']: x['points'] for x in y} for y in matchups]
    weekly_matchups = [{x['roster_id']: x['matchup_id'] for x in y} for y in matchups]
    weekly_opponents = np.array([get_opponents(w) for w in weekly_matchups])
    pts = np.array([list(p.values()) for p in pts_dict])

    pts_played = pts[:games]
    wins_played = np.array([pts_row > pts_row[opp] for pts_row, opp in zip(pts_played, weekly_opponents[:games])])
    rotis = rotisserie(pts_played)
    
    ptsAvg = pts_played.mean()
    pts_regress = pts_played.mean(axis=0)*0.5 + ptsAvg*0.5

    future_matchups = [np.array(list(groupby(m).values()))-1 for m in weekly_matchups[games:]]
    future_opponents = weekly_opponents[games:]
    stdev = pts_played.std()*1.03 # Add a little extra uncertainty
    # Unplayed games:
    games_left = season_weeks - games


    N = 5000
    rng = np.random.default_rng()

    pts_unplayed, wins_unplayed = simulate_remaining_weeks(
            games_left, n_teams, N, pts_regress, stdev, future_opponents)

    overall_pts = (pts_played.sum(axis=0).reshape((-1,1)) + pts_unplayed.sum(axis=0)).T
    overall_wins = (wins_played.sum(axis=0).reshape((-1, 1)) + wins_unplayed.sum(axis=0)).T
    seeds = getSeedsArray(n_teams, N, overall_pts, overall_wins)

    inds = np.arange(N, dtype=int)


    st.write("Projected outcomes for each game:")
    slots = [st.empty() for _ in range(games_left*2)]

    st.write("Use the buttons to see how playoff chances change depending on the results of each game.")

    # Make buttons:
    n_cols = int(n_teams / 2)
    buttons = []
    for j, match_ in enumerate(future_matchups):
        st.write("Week {}".format(week+j))
        cols = st.beta_columns(n_cols) 
        weekly_buttons = []
        for i, x in enumerate(match_):
            weekly_buttons.append(cols[i].radio(label='Winner',
                options=['Any', teams_canonical[x[0]], teams_canonical[x[1]]]
            )
        )
        buttons.append(weekly_buttons)
    

    # Filter simulations:
    for i, weekly_buttons in enumerate(buttons):
        for button in weekly_buttons:
            if button != 'Any':
                inds_match = np.nonzero(wins_unplayed[i, list(teams_canonical).index(button), :])
                inds = np.intersect1d(inds, inds_match, assume_unique=True)

    # Print weekly matchup projections
    for i, match_ in enumerate(future_matchups):
        slots[i*2].write("Week {}".format(week+i))
        dfWeek = pd.DataFrame(np.c_[
                wins_unplayed[i, :, inds].mean(axis=0)*100,
                pts_unplayed[i, :, inds].mean(axis=0)]
                , index=teams_canonical, columns=['Win Prob', 'Proj. Pts'])
        slots[i*2+1].dataframe(dfWeek.iloc[match_.flatten()].style.format("{:.1f}")\
        .background_gradient(cmap='RdBu_r', low=1.25, high=1.25, axis=0, subset=['Win Prob'])
        )


    avgWins = np.round(np.mean(overall_wins[inds], axis=0),2)
    avgPts = np.round(np.mean(overall_pts[inds], axis=0), 1)
    makePlayoffs = np.sum(seeds[inds] <= n_playoff_teams, axis=0)

    dfSS = makeSeeds(seeds[inds], teams_canonical, n_playoff_teams)

    dfAvg = pd.DataFrame(data=np.c_[avgWins, avgPts, makePlayoffs/len(inds)*100],
                    index=teams_canonical,
                    columns=['avgWins', 'avgPts', 'playoffPercent'])
    
    dfAvg = dfAvg.loc[dfSS.index, :]
    
    st.write("Playoff Seeding Chances")
    st.dataframe(dfSS.style.format("{:.1f}")\
        .background_gradient(cmap='Greens', low=0.0, high=0.7))

    
    st.write("Standings")
    dfAvg.sort_values('avgWins', inplace=True, ascending=False)
    st.dataframe(dfAvg.style.format("{:.1f}")\
        .background_gradient(cmap='RdBu_r', low=1, high=1, axis=0))



    # Choose playoff teams...
    # Bracket...
    # Byes?
