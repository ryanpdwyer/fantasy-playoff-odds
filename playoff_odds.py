import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import requests
import datetime
from collections import OrderedDict, defaultdict, Counter

currentYear = datetime.datetime.today().year

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

def get_opponents_no_off_one(d):
    x = np.array(list(groupby(d).values())) # Subtract 1 so we have indices
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
def get_league_espn(id, swid=None, espn_s2=None):
    if swid is not None:
        cookies = dict(swid=swid, espn_s2=espn_s2)
    else:
        cookies = None
    if swid is not None:
        return requests.get("https://fantasy.espn.com/apis/v3/games/ffl/seasons/2020/segments/0/leagues/{}".format(id), cookies=cookies).json()

@st.cache
def get_league_data_espn(id, swid=None, espn_s2=None):
    if swid is not None:
        cookies = dict(swid=swid, espn_s2=espn_s2)
    else:
        cookies = None
    return requests.get("https://fantasy.espn.com/apis/v3/games/ffl/seasons/2020/segments/0/leagues/{}".format(id), params=dict(view="mMatchup"), cookies=cookies).json()

@st.cache
def get_league_members_espn(id, swid=None, espn_s2=None):
    if swid is not None:
        cookies = dict(swid=swid, espn_s2=espn_s2)
    else:
        cookies = None
    return requests.get("https://fantasy.espn.com/apis/v3/games/ffl/seasons/2020/segments/0/leagues/{}".format(id), params=dict(view="mTeam"), cookies=cookies).json()


@st.cache
def get_league_espn_hist(id, year):
    if year != currentYear:
        return requests.get("https://fantasy.espn.com/apis/v3/games/ffl/leagueHistory/{}".format(id), params=dict(seasonId=year)).json()[0]
    else:
        return requests.get("https://fantasy.espn.com/apis/v3/games/ffl/leagueHistory/{}".format(id), params=dict(seasonId=year)).json()

@st.cache
def get_league_data_espn_hist(id, year):
    if year != currentYear:
        return requests.get("https://fantasy.espn.com/apis/v3/games/ffl/leagueHistory/{}".format(id), params=dict(seasonId=year, view="mMatchup")).json()[0]
    else:
        return requests.get("https://fantasy.espn.com/apis/v3/games/ffl/leagueHistory/{}".format(id), params=dict(seasonId=year, view="mMatchup")).json()

@st.cache
def get_league_members_espn_hist(id, year):
    req = requests.get("https://fantasy.espn.com/apis/v3/games/ffl/leagueHistory/{}".format(id), params=dict(seasonId=year, view="mTeam")).json()
    if year != currentYear:
        return req[0]
    else:
        return req

@st.cache
def get_matchups(id, week):
    return requests.get("https://api.sleeper.app/v1/league/{}/matchups/{}".format(id, week)).json()

def filter_rosters(rosters):
    if 'division' in rosters[0]['settings']:
        return {r['roster_id']: dict(wins=r['settings']["wins"], losses=r['settings']['losses'], ties=r['settings']['ties'],
            pts=(int(r['settings']['fpts'])+int(r['settings']['fpts_decimal'])/100),
            division=r['settings']['division']) for r in rosters}
    else:
        return {r['roster_id']: dict(wins=r['settings']["wins"], losses=r['settings']['losses'], ties=r['settings']['ties'],
            pts=(int(r['settings']['fpts'])+int(r['settings']['fpts_decimal'])/100),
            division=1) for r in rosters}

def intx(x):
    return int(x) if x != '' else x

@st.cache
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
    rotis_unplayed = np.zeros((games_left, n_teams, N), dtype=int)

    for i in range(games_left):
        pts_unplayed[i] = rng.normal(scale=stdev,
                        loc=pts_regress.reshape((-1, 1)), size=(n_teams, N))
        rotis_unplayed[i] = rotisserie(pts_unplayed[i].T).T
        wins_unplayed[i] = pts_unplayed[i] > pts_unplayed[i][future_opponents[i]]

    
    return pts_unplayed, wins_unplayed, rotis_unplayed

@st.cache
def getSeedsArray(n_teams, N, overall_pts, overall_wins, divisions):
    seeds = np.zeros((N, n_teams), dtype=int)
    for i, (pts_, wins_) in enumerate(zip(overall_pts, overall_wins)):
        inds_ = np.lexsort((pts_, wins_))[::-1]
        seeds[i, inds_] = np.arange(n_teams)+1
    
    return seeds

@st.cache
def getSeedsArrayDivisions(n_teams, N, overall_pts, overall_wins, divisions):
    seeds = np.zeros((N, n_teams), dtype=int)
    n_divisions = max(divisions)
    division_dict = groupby(dict(zip(np.arange(len(divisions), dtype=int), divisions)))
    for i, (pts_, wins_) in enumerate(zip(overall_pts, overall_wins)):
        division_winners = np.array([div[np.lexsort((pts_[div], wins_[div]))[-1]] for div in division_dict.values()], dtype=int)
        not_winners = np.array([x for x in range(n_teams) if x not in division_winners], dtype=int)
        
        division_winners_sorted = np.lexsort((pts_[division_winners], wins_[division_winners]))[::-1] # This is the correct overall ordering...
        not_winners_sorted = np.lexsort((pts_[not_winners], wins_[not_winners]))[::-1]

        inds_ = np.r_[division_winners[division_winners_sorted], not_winners[not_winners_sorted]].flatten()

        seeds[i, inds_] = np.arange(n_teams)+1
    
    return seeds

def makeSeeds(seeds, teams, n_playoff_teams):
    seedCounts = {team: Counter(s) for team, s in zip(teams, seeds.T)}
    dfSeeds = pd.DataFrame.from_dict(seedCounts, orient='index')
    dfSeedsPercent = dfSeeds.fillna(0) / len(seeds) * 100
    dfSeedsPercent['avgSeed'] = dfSeedsPercent[np.arange(1, len(teams)+1)].values @ np.arange(1, len(teams)+1) / 100
    dfSeedsPercent.sort_values('avgSeed', ascending=True, inplace=True)
    dfS = dfSeedsPercent[np.arange(1, n_playoff_teams+1)]
    dfSeedsPercent['Any'] = dfS.values.sum(axis=1)
    inds_out = ['Any']
    inds_out.extend(np.arange(1, n_playoff_teams+1))
    return dfSeedsPercent.loc[:, inds_out]

playoff_options = {"record_points": ("By record, using total points as a tiebreaker", getSeedsArray),
        "record_division_points": ("By record (division winners get top seeds), using total points as a tiebreaker", getSeedsArrayDivisions)}
class PlayoffFormats:
    def __init__(self, d=playoff_options):
        self.d = d
        self.options = list(d.keys())
    
    def __call__(self, opt):
        return self.d[opt][0]
    
    def getSeeds(self, opt):
        return self.d[opt][1]
    

playoff_formats = PlayoffFormats(playoff_options)


def playoffs_fast(pts, seedsRow, n_playoff_teams, season_weeks, std):
    seedsArray = seedsRow
    playoff_inds = np.where(seedsArray <= n_playoff_teams)[0]

    # Just the playoff teams
    ppts = pts[playoff_inds]
    pseedsRow = list(seedsArray[playoff_inds])

    # Regress expected points during the playoffs heavily back to the mean
    ppts_avg = (pts.mean()* 0.5 + ppts*0.5)/season_weeks

    if n_playoff_teams == 6:
        matchups = [{'Q1': (3, 6), 'Q2': (4, 5)}, {'S1': (1, 'Q2_W'), 'S2': (2, 'Q1_W')}, {"Championship": ('S1_W', 'S2_W'), "Third": ("S1_L", "S2_L")}]
    elif n_playoff_teams == 4:
        matchups =  [{'S1': (1, 4), 'S2': (2, 3)}, {"Championship": ('S1_W', 'S2_W'), "Third": ("S1_L", "S2_L")}]
    elif n_playoff_teams == 8:
        matchups = [{'Q1': (1, 8), 'Q2': (4, 5), 'Q3': (2, 7), 'Q4': (3, 6)}, {'S1': ('Q1_W', 'Q2_W'), 'S2': ('Q3_W', 'Q4_W')},
                    {"Championship": ('S1_W', 'S2_W'), "Third": ("S1_L", "S2_L")}]
    else:
        raise ValueError("Only 4, 6, or 8 playoff teams supported.")

    playoffPts = np.random.randn(n_playoff_teams, len(matchups))*std + ppts_avg.reshape(-1,1)

    results = {i: pseedsRow.index(i) for i in range(1, n_playoff_teams+1)} # Match seed to index...
    for j, matchup_weeks in enumerate(matchups):
        for key, val in matchup_weeks.items():
            s1, s2 = val
            r1 = results[s1]
            r2 = results[s2]
            if playoffPts[r1, j] > playoffPts[r2, j]:
                results[key+'_W'] = r1
                results[key+'_L'] = r2
            else:
                results[key+'_W'] = r2
                results[key+'_L'] = r1

    yyyPlayoffs = np.ones_like(ppts_avg, dtype=int)*5
    yyyPlayoffs[results['Championship_W']] = 1
    yyyPlayoffs[results['Championship_L']] = 2
    yyyPlayoffs[results['Third_W']] = 3
    yyyPlayoffs[results['Third_L']] = 4


    yyy = np.ones_like(seedsArray, dtype=int)*(n_playoff_teams+1)
    yyy[playoff_inds] = yyyPlayoffs

    return yyy

@st.cache
def makeAllPlayoffResults(overall_pts, seeds, n_playoff_teams, season_weeks, stdev):
    return np.array(
    [playoffs_fast(p,s, n_playoff_teams, season_weeks, stdev) for p, s in zip(overall_pts, seeds)])


def analyzePlayoffResults(playoffResults, teams):
    winsCount = {team: Counter(play) for team, play in zip(teams,
                                                    playoffResults.T)}
    dfPR = pd.DataFrame.from_dict(winsCount, orient='index').fillna(0)/len(playoffResults)*100
    dfPR.sort_values(1, ascending=False, inplace=True)
    dfPRO = dfPR.loc[:, [1, 2, 3]].rename(columns={1: "Champion", 2: "2nd", 3: "3rd" })
    dfPRO['Make Final'] = dfPRO["Champion"] + dfPRO["2nd"]
    dfPRO.round(1)
    try:
        del dfPRO['avgSeed']
    except:
        pass
    return dfPRO

# st.write('''<div style='height: 200px; background-image: url("https://lh5.googleusercontent.com/GAmGcsk5dPuofSN8eXAYinlFC8lxIQdvynYW7CgyoGRhOy_em36mmJ1pNtpchiz7Es-q86OUjA=w16383");'></div>''', unsafe_allow_html=True)

st.title("Fantasy Football Playoff Odds")
league_website = st.selectbox("League website", options=["Sleeper" , "ESPN"])

if league_website == 'ESPN':
    st.write("""If your ESPN league is completely private, do one of the following:
    
1) Ask your league manager to make your league viewable to the public (https://support.espn.com/hc/en-us/articles/360000064451-Making-a-Private-League-Viewable-to-the-Public), or
    
2) Enter swid and ESPN_S2 cookies (in Chrome, paste `chrome://settings/cookies/detail?site=espn.com` into the url bar and copy and paste the Content under the SWID and espn_s2 entries) here.
    """)
    swid = st.text_input("SWID")
    espn_s2 = st.text_input("espn_s2")
else:
    swid = None
    espn_s2 = None

url = st.text_input("League ID")
season_weeks = st.number_input("Regular season weeks", value=13)
playoff_format = st.selectbox("How are playoff seeds determined?",
                    options=playoff_formats.options,
                    format_func=playoff_formats,
                    index=0)
game_vs_league_median = st.checkbox("Extra game vs league median?")
historical = st.checkbox("See historical projections")

historical_games = 0
if historical:
    historical_games = st.number_input("Games to use", value=11)




if url != "" and season_weeks != '' and league_website == 'Sleeper':
    id = get_id(url)

    # Sleeper specific
    rjson = get_league(id)
    users = get_users(id)
    rosters = get_rosters(id)

    owners = OrderedDict((user['user_id'], user['display_name']) for user in users)
    # teams = list(owners.values())
    roster_owner = OrderedDict((r['roster_id'], r['owner_id']) for r in rosters)
    roster_display = OrderedDict((key, owners[val]) for key, val in roster_owner.items())

    ## Everything needs this list of canonical teams, number of teams
    ## Number of playoff teams
    teams_canonical = np.array(list(roster_display.values()))
    n_teams = len(teams_canonical)
    

    n_playoff_teams = rjson['settings']['playoff_teams']
    st.write("League name: {}".format(rjson['name']))
    # st.write("Teams: {}".format(rjson['settings']['num_teams']))
    st.write("Playoff teams: {}".format(rjson['settings']['playoff_teams']))


    # Sleeper specific
    rr = filter_rosters(rosters)
    rr_df = pd.DataFrame.from_dict({owners[roster_owner[key]]: val for key, val in rr.items()},
                    orient='index')
    
    divisions = np.zeros(n_teams, dtype=int)
    for id_, val in rr.items():
        divisions[id_-1] = val['division']


    games = (rr_df['wins'].iloc[0] + rr_df['losses'].iloc[0] + rr_df['ties'].iloc[0])

    # Contains wins, losses, points, almost everything
    # we need on a week-by-week basis

    matchups = [get_matchups(id, w) for w in range(1, season_weeks+1)]
    

    # Using roster_id as the canonical ordering
    pts_dict = [{x['roster_id']: x['points'] for x in y} for y in matchups]
    weekly_matchups = [{x['roster_id']: x['matchup_id'] for x in y} for y in matchups]
    weekly_opponents = np.array([get_opponents(w) for w in weekly_matchups])
    pts = np.array([list(p.values()) for p in pts_dict], dtype=float)

    # Once I have the pts array, and the matchups list,
    # I can generate everything else from scratch! ----



    # This ends the sleeper specific part of the code

def make_opp_arr(x):
    x = np.array(x)
    out = np.zeros(x.size, dtype=int)
    for i in range(x.shape[0]):
        out[x[i, 0]] = x[i, 1]
        out[x[i, 1]] = x[i, 0]
    return out 


if url != "" and season_weeks != '' and league_website == 'ESPN':
    id = get_id(url)

    st.write(id)
    league = get_league_espn(id, swid, espn_s2)
    league_data = get_league_data_espn(id, swid, espn_s2)
    member_info = get_league_members_espn(id, swid, espn_s2)

    # Maps id to full name...
    member_dict = {x['id']: x['firstName']+x['lastName'] for x in member_info['members']}
    

    od = OrderedDict((x['id'], x['abbrev']) for x in league['teams']) # Add more here...
    ## Everything needs this list of canonical teams, number of teams
    ## Number of playoff teams
    teams_canonical = np.array(list(od.values()))
    n_teams = len(teams_canonical)

    divisions = np.zeros(n_teams, dtype=int)
    for team in member_info['teams']:
        divisions[team['id']-1] = team['divisionId']

    pts = np.zeros((season_weeks, n_teams))
    weekly_matchups_dict = defaultdict(list)
    for game in league_data['schedule']:
        week = game['matchupPeriodId']
        if week <= season_weeks:
            pts[week-1, game['away']['teamId']-1] = game['away']['totalPoints']
            pts[week-1, game['home']['teamId']-1] = game['home']['totalPoints']
            weekly_matchups_dict[week].append((game['away']['teamId']-1, game['home']['teamId']-1))
    
    games = league['status']['currentMatchupPeriod'] - 1

    weekly_matchups = [weekly_matchups_dict[i] for i in range(1, season_weeks+1)]
    weekly_opponents = np.array([make_opp_arr(x) for x in weekly_matchups])


    

    n_playoff_teams = 6 # Where is this in the ESPN API? TODO
    st.write("League name: {}".format(league['settings']['name']))
    # st.write("Teams: {}".format(n_teams))
    st.write("Playoff teams: {}".format(n_playoff_teams))




    # Contains wins, losses, points, almost everything
    # we need on a week-by-week basis


if url != "" and season_weeks != '':

    if historical:
        games = historical_games

    # This stuff is largely reproducible...
    pts_played = pts[:games]
    wins_played = np.array([pts_row > pts_row[opp] for pts_row, opp in zip(pts_played, weekly_opponents[:games])])
    total_wins = wins_played.sum(axis=0)

    rotis = rotisserie(pts_played)
    # st.dataframe(
    #     pd.DataFrame(data=rotis.T, columns=np.arange(1, 12),
    #     index=teams_canonical).style\
    #     .format("{:.0f}")\
    #     .background_gradient(cmap='RdBu_r', low=1.25, high=1.25)
    # )
    rotis_win_pct = rotis.mean(axis=0)/(n_teams - 1)
    rotis_wins = rotis >= int(n_teams/2)

    if game_vs_league_median:
        total_wins += rotis_wins.sum(axis=0)
    


    
    # This is not sleeper specific...
    ptsAvg = pts_played.mean()
    pts_regress = pts_played.mean(axis=0)*0.5 + ptsAvg*0.5

    # 
    future_matchups = [np.array(list(groupby(m).values()))-1 for m in weekly_matchups[games:]]
    future_opponents = weekly_opponents[games:]
    stdev = pts_played.std()*1.03 # Add a little extra uncertainty
    
    # Unplayed games:
    games_left = max(0, season_weeks - games) 


    # Simulation
    N = 5000
    rng = np.random.default_rng()

    pts_unplayed, wins_unplayed, rotis_unplayed = simulate_remaining_weeks(
            games_left, n_teams, N, pts_regress, stdev, future_opponents)
    


    overall_pts = (pts_played.sum(axis=0).reshape((-1,1)) + pts_unplayed.sum(axis=0)).T
    overall_wins = (total_wins + wins_unplayed.sum(axis=0).T)

    if game_vs_league_median:
        overall_wins +=  (rotis_unplayed >= 5).sum(axis=0).T

    getSeeds = playoff_formats.getSeeds(playoff_format)
    
    seeds = getSeeds(n_teams, N, overall_pts, overall_wins, divisions)

    playoffResults = makeAllPlayoffResults(overall_pts, seeds, n_playoff_teams, season_weeks, stdev)



    st.header("Path to the playoffs")
    st.write("Choose your team to see how you can make the playoffs most easily.")
    playoff_path_team = st.selectbox("Team",
                options=teams_canonical)
    run_path = st.button("Run")
    if run_path:
        i_team = list(teams_canonical).index(playoff_path_team)
        inds2 = np.arange(N, dtype=int)
        my_seeds = seeds[:, i_team]
        playoff_percent = (my_seeds <= n_playoff_teams).mean()
        bye_possible = (n_playoff_teams == 6)
        if bye_possible:
            bye_chance = (my_seeds <= 2).mean()
        if (playoff_percent > 0.999):
            st.write(f"You have **already secured a playoff spot!** Congratulations!")
        else:
            st.write(f"Your current playoff chances are {playoff_percent*100:.1f}%.")
        
        if bye_possible:
            st.write(f"You have a {bye_chance*100:.1f}% chance of getting a first round bye.")



        
        for i in range(games_left):
            inds_match = np.nonzero(wins_unplayed[i, i_team, :])
            inds2 = np.intersect1d(inds2, inds_match, assume_unique=True)
        

        st.subheader("Win out")
        dfSS2 = makeSeeds(seeds[inds2], teams_canonical, n_playoff_teams)
        win_out = dfSS2.loc[playoff_path_team, "Any"]
        analyze_chances = True if playoff_percent < 0.999 else False
        analyze_byes = True if bye_possible else False
        analyze_byes_only = False
        if analyze_byes:
            bye_chance_win_out = (seeds[inds2, i_team] <= 2).mean()*100
        
        bye_language = f" and have a {bye_chance_win_out:.1f}% chance to get a bye." if analyze_byes else "."
        
        if win_out > 99.9 and playoff_percent < 0.98:
            st.write(f"You **control your own destiny** - if you win out, you will make the playoffs{bye_language} Projections:")
        elif win_out > 0.1 and playoff_percent < 0.98:
            st.write(f"You **need help to make the playoffs**. Even if you win out, you have a {win_out:.1f}% chance to make the playoffs{bye_language} Projections:")
        elif playoff_percent > 0.98:
            if analyze_byes:
                analyze_byes_only = True
                if bye_chance_win_out > 99.9:
                    st.write(f"You **control your own destiny for a bye** - if you win out, you will get a bye!")
                elif bye_chance_win_out >0.1:
                    st.write(f"You **need help to obtain a bye.** Even if you win out, you have a {bye_chance_win_out:.1f}% chance to get a bye.")
                else:
                    st.write(f"You will make the playoffs but have no chance to obtain a first round bye.")
                    analyze_byes_only = False
                    analyze_chances = False
            else:
                analyze_chances = False
        else:
            st.write(f"Even if you win out, you **cannot make the playoffs**. Better luck next year!")
            analyze_chances = False
            analyze_byes = False
        

        
        st.dataframe(dfSS2.style.format("{:.1f}")\
        .background_gradient(cmap='Greens', low=0.0, high=0.7))

        if analyze_chances and not analyze_byes_only:
            inds3 = np.arange(N, dtype=int)
            other_teams = list(range(n_teams))
            other_teams.pop(i_team)
            rooting_interests = []
            for i in range(games_left):
                rooting_week = []
                for j in other_teams:
                    inds_match = np.nonzero(wins_unplayed[i, j, :])
                    new_chances = (my_seeds[inds_match] <= n_playoff_teams).mean()
                    if (new_chances > (playoff_percent+0.005)):
                        rooting_week.append((j, (new_chances-playoff_percent)*100))
                rooting_interests.append(np.array(rooting_week))

            st.write("This week you should root for:")
            this_week = rooting_interests[0]
            this_week_df = pd.DataFrame({"Team": teams_canonical[this_week[:, 0].astype(int)],
                            "Opponent": teams_canonical[future_opponents[0][this_week[:, 0].astype(int)]],
                            "Chg. in playoff odds": this_week[:, 1]})
            this_week_df.sort_values("Chg. in playoff odds", ascending=False, inplace=True)
            st.write(this_week_df\
                .style.format("+{:.1f}", subset="Chg. in playoff odds")\
                .background_gradient(cmap='Greens', low=0.1, high=0.9, ))

        
        if analyze_byes_only:
            inds3 = np.arange(N, dtype=int)
            other_teams = list(range(n_teams))
            other_teams.pop(i_team)
            rooting_interests = []
            for i in range(games_left):
                rooting_week = []
                for j in other_teams:
                    inds_match = np.nonzero(wins_unplayed[i, j, :])
                    new_chances = (my_seeds[inds_match] <= 2).mean()
                    if (new_chances > (bye_chance + 0.005)):
                        rooting_week.append((j, (new_chances-bye_chance)*100))
                rooting_interests.append(np.array(rooting_week))

            st.write("This week you should root for:")
            this_week = rooting_interests[0]
            this_week_df = pd.DataFrame({"Team": teams_canonical[this_week[:, 0].astype(int)],
                            "Opponent": teams_canonical[future_opponents[0][this_week[:, 0].astype(int)]],
                            "Chg. in bye odds": this_week[:, 1]})
            this_week_df.sort_values("Chg. in bye odds", ascending=False, inplace=True)
            st.write(this_week_df\
                .style.format("+{:.1f}", subset="Chg. in bye odds")\
                .background_gradient(cmap='Greens', low=0.1, high=0.9, ))



    inds = np.arange(N, dtype=int)


    st.header("All Projections")
    st.write("Projected outcomes for each game:")
    if games_left <= 1:
        slots = [st.empty() for _ in range(games_left*2)]
    else:
        cols = st.beta_columns(2)
        slots = []
        for i in range(games_left):
            slots.append(cols[i % 2].empty())
            slots.append(cols[i % 2].empty())

    st.write("Use the buttons to see how playoff chances change depending on the results of each game.")

    # Make buttons:
    n_cols = int(n_teams / 2)
    buttons = []
    for j, match_ in enumerate(future_matchups):
        st.write("Week {}".format(games+1+j))
        cols = st.beta_columns(n_cols) 
        weekly_buttons = []
        for i, x in enumerate(match_):
            weekly_buttons.append(cols[i].radio(label='Winner',
                options=['Any', teams_canonical[x[0]], teams_canonical[x[1]]],
                key=f"week-{j}-matchup-{i}"
            )
        )
        buttons.append(weekly_buttons)
    

    # Filter simulations:
    for i, weekly_buttons in enumerate(buttons):
        for button in weekly_buttons:
            if button != 'Any':
                inds_match = np.nonzero(wins_unplayed[i, list(teams_canonical).index(button), :])
                inds = np.intersect1d(inds, inds_match, assume_unique=True)



    for i, match_ in enumerate(future_matchups):
        slots[i*2].write("Week {}".format(games+1+i))
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

    dfPRO = analyzePlayoffResults(playoffResults[inds], teams_canonical)
    
    st.subheader("Playoff Seeding Chances")
    st.dataframe(dfSS.style.format("{:.1f}")\
        .background_gradient(cmap='Greens', low=0.0, high=0.7))

    
    st.subheader("Predicted Standings")
    dfAvg.sort_values('avgWins', inplace=True, ascending=False)
    st.dataframe(dfAvg.style.format("{:.1f}")\
        .background_gradient(cmap='RdBu_r', low=1, high=1, axis=0))


    st.subheader("Current Standings")
    df_standings = pd.DataFrame(np.c_[total_wins, pts_played.sum(axis=0), rotis_win_pct, divisions], index=teams_canonical,
                            columns=['Wins', 'Pts', 'Rotis. Win %', 'Division'])
    # This should be generated from scratch...
    st.dataframe(df_standings.style.format("{:.0f}", subset=['Division'])\
                            .format("{:.1f}", subset=['Pts', 'Wins'])\
                            .format("{:.3f}", subset=['Rotis. Win %']))

    st.subheader("Playoff Outcomes")
    st.dataframe(dfPRO.style.format("{:.1f}")\
        .background_gradient(cmap='Greens', low=0.0, high=0.7))
    

    
        


    # Choose playoff teams...
    # Bracket...
    # Byes?

