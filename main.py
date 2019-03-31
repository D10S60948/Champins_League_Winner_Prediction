import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random

# --- Removing duplicated values from the given list ---
def remove_dupes(dup_list):
    new_list = []
    for elem in dup_list:
        if elem not in new_list:
            new_list.append(elem)
    return new_list


# --- Receives the fixtures and returns the predicted next round's qualifiers ---
def predict(fixtures):
    pred_set = []

    for index, row in fixtures.iterrows():
        pred_set.append({'Team_1': row['Team_1'], 'Team_2': row['Team_2'], 'Winner': None})

    pred_set = pd.DataFrame(pred_set)

    pred_set_backup = pred_set

    pred_set = pd.get_dummies(pred_set, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])

    missing_cols = set(final.columns) - set(pred_set.columns)
    for c in missing_cols:
        pred_set[c] = 0
    pred_set = pred_set[final.columns]

    # --- Remove winning team column ---

    pred_set = pred_set.drop(['Winner'], axis=1)

    qulifiers = []

    predictions = logreg.predict(pred_set)
    number_of_games_in_leg = len(fixtures)//2
    for i in range(fixtures.shape[0]):
        print(pred_set_backup.iloc[i, 0] + " and " + pred_set_backup.iloc[i, 1])
        if predictions[i] == 2:
            print("Winner: " + pred_set_backup.iloc[i, 1])
        elif predictions[i] == 1:
            print("Draw")
        elif predictions[i] == 0:
            print("Winner: " + pred_set_backup.iloc[i, 0])
        print('Probability of ' + pred_set_backup.iloc[i, 0] + ' winning: ',
              '%.3f' % (logreg.predict_proba(pred_set)[i][0]))
        print('Probability of Draw: ', '%.3f' % (logreg.predict_proba(pred_set)[i][1]))
        print('Probability of ' + pred_set_backup.iloc[i, 1] + ' winning: ',
              '%.3f' % (logreg.predict_proba(pred_set)[i][2]))
        print("")
        if i > number_of_games_in_leg-1:  # --- Inserting the winning team to the qualifiers ---
            if (predictions[i] == 2 and predictions[i - number_of_games_in_leg] != 2) or \
                    (predictions[i - number_of_games_in_leg] == 0 and predictions[i] == 1):
                qulifiers.append(pred_set_backup.iloc[i, 1])
            elif (predictions[i] == 0 and predictions[i - number_of_games_in_leg] != 0) or \
                    (predictions[i - number_of_games_in_leg] == 2 and predictions[i] == 1):
                qulifiers.append(pred_set_backup.iloc[i, 0])
            else:  # --- In case of a draw the team with the better odds will qualify ---
                if ((logreg.predict_proba(pred_set)[i][2] +
                     logreg.predict_proba(pred_set)[i - number_of_games_in_leg][0]) >
                        (logreg.predict_proba(pred_set)[i][0] +
                         logreg.predict_proba(pred_set)[i - number_of_games_in_leg][2])):
                    qulifiers.append(pred_set_backup.iloc[i, 1])
                else:
                    qulifiers.append(pred_set_backup.iloc[i, 0])
    if number_of_games_in_leg == 0:
        if predictions[i] == 2:
            return pred_set_backup.iloc[i, 1]
        elif predictions[i] == 1:
            if logreg.predict_proba(pred_set)[i][2] > logreg.predict_proba(pred_set)[i][0]:
                return pred_set_backup.iloc[i, 1]
            else:
                return pred_set_backup.iloc[i, 0]
        elif predictions[i] == 0:
            return pred_set_backup.iloc[i, 0]
    return qulifiers


# --- Given the qualifiers of a round it randomly (as in real life) sets the fixtures ---
def pairing_the_qualifiers(the_qualifiers):
    pairs = {}
    for p in range(len(the_qualifiers) // 2):
        pairs[p + 1] = (the_qualifiers.pop(random.randrange(len(the_qualifiers))),
                        the_qualifiers.pop(random.randrange(len(the_qualifiers))))

    df_fixtures = pd.DataFrame.from_dict(pairs)
    df_fixtures = df_fixtures.transpose()
    df_fixtures.rename(columns={0: 'Team_1', 1: 'Team_2'}, inplace=True)

    return df_fixtures


def adding_2nd_leg(qualifing_teams):
    rows_before = len(qualifing_teams)
    for i in range(rows_before):
        row_number = i+rows_before+1
        qualifing_teams.loc[row_number] = [qualifing_teams.loc[i+1, 'Team_2'], qualifing_teams.loc[i+1, 'Team_1']]
    return qualifing_teams


def print_fixtures(to_print):
    print("\n")
    print(to_print)
    print("\n")


# -----------------------
# --- Data --------------
# -----------------------

# --- Reading the data --

all_games_names = ['Round', 'Season', 'Team_1', 'T1_Goals', 'Team_2', 'T2_Goals']

all_games = pd.read_csv('data/all games.csv', names=all_games_names, encoding="ISO-8859-1")

df_all_games = pd.DataFrame(all_games)

# --- Manipulating the data - Setting the winner ---

# --- 1: Home team won || 2: Away team won || 0: Draw
df_all_games['Winner'] = np.where(df_all_games['T1_Goals'] > df_all_games['T2_Goals'], df_all_games['Team_1'],
                                  np.where(df_all_games['T1_Goals'] == df_all_games['T2_Goals'], 'Draw',
                                           df_all_games['Team_2']))

# --- Adding current's tournament teams ---

this_year_participants = ["Roma", "Porto", "Manchester United", "Paris Saint-Germain", "Tottenham", "Borussia Dortmund",
                          "Ajax", "Lyon", "Barcelona", "Liverpool", "Bayern Munich", "Atletico Madrid", "Juventus",
                          "Schalke", "Manchester City", "Real Madrid"]

# --- Narrowing to team participating in the this year competition ---

df_teams_home = df_all_games[df_all_games['Team_1'].isin(this_year_participants)]
df_teams_away = df_all_games[df_all_games['Team_2'].isin(this_year_participants)]
df_teams = pd.concat((df_teams_home, df_teams_away))

df_all_games = df_teams.drop_duplicates()

df_all_games.drop(['Round', 'Season', 'T1_Goals', 'T2_Goals'], axis=1, inplace=True)

df_all_games_final = df_all_games.reset_index(drop=True)
df_all_games_final.loc[df_all_games_final.Winner == df_all_games_final.Team_1, 'Winner'] = 0
df_all_games_final.loc[df_all_games_final.Winner == 'Draw', 'Winner'] = 1
df_all_games_final.loc[df_all_games_final.Winner == df_all_games_final.Team_2, 'Winner'] = 2


final = pd.get_dummies(df_all_games_final, prefix=['Team_1', 'Team_2'], columns=['Team_1', 'Team_2'])

# -------------------------------------
# --- Training and testing ------------
# -------------------------------------

# --- Splitting the data from the results

info = final.drop("Winner", axis=1)
results = final["Winner"]
results = results.astype('int')

# --- Splitting to the training set and testing set

X_train, X_test, y_train, y_test = train_test_split(info, results, test_size=0.25, random_state=42)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
score_train = logreg.score(X_train, y_train)
score_test = logreg.score(X_test, y_test)

print("Training set accuracy: ", '%.2f'%(score_train))
print("Testing set accuracy: ", '%.2f'%(score_test))
print("\n")

# ----------------------------------
# --- Preparing this year's data ---
# ----------------------------------

# --- Adding the fixtures and club's rankings ---

fixtures_names = ['Round', 'Team_1', 'T1_position', 'Team_2', 'T2_position']
last16_fixtures = pd.read_csv('data/fixtures.csv', names=fixtures_names)

quarter_final_qualifiers = predict(last16_fixtures)
quarter_final_fixtures = pairing_the_qualifiers(quarter_final_qualifiers)
quarter_final_fixtures = adding_2nd_leg(quarter_final_fixtures)
print_fixtures(quarter_final_fixtures)

semi_final_qualifiers = predict(quarter_final_fixtures)
semi_final_fixtures = pairing_the_qualifiers(semi_final_qualifiers)
semi_final_fixtures = adding_2nd_leg(semi_final_fixtures)
print_fixtures(semi_final_fixtures)

final_qualifiers = predict(semi_final_fixtures)
final_fixtures = pairing_the_qualifiers(final_qualifiers)
print_fixtures(final_fixtures)

the_champions_league_winner = predict(final_fixtures)

print("The Champions League 2018/2019 winner is " + the_champions_league_winner)

file = open("Winners.txt", "a+")
file.write(the_champions_league_winner)
file.write("\n")
file.close()
