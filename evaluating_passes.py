"""
Run code to build xPass model and use this to identify and analyse midfielders
good at making difficult passes.

To run, you just need to put this script in a folder containing the Wyscout 
data in a directory called 'data/'

If the Wyscout data is elsewhere, change the variable data_folder to point to
the actual location of the data
"""

import json
import os
import numpy as np
import pandas as pd

#Plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

#Statistical fitting of models
import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.calibration import calibration_curve


###############################################################################
# Library of functions to load Wyscout data                                   #
###############################################################################
def load_wyscout_event_data(data_folder='data/'):
    """
    Given a folder location holding Wyscout data, load all json files for event data and
    put all data in a single dataframe, which is returned.
    """
    
    data_folder = data_folder.rstrip('/').strip('"')
    
    event_files = []
    events = []
    
    for root, dirs, files in os.walk(f'{data_folder}/events'):
        for file in files:
            event_files.append(f'{root}/{file}')

    for file_location in event_files:
        with open(file_location, encoding="utf8") as json_data:
            event_df_for_file = pd.DataFrame(json.load(json_data))
            events.append(event_df_for_file)
    
    events_df = pd.concat(events) 
    events_df = events_df.reset_index(drop=True)
    
    return events_df


def get_competition_id(competition_name, data_folder='data'):
    """
    Open competitions data json and get competition id for requested competition
    """
    data_folder = data_folder.rstrip('/').strip('"')

    with open(f'{data_folder}/competitions.json', encoding="utf8") as json_data:
        competition_data = pd.DataFrame(json.load(json_data))
    
    competition_id = competition_data[competition_data['name'] == competition_name]['wyId'].values[0]
    
    return competition_id


def get_wyscout_match_ids_for_competition(competition_id, data_folder='data'):
    """
    Given a competition id and a folder location holding Wyscout data, load all 
    json files for match data and get match ids for all matches from desired 
    competition
    """
    data_folder = data_folder.rstrip('/').strip('"')
    
    match_files = []
    matches = []
    
    for root, dirs, files in os.walk(f'{data_folder}/matches'):
        for file in files:
            match_files.append(f'{root}/{file}')

    for file_location in match_files:
        with open(file_location, encoding="utf8") as json_data:
            match_df_for_file = pd.DataFrame(json.load(json_data))
            matches.append(match_df_for_file)
    
    matches_df = pd.concat(matches) 
    matches_df = matches_df.reset_index(drop=True)
    
    matches_for_competition = matches_df[matches_df['competitionId'] == competition_id]
    match_ids_for_competition = matches_for_competition['wyId'].unique()

    return match_ids_for_competition


def get_pass_data_for_competition(pass_data, competition_to_evaluate,
                                  data_folder='data/'):

    pass_data = pass_data.copy()

    competition_to_evaluate_id = get_competition_id(competition_to_evaluate,
                                                   data_folder=data_folder)
    competition_match_ids = get_wyscout_match_ids_for_competition(competition_to_evaluate_id,
                                                                 data_folder=data_folder)
    pass_data_for_competition = pass_data[pass_data['matchId'].isin(competition_match_ids)].copy()
    
    return pass_data_for_competition


def get_player_data(data_folder='data/'):
    """
    Get Wyscout player data and return in a dataframe
    """
    data_folder = data_folder.strip('/').strip('"')
    with open(f'{data_folder}/players.json', encoding="utf8") as json_data:
        player_data = pd.DataFrame(json.load(json_data))
    
    player_data['player_position'] = player_data['role'].apply(lambda x: x['name'])

    return player_data


###############################################################################
# Pass and event data pipeline and filters library to generate features for   #
# pass data using the Wyscout event data                                      #
###############################################################################
class EventDataPipeline:
    """
    Pipeline to generate features from event data.
    
    First instantiate, then provide list of filters to use to add_filters and
    finally run run_get_event_data_pipeline, providing df of event data.
    """
    def __init__(self):
        self.filters = list()
    
    def add_filters(self, filters_to_add):
        self.filters.extend(filters_to_add)
    
    def run_get_event_data_pipeline(self, event_data):
        event_data = event_data.copy()
        for f in self.filters:
            event_data = f(event_data)
        
        return event_data
    

class PassDataPipeline:
    """
    Pipeline to generate features for pass data.
    
    First instantiate, then provide list of filters to use to add_filters and
    finally run run_get_pass_data_pipeline, providing df of event data.
    """
    def __init__(self):
        self.filters = list()

    def add_filters(self, filters_to_add):
        self.filters.extend(filters_to_add)

    def run_get_pass_data_pipeline(self, event_data):
        pass_data = event_data[event_data['eventName'] == 'Pass'].copy()
        for f in self.filters:
            pass_data = f(pass_data)

        return pass_data


###############################################################################
# Event data pipeline filters                                                 #
###############################################################################
def get_shot_assist_flag(event_data):
    """
    Given event data, derive a flag to demonstrate whether an event is an 
    assist for a shot
    """
    event_data = event_data.copy()
    
    next_event_in_same_match = (event_data['matchId'].diff() == 0.0) | (event_data['matchId'].diff().isnull())
    next_event_by_same_team = (event_data['teamId'].diff() == 0.0) | (event_data['teamId'].diff().isnull())
    next_event_a_shot = event_data['eventName'].shift(-1) == 'Shot'
    
    event_data['shot_assist'] = next_event_in_same_match & next_event_by_same_team & next_event_a_shot
    
    return event_data
        

###############################################################################
# Pass data pipeline filters                                                  #
###############################################################################
def remove_rows_with_no_pass_end_position(pass_data):
    """
    All passes should have an end location. If any rows don't, remove them from the data
    """
    pass_data = pass_data.copy()
    number_of_coords = pass_data['positions'].apply(lambda x: len(x))
    pass_data = pass_data[number_of_coords == 2]
    pass_data = pass_data.reset_index(drop=True)
    
    return pass_data


def get_success_flag_from_tags(pass_data):
    """
    Extract a success flag from tags field of pass data:
        1801 means success
        1802 mean not a success
    """
    all_tags = pass_data['tags'].copy().apply(
            lambda x: [tag['id'] for tag in x])
    pass_data['success'] = all_tags.apply(
            lambda x: True if 1801 in x and 1802 not in x else False)

    return pass_data


def get_start_coordinates_of_pass_from_position(pass_data):
    """
    Extract the start co-ordinates of a pass from the position field of the 
    event data
    """
    pass_data = pass_data.copy()
    pass_data['pass_start_x'] = pass_data['positions'].apply(
            lambda coords: coords[0]['x'])
    pass_data['pass_start_y'] = pass_data['positions'].apply(
            lambda coords: coords[0]['y'])

    return pass_data


def get_end_coordinates_of_pass_from_position(pass_data):
    """
    Exctract the end co-ordinates of a pass from the position field of the 
    event data
    If there is no end co-ordinates, then assume pass end is same as pass start

    Note, this doesn't make any attempt to work out the intended end 
    co-ordinates of a failed pass
    """
    pass_data = pass_data.copy()

    # For some reason there are two passes for which there are only one co-ordinate
    number_of_coords = pass_data['positions'].apply(lambda x: len(x))
    inferred_end_coordiates = pass_data[number_of_coords == 1]['positions'].apply(
        lambda coords: [{'y': coords[0]['y'], 'x': coords[0]['x']},
                        {'y': coords[0]['y'], 'x': coords[0]['x']}])
    pass_data['positions'].update(inferred_end_coordiates)

    pass_data['pass_end_x'] = pass_data['positions'].apply(
            lambda coords: coords[1]['x'])
    pass_data['pass_end_y'] = pass_data['positions'].apply(
            lambda coords: coords[1]['y'])

    return pass_data


def get_start_coordinates_squared(pass_data):
    """
    Derive new features of start coordinates squared from start coordinates.
    Note that this filter must be applied after you have already run 
    get_start_coordinates_of_pass_from_position()
    """
    pass_data = pass_data.copy()

    pass_data['pass_start_x_squared'] = pass_data['pass_start_x'] ** 2
    pass_data['pass_start_y_squared'] = pass_data['pass_start_y'] ** 2

    return pass_data


def get_start_coordinates_in_metres(pass_data):
    """
    Derive the start co-ordinates in metres from the start co-ordinates.
    Note that this filter must be applied after you have already run 
    get_start_coordinates_of_pass_from_position()
    """
    pass_data = pass_data.copy()
    pass_data['pass_start_x_metres'] = pass_data['pass_start_x'] * 105 / 100
    pass_data['pass_start_y_metres'] = pass_data['pass_start_y'] * 65 / 100

    return pass_data


def get_product_of_start_coords(pass_data):
    """
    Derive new feature: product of x and y start co-ordinates for pass
    """
    pass_data = pass_data.copy()
    pass_data['start_coords_product'] = pass_data['pass_start_x'] * pass_data['pass_start_y']
    
    return pass_data


def get_pass_type(pass_data):
    """
    Derive boolean fields for each pass type
    """
    pass_data = pass_data.copy()
    for pass_type in ['Simple pass', 'High pass', 'Head pass', 'Smart pass', 'Launch', 'Cross', 'Hand pass']:
        pass_data[pass_type.lower().replace(' ', '_')] = pass_data['subEventName'] == pass_type
    
    return pass_data


def get_angle_and_distance_of_pass(pass_data):
    """
    Derive the angle at which the pass was made and the distance of it
    """
    pass_data = pass_data.copy()
    
    x_start_in_metres = pass_data['pass_start_x'] * 105 / 100
    y_start_in_metres = pass_data['pass_start_y'] * 65 / 100
    x_end_in_metres = pass_data['pass_end_x'] * 105 / 100
    y_end_in_metres = pass_data['pass_end_y'] * 65 / 100
    
    dx = x_end_in_metres - x_start_in_metres
    dy = y_end_in_metres - y_start_in_metres
    pass_directly_forwards = dx == 0
    pass_backwards = dx < 0
    
    # If pass directly forwards, angle is 0
    angle_of_pass = pd.Series([0] * len(pass_data))
    angle_of_pass[~pass_directly_forwards] = np.arctan(
            abs(dy[~pass_directly_forwards] / dx[~pass_directly_forwards]))
    angle_of_pass[pass_backwards] = np.pi - angle_of_pass
    
    pass_data = pass_data.assign(angle_of_pass=angle_of_pass)
    
    pass_data['distance_of_pass'] = np.sqrt(dx**2 + dy**2)
    
    return pass_data


def get_part_of_field_for_pass_start_and_end(pass_data):
    """
    Derive new features for start third of field, end third of field and a 
    score for progression up the pitch, which is simply end third - start third
    """
    pass_data = pass_data.copy()
    for start_or_end in ['start', 'end']:
        
        in_first_third = pass_data[f'pass_{start_or_end}_x'] <= 33.3
        in_middle_third = (pass_data[f'pass_{start_or_end}_x'] > 33.3) & (pass_data[f'pass_{start_or_end}_x'] <= 66.6)
        in_final_third = pass_data[f'pass_{start_or_end}_x'] > 66.6

        pass_data[f'{start_or_end}_third_of_field'] = 0
        pass_data.loc[in_first_third, f'{start_or_end}_third_of_field'] = 1
        pass_data.loc[in_middle_third, f'{start_or_end}_third_of_field'] = 2
        pass_data.loc[in_final_third, f'{start_or_end}_third_of_field'] = 3

    pass_data['pitch_progression_score'] = pass_data['end_third_of_field'] - pass_data['start_third_of_field']
    
    return pass_data


def get_end_position_to_goal_details(pass_data):
    """
    Get the distance of the end position from goal and the angle made by lines 
    drawn from the end position to the goalposts.
    
    Note these features are inspired by work in SoccermaticsForPython:
        https://github.com/Friends-of-Tracking-Data-FoTD/SoccermaticsForPython
    """
    
    pass_data = pass_data.copy()
    end_x_from_goal = (100 - pass_data['pass_end_x']) * 105 / 100
    end_y_from_goal_centre = abs(50 - pass_data['pass_end_y']) * 65 / 100

    pass_data['end_distance_from_goal'] = np.sqrt(
            end_x_from_goal ** 2 + end_y_from_goal_centre ** 2)
    angle = np.arctan(7.32 * end_x_from_goal / (end_x_from_goal ** 2 + end_y_from_goal_centre ** 2 - (7.32 / 2) ** 2))
    angle_less_than_0 = angle < 0
    angle[angle_less_than_0] = np.pi + angle
    pass_data['end_angle_between_goalpost'] = angle

    return pass_data


###############################################################################
# Plotting library                                                            #
###############################################################################
def createPitch(length, width, unity, linecolor):
    # Code by @JPJ_dejong
    # This was taken directly from FCPython, in Soccermatics for Python Friends
    # of tracking repository: 
    # https://github.com/Friends-of-Tracking-Data-FoTD/SoccermaticsForPython

    """
    creates a plot in which the 'length' is the length of the pitch (goal to goal).
    And 'width' is the width of the pitch (sideline to sideline). 
    Fill in the unity in meters or in yards.

    """
    #Set unity
    if unity == "meters":
        # Set boundaries
        if length >= 120.5 or width >= 75.5:
            return(str("Field dimensions are too big for meters as unity, didn't you mean yards as unity?\
                       Otherwise the maximum length is 120 meters and the maximum width is 75 meters. Please try again"))
        #Run program if unity and boundaries are accepted
        else:
            #Create figure
            fig=plt.figure()
            #fig.set_size_inches(7, 5)
            ax=fig.add_subplot(1,1,1)
           
            #Pitch Outline & Centre Line
            plt.plot([0,0],[0,width], color=linecolor)
            plt.plot([0,length],[width,width], color=linecolor)
            plt.plot([length,length],[width,0], color=linecolor)
            plt.plot([length,0],[0,0], color=linecolor)
            plt.plot([length/2,length/2],[0,width], color=linecolor)
            
            #Left Penalty Area
            plt.plot([16.5 ,16.5],[(width/2 +16.5),(width/2-16.5)],color=linecolor)
            plt.plot([0,16.5],[(width/2 +16.5),(width/2 +16.5)],color=linecolor)
            plt.plot([16.5,0],[(width/2 -16.5),(width/2 -16.5)],color=linecolor)
            
            #Right Penalty Area
            plt.plot([(length-16.5),length],[(width/2 +16.5),(width/2 +16.5)],color=linecolor)
            plt.plot([(length-16.5), (length-16.5)],[(width/2 +16.5),(width/2-16.5)],color=linecolor)
            plt.plot([(length-16.5),length],[(width/2 -16.5),(width/2 -16.5)],color=linecolor)
            
            #Left 5-meters Box
            plt.plot([0,5.5],[(width/2+7.32/2+5.5),(width/2+7.32/2+5.5)],color=linecolor)
            plt.plot([5.5,5.5],[(width/2+7.32/2+5.5),(width/2-7.32/2-5.5)],color=linecolor)
            plt.plot([5.5,0.5],[(width/2-7.32/2-5.5),(width/2-7.32/2-5.5)],color=linecolor)
            
            #Right 5 -eters Box
            plt.plot([length,length-5.5],[(width/2+7.32/2+5.5),(width/2+7.32/2+5.5)],color=linecolor)
            plt.plot([length-5.5,length-5.5],[(width/2+7.32/2+5.5),width/2-7.32/2-5.5],color=linecolor)
            plt.plot([length-5.5,length],[width/2-7.32/2-5.5,width/2-7.32/2-5.5],color=linecolor)
            
            #Prepare Circles
            centreCircle = plt.Circle((length/2,width/2),9.15,color=linecolor,fill=False)
            centreSpot = plt.Circle((length/2,width/2),0.8,color=linecolor)
            leftPenSpot = plt.Circle((11,width/2),0.8,color=linecolor)
            rightPenSpot = plt.Circle((length-11,width/2),0.8,color=linecolor)
            
            #Draw Circles
            ax.add_patch(centreCircle)
            ax.add_patch(centreSpot)
            ax.add_patch(leftPenSpot)
            ax.add_patch(rightPenSpot)
            
            #Prepare Arcs
            leftArc = Arc((11,width/2),height=18.3,width=18.3,angle=0,theta1=308,theta2=52,color=linecolor)
            rightArc = Arc((length-11,width/2),height=18.3,width=18.3,angle=0,theta1=128,theta2=232,color=linecolor)
            
            #Draw Arcs
            ax.add_patch(leftArc)
            ax.add_patch(rightArc)
            #Axis titles

    #check unity again
    elif unity == "yards":
        #check boundaries again
        if length <= 95:
            return(str("Didn't you mean meters as unity?"))
        elif length >= 131 or width >= 101:
            return(str("Field dimensions are too big. Maximum length is 130, maximum width is 100"))
        #Run program if unity and boundaries are accepted
        else:
            #Create figure
            fig=plt.figure()
            #fig.set_size_inches(7, 5)
            ax=fig.add_subplot(1,1,1)
           
            #Pitch Outline & Centre Line
            plt.plot([0,0],[0,width], color=linecolor)
            plt.plot([0,length],[width,width], color=linecolor)
            plt.plot([length,length],[width,0], color=linecolor)
            plt.plot([length,0],[0,0], color=linecolor)
            plt.plot([length/2,length/2],[0,width], color=linecolor)
            
            #Left Penalty Area
            plt.plot([18 ,18],[(width/2 +18),(width/2-18)],color=linecolor)
            plt.plot([0,18],[(width/2 +18),(width/2 +18)],color=linecolor)
            plt.plot([18,0],[(width/2 -18),(width/2 -18)],color=linecolor)
            
            #Right Penalty Area
            plt.plot([(length-18),length],[(width/2 +18),(width/2 +18)],color=linecolor)
            plt.plot([(length-18), (length-18)],[(width/2 +18),(width/2-18)],color=linecolor)
            plt.plot([(length-18),length],[(width/2 -18),(width/2 -18)],color=linecolor)
            
            #Left 6-yard Box
            plt.plot([0,6],[(width/2+7.32/2+6),(width/2+7.32/2+6)],color=linecolor)
            plt.plot([6,6],[(width/2+7.32/2+6),(width/2-7.32/2-6)],color=linecolor)
            plt.plot([6,0],[(width/2-7.32/2-6),(width/2-7.32/2-6)],color=linecolor)
            
            #Right 6-yard Box
            plt.plot([length,length-6],[(width/2+7.32/2+6),(width/2+7.32/2+6)],color=linecolor)
            plt.plot([length-6,length-6],[(width/2+7.32/2+6),width/2-7.32/2-6],color=linecolor)
            plt.plot([length-6,length],[(width/2-7.32/2-6),width/2-7.32/2-6],color=linecolor)
            
            #Prepare Circles; 10 yards distance. penalty on 12 yards
            centreCircle = plt.Circle((length/2,width/2),10,color=linecolor,fill=False)
            centreSpot = plt.Circle((length/2,width/2),0.8,color=linecolor)
            leftPenSpot = plt.Circle((12,width/2),0.8,color=linecolor)
            rightPenSpot = plt.Circle((length-12,width/2),0.8,color=linecolor)
            
            #Draw Circles
            ax.add_patch(centreCircle)
            ax.add_patch(centreSpot)
            ax.add_patch(leftPenSpot)
            ax.add_patch(rightPenSpot)
            
            #Prepare Arcs
            leftArc = Arc((11,width/2),height=20,width=20,angle=0,theta1=312,theta2=48,color=linecolor)
            rightArc = Arc((length-11,width/2),height=20,width=20,angle=0,theta1=130,theta2=230,color=linecolor)
            
            #Draw Arcs
            ax.add_patch(leftArc)
            ax.add_patch(rightArc)
                
    #Tidy Axes
    plt.axis('off')
    
    return fig,ax


def plot_event_heatmap(y_events, x_events, title, no_of_bins=10, file_name=None):
    """
    Given series of event positions (one series for y and one for x coordinates) plot a heatmap 
    for those events, with the pitch split into no_of_bins by no_of_bins areas
    """
    histogram_values = np.histogram2d(y_events, x_events, bins=no_of_bins)
    fig, ax = createPitch(105, 65, 'meters', 'black')
    pos=ax.imshow(histogram_values[0], extent=(0, 105, 0, 65), aspect='auto', cmap=plt.cm.Reds)
    fig.colorbar(pos, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    
    if file_name is not None:
        fig.savefig(f'{file_name}.pdf', dpi=300)
    
    plt.show()


def plot_relationship_scatter(pass_data, variable_to_plot, 
                              curve_parameters=None, title=None,
                              file_name=None):
    """
    Plot the relationship between the variable to plot and the probability of a 
    pass success, based on the data provided in the pass_data DataFrame.
    If curve_parameters is also present, then plot the logistic regression 
    curve with these values too
    
    Parameters
    ----------
    pass_data (pd.DataFrame)
        DataFrame containing event data relating to passes, detailing whether 
        each pass is successful, and 
        containing the variable to plot
    
    variable_to_plot (str)
        The name of a field in the pass_data for which to plot a relationship 
        between it and pass success probability
    
    curve_parameters (list)
        Parameters for a logistic regression to plot.
    
    title (str)
        Optional. If provided, include as a title for the plot
    
    file_name (str)
        Optional. If provided, save output with this file name
    """
    passcount_dist = np.histogram(pass_data[variable_to_plot], bins=40, range=[0, 100])
    successcount_dist = np.histogram(pass_data[pass_data['success']][variable_to_plot], 
                                     bins=40, range=[0, 100])
    prob_success = np.divide(successcount_dist[0], passcount_dist[0])
    variable_vals = passcount_dist[1]
    mid_variable_vals = (variable_vals[:-1] + variable_vals[1:]) / 2
    
    fig, ax = plt.subplots(num=2)
    ax.plot(mid_variable_vals, prob_success, linestyle='none', marker= '.', 
            markerSize= 12, color='black')
    
    if curve_parameters is not None:
        exponential_value = np.array([0.0] * len(mid_variable_vals))
        for i, param  in enumerate(curve_parameters):
            exponential_value += param * pow(mid_variable_vals, i)
        xPass_prob = 1 / (1 + np.exp(exponential_value))
        ax.plot(mid_variable_vals, xPass_prob, linestyle='solid', color='black')
    
    ax.set_ylabel('Probability pass success')
    ax.set_xlabel(variable_to_plot)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if title:
        plt.title(title)
    
    if file_name is not None:
        fig.savefig(f'{file_name}.pdf', dpi=300)
        
    plt.show()


def plot_xpass_per_pass_vs_total_pass_count(player_data_with_total_xPass):
    """
    A specific function, to produce a plot showing relationship between total
    pass per player and average xPass added per player.
    
    Also highlights some players positions in this plot
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.set_facecolor('#333333')
    fig.set_facecolor('#333333')
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.scatter(player_data_with_total_xPass['xPass_per_pass'], player_data_with_total_xPass['pass_attempt_count'],
               color='#b1cefc', alpha=0.5)
    for player, text_position in [
        ['David Silva', (0.12, 400)],
        ['R. Mahrez', (0.12, 350)],
        ['J. Milner', (0.12, 300)],
        ['M. Noble', (0.12, 250)],
        ['F. Delph', (0.12, 200)],
        ['R. Pereyra', (0.12, 150)],
        ['A. Maitland-Niles', (0.12, 100)],
        ['Mohamed Elneny', (0.12, 50)]]:
        this_player_data = player_data_with_total_xPass[player_data_with_total_xPass['shortName'] == player]
        player_xPass_per_game = this_player_data['xPass_per_pass']
        player_pass_count = this_player_data['pass_attempt_count']
        plt.scatter(player_xPass_per_game, player_pass_count, color='#b1cefc')
        plt.annotate(player, (player_xPass_per_game, player_pass_count), 
                     xytext=text_position, color='white',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='white'))
    plt.vlines(x=0, ymin=0, ymax=600, color='#b1cefc', alpha=0.5)
    plt.hlines(y=250, xmin=-0.25, xmax=0.1, color='#b1cefc', alpha=0.5)
    plt.title('Midfielders: value added against number of difficult passes attempted',
             color='white')
    plt.show()


def plot_heatmaps_for_player(player_id, pass_data, player_name):
    """
    Plot heatmaps showing:
        all passes
        all completed passes
        all failed passes
        all difficult passes
        all completed difficult passes
        all failed difficult passes
    for a player given
    """
    pass_data_for_player = pass_data[pass_data['playerId'] == player_id_to_plot]

    x_start_metres = pass_data_for_player['pass_start_x'] * 105 / 100
    y_start_metres = pass_data_for_player['pass_start_y'] * 65 / 100
    # All passes
    plot_event_heatmap(y_start_metres, x_start_metres, f'Number of passes made by {player_name}',
                       no_of_bins=5)
    # Completed passes
    plot_event_heatmap(y_start_metres[pass_data_for_player['success']], 
                       x_start_metres[pass_data_for_player['success']], 
                       'Number of passes successful', no_of_bins=5)
    # Failed passes
    plot_event_heatmap(y_start_metres[~pass_data_for_player['success']], 
                       x_start_metres[~pass_data_for_player['success']], 
                       'Number of passes not successful', no_of_bins=5)
    
    # Difficult passes
    plot_event_heatmap(y_start_metres[pass_data_for_player['xPass'] <= 0.8], 
                       x_start_metres[pass_data_for_player['xPass'] <= 0.8], 
                       'Number of difficult passes', no_of_bins=5)

    plot_event_heatmap(y_start_metres[pass_data_for_player['xPass'] <= 0.8][pass_data_for_player['success']], 
                       x_start_metres[pass_data_for_player['xPass'] <= 0.8][pass_data_for_player['success']], 
                       'Number of difficult successful passes', no_of_bins=5)

    plot_event_heatmap(y_start_metres[pass_data_for_player['xPass'] <= 0.8][~pass_data_for_player['success']], 
                       x_start_metres[pass_data_for_player['xPass'] <= 0.8][~pass_data_for_player['success']], 
                       'Number of difficult failed passes', no_of_bins=5)
    
    
def plot_probability_difference_heatmap_for_player(player_id_to_plot, player_name, pass_data,
                                                   player_data, position_of_player):
    """
    For a given player plot a heatmap showing difference in their pass success
    rate and the overall pass success rate for players in their position
    """

    pass_data_for_player = pass_data[pass_data['playerId'] == player_id_to_plot]

    x_start_metres = pass_data_for_player['pass_start_x'] * 105 / 100
    y_start_metres = pass_data_for_player['pass_start_y'] * 65 / 100

    histogram_values_all = np.histogram2d(y_start_metres, x_start_metres, bins=5, range=[[0, 65],[0, 105]])
    histogram_values_success = np.histogram2d(y_start_metres[pass_data_for_player['success']], 
                                              x_start_metres[pass_data_for_player['success']], 
                                              bins=5, range=[[0, 65],[0, 105]])
    histogram_values_probability = histogram_values_success[0] / histogram_values_all[0]
    
    pass_data_with_player_data = pass_data.merge(player_data, left_on='playerId', right_on='wyId')
    pass_data_for_position = pass_data_with_player_data[pass_data_with_player_data['player_position'] == position_of_player]
    x_start_metres = pass_data_for_position['pass_start_x'] * 105 / 100
    y_start_metres = pass_data_for_position['pass_start_y'] * 65 / 100

    all_histogram_values_all = np.histogram2d(y_start_metres, x_start_metres, bins=5, range=[[0, 65],[0, 105]])
    all_histogram_values_success = np.histogram2d(y_start_metres[pass_data_for_position['success']], 
                                                  x_start_metres[pass_data_for_position['success']], 
                                                  bins=5, range=[[0, 65],[0, 105]])
    all_histogram_values_probability = all_histogram_values_success[0] / all_histogram_values_all[0]

    difference_in_probability = histogram_values_probability - all_histogram_values_probability
    # Remove values where player has fewer than 10 passes, these are not statistically relevant
    difference_in_probability = [[p if histogram_values_all[0][c][i] > 10 else 0 for i, p in enumerate(row)] for c, row in enumerate(difference_in_probability)]

    fig, ax = createPitch(105, 65, 'meters', 'black')
    pos=ax.imshow(difference_in_probability, extent=(0, 105, 0, 65), aspect='auto', vmin=-0.3, vmax=0.3, cmap=plt.cm.PRGn)
    fig.colorbar(pos, ax=ax)
    ax.set_title(f'{player_name} compared to all players')
    plt.tight_layout()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


###############################################################################
# Logistic regression and expected pass                                       #
###############################################################################
def calculate_and_plot_logistic_polynomial_regression_curve(pass_data, 
                                                            variables=list(), 
                                                            plot=True,
                                                            file_name=None):
    """
    Calculate and print logistic regression model for variables provided, and 
    optionally plot curve with real data
    
    Parameters
    ----------
    pass_data (pd.DataFrame)
        DataFrame containing event data relating to passes, detailing whether 
        each pass is successful, and containing the variables for the logistic 
        regression
    
    variables (list)
        Fields in the DataFrame to use in logistic regression model
    
    plot (bool)
        Whether to product a plot. If making a plot it must either be for a 
        single variable model or for a model
        where all variables are powers of a single variable
    
    file_name (str)
        Optional. If provided, save plot with this file name     
    """
    test_model = smf.glm(formula=f"success ~ {' + '.join(variables)}", 
                         data=pass_data, family=sm.families.Binomial()).fit()
    print(test_model.summary())
    if plot:
        plot_relationship_scatter(pass_data, variables[0], 
                                  curve_parameters=test_model.params,
                                  file_name=file_name)
    
    return test_model.params()


def fit_logistic_regression_model(pass_data, variables=list()):
    """
    Calculate and print logistic regression model for variables provided
    
    Parameters
    ----------
    pass_data (pd.DataFrame)
        DataFrame containing event data relating to passes, detailing whether 
        each pass is successful, and containing the variables for the logistic 
        regression
    
    variables (list)
        Fields in the DataFrame to use in logistic regression model
    """
    test_model = smf.glm(formula=f"success ~ {' + '.join(variables)}", data=pass_data, 
                         family=sm.families.Binomial()).fit()
    print(test_model.summary())
    
    return test_model.params


def calculate_xPass(pass_to_evaluate, model_parameters): 
    """
    Given a row of event data for a pass, b_values for a logistic regression and 
    the model variables, calculate the xPass for the pass.
    """
    bsum = model_parameters[0]
    for i in model_parameters[1:].index:
        variable_name = i.split('[')[0]
        b_val = model_parameters[i]
        bsum = bsum + b_val * pass_to_evaluate[variable_name]
    
    xPass = 1 / (1 + np.exp(bsum))

    return xPass


def plot_calibration_curve(pass_data_with_predictions, file_name=None):
    """
    Plot calibration curve - given pass data with predictions produce plot to show how
    well calibrated the model that made those predictions is
    """
    
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    fraction_of_positives, mean_predicted_value = calibration_curve(pass_data_with_predictions['success'],
                                                                    pass_data_with_predictions['xPass'],
                                                                    n_bins=10)
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
             label="model")
    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')
    
    if file_name is not None:
        plt.savefig(f'{file_name}.pdf', dpi=300, bbox_inches="tight")
    plt.show()
    

def fit_and_evaluate_model(pass_data, model_variables, plot_file_name=None):
    """
    Given a feature vector for pass data, and a list of variables from that vector to 
    use in the model:
        * fit a logistic regression
        * calculate xPass for the rows in the pass data
        * plot a calibration curve to see how well calibrated model is
    
    Return pass data with xPass as an additional field
    """
    
    pass_data = pass_data.copy()
    
    model_params = fit_logistic_regression_model(pass_data, model_variables)
    pass_data['xPass'] = pass_data.apply(lambda x: calculate_xPass(x, model_params), axis=1)
    plot_calibration_curve(pass_data, file_name=plot_file_name)
    
    return pass_data


# Set the folder containing Wyscout data
data_folder = 'data/'

# Load event data
event_data = load_wyscout_event_data(data_folder)

# Get pass data and derive features of possible expected pass models
# This code would load the pass data in one go. As this has all been a bit exploratory, we've built this up bit by bit instead
event_data_pipeline = EventDataPipeline()
event_data_pipeline.add_filters([get_shot_assist_flag])
event_data = event_data_pipeline.run_get_event_data_pipeline(event_data)

pass_data_pipeline = PassDataPipeline()
pass_data_pipeline.add_filters([
        remove_rows_with_no_pass_end_position, get_success_flag_from_tags, 
        get_start_coordinates_of_pass_from_position, get_end_coordinates_of_pass_from_position,
        get_start_coordinates_squared, get_product_of_start_coords, get_pass_type,
        get_angle_and_distance_of_pass, get_part_of_field_for_pass_start_and_end,
        get_end_position_to_goal_details])
pass_data = pass_data_pipeline.run_get_pass_data_pipeline(event_data)

# Look at heatmaps for the pitch for all passes, completed passes and failed passes
"""
Let's start by simply plotting a heatmap of pass start co-ordinates for all passes, 
completed passes and uncompleted passes to see how these differ. We expect to see 
more passes fail higher up the pitch and from positions where balls are typically 
played into the box (ie, crosses, through balls, etc).
"""
x_start_metres = pass_data['pass_start_x'] * 105 / 100
y_start_metres = pass_data['pass_start_y'] * 65 / 100
# All passes
plot_event_heatmap(y_start_metres, x_start_metres, 'Number of passes',
                   file_name='heatmap_all_passes')
# Completed passes
plot_event_heatmap(y_start_metres[pass_data['success']], 
                   x_start_metres[pass_data['success']], 
                   'Number of passes successful',
                   file_name='heatmap_completed_passes')
# Failed passes
plot_event_heatmap(y_start_metres[~pass_data['success']], 
                   x_start_metres[~pass_data['success']], 
                   'Number of passes not successful',
                   file_name='heatmap_failed_passes')


# Now look at the relationship between coordinates and pass probability
plot_relationship_scatter(pass_data, 'pass_start_x',
                         title='Probability of pass success based on position in length of pitch',
                         file_name='relationship_between_x_and_pass_success')
"""
In the x-direction (ie, along the length of the pitch) this confirms what we can 
see in the plot of the pitch above:

Near a team's own goal line passes have a middling probability of success. Here 
passes may be defensive clearances
In the middle of the park, passes are most likely to be successful. Here teams 
are more likely to have the ball under less pressure from opponents
In the final third probability of pass success increases. Passes need to be 
more creative here to break through an opponent defensive line, and many will be 
crosses which tend to be less successful.
This looks like a quadratic relationship, so likely that including x**2 will 
improve the model
"""
plot_relationship_scatter(pass_data, 'pass_start_y', 
                          title='Probability of pass success based on position in width of pitch',
                          file_name='relationship_between_x_and_pass_success')
"""
In the y-direction (ie, along the width of the pitch):

Passes are less likely to be successful nearer the touch lines. This is not 
surprising as here we're more likely to see crosses, or other balls into the box
Passes are more likely to be successful in the centre of the width. Based on 
the heatmaps above these are likely to be dominated by passes in the centre of 
the park, where there is often an emphasis on keeping posession.
This looks like a quadratic relationship, so likely that including y**2 will 
improve the model
"""

# Fitting logistic regression models

# First the original models considering only either x or y variables
# calculate_and_plot_logistic_polynomial_regression_curve(pass_data, ['pass_start_x'])
# calculate_and_plot_logistic_polynomial_regression_curve(pass_data, ['pass_start_x', 'pass_start_x_squared'])
# calculate_and_plot_logistic_polynomial_regression_curve(pass_data, ['pass_start_y'])
# calculate_and_plot_logistic_polynomial_regression_curve(pass_data, ['pass_start_y', 'pass_start_y_squared'])

# Considering calibration, we build the model up with more features and see
# calibration improve
# pass_data_for_simplest_model = fit_and_evaluate_model(pass_data,
#                                                       ['pass_start_x', 'pass_start_y'])

# pass_data_for_coordinates_squared = fit_and_evaluate_model(
#     pass_data, ['pass_start_x', 'pass_start_y', 'pass_start_x_squared', 'pass_start_y_squared'])

# pass_data_for_pass_type = fit_and_evaluate_model(
#     pass_data, 
#     ['pass_start_x', 'pass_start_y', 'pass_start_x_squared', 'pass_start_y_squared', 
#      'start_coords_product', 'simple_pass', 'high_pass', 'head_pass',
#      'smart_pass', 'launch', 'cross', 'hand_pass'])
    
pass_data_with_xPass = fit_and_evaluate_model(
    pass_data,
    ['pass_start_x', 'pass_start_y', 'pass_start_x_squared', 'pass_start_y_squared',
     'start_coords_product', 'simple_pass', 'high_pass', 'head_pass',
     'smart_pass', 'launch', 'cross', 'hand_pass', 'start_third_of_field',
     'end_third_of_field', 'pitch_progression_score', 'angle_of_pass',
     'distance_of_pass', 'end_distance_from_goal', 'end_angle_between_goalpost'],
     plot_file_name='calibration_curve_all_features_model')
"""
As we incrementally add features,the calibration curve improves, so we pick the
final model to use.
Important to note that P values are close to zero for all features, so all have
a relationship to pass success.
"""


# With this model, we lets look try to identify good passers.
# We define a good passer for the purposes of this as a passer who succeeds 
# with difficult passes more than average

# We'll focus on the EPL
competition_to_evaluate = 'English first division'

pass_data_for_competition = get_pass_data_for_competition(pass_data_with_xPass,
                                                          competition_to_evaluate,
                                                          data_folder)
pass_data_for_competition['pass_outcome_vs_xPass'] = pass_data_for_competition['success'] - pass_data_for_competition['xPass']

"""
How to find good passing midfielders:

Identify hardest passes, and look for players in a certain position who are best at making these
Hardest passes can be passes which model thinks have a less than X% chance of success
When looking for players who make these - what proportion of hard passes attempted do they complete?
What is their xPass difference?
Where on the pitch are they making difficult passes?
Plot heatmap for player to compare with overall
"""
fig, ax = plt.subplots()
ax.violinplot(pass_data_for_competition['xPass'])
ax.hlines(y=0.8, xmin=0.8, xmax=1.2)
plt.show()
# Looking at this violon plot for xPass for each pass in the competition data, 
# we can see that the majority of passes are above 0.8 xPass. So let's restrict 
# ourselves to looking at passes which have a 80% chance or lower of being 
# successful accoring to our model

low_probability_passes = pass_data_for_competition[pass_data_for_competition['xPass'] <= 0.8]
player_data = get_player_data(data_folder)

# Get count of passes per player, xPass added value per player and average xPass added per pass - just looking at midfielders
total_x_pass = low_probability_passes[['playerId', 'pass_outcome_vs_xPass']].groupby('playerId').sum()
total_x_pass.reset_index(inplace=True)
total_passes = low_probability_passes[['playerId', 'eventId']].groupby('playerId').count()
total_passes.reset_index(inplace=True)
total_passes = total_passes.rename(columns={'eventId': 'pass_attempt_count'})
player_data_with_total_xPass = total_x_pass.merge(player_data, left_on='playerId', right_on='wyId')
player_data_with_total_xPass = player_data_with_total_xPass.merge(total_passes, on='playerId')
player_data_with_total_xPass = player_data_with_total_xPass[player_data_with_total_xPass['player_position'] == 'Midfielder']
player_data_with_total_xPass['xPass_per_pass'] = player_data_with_total_xPass['pass_outcome_vs_xPass'] / player_data_with_total_xPass['pass_attempt_count']

player_data_with_total_xPass = player_data_with_total_xPass[player_data_with_total_xPass['pass_attempt_count'] >= 46]
player_data_with_total_xPass = player_data_with_total_xPass.sort_values(by='pass_outcome_vs_xPass', ascending=False)

plot_xpass_per_pass_vs_total_pass_count(player_data_with_total_xPass)
    
player_name_to_plot = 'Mohamed Elneny'
player_id_to_plot = 120339
plot_heatmaps_for_player(player_id_to_plot, pass_data_for_competition, player_name_to_plot)
plot_probability_difference_heatmap_for_player(player_id_to_plot, player_name_to_plot, pass_data_for_competition,
                                               player_data, 'Midfielder')
