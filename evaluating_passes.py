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


###############################################################################
# Library of functions to load Wyscout data                                   #
###############################################################################
def load_wyscout_event_data(data_folder='data/'):
    """
    Given a folder location holding Wyscout data, load all json files for event 
    data and put all data in a single dataframe, which is returned.
    """

    data_folder = data_folder.strip('/')

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


###############################################################################
# Pass data pipeline and filters library to generate features for pass data   #
# using the Wyscout event data                                                #
###############################################################################
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

def plot_event_heatmap(y_events, x_events, title, no_of_bins=10):
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
    plt.show()


def plot_relationship_scatter(pass_data, variable_to_plot, curve_parameters=None):
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
    plt.show()


def calculate_and_plot_logistic_polynomial_regression_curve(pass_data, 
                                                            variables=list(), 
                                                            plot=True):
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
        
    """
    test_model = smf.glm(formula=f"success ~ {' + '.join(variables)}", 
                         data=pass_data, family=sm.families.Binomial()).fit()
    print(test_model.summary())
    if plot:
        plot_relationship_scatter(pass_data, variables[0], 
                                  curve_parameters=test_model.params)



event_data = load_wyscout_event_data()

pass_data_pipeline = PassDataPipeline()
pass_data_pipeline.add_filters([remove_rows_with_no_pass_end_position,
        get_success_flag_from_tags, get_start_coordinates_of_pass_from_position,
        get_end_coordinates_of_pass_from_position, get_start_coordinates_squared])
pass_data = pass_data_pipeline.run_get_pass_data_pipeline(event_data)
