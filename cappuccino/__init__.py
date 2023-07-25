import subprocess
import sys
import os
# from boundary_checker import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import statistics as stats
import pandas as pd
from scipy import signal
import sys
from datetime import datetime
import h5py
from scipy.stats import pearsonr   
from tqdm import tqdm
import traceback
import hdf5plugin
import argparse


def main():
    """This is the main function.

    Args:
        candidates_df_name (_type_): _description_
    """


    ## load all parameters of the pipeline
    parser = argparse.ArgumentParser(
                    prog='Cappuccino',
                    description='A cross-correlation based filter to find ET signals in GBT data.',
                    epilog="Documentation: https://bsrc-cappuccino.readthedocs.io/en/latest/")

    
    parser.add_argument('-b', '--block_size',dest='block_size', help="block size",default=500,action="store")
    parser.add_argument('-p', '--pearson_threshold',dest='pearson_threshold', help="pearson threshold",default=.3,action="store")
    parser.add_argument('-s', '--significance_level',dest='significance_level', help="mimimum SNR for a signal",default=10,action="store")
    parser.add_argument('-e', '--edge',dest='edge', help="maximum drift rate in units of frequency bin (~2.79 Hz)",default=50,action="store")

    args = vars(parser.parse_args())

    # block size is how large of a frequency range we iterate over while searching for signals
    block_size = int(args["block_size"])
    # the minimum correlation score required to reject a signal as being strongly correlated.
    pearson_threshold = float(args['pearson_threshold'])
    # significance level is how strong a signal must be so that we flag the area and do a more detailed filtering.
    significance_level = int(args["significance_level"])
    # edge is the maximum extent we will slide the two boundaries to match them up. On average there is a 30s slew time between observation. 
    # --> If we assume a max drift rate of 4 Hz/s, this corresponds to 120Hz, which is ~43 frequency bins
    edge = int(args["edge"])



    # check if candidates database is set up, if not then initialize it. This is where the candidates will be stored
    main_dir = os.getcwd() + "/"
    df_name = f'second_updated_candidate_events_sigma_{significance_level}_pearsonthreshold_{int(pearson_threshold*10)}_blocksize_{block_size}_edge_{edge}.csv'

    db_exists = os.path.exists(main_dir+df_name)
    if db_exists == False:
        print("Creating candidates database as ",df_name)
        candidates = pd.DataFrame(columns=["Primary File","Frequency","All Files","Score"])
        candidates.to_csv(main_dir+df_name,index=False)
    else:
        print("Candidates database already exists:",df_name)


    # define the target list you want to search through. These should be folders in the current directory, with .h5 files of entire cadences in each of them
    target_list = ['AND_II','AND_I', 'AND_X', 'AND_XI', 'AND_XIV', 'AND_XVI', 'AND_XXIII', 'AND_XXIV', 'BOL520', 'CVNI', 'DDO210', 'DRACO', 'DW1','HERCULES', 'HIZSS003', 'IC0010', 'IC0342', 'IC1613', 'LEOA', 'LEOII', 'LEOT', 'LGS3', 'MAFFEI1', 'MAFFEI2', 'MESSIER031', 'MESSIER033', 'MESSIER081', 'MESSIER101', 'MESSIER49', 'MESSIER59', 'MESSIER84', 'MESSIER86', 'MESSIER87', 'NGC0185', 'NGC0628', 'NGC0672 ', 'NGC1052', 'NGC1172 ', 'NGC1400', 'NGC1407', 'NGC2403','NGC2683', 'NGC2787', 'NGC3193', 'NGC3226', 'NGC3344', 'NGC3379', 'NGC4136', 'NGC4168', 'NGC4239', 'NGC4244', 'NGC4258', 'NGC4318', 'NGC4365', 'NGC4387', 'NGC4434', 'NGC4458', 'NGC4473', 'NGC4478', 'NGC4486B', 'NGC4489', 'NGC4551', 'NGC4559', 'NGC4564', 'NGC4600', 'NGC4618', 'NGC4660', 'NGC4736', 'NGC4826', 'NGC5194', 'NGC5195', 'NGC5322', 'NGC5638', 'NGC5813', 'NGC5831', 'NGC584', 'NGC5845', 'NGC5846', 'NGC596', 'NGC636', 'NGC6503', 'NGC6822', 'NGC6946', 'NGC720', 'NGC7454 ', 'NGC7640', 'NGC821', 'PEGASUS', 'SAG_DIR', 'SEXA', 'SEXB', 'SEXDSPH', 'UGC04879', 'UGCA127', 'UMIN']
    candidates = pd.read_csv(main_dir+df_name)

    # iterate through each target, grabbing the correct files. Files get grouped in cadences by node number and put in a list. 
    for target in target_list:        
        print("Running boundary checker for target:",target)
        unique_h5_files,unique_nodes = get_all_h5_files(target)
        # print total number of files in target folder
        count = sum( [ len(listElem) for listElem in unique_h5_files])
        print(f"{count} files")
        # change back into main directory
        os.chdir(main_dir)

        try:
            # iterate through each node (cadence)
            for i in range(0,len(unique_h5_files)):
                # for the moment skip spliced files due to their size
                if unique_nodes[i] != "splic":
                    # grab the specific cadence to look at
                    h5_files = unique_h5_files[i]
                    # pass the files into the boundary_checker wrapper function. Returns flagged frequencies and respective scores
                    low_correlation_frequencies,scores= main_boundary_checker(h5_files,pearson_threshold,block_size,significance_level,edge)

                    # append all flagged frequencies to the candidates database
                    for i in range(0,len(low_correlation_frequencies)):
                        freq = low_correlation_frequencies[i]
                        score = scores[i]
                        candidates.loc[len(candidates.index)] = [h5_files[0],freq,h5_files,score]
                    print(candidates)
                    
                    # update candidates database
                    candidates.to_csv(main_dir+df_name,index=False)
                else: 
                    print("skipping spliced files")
        except Exception:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ERROR ON TARGET {target} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(traceback.print_exc())


def get_all_h5_files(target):
    """Returns a list containaing cadences grouped together as tuples, as well as a list of all unique nodes

    Args:
        target (str): Galaxy/Star (or overarching file folder) you are looking at

    :Returns:
        - h5_list (list): list containaing cadences grouped together as tuples
        - unique_nodes (list): list of all unique nodes
    """


    # initialize list to store h5 files
    h5_list = []

    # first change directory into the target directory
    os.chdir(target)
    data_dir = os.getcwd() + "/"

    # we want to get all the unique nodes
    unique_nodes = get_unique_nodes(data_dir)

    for node in unique_nodes:
    # then loop through and grab all the file names
        node_set = grab_node_file_list(data_dir,node)
        h5_list.append(node_set)

    return h5_list, unique_nodes


def get_unique_nodes(data_dir):
    """Grabs the unique blc nodes in a given directory

    Args:
        data_dir (str): Data directory to search through

    Returns:
        unique_nodes (list): List of all unique nodes in the directory, sorted.
    """
    node_list = []
    for dirname, _, filenames in os.walk(data_dir):
        for filename in filenames:
            # we remove the start and end nodes as these have low sensitivity
            if "blc" in filename and (filename[4] != '7') and (filename[4] != '0'):
                node_list.append(filename[:5])

    node_set = set(node_list)
    print("Unique nodes:", node_set)

    unique_nodes = sorted(node_set)
    unique_nodes.sort()
    return unique_nodes

def grab_node_file_list(data_dir,node_number):
    """Returns the list of h5 files associated with a given node

    Args:
        data_dir (str): Data directory to search through
        node_number (str): Node number to filter on

    Returns:
        data_list (list): List of h5 files that make up the cadence, sorted chronlogically
    """


    ## h5 list
    data_list = []
    for dirname, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename[-3:] == '.h5' and node_number in filename:
                data_list.append(data_dir + filename)
                
    data_list = sorted(data_list, key=lambda x: (x,x.split('_')[5]))

    return data_list


def get_boundary_data(hf_ON,hf_OFF,lower,upper,edge):
    """Takes two observations, returns frequency snippets closest in time to each other.

    Args:
        hf_ON (str): The observation that came chronologically first.
        hf_OFF (str): The observation that came second.
        lower (int): Lower index on the frequency range.
        upper (int): Upper index on the frequency range.
        edge (_type_): Additional frequency range to account for drift between first and second observations.

    Returns:
        row_ON (numpy array): The frequency values in specificed range from last time bin in first observation.
        row_OFF (numpy array): The frequency values in specificed range from first time bin in second observation.

    """

    row_ON = np.squeeze(hf_ON['data'][-1:,:,lower:upper],axis=1)[0]

    # we also grab the additional edge data to iterate over when calculating correlation
    row_OFF = np.squeeze(hf_OFF['data'][:1,:,lower-edge:upper+edge],axis=1)[0]

    return row_ON,row_OFF


def get_last_time_row(file):
    """Grabs the last time bin from a file. Used to iterate over to find 'hotspots' (regions with a strong signal)

    Args:
        file (str): Observation in question

    Returns:
        last_time_row (numpy array): Frequency values for last time bin in observation
    """
    hf = h5py.File(file, 'r')
    data = hf.get('data')
    last_time_row = data[-1]
    return last_time_row[0]

# def get_freq_slices(last_time_row,f_start,f_end):
#     """Grabs specific frequency range from 

#     Args:
#         last_time_row (_type_): _description_
#         f_start (_type_): _description_
#         f_end (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     freq_block = last_time_row[f_start:f_end]
#     return freq_block

def get_snr(sliced,sigma_multiplier):
    """Checks for any high SNR bins in the given frequency snippet and flags them.

    Args:
        sliced (numpy array): frequency snippet from observation
        sigma_multiplier (int): SNR threshold for a signal to count as significant

    Returns:
        snr (boolean): True if there is a high SNR signal, False if not
        threshold (int): Threshold that normalized data needs to be above in order to count as signal. 
    """

    snr = False
    # divide by max to make numbers smaller
    sliced = sliced/np.max(sliced)

    # remove top 30 percent of values to get real baseline (in case there are many high value signals). 
    lower_quantile = np.quantile(sliced,.85)
    lower_slice = sliced[sliced < lower_quantile]

    # get median and standard deviation of baseline
    median = np.median(lower_slice)
    sigma = np.std(lower_slice)

    # calcualate threshold as median of baseline + SNR * standard deviation
    threshold = median+sigma_multiplier*sigma
    if np.max(sliced) > threshold:
        snr = True

    return snr, threshold

def get_first_round_snr(sliced,first_round_multiplier):
    """Preliminary filter to find any regions with a certain SNR that is smaller than the specificed cutoff.
        Calculating the quantile of a lot of regions is time intensive, so better to narrow down search field first. 

    Args:
        sliced (numpy array): frequency snippet from observation
        first_round_multiplier (int): Lower SNR required for regions to get passed on to next round of filtering

    Returns:
        snr (boolean): True if there is a high SNR signal, False if not
        threshold (int): Threshold that normalized data needs to be above in order to count as signal. 
    """

    snr = False
    sliced = sliced/np.max(sliced)

    median = np.median(sliced)
    sigma = np.std(sliced)
    threshold = median+first_round_multiplier*sigma

    if threshold <= 1:
        snr = True

    return snr, threshold

    
def find_hotspots(row,number,block_size,significance_level):
    """Wrapper function for hotspot finding.

    Args:
        row (numpy array): Last row of first observation in cadence. Will be the one iterated over to check for hotspots
        number (int): Number of distinct regions of block_size in the row
        block_size (int): Number of frequency bins in the region
        significance_level (int): SNR threshold for signal to count as significant

    Returns:
        hotspots (list): List of block numbers as integers with a high signal in them
    """

    # list of all block regions that pass first round filtering
    first_round = []
    # list of all block regions that pass second round filtering
    hotspots = []
    # lower SNR required for first round filtering
    first_round_multiplier = 5

    # iterate through regions for first filter
    for i in tqdm(range(0,number)):
        slice_ON = row[i*block_size:(i+1)*block_size:]
        snr,threshold = get_first_round_snr(slice_ON,first_round_multiplier)
        if snr:
            first_round.append(i)

    # iterate through remaining regions for second filter
    for i in tqdm(first_round):
        slice_ON = row[i*block_size:(i+1)*block_size:]
        snr,threshold = get_snr(slice_ON,significance_level)
        if snr:
            hotspots.append(i)

    # return all regions with a signal above specificed significance_level
    return hotspots

def get_file_properties(f):
    """Get file properties of given h5 file.

    Args:
        f (h5 object): h5 file corresponding to desired observation

    Returns:
        fch1 (float): start frequency of observation in Mhz
        foff (float): frequency of each bin in Mhz

    """
    tstart=f['data'].attrs['tstart']
    fch1=f['data'].attrs['fch1']
    foff=f['data'].attrs['foff']
    nchans=f['data'].attrs['nchans']
    ra=f['data'].attrs['src_raj']
    decl=f['data'].attrs['src_dej']
    target=f['data'].attrs['source_name']
    # print("tstart %0.6f fch1 %0.10f foff %0.30f nchans %d cfreq %0.10f src_raj %0.10f src_raj_degs %0.10f src_dej %0.10f target %s" % (tstart,fch1,foff,nchans,(fch1+((foff*nchans)/2.0)),ra,ra*15.0,decl,target))
    print("Start Channel: %0.10f Frequency Bin: %0.30f # Channels: %d" % (fch1,foff,nchans))

    return fch1, foff

def get_correct_index(freq,fch1,foff):
    """Converts frequency to index integer for numpy array

    Args:
        freq (float): frequency region to be converted to index integer
        fch1 (float): start frequency of observation in Mhz
        foff (float): frequency of each bin in Mhz

    Returns:
        lower_bound (int): Lower index flanking freq
        upper_bound (int): Upper index flanking freq

    """
    distance = fch1 - freq
    number = int(np.round(-distance/foff))
    bound = 250
    return number+bound, number-bound

def filter_hotspots(hotspots,fch1,foff):
    """Filters out hotspots in RFI heavy regions. 

    Args:
        hotspots (list): List of hotspot regions found previously
        fch1 (float): start frequency of observation in Mhz
        foff (float): frequency of each bin in Mhz

    Returns:
        all_indexes: Remaining hotspots after filtering
    """

    # define regions that are RFI heavy:
    bad_regions = [[700,830],[1160,1340],[1370,1390],[1520,1630],[1675,1695],[1915,1935],[2025,2035],[2100,2120],[2180,2280],[2300,2360],[3800,4200],[4680,4800],[11000,11180]]

    # first convert hotspots indexes to frequency channels
    hotspots_frequencies = np.array([int((fch1+foff*(i*500))) for i in hotspots])


    all_indexes = []
    # iterate through bad regions and remove all hotspots in them
    for i in bad_regions:
        bottom = int(i[0])
        top = int(i[1])
        indexes = np.where(np.logical_and(bottom<hotspots_frequencies, hotspots_frequencies<top))
        indexes = indexes[0]
        indexes = [int(i) for i in indexes]
        for i in indexes:
            all_indexes.append(i)

    # return filtered hotspots
    return all_indexes


def check_hotspots(hf_obs1,hf_obs2,hf_obs3,hf_obs4,hf_obs5,hf_obs6,filtered_hotspots,pearson_threshold,significance_level,edge):
    low_correlations = []
    scores = []
    threshold = .5
    # we iterate through all of the hotspots
    for i in tqdm(filtered_hotspots):

        # define the block region we are looking at
        lower = i * 500
        upper = (i+1) * 500 

        
        try:

            # load the observations for off and ON. Last time bin for first obs, first time bin for seond obs
            # also check other boundaries


            row_ON, row_OFF = get_boundary_data(hf_obs1,hf_obs2,lower,upper,edge)
            same_signal_number = check_same_signal_number(row_ON,row_OFF,significance_level)


            # first just check if there are same number of signals
            if same_signal_number==False:
                Obs1 = np.squeeze(hf_obs1['data'][:,:,lower:upper],axis=1)
                # print(Obs1.shape)
                print("checking blips")
                not_constant_signal,threshold_blips = blip_checker(Obs1)                

                if not_constant_signal ==False :
                    # if it passes the initial check, perform pearson correlation

                    print("Checking correlation")
                    max_corr, current_shift = pearson_slider(row_ON,row_OFF,pearson_threshold,edge)

                    if max_corr < pearson_threshold:
                        # Obs1 = np.squeeze(hf_obs1['data'][:,:,lower:upper],axis=1)
                        # check if it is broadband RFI:
                        print("checking broadband")
                        is_broadband = check_broadband(Obs1)
                        if is_broadband == False:
                            
                            # load entire observation and see if time-summing the signal produces different result
                            Obs2 = np.squeeze(hf_obs2['data'][:,:,lower-edge:upper+edge],axis=1)
                            # will catch weak signals
                            fails_sum,integrated_pearson_score = second_filter(Obs1,Obs2,pearson_threshold,edge)
                
                
                            # then do a check that there is still a strong signal somewhere higher up in the observation --> so not just a little point right at the boundary
                            # time sum whole observation, and see if signal stands out
                
                            # also check if there was just a dim signal in the first OFF, maybe it gets stronger in second bin
                            second_row_corr,current_shift = pearson_slider(row_ON,Obs2[8],pearson_threshold,edge)
                            # can also check if same # of signals in middle of next row
                            check_same_signal_number_middle = check_same_signal_number(row_ON,Obs2[8],significance_level)

                            print("fails sum:",fails_sum)
                            print("not_constant_signal:",not_constant_signal)
                            print("second_row_corr",second_row_corr)
            
                            # if it passes these tests, check if it zero drift rate. Compare first time from first ON observation to last time from last OFF.
                            # if it has high correlation at drift rate = 0, then probably not good 
                            if fails_sum and second_row_corr < pearson_threshold and check_same_signal_number_middle == False:
                
                                print("Checking Drift Rate")
                
                                # sum whole observation and see if zero drift rate, if necessary
                                obs1_int = Obs1.sum(axis=0)
                
                                # load last OFF observation
                                Obs6 = np.squeeze(hf_obs6['data'][:,:,lower-edge:upper+edge],axis=1)
                                obs6_int = Obs6.sum(axis=0)
                
                                first_last_corr,current_shift = pearson_slider(obs1_int, obs6_int,pearson_threshold,edge)
                
                                if current_shift != 0 and abs(current_shift) != 1:
                
                                    # Final check will be to see if all the on signals are in the same place in all the off signals
                                    # time intensive bc we are loading all the data, so this is the very last check
                                    print("Second drift check")
                                    Obs2 = np.squeeze(hf_obs2['data'][:,:,lower:upper],axis=1)
                                    obs2_int = Obs2.sum(axis=0)
                
                                    Obs3 = np.squeeze(hf_obs3['data'][:,:,lower:upper],axis=1)
                                    obs3_int = Obs3.sum(axis=0)
                
                                    Obs4 = np.squeeze(hf_obs4['data'][:,:,lower:upper],axis=1)
                                    obs4_int = Obs4.sum(axis=0)
                
                                    Obs5 = np.squeeze(hf_obs5['data'][:,:,lower:upper],axis=1)
                                    obs5_int = Obs5.sum(axis=0)
                
                                    Obs6 = np.squeeze(hf_obs6['data'][:,:,lower:upper],axis=1)
                                    obs6_int = Obs6.sum(axis=0)
                                    
                                    on_sum = obs1_int+obs3_int+obs5_int
                                    off_sum = obs2_int+obs4_int+obs6_int
                                    whole_sum = obs1_int+obs3_int+obs5_int+obs2_int+obs4_int+obs6_int

                                    on_sum = on_sum/np.max(on_sum)
                                    off_sum = off_sum/np.max(off_sum)
                                    whole_sum = whole_sum/np.max(whole_sum)

                                        
                                        

                                    zero_drift = drift_index_checker(whole_sum, row_ON,significance_level)
                                    
                                    signal_stays_strong = True

                                    snr_obs3_0, threshold30 = get_snr(Obs3[0],significance_level)
                                    snr_obs3_16, threshold316 = get_snr(Obs3[-1],significance_level)
                                    snr_obs5_0, threshold50 = get_snr(Obs5[0],significance_level)
                                    snr_obs5_16, threshold516 = get_snr(Obs5[-1],significance_level)


                                    number_strong_boundaries = 0
                                    print("checking if signal peters out")
                                    for snr in [snr_obs3_0,snr_obs3_16,snr_obs5_0,snr_obs5_16]:
                                        print(snr)
                                        if snr == True:
                                            number_strong_boundaries += 1

                                    print(number_strong_boundaries)
                                    if number_strong_boundaries <=1:
                                        signal_stays_strong = False

                                    print('zero drift',zero_drift)
                                    print('signal',signal_stays_strong)
                                    if zero_drift == False and signal_stays_strong == True: 
                                        # check other boundaries
                                        # boundary 2/5
                                        row_ON2, row_OFF2 = get_boundary_data(hf_obs2,hf_obs3,lower,upper,edge)

                                        

                                        max_corr2, current_shift2 = pearson_slider(row_ON2,row_OFF2,pearson_threshold,edge)
                                        # boundary 3/5
                                        row_ON3, row_OFF3 = get_boundary_data(hf_obs3,hf_obs4,lower,upper,edge)
                                        max_corr3, current_shift3 = pearson_slider(row_ON3,row_OFF3,pearson_threshold,edge)
                                        # boundary 4/5
                                        row_ON4, row_OFF4 = get_boundary_data(hf_obs4,hf_obs5,lower,upper,edge)
                                        max_corr4, current_shift4 = pearson_slider(row_ON4,row_OFF4,pearson_threshold,edge)
                                        # boundary 5/5
                                        row_ON5, row_OFF5 = get_boundary_data(hf_obs5,hf_obs6,lower,upper,edge)
                                        max_corr5, current_shift5 = pearson_slider(row_ON5,row_OFF5,pearson_threshold,edge)
                                        # count how many had low pearson values
                                        score = 0

                                        for corr in [max_corr,max_corr2,max_corr3,max_corr4,max_corr5]:
                                            print('co:',corr)
                                            if corr < pearson_threshold:
                                                score +=1 
                
                
                                        print('score:',score)
                                        print(f"max correlation was {max_corr}")
                                        print(f"integrated correlation was {integrated_pearson_score}")
                                        print('current shift:',current_shift)
                                        print('firstlastcorr:',first_last_corr)

                                        # only return ones with low correlation on all boundaries
                                        if score == 5:
                                            low_correlations.append(i)
                                            scores.append(score)
        
        except Exception:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ERROR ON BLOCK # {i} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(traceback.print_exc())
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    

    return low_correlations, scores


def check_broadband(obs1):
    obs1_freq_integrated = obs1.sum(axis=1,dtype='float')
    obs1_time_integrated = obs1.sum(axis=0,dtype='float')

    blip_or_broadband = False

    # broadband will have a very high snr, can use ~100
    broadband_threshold = 10
    freq_snr, threshold_freq = get_snr(obs1_freq_integrated,broadband_threshold)

    # also check it isnt a blip
    time_snr,threshold_time = get_snr(obs1_time_integrated,5)

    if freq_snr == True or time_snr==False:
    # if freq_snr == True:
        blip_or_broadband = True

    return blip_or_broadband


def check_same_signal_number(row_ON,row_OFF,significance_level):
    
    row_ON = row_ON/np.max(row_ON)
    row_OFF = row_OFF/np.max(row_OFF)
    same_signal_number = False

    snr1, threshold1 = get_snr(row_ON,significance_level)
    snr6, threshold6 = get_snr(row_OFF,significance_level)

    indicesON = np.where(np.array(row_ON) > threshold1)[0].tolist()
    indicesOFF = np.where(np.array(row_OFF) > threshold6)[0].tolist()

    if len(indicesON) == len(indicesOFF):
        # print("same number of strong signals")
        same_signal_number = True

    return same_signal_number


def blip_checker(obs1):
    blip_threshold = 5

    not_constant_signal = False
    # make sure there is a signal and not just blips
    for i in [0,2,4,6,8,10,12,14]:
        int_snr,threshold2 = get_snr(obs1[i],blip_threshold)
        if int_snr == False:
            not_constant_signal = True

    return not_constant_signal, (threshold2)

def pearson_slider(boundary_ON,boundary_OFF,pearson_threshold,edge):

    # this initializes the psosible edges we iterate over for the pearson. We start at zero and work our way out, so as to be time efficient.
    possible_drifts = [0]
    for i in range(1,edge):
        possible_drifts.append(i)
        possible_drifts.append(-i)

    max_corr = 0
    max_pearson = 0
    current_shift = 0

    x = boundary_ON

    for i in possible_drifts:
        current_shift = i
        if i != edge:
            y = boundary_OFF[(edge+i):-(edge-i)]
        else: 
            y = boundary_OFF[(edge+i):]

        # print(len(x),len(y))

        x= x / np.max(x)
        y = y / np.max(y)
        pearson = pearsonr(x,y)

        pearson  = pearson[0]
        if pearson > max_pearson:
            max_pearson = pearson
            if pearson > pearson_threshold:
                # print("ran iterations:",i)
                break

    return max_pearson, current_shift

# this performs a similar step as pearson but will check time sum and i
def second_filter(obs1,obs2,pearson_threshold,edge):

    still_good = False
    # first try summing and checking correlation again --> possible the signal was too weak the first time
    obs1_time_integrated = obs1.sum(axis=0,dtype='float')
    obs2_time_integrated = obs2.sum(axis=0,dtype='float')
    integrated_pearson_score, current_shift = pearson_slider(obs1_time_integrated,obs2_time_integrated,pearson_threshold,edge)
    
    if integrated_pearson_score < pearson_threshold:
        still_good = True

    return still_good, integrated_pearson_score

def drift_index_checker(whole_sum, row_ON,significance_level):
    print("Checking Drift")
    zero_drift = False

    # we can also check if when we sum the entire observation, we pick up the signal that set off the hotspot. 
    # Will only do this if there are same number of peaks in on ROw and summed, in case there was a genuine signal in the ON row

    hotspot_snr, hotspot_threshold = get_snr(row_ON,significance_level)
    summed_snr, summed_threshold = get_snr(whole_sum,significance_level)

    hotspot_indices = np.where(np.array(row_ON) > hotspot_threshold)[0].tolist()
    summed_indices = np.where(np.array(whole_sum) > summed_threshold)[0].tolist()

    # do same filtering as above
    if len(hotspot_indices) != 0 and len(summed_indices) != 0:

        filtered_hotspot_indices = [hotspot_indices[0]]
        for i in hotspot_indices[1:]:
            if abs(filtered_hotspot_indices[-1] - i) <2:
                filtered_hotspot_indices.pop()
                filtered_hotspot_indices.append(i-.5)
            else:
                filtered_hotspot_indices.append(i)


        filtered_summed_indices = [summed_indices[0]]
        for i in summed_indices[1:]:
            if abs(filtered_summed_indices[-1] - i) <2:
                filtered_summed_indices.pop()
                filtered_summed_indices.append(i-.5)
            else:
                filtered_summed_indices.append(i)

        # check if all hotspots picked up in ON row are in the summed
        print(filtered_summed_indices, filtered_hotspot_indices)
        all = 0
        for i in filtered_hotspot_indices:
            for j in filtered_summed_indices:
                if abs(i-j) < 2:
                    all +=1 
        if all == len(filtered_hotspot_indices):
            print("ZERO DRIFT")
            zero_drift = True

    return zero_drift


def main_boundary_checker(h5_files,pearson_threshold,block_size,significance_level,edge):
        # load data
    hf_ON = h5py.File(h5_files[0], 'r')
    hf_OFF = h5py.File(h5_files[1], 'r')
    hf_ON2 = h5py.File(h5_files[2], 'r')
    hf_OFF2 = h5py.File(h5_files[3], 'r')
    hf_ON3 = h5py.File(h5_files[4], 'r')
    hf_OFF3 = h5py.File(h5_files[5], 'r')



    # grab last row to find hotspots
    last_time_row_ON = get_last_time_row(h5_files[0])

    # find file frequency information
    fch1,foff = get_file_properties(hf_ON)

    # calculate number of iterations needed and find hotspots
    number = int(np.round(len(last_time_row_ON)/block_size))
    # record interesting freq chunks
    hotspots_ON = find_hotspots(last_time_row_ON,number,block_size,significance_level)
    # hotspots_OFF = find_hotspots(last_time_row_OFF)
    print("# of hotspots:",len(hotspots_ON))

    # filter the hotspots
    filtered_indexes = filter_hotspots(hotspots_ON,fch1,foff)
    filtered_hotspots = np.delete(hotspots_ON, filtered_indexes)
    print("# post filtering",len(filtered_hotspots))

    # find regions of low correlation
    low_correlations,scores = check_hotspots(hf_ON,hf_OFF,hf_ON2,hf_OFF2,hf_ON3,hf_OFF3,filtered_hotspots,pearson_threshold,significance_level,edge)
    print(f"Found {len(low_correlations)} regions of low correlation")

    # save regions to csv file, in current directory

    # return frequency positions of candidates
    return [fch1+foff*(i*500) for i in low_correlations],scores

