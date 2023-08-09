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
import scipy.stats  
from tqdm.auto import tqdm
import traceback
import hdf5plugin
import argparse
import pickle
import gc



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

    
    parser.add_argument('-b', '--block_size',dest='block_size', help="block size",default=2048,action="store")
    parser.add_argument('-p', '--pearson_threshold',dest='pearson_threshold', help="pearson threshold",default=.7,action="store")
    parser.add_argument('-s', '--significance_level',dest='significance_level', help="mimimum SNR for a signal",default=10,action="store")
    parser.add_argument('-e', '--edge',dest='edge', help="maximum drift rate in units of frequency bin (~2.79 Hz)",default=50,action="store")
    parser.add_argument('-f', '--files',dest='files', help="path to text file with files list",default='',action="store")
    parser.add_argument('-d', '--directory',dest='directory', help="directory with cadences in it",default='',action="store")
    parser.add_argument('-n', '--number',dest='number', help="batch number of archival GBT data compilation",default=0,action="store")
    parser.add_argument('-t', '--target',dest='target', help="target desired: [Target Name, Day, Node]",default='',action="store")

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

    directory = (args["directory"])

    files = args["files"]
    
    target_line = args["target"]


    batch_number = int(args["number"])

    print("block_size:",block_size)
    # check if candidates database is set up, if not then initialize it. This is where the candidates will be stored
    # main_dir = os.getcwd() + "/"

    main_dir = '/mnt_blpc1/datax/scratch/calebp/cappuccino_runs/'


    if files == '' and target_line == '' and directory == '' and batch_number == '':
        df_name = f'default_candidate_events_sigma_{significance_level}_pearsonthreshold_{int(pearson_threshold*10)}_blocksize_{block_size}_edge_{edge}.csv'
        failures_name = f'failed_default_candidate_events_sigma_{significance_level}_pearsonthreshold_{int(pearson_threshold*10)}_blocksize_{block_size}_edge_{edge}.csv'
    elif target_line != '':
        n = len(target_line)
        target_info = target_line[1:n-1]
        target_info = target_info.split(',')
        target_name = target_info[0]
        target_date = target_info[1]
        target_node = target_info[2]
        df_name = f'target_{target_name}_date_{target_date}_node_{target_node}_sig_{significance_level}_pearsonthreshold_{int(pearson_threshold*10)}_blocksize_{block_size}_edge_{edge}.csv'
        failures_name = f'failed_target_{target_name}_date_{target_date}_node_{target_node}_sig_{significance_level}_pearsonthreshold_{int(pearson_threshold*10)}_blocksize_{block_size}_edge_{edge}.csv'
    else:
        df_name = f'all_batches_number_{batch_number}_sig_{significance_level}_pearsonthreshold_{int(pearson_threshold*10)}_blocksize_{block_size}_edge_{edge}.csv'
        failures_name = f'failed_all_batches_number_{batch_number}_sig_{significance_level}_pearsonthreshold_{int(pearson_threshold*10)}_blocksize_{block_size}_edge_{edge}.csv'


    # initialize candidates db
    db_exists = os.path.exists(main_dir+df_name)
    if db_exists == False:
        print("Creating candidates database as ",df_name)
        candidates = pd.DataFrame(columns=["Target","Frequency","Block Index","All Files","Score","Max Ranges","Start Freq","Bin Freq"])
        candidates.to_csv(main_dir+df_name,index=False)
    else:
        print("Candidates database already exists:",df_name)

    # initialize failures db
    db_exists = os.path.exists(main_dir+failures_name)
    if db_exists == False:
        print("Creating failures database as ",failures_name)
        failure_database = pd.DataFrame(columns=["Index","All Files"])
        failure_database.to_csv(main_dir+failures_name,index=False)
    else:
        print("Failures database already exists:",failures_name)


    # define the target list you want to search through. These should be folders in the current directory, with .h5 files of entire cadences in each of them
    # target_list = ['AND_II','AND_I', 'AND_X', 'AND_XI', 'AND_XIV', 'AND_XVI', 'AND_XXIII', 'AND_XXIV', 'BOL520', 'CVNI', 'DDO210', 'DRACO', 'DW1','HERCULES', 'HIZSS003', 'IC0010', 'IC0342', 'IC1613', 'LEOA', 'LEOII', 'LEOT', 'LGS3', 'MAFFEI1', 'MAFFEI2', 'MESSIER031', 'MESSIER033', 'MESSIER081', 'MESSIER101', 'MESSIER49', 'MESSIER59', 'MESSIER84', 'MESSIER86', 'MESSIER87', 'NGC0185', 'NGC0628', 'NGC0672 ', 'NGC1052', 'NGC1172 ', 'NGC1400', 'NGC1407', 'NGC2403','NGC2683', 'NGC2787', 'NGC3193', 'NGC3226', 'NGC3344', 'NGC3379', 'NGC4136', 'NGC4168', 'NGC4239', 'NGC4244', 'NGC4258', 'NGC4318', 'NGC4365', 'NGC4387', 'NGC4434', 'NGC4458', 'NGC4473', 'NGC4478', 'NGC4486B', 'NGC4489', 'NGC4551', 'NGC4559', 'NGC4564', 'NGC4600', 'NGC4618', 'NGC4660', 'NGC4736', 'NGC4826', 'NGC5194', 'NGC5195', 'NGC5322', 'NGC5638', 'NGC5813', 'NGC5831', 'NGC584', 'NGC5845', 'NGC5846', 'NGC596', 'NGC636', 'NGC6503', 'NGC6822', 'NGC6946', 'NGC720', 'NGC7454 ', 'NGC7640', 'NGC821', 'PEGASUS', 'SAG_DIR', 'SEXA', 'SEXB', 'SEXDSPH', 'UGC04879', 'UGCA127', 'UMIN']
    target_list = [directory]
    candidates = pd.read_csv(main_dir+df_name)
    failures_db = pd.read_csv(main_dir+failures_name)
    # iterate through each target, grabbing the correct files. Files get grouped in cadences by node number and put in a list. 




    # if we pass in a directory of files it just iterates over the files in galaxy directory
    if files == '' and target_line == '' and directory == '' and batch_number == '':

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
                        print("Now running on file ",h5_files[0])
                        low_correlation_frequencies,low_correlation_indexes,scores,ranges,failure_frequencies,fch1,foff= main_boundary_checker(h5_files,pearson_threshold,block_size,significance_level,edge)

                        # append all flagged frequencies to the candidates database
                        for i in range(0,len(low_correlation_frequencies)):
                            freq = low_correlation_frequencies[i]
                            block_index = low_correlation_indexes[i]
                            max_ranges = ranges[i]
                            score = scores[i]
                            primary_file = h5_files[0]
                            name = primary_file.split('/')[-1]
                            target = name.split("_")[-2]

                            candidates.loc[len(candidates.index)] = [target,freq,block_index,h5_files,score,max_ranges,fch1,foff]
                            
                        for i in range(0,len(failure_frequencies)):
                            failure_freq = failure_frequencies[i]
                            failures_db.loc[len(failures_db.index)] = [failure_freq,h5_files]

                        print(candidates)
                        
                        # update candidates database
                        candidates.to_csv(main_dir+df_name,index=False)
                        failures_db.to_csv(main_dir+failures_name,index=False)

                    else: 
                        print("skipping spliced files")
            except Exception:
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ERROR ON TARGET {target} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

                # keep track of targets that straight up fail
                failures_db.loc[len(failures_db.index)] = ["All",h5_files]
                failures_db.to_csv(main_dir+failures_name,index=False)

                print(traceback.print_exc())
    
    else:
        print("running on input files, batch #",batch_number)
        # load all files
        with open('/mnt_blpc1/datax/scratch/calebp/boundaries/cappuccino/all_batches_all_cadences_1000.pkl', 'rb') as f:
            reloaded_batches = pickle.load(f)
        
        # choose subset of all cadences (batches of 5000)
        all_file_paths = reloaded_batches[batch_number] 

        if target_line != '':

            print("TARGET:",target_name,target_date,target_node)
            all_file_paths = [find_cadence(target_name,target_date,target_node,reloaded_batches)]

        try:
            # iterate through each node (cadence)
            for i in range(0,len(all_file_paths)):
                h5_files = all_file_paths[i]

                primary_file = h5_files[0]
                name = primary_file.split('/')[-1]
                target = name.split("_")[-2]

                if target == "OFF":
                    temp = h5_files[0]
                    h5_files = h5_files[1:]
                    h5_files.append(temp)
                    primary_file = h5_files[0]
                    name = primary_file.split('/')[-1]
                    target = name.split("_")[-2]


                print(f'Cadence {i} out of {len(all_file_paths)}:')
                # grab the specific cadence to look at
                # pass the files into the boundary_checker wrapper function. Returns flagged frequencies and respective scores
                print("Now running on file ",h5_files[0])
                low_correlation_frequencies,low_correlation_indexes,scores,ranges,failure_frequencies,fch1,foff= main_boundary_checker(h5_files,pearson_threshold,block_size,significance_level,edge)

                # append all flagged frequencies to the candidates database
                for i in range(0,len(low_correlation_frequencies)):
                    freq = low_correlation_frequencies[i]
                    block_index = low_correlation_indexes[i]
                    max_ranges = ranges[i]
                    score = scores[i]
                    
                    candidates.loc[len(candidates.index)] = [target,freq,block_index,h5_files,score,max_ranges,fch1,foff]
                    
                for i in range(0,len(failure_frequencies)):
                    failure_freq = failure_frequencies[i]
                    failures_db.loc[len(failures_db.index)] = [failure_freq,h5_files]

                print(candidates)
                
                # update candidates database
                candidates.to_csv(main_dir+df_name,index=False)
                failures_db.to_csv(main_dir+failures_name,index=False)

        except Exception:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ERROR ON batch {batch_number} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

            # keep track of targets that straight up fail
            failures_db.loc[len(failures_db.index)] = ["All",h5_files]
            failures_db.to_csv(main_dir+failures_name,index=False)

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
        node_set = get_node_file_list(data_dir,node)
        h5_list.append(node_set)

    return h5_list, unique_nodes


def find_cadence(target,time,node,reloaded_batches):
    for batch in range(0,21):
        for cadence in reloaded_batches[batch]:
            combined_string = " ".join(cadence)
            if combined_string.count(target) >= 3 and time in combined_string and node in combined_string:
                return cadence

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

def get_node_file_list(data_dir,node_number):
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

    
# def find_hotspots(row,number,block_size,significance_level):
#     """Wrapper function for hotspot finding.

#     Args:
#         row (numpy array): Last row of first observation in cadence. Will be the one iterated over to check for hotspots
#         number (int): Number of distinct regions of block_size in the row
#         block_size (int): Number of frequency bins in the region
#         significance_level (int): SNR threshold for signal to count as significant

#     Returns:
#         hotspots (list): List of block numbers as integers with a high signal in them
#     """

#     # list of all block regions that pass first round filtering
#     first_round = []
#     # list of all block regions that pass second round filtering
#     hotspots = []
#     # lower SNR required for first round filtering
#     first_round_multiplier = 5

#     # iterate through regions for first filter
#     for i in tqdm(range(0,number)):
#         slice_ON = row[i*block_size:(i+1)*block_size:]
#         snr,threshold = get_first_round_snr(slice_ON,first_round_multiplier)
#         if snr:
#             first_round.append(i)

#     # iterate through remaining regions for second filter
#     for i in tqdm(first_round):
#         slice_ON = row[i*block_size:(i+1)*block_size:]
#         snr,threshold = get_snr(slice_ON,significance_level)
#         if snr:
#             hotspots.append(i)

#     # return all regions with a signal above specificed significance_level
#     return hotspots


def find_warmspots(row,number,block_size):
    first_round = []
    first_round_multiplier = 5
    # iterate

        
    for i in tqdm(range(0,number)):
        slice_ON = row[i*block_size:(i+1)*block_size:]
        snr,threshold = get_first_round_snr(slice_ON,first_round_multiplier)
        if snr:
            first_round.append(i)
    
    return first_round

def find_hotspots(row,first_round,block_size,significance_level):
    hotspots = []

    for i in tqdm(first_round):
        slice_ON = row[i*block_size:(i+1)*block_size:]
        snr,threshold = get_snr(slice_ON,significance_level)
        if snr:
            hotspots.append(i)

    
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

def filter_hotspots(hotspots,fch1,foff,block_size):
    """Filters out hotspots in RFI heavy regions. 

    Args:
        hotspots (list): List of hotspot regions found previously
        fch1 (float): start frequency of observation in Mhz
        foff (float): frequency of each bin in Mhz

    Returns:
        all_indexes: Remaining hotspots after filtering
    """

    # define regions that are RFI heavy:
    bad_regions = [[700,1100],[1160,1340],[1370,1390],[1520,1630],[1670,1705],[1915,2000],[2025,2035],[2100,2120],[2180,2280],[2300,2360],[2485,2500],[2800,4400],[4680,4800],[8150,8350],[9550,9650],[10700,12000]]
    # first convert hotspots indexes to frequency channels
    hotspots_frequencies = np.array([int((fch1+foff*(i*block_size))) for i in hotspots])


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


def check_hotspots(hotspot_slice_data,first_off,hf_obs1,hf_obs2,hf_obs3,hf_obs4,hf_obs5,hf_obs6,filtered_hotspots,pearson_threshold,significance_level,edge,block_size,filtered_hotspots_indexes,all_filtered_hotspots):
    """This is the primary filter wrapper, which contains the layers of filters that each cadence must pass through in order to be flagged.

    \b
    Args:
        hf_obs1 (h5 object): h5 Object for first observation |
        hf_obs2 (h5 object): h5 Object for second observation |
        hf_obs3 (h5 object): h5 Object for third observation |
        hf_obs4 (h5 object): h5 Object for fourth observation |
        hf_obs5 (h5 object): h5 Object for fifth observation |
        hf_obs6 (h5 object): h5 Object for sixth observation |
        filtered_hotspots (list): List of integers corresponding to hotspot number |
        pearson_threshold (int): Correlation threshold for a signal to be considered significant   
        significance_level (int): Minium SNR for a signal to be considered a "signal"
        edge (int): Max range of sliding for pearson correlation. Akin to max drift rate
        block_size (int): Size of hotspot regionl
    
    \b
    Returns:
        low_correlations (list): List of integers corresponding to regions that were flagged
        scores (list): Outdated, list of the score of each cadence --> how many boundaries had low correlation.
    """

    # first initialize the low correlation list
    low_correlations = []
    failures = []
    ranges = []
    scores = []

    # we iterate through all of the hotspots
    for i in tqdm(filtered_hotspots):

        # define the block region we are looking at 
        lower = i * block_size
        upper = (i+1) * block_size 

        
        try:
            dt = datetime.now()
            print('time start',dt.microsecond/1000)
            # load the observations for off and ON. Last time bin for first obs, first time bin for seond obs
            # find the correct obs based on the filter index
            observations_ON = [hf_obs1,hf_obs3,hf_obs5]
            observations_OFF = [hf_obs2,hf_obs4,hf_obs6]
            
            hotspot_index = all_filtered_hotspots.index(i)
            hotspot_slice = filtered_hotspots_indexes[hotspot_index]

            hotspot_slices = [0,1,2]
            print(hotspot_slice)

            primary_hf_ON = observations_ON[hotspot_slice]
            primary_hf_OFF = observations_OFF[hotspot_slice]

            hotspot_slices.remove(hotspot_slice)

            hf_ON_2 = observations_ON[hotspot_slices[0]]
            hf_OFF_2 = observations_OFF[hotspot_slices[0]]

            hf_ON_3 = observations_ON[hotspot_slices[1]]
            hf_OFF_3 = observations_OFF[hotspot_slices[1]]

            if hotspot_slice == 0:
                row_ON = hotspot_slice_data[hotspot_slice][lower:upper]
                row_OFF = first_off[lower-edge:upper+edge]
            else:
                row_OFF = np.squeeze(primary_hf_OFF['data'][:1,:,lower-edge:upper+edge],axis=1)[0]
                row_ON = np.squeeze(primary_hf_ON['data'][-1:,:,lower:upper],axis=1)[0]

            # row_ON, row_OFF = get_boundary_data(primary_hf_ON,primary_hf_OFF,lower,upper,edge)           
            # first just check if there are same number of signals in the first ON and OFF. If there aren't pass it on.

            # normalize the data
            row_ON = row_ON/np.max(row_ON)
            row_OFF = row_OFF/np.max(row_OFF)

            same_signal_number, initial_signal_number = check_same_signal_number(row_ON,row_OFF,significance_level,"ON")

            dt = datetime.now()
            print('time signal',dt.microsecond/1000)


            if same_signal_number==False:
                print("checking correlation")

                # if it passes these two initial checks, perform pearson correlation check
                max_corr, current_shift = pearson_slider(row_ON,row_OFF,pearson_threshold,edge)
                dt = datetime.now()
                print('time corr',dt.microsecond/1000)

                if max_corr < pearson_threshold:
                    # Now check if it is broadband RFI:
                    primary_obs = np.squeeze(primary_hf_ON['data'][:,:,lower:upper],axis=1)
                    is_broadband = check_broadband(primary_obs)
                    is_blip = check_blip(primary_obs)

                    #sum primary_obs
                    primary_time_integrated = primary_obs.sum(axis=0,dtype='float')

                    # we can also check the last ON row with the entire ON observation summed
                    not_drifting = drift_index_checker(primary_obs[0], row_ON,significance_level,3)
                    print("not drifting",not_drifting)
                    print("is broadband",is_broadband)
                    print("is_blip",is_blip)

                    dt = datetime.now()
                    print('time drift',dt.microsecond/1000)

                    if is_broadband == False and not_drifting == False and is_blip == False:
                        # load entire observation and see if time-summing the signal produces different result --> signal might be weak
                        secondary_obs = np.squeeze(primary_hf_OFF['data'][:,:,lower-edge:upper+edge],axis=1)
                        secondary_time_integrated = secondary_obs.sum(axis=0,dtype='float')

                        # first check same signal number:
                        check_same_signal_number_integrated, integrated_signal_number = check_same_signal_number(primary_time_integrated,secondary_time_integrated,significance_level,"ON")

                        print('same_signal integrated',check_same_signal_number_integrated)

                        if check_same_signal_number_integrated == False:

                            # then check if integreated pearson correlation is high
                            passes_integrated,integrated_pearson_score = second_filter(primary_time_integrated,secondary_time_integrated,pearson_threshold,edge)
                            print('passes integrated',passes_integrated)

                            if passes_integrated:
                                # also check if there was just a dim signal in the first OFF, maybe it gets stronger in second bin.                             
                                #can also check if same # of signals in middle of next row
                                same_signal_number_middle, middle_signal_number = check_same_signal_number(row_ON,secondary_obs[8],significance_level,"ON")
                                print('same_signal_number_middle',same_signal_number_middle)


                                if same_signal_number_middle == False:
                                    # finally do a correlation check --> maybe that is problem
                                    second_row_corr,current_shift = pearson_slider(row_ON,secondary_obs[8],pearson_threshold,edge)
                                    print('second_row_corr',second_row_corr)


                                    if second_row_corr < pearson_threshold:
                                    # if it passes these tests, check if it zero drift rate. Compare first time from first ON observation to last OFF.
                                    # if it has high correlation at drift rate = 0, then probably not good 
                                        print("Checking Drift Rate and Signal Strength")
                        
                                        # sum whole observation and see if zero drift rate, if necessary
                        
                                        # also check that there is still a legitamate signal at other boundaries --> i.e. it does not just peter out over time
                                        signal_stays_strong = True

                                        # we check the other 5 boundaries in ON observations only 

                                        obs1_row1 = np.squeeze(primary_hf_ON['data'][:1:,:,lower:upper],axis=1)[0]
                
                                        obs2_row1 = np.squeeze(hf_ON_2['data'][:1,:,lower:upper],axis=1)[0]
                                        obs2_row16 = np.squeeze(hf_ON_2['data'][-1:,:,lower:upper],axis=1)[0]
                                        
                                        obs3_row1 = np.squeeze(hf_ON_3['data'][:1:,:,lower:upper],axis=1)[0]
                                        obs3_row16 = np.squeeze(hf_ON_3['data'][-1::,:,lower:upper],axis=1)[0]

                                        on_boundaries = [obs1_row1,obs2_row1,obs2_row16,obs3_row1,obs3_row16]
                                        on_boundaries_snrs = []
                                        number_of_peaks = []
                                        for boundary in on_boundaries:
                                            boundary_snr, boundary_threshold = get_snr(boundary,significance_level)
                                            quant = np.quantile(boundary,.85)
                                            lower_slice = boundary[boundary < quant]
                                            sigma = np.std(lower_slice)
                                            peaks, properties = signal.find_peaks(boundary, prominence=significance_level*sigma, width=1)
                                            if len(peaks)>0:
                                                boundary_snr = True

                                            on_boundaries_snrs.append(boundary_snr)
                                            number_of_peaks.append(len(peaks))

                                        # we require that there must be a strong signal at the boundary of at least two other ON observations
                                        number_strong_boundaries = 0
                                        for b in range(0,len(on_boundaries_snrs)):
                                            if on_boundaries_snrs[b] == True and number_of_peaks[b] >= initial_signal_number:
                                                number_strong_boundaries += 1

                                        if number_strong_boundaries <=1:
                                            signal_stays_strong = False

                                        boundaries_summed = obs2_row1+obs2_row16+obs3_row1+obs3_row16
                                        boundaries_summed = boundaries_summed/np.max(boundaries_summed)
                                        boundary_drift = drift_index_checker(boundaries_summed, row_ON,significance_level,significance_level)

                                        if boundary_drift==False and signal_stays_strong:
                        
                                            # Final check will be to see if all the signal that set off the hotspot are in the same place when you sum the whole observation
                                            # time intensive bc we are loading all the data, so this is a very last check
                                            print("Second drift check")
                                            Obs1 = np.squeeze(hf_obs1['data'][:,:,lower:upper],axis=1)
                                            obs1_int = Obs1.sum(axis=0)

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
                                            whole_sum = obs1_int+obs3_int+obs5_int+obs2_int+obs4_int+obs6_int
                                                                                    # subtract primary obs so it does not bias it
                                            whole_sum -= primary_obs.sum(axis=0)


                                            on_sum = on_sum/np.max(on_sum)
                                            whole_sum = whole_sum/np.max(whole_sum)
                                            row_ON = row_ON/np.max(row_ON)

                                            zero_drift = drift_index_checker(whole_sum, row_ON,significance_level,10)
                                            

                                            # calculate k-score
                                            cadence_max = np.max([np.max(Obs1),np.max(Obs2),np.max(Obs3),np.max(Obs4),np.max(Obs5),np.max(Obs6)])
                
                                            obs1_values = (Obs1/cadence_max).flatten()
                                            obs2_values = (Obs2/cadence_max).flatten()
                                            obs3_values = (Obs3/cadence_max).flatten()
                                            obs4_values = (Obs4/cadence_max).flatten()
                                            obs5_values = (Obs5/cadence_max).flatten()
                                            obs6_values = (Obs6/cadence_max).flatten()
                                        
                                        
                                            k1 = scipy.stats.kurtosis(obs1_values)
                                            k2 = scipy.stats.kurtosis(obs2_values)
                                            k3 = scipy.stats.kurtosis(obs3_values)
                                            k4 = scipy.stats.kurtosis(obs4_values)
                                            k5 = scipy.stats.kurtosis(obs5_values)
                                            k6 = scipy.stats.kurtosis(obs6_values)

                                            # a good candidate should have a higher kurtosis in ON observation than OFF observation.
                                            k_score = (k1+k3+k5)/(k2+k4+k6)
                                            stronger_k_in_ONs = 0
                                            strong_k_in_ONs = 0

                                            # Here we want to get rid of very weak signals that are basically blips in one observation
                                            # we check first if there are at least 2 ON observations that have > k than their following OFF --> Weak blips won't pass this, but its also possible that observations with multiple signals won't too
                                            # so we add another option --> That there are at least 2 ONs that have some high kurtosis --> Weak blips won't pass this, but observations with multiple signals will.
                                            
                                            if k1 > k2:
                                                stronger_k_in_ONs +=1 
                                            if k3 > k4:
                                                stronger_k_in_ONs +=1 
                                            if k5 > k6:
                                                stronger_k_in_ONs +=1 


                                            if k1 > 1:
                                                strong_k_in_ONs +=1 
                                            if k3 > 1:
                                                strong_k_in_ONs +=1 
                                            if k5 > 1:
                                                strong_k_in_ONs +=1 


                                            # if it passes both drift rate and the check that the signal stays strong, we do a last check of the pearson correlation at all boundaries.
                                            if zero_drift == False and (stronger_k_in_ONs >=2 or strong_k_in_ONs>=2): 
                                                # check other boundaries
                                                # first load boundary data

                                                # make sure to label correctly which ones come from ON target observations and which ones from OFF target observations
                                                # we will end up returning the number of signals in the OFF target observations, as another check to see if they are > than the ON
                                                row_OFF2, row_ON2 = get_boundary_data(hf_obs2,hf_obs3,lower,upper,edge)
                                                row_ON3, row_OFF3 = get_boundary_data(hf_obs3,hf_obs4,lower,upper,edge)
                                                row_OFF4, row_ON4 = get_boundary_data(hf_obs4,hf_obs5,lower,upper,edge)
                                                row_ON5, row_OFF5 = get_boundary_data(hf_obs5,hf_obs6,lower,upper,edge)

                                                # check signal numbers at boundaries
                                                same_signal_2, num_signals_2 = check_same_signal_number(row_ON2,row_OFF2,significance_level,"OFF")
                                                # boundary 3/5
                                                same_signal_3,num_signals_3 = check_same_signal_number(row_ON3,row_OFF3,significance_level,"OFF")
                                                # boundary 4/5
                                                same_signal_4,num_signals_4 = check_same_signal_number(row_ON4,row_OFF4,significance_level,"OFF")
                                                # boundary 5/5
                                                same_signal_5,num_signals_5 = check_same_signal_number(row_ON5,row_OFF5,significance_level,"OFF")

                                                same_signals = [same_signal_2,same_signal_3,same_signal_4,same_signal_5]

                                                # this is num of signals in the ON boundary --> will be used to check that signal is not drifting out of frame
                                                num_signals = [num_signals_2,num_signals_3,num_signals_4,num_signals_5]

                                                num_same_signals = 0

                                                for num in range(0,4):
                                                    # its not okay to have less signals (good signals might drift out of region), but it is okay to have = or > than initial
                                                    print(same_signals[num])
                                                    print('num',num_signals[num])
                                                    print('initial',initial_signal_number)
                                                    if same_signals[num] == True and num_signals[num] >= initial_signal_number:
                                                        num_same_signals +=1

                                                # also calculate another statistic: The range of the max vaules in the ON observations --> a strong signal should not have a large range
                                                # won't filter on this, but will append to table.

                                                observations = [Obs1/np.max(Obs1),Obs3/np.max(Obs3),Obs5/np.max(Obs5)]

                                                obs_time_maxes = []
                                                for number in [0,1,2]:
                                                    time_maxes = []
                                                    for time in range(16):
                                                        time_max = np.max(observations[number][time])
                                                        time_maxes.append(time_max)
                                                    obs_time_maxes.append(time_maxes)

                                                range_maxes = [np.ptp(obs_time_maxes[0]),np.ptp(obs_time_maxes[1]),np.ptp(obs_time_maxes[2])]

                                                # only return ones with low correlation on all boundaries.
                                                # only one of the boundaries would have floated out, in theory?
                                                if num_same_signals == 0:
                                                    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX SIGNAL FOUND XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
                                                    low_correlations.append(i)
                                                    scores.append(k_score)
                                                    ranges.append(range_maxes)
        
        except Exception:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ERROR ON BLOCK # {i} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(traceback.print_exc())
            failures.append(i)
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    

    return low_correlations, scores, ranges, failures


def check_broadband(obs1):
    """Filter to check for broadband and blip signals in the data

    Args:
        obs1 (numpy array): 2D array representing observation 1

    Returns:
        blip_or_broadband (Boolean): True if there is blip or broadband present, false if not
    """

    # integrated observations both over time and frequency
    obs1_freq_integrated = obs1.sum(axis=1,dtype='float')
    # obs1_time_integrated = obs1.sum(axis=0,dtype='float')

    blip_or_broadband = False

    # broadband will come up as a spike in the frequency integrated observation. A genuine signal would not.
    broadband_threshold = 30
    freq_snr, threshold_freq = get_snr(obs1_freq_integrated,broadband_threshold)

    # also check it isnt a blip
    # a blip will no longer come up as a significant signal when time summing. 
    # this should potentially be changed, as heavily drifting signals might not pass this check

    # time_snr,threshold_time = get_snr(obs1_time_integrated,5)

    if freq_snr == True:
        blip_or_broadband = True

    return blip_or_broadband


def check_same_signal_number(row_ON,row_OFF,significance_level,on_off):
    
    row_ON = row_ON/np.max(row_ON)
    row_OFF = row_OFF/np.max(row_OFF)


    same_signal_number = False

    
    snr1, threshold1 = get_snr(row_ON,significance_level)
    snr6, threshold6 = get_snr(row_OFF,significance_level-2)

    # plt.plot(row_ON)
    # plt.axhline(y=threshold1)
    # plt.show()
    # plt.plot(row_OFF)
    # plt.axhline(y=threshold6)
    # plt.show()

    # get quantiles of the data
    lower_quantile_ON = np.quantile(row_ON,.85)
    lower_slice_ON = row_ON[row_ON < lower_quantile_ON]
    sigma_ON = np.std(lower_slice_ON)

    lower_quantile_OFF = np.quantile(row_OFF,.85)
    lower_slice_OFF = row_OFF[row_OFF < lower_quantile_OFF]
    sigma_OFF = np.std(lower_slice_OFF)


    peaks_ON, properties = signal.find_peaks(row_ON, prominence=10*sigma_ON, width=1)
    peaks_OFF, properties = signal.find_peaks(row_OFF, prominence=10*sigma_OFF, width=1)


    indicesON = np.where(np.array(row_ON) > threshold1)[0].tolist()
    indicesOFF = np.where(np.array(row_OFF) > threshold6)[0].tolist()


    indicesON = peaks_ON
    indicesOFF = peaks_OFF
    filtered_indicesON = indicesON
    filtered_indicesOFF = indicesOFF


    
    if len(indicesON) != 0:
        # first need to filter the signal somewhat in case it is spread out

        filtered_indicesON = [indicesON[0]]
        for i in indicesON[1:]:
            if abs(filtered_indicesON[-1] - i) <10:
                last = filtered_indicesON[-1]
                filtered_indicesON.pop()
                filtered_indicesON.append(np.mean([last,i]))
            else:
                filtered_indicesON.append(i)
                
    if len(indicesOFF) != 0:
        filtered_indicesOFF = [indicesOFF[0]]
        for i in indicesOFF[1:]:
            if abs(filtered_indicesOFF[-1] - i) <10:
                last = filtered_indicesOFF[-1]
                filtered_indicesOFF.pop()
                filtered_indicesOFF.append(np.mean([last,i]))
            else:
                filtered_indicesOFF.append(i)

        
    all = 0
    for i in filtered_indicesON:
        for j in filtered_indicesOFF:
            if abs(i-j) < 100:
                all +=1 
    if all >= len(filtered_indicesON):
        same_signal_number = True

    if on_off == "OFF":
        num_signals = len(filtered_indicesOFF)
    if on_off == "ON":
        num_signals = len(filtered_indicesON)


    return same_signal_number, num_signals



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
    """Function to calculate the pearson correlation between two frequency arrays

    Args:
        boundary_ON (numpy array): Array for last time slice of an ON observation
        boundary_OFF (numpy array): Array for first time slice of an OFF observation
        pearson_threshold (int): Correlation threshold for a signal to be considered significant   
        edge (int): Max range of sliding for pearson correlation. Akin to max drift rate

    Returns:
        max_pearson (float): Maximum pearson correlation achieved.
        current_shift (int): Shift between ON and OFF observation required to obtain a pearson score above the pearson threshold
    """

    # this initializes the psosible edges we iterate over for the pearson. We start at zero and work our way out, so as to be time efficient.
    # most signals have close to zero drift
    possible_drifts = [0]
    for i in range(1,edge):
        possible_drifts.append(i)
        possible_drifts.append(-i)

    max_pearson = 0
    current_shift = 0

    x = boundary_ON

    # take a sliding slice of the OFF observation to compare to the ON 
    for i in possible_drifts:
        current_shift = i
        if i != edge:
            y = boundary_OFF[(edge+i):-(edge-i)]
        else: 
            y = boundary_OFF[(edge+i):]

        # divide by max to normalize
        x= x / np.max(x)
        y = y / np.max(y)
        pearson = pearsonr(x,y)
        pearson  = pearson[0]

        # keep track of highest pearson value yet achieved
        if pearson > max_pearson:
            max_pearson = pearson
            # if pearson ever crosses the pearson threshold, halt the process to save time.
            if pearson > pearson_threshold:
                break

    return max_pearson, current_shift

# this performs a similar step as pearson but will check time sum and i
def second_filter(obs1_time_integrated,obs2_time_integrated,pearson_threshold,edge):
    """Takes two observations and integrates them over time to see if that will produce a stronger signal that does have a high correlation.

    Args:
        obs1 (numpy array): 2D array representing observation 1
        obs2 (numpy array): 2D array representing observation 2
        pearson_threshold (int): Correlation threshold for a signal to be considered significant   
        edge (int): Max range of sliding for pearson correlation. Akin to max drift rate

    Returns:
        still_good (Boolean): True if integrated pearson score is < pearson threshold |
        integrated_pearson_score (int): The maximum correlation achieved from the integrated correlation
    """

    still_good = False
    # first try summing and checking correlation again --> possible the signal was too weak the first time
    integrated_pearson_score, current_shift = pearson_slider(obs1_time_integrated,obs2_time_integrated,pearson_threshold,edge)
    
    if integrated_pearson_score < pearson_threshold:
        still_good = True

    return still_good, integrated_pearson_score

def check_blip(obs1):
    # for moment we can just check rows above:
    
    blip_threshold = 6

    not_constant_signal = False
    snrs = []
    # make sure there is a signal and not just blips
    for i in [0,8]:
        int_snr,threshold2 = get_snr(obs1[i],blip_threshold)
        snrs.append(int_snr)

    if snrs[0] == False and snrs[1] == False:
        not_constant_signal = True

    return not_constant_signal


def drift_index_checker(whole_sum, row_ON,significance_level,min_distance):
    """Checks if drift rate == 0. Compares all signals that set off hotspot to those in the full observation summed

    Args:
        whole_sum (numpy array): 2D array representing entire cadence summed 
        row_ON (numpy array): 1D array representing last time row of first observation
        significance_level (int): Minimum SNR for signal to be considered present

    Returns:
        zero_drift (Boolean): True if signal has zero drift, False if not
    """

    whole_sum = whole_sum/np.max(whole_sum)
    row_ON = row_ON/np.max(row_ON)

    zero_drift = False

    # we check if when we sum the entire observation, we pick up the signal that set off the hotspot. 
    # Will only do this if there are same number of peaks in on ROw and summed, in case there was a genuine signal in the ON row

    # get the peaks in the last row and the summed cadence
    hotspot_snr, hotspot_threshold = get_snr(row_ON,significance_level)
    summed_snr, summed_threshold = get_snr(whole_sum,significance_level)

    hotspot_indices = np.where(np.array(row_ON) > hotspot_threshold)[0].tolist()
    summed_indices = np.where(np.array(whole_sum) > summed_threshold)[0].tolist()

    # average any points very close together
    print(hotspot_indices,summed_indices)


    if len(hotspot_indices) != 0 and len(summed_indices) != 0:

        filtered_hotspot_indices = [hotspot_indices[0]]
        for i in hotspot_indices[1:]:
            if abs(filtered_hotspot_indices[-1] - i) <10:
                filtered_hotspot_indices.pop()
                filtered_hotspot_indices.append(i-.5)
            else:
                filtered_hotspot_indices.append(i)


        filtered_summed_indices = [summed_indices[0]]
        for i in summed_indices[1:]:
            if abs(filtered_summed_indices[-1] - i) <5:
                filtered_summed_indices.pop()
                filtered_summed_indices.append(i-.5)
            else:
                filtered_summed_indices.append(i)

        # check if all hotspots picked up in ON row are in the summed
        all = 0
        for i in filtered_hotspot_indices:
            for j in filtered_summed_indices:
                if abs(i-j) < min_distance:
                    all +=1 
        if all >= len(filtered_hotspot_indices):
            zero_drift = True
        print(filtered_hotspot_indices,filtered_summed_indices)

    return zero_drift


def main_boundary_checker(h5_files,pearson_threshold,block_size,significance_level,edge):
    """Main wrapper function for all the hotspot finding and filtering that takes place

    Args:
        h5_files (list): list of h5 objects for each observation in the cadence
        filtered_hotspots (list): List of integers corresponding to hotspot number |
        pearson_threshold (int): Correlation threshold for a signal to be considered significant   
        block_size (int): Size of hotspot region
        significance_level (int): Minium SNR for a signal to be considered a "signal"
        edge (int): Max range of sliding for pearson correlation. Akin to max drift rate
    
    Returns:
        low_correlation_frequencies (list): List of all frequencies that were flagged
        scores (list): Outdated, corresponding scores for all of those frequencies

    """
    
    # load data
    hf_ON = h5py.File(h5_files[0], 'r')
    hf_OFF = h5py.File(h5_files[1], 'r')
    hf_ON2 = h5py.File(h5_files[2], 'r')
    hf_OFF2 = h5py.File(h5_files[3], 'r')
    hf_ON3 = h5py.File(h5_files[4], 'r')
    hf_OFF3 = h5py.File(h5_files[5], 'r')



    # grab specific rows which will be used to find hotspots
    obs1_row_16 = np.squeeze(hf_ON['data'][15:16,:,:])

    obs3_row_8 = np.squeeze(hf_ON2['data'][15:16,:,:])

    obs5_row_8 = np.squeeze(hf_ON3['data'][15:16,:,:])

    primary_off = np.squeeze(hf_OFF['data'][0:1,:,:])

    last_time_row_ON = obs1_row_16


    # find file frequency information
    fch1,foff = get_file_properties(hf_ON)

    # calculate number of iterations needed and find hotspots
    number = int(np.round(len(last_time_row_ON)/block_size))

    # record interesting freq chunks as 'warmspots'. This is the initial pass.
    hotspot_slices = [last_time_row_ON,obs3_row_8, obs5_row_8]

    number = int(np.round(len(last_time_row_ON)/block_size))

    all_warmspots = []

    for i in range(0,len(hotspot_slices)):
        warmspots = find_warmspots(hotspot_slices[i],number,block_size)
        print(f"slice {i} found {len(warmspots)} warmspots")
        all_warmspots = all_warmspots+warmspots

    # keep only the unique blocks
    warmspots = [*set(all_warmspots)]

    # next filter out warmspots that fall in bad regions
    filtered_indexes = filter_hotspots(warmspots,fch1,foff,block_size)
    filtered_warmspots = np.delete(warmspots, filtered_indexes)
    print("# warmspots post filtering",len(filtered_warmspots))

    # now sort through these warmspots and find hotspots --> higher signal
    all_hotspots = []
    for i in range(0,len(hotspot_slices)):
        hotspots = find_hotspots(hotspot_slices[i],filtered_warmspots,block_size,significance_level)
        print(len(hotspots))
        all_hotspots=all_hotspots+hotspots

    filtered_hotspots = [*set(all_hotspots)]
    print("# of hotspots:",len(filtered_hotspots))

    filtered_hotspots_slice_indexes = []
    for spot in filtered_hotspots:
        for i in range(0,len(hotspot_slices)):    
            row = hotspot_slices[i]
            slice_ON = row[spot*block_size:(spot+1)*block_size:]
            snr,threshold = get_snr(slice_ON,significance_level)
            if snr:
                filtered_hotspots_slice_indexes.append(i)
                break

    # delete variables to clear up memory
    del hotspot_slices
    del obs5_row_8
    del obs3_row_8
    gc.collect()

    # find regions of low correlation
    low_correlations,scores,ranges,failures = check_hotspots([last_time_row_ON],primary_off,hf_ON,hf_OFF,hf_ON2,hf_OFF2,hf_ON3,hf_OFF3,filtered_hotspots,pearson_threshold,significance_level,edge,block_size,filtered_hotspots_slice_indexes,filtered_hotspots)
    print(f"Found {len(low_correlations)} regions of low correlation")
    print(low_correlations)
    # save regions to csv file, in current directory
    failure_frequencies = [fch1+foff*(i*block_size) for i in failures]
    # return frequency positions of candidates
    low_correlation_freqs = [fch1+foff*(i*block_size) for i in low_correlations]

    return low_correlation_freqs,low_correlations,scores,ranges,failure_frequencies,fch1,foff

