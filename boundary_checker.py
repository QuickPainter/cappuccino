# Load required packages
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
import statistics as stats
import blimpy
import pandas as pd
import pickle
from scipy import signal
import scipy
import sys

from datetime import datetime
import h5py
from scipy.stats import pearsonr   
from tqdm import tqdm

# Define Global Constants

# define the edges you want to account for due to drift rate
BLOCK_SIZE = 500
EDGE = 50
SIGNIFICANCE_LEVEL = 10
PEARSON_THRESHOLD = .3

possible_drifts = [0]
for i in range(1,EDGE):
    possible_drifts.append(i)
    possible_drifts.append(-i)

#,[2180,2200],[4000,4200],[10800,11000]]
bad_regions = [[700,830],[1370,1390],[1520,1630],[1160,1340],[1675,1695],[2180,2280],[2025,2035],[2100,2120],[1915,1935],[2300,2360],[3800,4200],[4680,4800],[11000,11180]]


# Define Necessary Functions
def get_boundary_data(hf_ON,hf_OFF,lower,upper):

    row_ON = np.squeeze(hf_ON['data'][-1:,:,lower:upper],axis=1)[0]
    row_OFF = np.squeeze(hf_OFF['data'][:1,:,lower-EDGE:upper+EDGE],axis=1)[0]

    return row_ON,row_OFF


def get_last_time_row(file):
    hf = h5py.File(file, 'r')
    data = hf.get('data')
    last_time_row = data[-1]
    return last_time_row[0]

def get_freq_slices(last_time_row,f_start,f_end):
    freq_block = last_time_row[f_start:f_end]
    return freq_block

def get_snr(sliced,sigma_multiplier):
    snr = False
    # divide by max to make numbers smaller
    sliced = sliced/np.max(sliced)

    lower_quantile = np.quantile(sliced,.7)
    lower_slice = sliced[sliced < lower_quantile]
    median = np.median(lower_slice)
    sigma = np.std(lower_slice)

    # print('sigma',sigma)
    # print('median',median)
    # print('sigma_multiplier',sigma_multiplier)
    threshold = median+sigma_multiplier*sigma
    if np.max(sliced) > threshold:
        snr = True
        # plt.axhline(y=threshold)
        # plt.plot(slice)
        # plt.show()

    return snr, threshold

def get_first_round_snr(sliced,first_round_multiplier):
    snr = False
    # divide by max to make numbers smaller
    sliced = sliced/np.max(sliced)

    median = np.median(sliced)
    sigma = np.std(sliced)

    # print('sigma',sigma)
    # print('median',median)
    # print('sigma_multiplier',sigma_multiplier)
    threshold = median+first_round_multiplier*sigma

    if threshold <= 1:
        snr = True
        # plt.axhline(y=threshold)
        # plt.plot(slice)
        # plt.show()

    return snr, threshold


def get_snr(sliced,sigma_multiplier):
    snr = False
    # divide by max to make numbers smaller
    sliced = sliced/np.max(sliced)

    lower_quantile = np.quantile(sliced,.85)
    lower_slice = sliced[sliced < lower_quantile]
    median = np.median(lower_slice)
    sigma = np.std(lower_slice)

    # print('sigma',sigma)
    # print('median',median)
    # print('sigma_multiplier',sigma_multiplier)
    threshold = median+sigma_multiplier*sigma

    if threshold <= 1:
        snr = True
        # plt.axhline(y=threshold)
        # plt.plot(slice)
        # plt.show()

    return snr, threshold
    
def find_hotspots(row,number):
    first_round = []
    hotspots = []
    first_round_multiplier = 5
    # iterate
    hotspot_size = 2000

    for i in tqdm(range(0,number)):
        slice_ON = row[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE:]
        snr,threshold = get_first_round_snr(slice_ON,first_round_multiplier)
        if snr:
            first_round.append(i)

    for i in tqdm(first_round):
        slice_ON = row[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE:]
        snr,threshold = get_snr(slice_ON,SIGNIFICANCE_LEVEL)
        if snr:
            hotspots.append(i)

    
    return hotspots

def get_file_properties(f):
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
    distance = fch1 - freq
    number = int(np.round(-distance/foff))
    bound = 250
    return number+bound, number-bound

def filter_hotspots(hotspots,fch1,foff):
    '''
    Take out hotspots that are falling in especially heavy RFI
    '''

    ## first convert hotspots ids to frequency channels
    hotspots_frequencies = np.array([int((fch1+foff*(i*500))) for i in hotspots])
    all_indexes = []
    for i in bad_regions:
        # print(bad_regions)
        bottom = int(i[0])
        top = int(i[1])
        indexes = np.where(np.logical_and(bottom<hotspots_frequencies, hotspots_frequencies<top))
        indexes = indexes[0]
        indexes = [int(i) for i in indexes]
        for i in indexes:
            all_indexes.append(i)


    return all_indexes


'''
Event Detection algorithms
Nice and short :)

 '''

def check_hotspots(hf_obs1,hf_obs2,hf_obs3,hf_obs4,hf_obs5,hf_obs6,filtered_hotspots):
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


            row_ON, row_OFF = get_boundary_data(hf_obs1,hf_obs2,lower,upper)
            same_signal_number = check_same_signal_number(row_ON,row_OFF)


            # first just check if there are same number of signals
            if same_signal_number==False:
                Obs1 = np.squeeze(hf_obs1['data'][:,:,lower:upper],axis=1)
                # print(Obs1.shape)
                print("checking blips")
                not_constant_signal,threshold_blips = blip_checker(Obs1)                

                if not_constant_signal ==False :
                    # if it passes the initial check, perform pearson correlation

                    print("Checking correlation")
                    max_corr, current_shift = pearson_slider(row_ON,row_OFF)

                    if max_corr < PEARSON_THRESHOLD:
                        # Obs1 = np.squeeze(hf_obs1['data'][:,:,lower:upper],axis=1)
                        # check if it is broadband RFI:
                        print("checking broadband")
                        is_broadband = check_broadband(Obs1)
                        if is_broadband == False:
                            
                            # load entire observation and see if time-summing the signal produces different result
                            Obs2 = np.squeeze(hf_obs2['data'][:,:,lower-EDGE:upper+EDGE],axis=1)
                            # will catch weak signals
                            fails_sum,integrated_pearson_score = second_filter(Obs1,Obs2)
                
                            # then do a check that there is still a strong signal somewhere higher up in the observation --> so not just a little point right at the boundary
                            # time sum whole observation, and see if signal stands out
                
                            # also check if there was just a dim signal in the first OFF, maybe it gets stronger in second bin
                            second_row_corr,current_shift = pearson_slider(row_ON,Obs2[8])
                            # can also check if same # of signals in middle of next row
                            check_same_signal_number_middle = check_same_signal_number(row_ON,Obs2[8])
                            # plt.plot(row_ON)
                            # plt.show()
                            # plt.plot(Obs2[8])
                            # plt.show()

                            print("fails sum:",fails_sum)
                            print("not_constant_signal:",not_constant_signal)
                            print("second_row_corr",second_row_corr)
            
                            # if it passes these tests, check if it zero drift rate. Compare first time from first ON observation to last time from last OFF.
                            # if it has high correlation at drift rate = 0, then probably not good 
                            if fails_sum and second_row_corr < PEARSON_THRESHOLD and check_same_signal_number_middle == False:
                
                                print("Checking Drift Rate")
                
                                # sum whole observation and see if zero drift rate, if necessary
                                obs1_int = Obs1.sum(axis=0)
                
                                # load last OFF observation
                                Obs6 = np.squeeze(hf_obs6['data'][:,:,lower-EDGE:upper+EDGE],axis=1)
                                obs6_int = Obs6.sum(axis=0)
                
                                first_last_corr,current_shift = pearson_slider(obs1_int, obs6_int)
                
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

                                        
                                        

                                    zero_drift = drift_index_checker(whole_sum, row_ON)
                                    
                                    signal_stays_strong = True

                                    snr_obs3_0, threshold30 = get_snr(Obs3[0],SIGNIFICANCE_LEVEL)
                                    snr_obs3_16, threshold316 = get_snr(Obs3[-1],SIGNIFICANCE_LEVEL)
                                    snr_obs5_0, threshold50 = get_snr(Obs5[0],SIGNIFICANCE_LEVEL)
                                    snr_obs5_16, threshold516 = get_snr(Obs5[-1],SIGNIFICANCE_LEVEL)


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
                                        row_ON2, row_OFF2 = get_boundary_data(hf_obs2,hf_obs3,lower,upper)

                                        

                                        max_corr2, current_shift2 = pearson_slider(row_ON2,row_OFF2)
                                        # boundary 3/5
                                        row_ON3, row_OFF3 = get_boundary_data(hf_obs3,hf_obs4,lower,upper)
                                        max_corr3, current_shift3 = pearson_slider(row_ON3,row_OFF3)
                                        # boundary 4/5
                                        row_ON4, row_OFF4 = get_boundary_data(hf_obs4,hf_obs5,lower,upper)
                                        max_corr4, current_shift4 = pearson_slider(row_ON4,row_OFF4)
                                        # boundary 5/5
                                        row_ON5, row_OFF5 = get_boundary_data(hf_obs5,hf_obs6,lower,upper)
                                        max_corr5, current_shift5 = pearson_slider(row_ON5,row_OFF5)
                                        # count how many had low pearson values
                                        score = 0

                                        for corr in [max_corr,max_corr2,max_corr3,max_corr4,max_corr5]:
                                            print('co:',corr)
                                            if corr < PEARSON_THRESHOLD:
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
        
        except:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ERROR ON BLOCK # {i} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
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


def check_same_signal_number(row_ON,row_OFF):
    
    row_ON = row_ON/np.max(row_ON)
    row_OFF = row_OFF/np.max(row_OFF)
    same_signal_number = False

    snr1, threshold1 = get_snr(row_ON,SIGNIFICANCE_LEVEL)
    snr6, threshold6 = get_snr(row_OFF,SIGNIFICANCE_LEVEL)

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

def pearson_slider(boundary_ON,boundary_OFF):
    max_corr = 0
    max_pearson = 0
    current_shift = 0

    x = boundary_ON

    for i in possible_drifts:
        current_shift = i
        if i != EDGE:
            y = boundary_OFF[(EDGE+i):-(EDGE-i)]
        else: 
            y = boundary_OFF[(EDGE+i):]

        # print(len(x),len(y))

        x= x / np.max(x)
        y = y / np.max(y)
        pearson = pearsonr(x,y)

        pearson  = pearson[0]
        if pearson > max_pearson:
            max_pearson = pearson
            if pearson > PEARSON_THRESHOLD:
                # print("ran iterations:",i)
                break

    return max_pearson, current_shift

# this performs a similar step as pearson but will check time sum and i
def second_filter(obs1,obs2):

    still_good = False
    # first try summing and checking correlation again --> possible the signal was too weak the first time
    obs1_time_integrated = obs1.sum(axis=0,dtype='float')
    obs2_time_integrated = obs2.sum(axis=0,dtype='float')
    integrated_pearson_score, current_shift = pearson_slider(obs1_time_integrated,obs2_time_integrated)
    
    if integrated_pearson_score < PEARSON_THRESHOLD:
        still_good = True

    return still_good, integrated_pearson_score

def drift_index_checker(whole_sum, row_ON):
    print("Checking Drift")
    zero_drift = False


    # row_ON = row_ON/np.max(row_ON)
    # obs1_int = obs1_int/np.max(obs1_int)
    # obs6_int = obs6_int/np.max(obs6_int)


    # # check if both the ons and offs have signal peaks in all the same places
    # snr1, threshold1 = get_snr(obs1_int,SIGNIFICANCE_LEVEL)
    # snr6, threshold6 = get_snr(obs1_int,SIGNIFICANCE_LEVEL)

    # indices1 = np.where(np.array(obs1_int) > threshold1)[0].tolist()
    # indices6 = np.where(np.array(obs6_int) > threshold6)[0].tolist()

    # # we need to avg the two that are close together
    # print(indices1,indices6)
    # if len(indices1) != 0 and len(indices6) != 0:

    #     filtered_indices1 = [indices1[0]]
    #     for i in indices1[1:]:
    #         if abs(filtered_indices1[-1] - i) <2:
    #             filtered_indices1.pop()
    #             filtered_indices1.append(i-.5)
    #         else:
    #             filtered_indices1.append(i)


    #     filtered_indices6 = [indices6[0]]
    #     for i in indices6[1:]:
    #         if abs(filtered_indices6[-1] - i) <2:
    #             filtered_indices6.pop()
    #             filtered_indices6.append(i-.5)
    #         else:
    #             filtered_indices6.append(i)

    #     if abs(np.mean(filtered_indices1)-np.mean(filtered_indices6)) < 2:
    #         zero_drift = True

    # we can also check if when we sum the entire observation, we pick up the signal that set off the hotspot. 
    # Will only do this if there are same number of peaks in on ROw and summed, in case there was a genuine signal in the ON row

    hotspot_snr, hotspot_threshold = get_snr(row_ON,SIGNIFICANCE_LEVEL)
    summed_snr, summed_threshold = get_snr(whole_sum,SIGNIFICANCE_LEVEL)

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


def main_boundary_checker(obs1,obs2,obs3,obs4,obs5,obs6):
    print("Now running on ",(obs1,obs2,obs3,obs4,obs5,obs6))

    
    # load data
    hf_ON = h5py.File(obs1, 'r')
    hf_OFF = h5py.File(obs2, 'r')
    hf_ON2 = h5py.File(obs3, 'r')
    hf_OFF2 = h5py.File(obs4, 'r')
    hf_ON3 = h5py.File(obs5, 'r')
    hf_OFF3 = h5py.File(obs6, 'r')



    # grab last row to find hotspots
    last_time_row_ON = get_last_time_row(obs1)

    # find file frequency information
    fch1,foff = get_file_properties(hf_ON)

    # calculate number of iterations needed and find hotspots
    number = int(np.round(len(last_time_row_ON)/BLOCK_SIZE))
    # record interesting freq chunks
    hotspots_ON = find_hotspots(last_time_row_ON,number)
    # hotspots_OFF = find_hotspots(last_time_row_OFF)
    print("# of hotspots:",len(hotspots_ON))
    # filter the hotspots
    filtered_indexes = filter_hotspots(hotspots_ON,fch1,foff)
    filtered_hotspots = np.delete(hotspots_ON, filtered_indexes)
    print("# post filtering",len(filtered_hotspots))

    # find regions of low correlation
    low_correlations,scores = check_hotspots(hf_ON,hf_OFF,hf_ON2,hf_OFF2,hf_ON3,hf_OFF3,filtered_hotspots)
    print(f"Found {len(low_correlations)} regions of low correlation")

    # save regions to csv file, in current directory

    # return frequency positions of candidates
    return [fch1+foff*(i*500) for i in low_correlations],scores


if __name__ == '__main__':
    obs1 = sys.argv[1]
    obs2 = sys.argv[2]
    obs3 = sys.argv[3]
    obs4 = sys.argv[4]
    obs5 = sys.argv[5]
    obs6 = sys.argv[6]


    # initialize a pandas database to store candidates
    current_dir = os.getcwd() + "/"


    main_boundary_checker(obs1,obs2,obs3,obs4,obs5,obs6)

    

