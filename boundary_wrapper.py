import subprocess
import sys
import os
from boundary_checker import *

def main(candidates_df_name):
    main_dir = os.getcwd() + "/"
   
    # target_list = ['NGC1052', 'NGC1172 ', 'NGC1400', 'NGC1407', 'NGC2403']
    # target_list = ['NGC584']
    # target_list = ['HERCULES', 'HIZSS003', 'IC0010', 'IC0342', 'IC1613', 'LEOA', 'LEOII', 'LEOT', 'LGS3', 'MAFFEI1', 'MAFFEI2', 'MESSIER031', 'MESSIER033', 'MESSIER081', 'MESSIER101', 'MESSIER49', 'MESSIER59', 'MESSIER84', 'MESSIER86', 'MESSIER87', 'NGC0185', 'NGC0628', 'NGC0672 ', 'NGC1052', 'NGC1172 ', 'NGC1400', 'NGC1407', 'NGC2403']
    # load candidates database

    target_list = ['AND_II','AND_I', 'AND_X', 'AND_XI', 'AND_XIV', 'AND_XVI', 'AND_XXIII', 'AND_XXIV', 'BOL520', 'CVNI', 'DDO210', 'DRACO', 'DW1','HERCULES', 'HIZSS003', 'IC0010', 'IC0342', 'IC1613', 'LEOA', 'LEOII', 'LEOT', 'LGS3', 'MAFFEI1', 'MAFFEI2', 'MESSIER031', 'MESSIER033', 'MESSIER081', 'MESSIER101', 'MESSIER49', 'MESSIER59', 'MESSIER84', 'MESSIER86', 'MESSIER87', 'NGC0185', 'NGC0628', 'NGC0672 ', 'NGC1052', 'NGC1172 ', 'NGC1400', 'NGC1407', 'NGC2403','NGC2683', 'NGC2787', 'NGC3193', 'NGC3226', 'NGC3344', 'NGC3379', 'NGC4136', 'NGC4168', 'NGC4239', 'NGC4244', 'NGC4258', 'NGC4318', 'NGC4365', 'NGC4387', 'NGC4434', 'NGC4458', 'NGC4473', 'NGC4478', 'NGC4486B', 'NGC4489', 'NGC4551', 'NGC4559', 'NGC4564', 'NGC4600', 'NGC4618', 'NGC4660', 'NGC4736', 'NGC4826', 'NGC5194', 'NGC5195', 'NGC5322', 'NGC5638', 'NGC5813', 'NGC5831', 'NGC584', 'NGC5845', 'NGC5846', 'NGC596', 'NGC636', 'NGC6503', 'NGC6822', 'NGC6946', 'NGC720', 'NGC7454 ', 'NGC7640', 'NGC821', 'PEGASUS', 'SAG_DIR', 'SEXA', 'SEXB', 'SEXDSPH', 'UGC04879', 'UGCA127', 'UMIN']
    candidates = pd.read_csv(main_dir+candidates_df_name)

    for target in target_list:        
        print("Running boundary checker for target:",target)
        unique_h5_files,unique_nodes = get_all_h5_files(target)
        # print total number of files
        count = sum( [ len(listElem) for listElem in unique_h5_files])
        print(f"{count} files")
        os.chdir(main_dir)
        # print("All files:",unique_h5_files)
        # reformed_unique_h5 = []

        # # make sure the spliced files don't get piled together by accident
        # for node in range(0,len(unique_nodes)):
        #     if unique_nodes[node] == "splic":
        #         for file in unique_h5_files[node]:
        #             reformed_unique_h5.append(file)
        #     else:
        #         reformed_unique_h5.append(unique_h5_files[node])

        # print("Reformed files:",reformed_unique_h5)


    # skip spliced for now
        try:
            # if unique_nodes[0] != "splic":
            print(unique_nodes)
            for i in range(0,len(unique_h5_files)):
                if unique_nodes[i] != "splic":
                    h5_files = unique_h5_files[i]
                    obs1 = h5_files[0]
                    obs2 = h5_files[1]
                    obs3 = h5_files[2]
                    obs4 = h5_files[3]
                    obs5 = h5_files[4]
                    obs6 = h5_files[5]
                    all_files = (obs1,obs2,obs3,obs4,obs5,obs6)
                    low_correlation_frequencies,scores= main_boundary_checker(obs1,obs2,obs3,obs4,obs5,obs6)
    
                    for i in range(0,len(low_correlation_frequencies)):
                        freq = low_correlation_frequencies[i]
                        score = scores[i]
                        file_location = obs1
                        candidates.loc[len(candidates.index)] = [file_location,freq,all_files,score]
                    print(candidates)

                    candidates.to_csv(main_dir+df_name,index=False)
                else: 
                    print("skipping spliced files")
            # else:
            #     print(f"Sxkipping {target} b/c spliced")
        except:
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print(f"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ERROR ON TARGET {target} XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    

    

    


def get_all_h5_files(target):
    '''
    We want to return groups of cadences
    '''


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
    node_list = []
    for dirname, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if "blc" in filename and (filename[4] != '7') and (filename[4] != '0'):
                node_list.append(filename[:5])

    node_set = set(node_list)
    print(node_set)

    unique_nodes = sorted(node_set)
    unique_nodes.sort()
    return unique_nodes

def grab_node_file_list(data_dir,node_number):
    '''
    returns h5 and dat file path from given directory, ordered correctly
    '''
    
    ## h5 list
    data_list = []
    for dirname, _, filenames in os.walk(data_dir):
        for filename in filenames:
            if filename[-3:] == '.h5' and node_number in filename:
                data_list.append(data_dir + filename)
                
    data_list = sorted(data_list, key=lambda x: (x,x.split('_')[5]))

    return data_list

if __name__ == '__main__':
    
    # check if candidates database is set up, if not then initialize it
    main_dir = os.getcwd() + "/"
    df_name = f'second_updated_candidate_events_sigma_{SIGNIFICANCE_LEVEL}_pearsonthreshold_{int(PEARSON_THRESHOLD*10)}_blocksize_{BLOCK_SIZE}_edge_{EDGE}.csv'

    db_exists = os.path.exists(main_dir+df_name)
    if db_exists == False:
        print("Creating candidates database as ",df_name)
        candidates = pd.DataFrame(columns=["Primary File","Frequency","All Files","Score"])
        candidates.to_csv(main_dir+df_name,index=False)

    main(df_name)
