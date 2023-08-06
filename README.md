# Cappuccino: A Cross-Correlation Based Filtering Pipeline for Identifying Technosignatures in GBT Data 


![alt text](https://cdn.sci.news/images/enlarge10/image_11611e-Breakthrough-Listen.jpg)

## Code Author

Caleb Painter

## Background

This directory contains a Python-based pipeline for processing and analyzing large amounts of GBT data to search for radio technosignatures. For a more detailed overview of the search for radio technosignatures, please see the [@UCBerkeleySETI](https://github.com/UCBerkeleySETI) page, [here](https://github.com/UCBerkeleySETI/breakthrough/tree/master/GBT).  

To summarize, the current state of techno signatures searches using Green Bank data faces two problems:
1. Human technology produces Radio Frequency Interference (RFI) that we must distinguish from genuine signals.
2. The volume of data being worked with is very large.

To address the first problem, the basic observing strategy is structured in the following pattern:
* Observe a target ("star A") from the primary target list for 5 minutes
* Observe another target ("star B") at a nearby position on the sky for 5 minutes
* Return to star A for 5 minutes
* Observe another target ("star C") from our secondary list for 5 minutes
* Another 5 minutes on star A
* 5 minutes on another secondary target ("star D")

Using this type of observing cadence, we're able to eliminate RFI by checking if signals appear in both the ON (primary star) and OFF (secondary stars) observations. Genuine signals would only appear in the ONs. In the plots below, RFI can be seen in the left (as it is present in every observation), whereas a strong candidate can be seen on the right (as it is only present in the ON observation).

![alt text](/notebooks/images/good_bad.png)

The primary algorithm used by the Breakthrough Listen group to find signals at the moment is [**turboSETI**](https://github.com/UCBerkeleySETI/turbo_seti), which is a fast, effective approach to finding narrowband, doppler drifting signals. However, the pipeline in its current format still faces two primary issues:
1. It misses promising signals that should be flagged.
   - This can be seen in Peter Ma's paper, where he uses an ML approach to find signals. One such example from HIP13402 is:
    ![alt text](/notebooks/images/Peter_signal.png)
2. It returns candidate events that are clearly RFI, and should not have passed the filters.

**Cappuccino** takes a different approach than **turboSETI** in order to find strong candidate signals, to avoid running into these problems. The basis of the method is a cross-correlation based filter: it checks the correlation at the boundary between the ON and OFF observations. Strong candidates should have very low correlation at this boundary, as the signal should be present in the ON but not in the OFF. RFI would have very high correlation at the boundaries, as the signal should be continuous even as the telescope switches target.  

There are several advantages of this method over **turboSETI**'s approach. Firstly, it is not biased toward any type of signal. It is successful at catching narrowband drifting signals, like the kind **turboSETI** searches for, but is not limited to these. The only requirements are that the signal is continuous in time, but does not need to have a linear drift. In fact, it can display any kind of behavior in the frequency domain. 

# Walkthrough

-------------------
## Setup

This code is meant to be run on data from Berkeley's Breakthrough Listen group, and is most easily run on the Breakthrough Listen computer clusters. It is possible to download the data files you want locally though, and run it on your machine. All that is needed is a basic python environment. 

-------------------
### Dependencies

- Python 3.7+
- numpy
- pandas
- hf5py
- scipy
- pickle
- os
- sys

&nbsp;

-------------------
## Usage

### Expected Input File Format

Cappuccino is built to read in and analyze HDF5 files. Unlike **turboSETI**, **cappuccino** runs on the entire cadence at a time, not individual observations. In doing so, it takes direct advantage of the ON/OFF cadence described above. There are three options for inputting file formats:
1. Giving the directory path containing the .h5 files.
2. Giving the path to a text file which holds the .h5 file paths.
3. Giving a batch number to be run on, in which case it will run on one of the 1000 batches that make up the archival GBT observation database.

If these options are not sufficient, the code is well documented and users can edit it to pass in their own files.

### Usage as a Command Line

Run with data: `cappuccino <PATH_TO_TEXT_FILE_WITH_HDF5_FILES> [OPTIONS]`

For an explanation of the program parameters: `cappuccino -h`

To briefly summarize them:
- `block size`: Cappuccino functions by dividing the .h5 files into promising frequency snippets, and analyzing those frequency snippets for candidate signals. Block size defines the size of that frequency snippet, with units of frequency bins. The default is 4096 (or around 11.5 kHz for a standard 2.8Hz bin size). 
- `Pearson threshold`: The minimum Pearson correlation metric required for two time slices (typically the last one in the ON observation and first one in the OFF observation) to have a 'high' correlation, and be rejected as RFI. The default has been set to .3, after testing on different signal strengths and types.
- `significance level`: In essence, the minimum SNR a signal must have to be flagged in the first place. The significance level defines the number of standard deviations above the baseline a signal must be. The baseline is calculated as the median of the lower 85% of values (to avoid having high values skew the baseline).The default is 10.
- `edge`: The maximum number of frequency bins that two time slices will be displaced in frequency to account for drift rate as the telescope slews from one target to another. The default is 50, which represents 140Hz in the standard frequency binning. It must be remembered that this does not represent an upper limit on the range of drift rates candidates can exhibit -- it is a lower limit on how much drift an RFI source can have without being caught. Setting a lower number will give more false positives, and a higher number will give less but at the cost of computation time.
- `files`: Described above, the directory path containing the .h5 files.
- `directory`: Described above, the path to a text file which holds the .h5 file paths.
- `number`: Described above, the specific batch of GBT data.


### Usage as a Python Package

```
from turbo_seti.find_doppler.find_doppler import FindDoppler
fdop = FindDoppler(datafile=my_HDF5_file, ...)
fdop.search(...)
```



