# Cappuccino: A Cross-Correlation Based Filtering Pipeline for Identifying Technosignatures in GBT Data 


![alt text](https://cdn.sci.news/images/enlarge10/image_11611e-Breakthrough-Listen.jpg)

## Code Author

Caleb Painter

## Background

This directory contains a Python-based pipeline for processing and analyzing large amounts of GBT data to search for radio technosignatures. For a more detailed overview of the search for radio technosignatures, please see the [@UCBerkeleySETI](https://github.com/UCBerkeleySETI) page, [here](https://github.com/UCBerkeleySETI/breakthrough/tree/master/GBT).  

To summarize, the current state of technosignaturs searches using Green Bank data faces two problems:
1. Human technology produces Radio Frequency Interference (RFI) that we must distinguish from genuine signals.
2. The volume of data being worked with is very large.

To address the first problem, the basic observing strategy is structured in the following pattern:
* Observe a target ("star A") from the primary target list for 5 minutes
* Observe another target ("star B") at a nearby position on the sky for 5 minutes
* Return to star A for 5 minutes
* Observe another target ("star C") from our secondary list for 5 minutes
* Another 5 minutes on star A
* 5 minutes on another secondary target ("star D")

Using this type of observing cadence, we're able to eliminate RFI by checking if signals appear in both the ON (primary star) and OFF (secondary stars) observations. Genuine signals would only appear in the ONs. 

The primary algorithm used by the Breakthrough Listen group to find signals at the moment is [**turboSETI**](https://github.com/UCBerkeleySETI/turbo_seti), which is a fast, effective approach to finding narrowband, doppler drifting signals. However, the pipeline in its current format still faces two primary issues:
1. It misses promising signals that should be flagged.
   - This can be seen in Peter Ma's paper, where he uses an ML approach to find signals. One such example from HIP13402 is:
    ![alt text](/notebooks/images/Peter_signal.png)
3. It returns candidate events that are clearly RFI, and should not have passed the filters.


# Walkthrough

## Setup

This code is meant to be run on data from Berkeley's Breakthrough Listen group, and is most easily run on the Breakthrough Listen computer clusters. It is possible to download the data files you want locally though, and run it on your machine. All that is needed is a basic python environment. 




