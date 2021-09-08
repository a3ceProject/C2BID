C2BID

In this repository, you can find four scripts (features_extract.py, clustering_process.py, history_path.py, prob_model.py, and RRCF.py). We also provide examples of input and output data, available at https://drive.google.com/drive/folders/1OW_IWoQJAfGClLMlgaWLHJ_LDSDy0BER?usp=sharing

Scripts prerequisites:

Python 3
Pandas (https://pandas.pydata.org/getting_started.html)
Numpy (https://numpy.org)
Sklearn (https://scikit-learn.org/stable/index.html)
You will also need permission to create folders in your working directory.

################## featureS_extract.py ##################

The features_extract.py script is used to extract the features from a file CSV file with traffic information (flows). This script receives the CSV file as input and returns CSV files, one per time window, with the features extracted and organized by entities (IP addresses).


    [FILE PATH] The file name 
    [TimeWindows] time windows (default 10), for example: 10,20,30
    [method] 0 OutGene (defaul); 1  the day  ports; 2 Junction of ports 0 and 1 ;3 DN3_x, 4 DN3_x_Outgene, 5 FlowHacker
    [X] Port number to be used in DN3_X method, defaul: 100
    
    --help: to get help 
    --file: get possible files 

NOTE: 
	1) To add a new method, you must create a new function an add it in Line 87
	2) File name,  days, and dates to be considered and must use the port (src and dst) can be updated in Line 19
	3) If you specify the [method], you must also specify [TimeWindows].

Usage example (from the command line):

Linux/MAC OS:

	- python3 features_extract.py [FILE PATH] [TimeWindows] [method] [X]
	- python3 features_extract.py new_friday-02-03-18.csv 10,30,120 3 100


################## clustering_process.py ##################

The clustering_process.py script is used to perform clustering using as input the features files created by features_extract.py script. This script receives several CSV files with features and returns CSV files with clustering results.


    [FILE PATH]: The path to the file. The name of the file must respect the format: [WINDOW]_features_[DAY]_[TIME].csv
    Fast mode: directory for files to analyze 
    [WINDOW]: the size of the time window in minutes
    [DAY]:  Day
    [TIME]: HH/MM/SS from the start of the time window
    
   
    --help: to get help 
    --fast: fast mode, only produces the CSV files where there are outliers (clusters with just one entity), you can read more than one file 

Linux/MAC OS:

	-python3 clustering_process.py [FILE PATH] 
	-python3 clustering_process.py --fast day1/10.0min


################## history_path.py ##################
The history_path.py script is used to construct the history path using clustering files created by the clustering_process.py script. This script receives several CSV files with clustering and JobLib files with the classification used  and returns CSV files with history paths.

    [FILE PATH] location of .csv files 
         The directory should contain:
            - .csv files with features, with the [WINDOW]_features_[DAY]_[TIME].csv format
            -directory named 'results,' with two sub-directory: 'Ext' and 'Int' each with:
                -csv files with clusters per time window, with format [WINDOW]_Int_[DAY]_[END_TIME]_cluster.csv or [WINDOW]_Ext_[DAY]_[TIME]_cluster.csv
                -directory named 'classifier' with the classifiers, with format [WINDOW]_Int_[DAY]_[END_TIME]_classifier.joblib or [WINDOW]_Ext_[DAY]_[TIME]_classifier.joblib    
        [WINDOW]: min size of time window 
        [DAY]: DD day
        [TIME]: HH/MM/SS from start of time window
    [TRESHOLD] minimum overlap value, by default 0.5. NOTE: Treshold values below 0.5 can lead to rating problems
    [COMP] features comparison mode, R- compare time window based on relative values, A- compare based on absolute values, by default A 
        
    --help: to get help 
    --fast: fast mode only produces CSV files with cluster numbers and only analyzes Int (internal) clusters

Linux/MAC OS:

	-python3 history_path.py [FILE PATH] [COMP] [TRESHOLD] 
	-python3 history_path.py --fast day1/10.0min 0.5 R



################## prob_model.py ##################
The prob_model.py script is used to analyze the history path files created by the history_path.py script. This script receives several CSV files with history path (one per time window) and prints several evaluation metrics and marked IPs.

    
    [FILE PATH] The file name 
    [Graph] Show graphs (T -yes, F -no, default: F)
    [Filter Sensitivity] Filter sensitivity to be applied in the detection of outliers, 
                            the smaller the value, the less value should be positive
                            (default: 1)
     [Cluster Sensitivity] Algorithm sensitivity to be applied in the clustering algorithm, 
                            the higher the value, the lower the positive value, the lower the positive value, the lower the value of 100
                            (default: 99.7)
    [TimeWindows] time windows (min) (default 10,30,120 ), for example: 10,20,30

    --help: to get help

Note:
	1) Parameters may be omitted, provided that the following are also omitted
	2) Change Victims in Line 21 and 528
	3) change White List in Line 34
	4) Change Days in Line 504 and 505
	5) for more significant changes, go to the time subdivisions (Line 287) function for relative height (Line 362)

Linux/MAC OS:

	-python3 prob_model.py [FILE PATH] [Graph] [Filter Sensitivity] [Cluster Sensitivity] [TimeWindows]   
	-python3 prob_model.py day1/10.0min/results/Int/history_path_rel/Int_1_10.0_0.5_cluster_history_path.csv

To evaluate the approaches, some metrics were defined:

	- True Positives (TP) -  entities correctly classified as outliers

	- False Positives (FP) - entities wrongly classified as outliers

	- True Negatives (TN) - entities correctly classified as 'normal'

	- False Negatives (FN) - entities wrongly classified as 'normal'

	- Precision (PREC)- TP/(TP+FP)

	- Recall (REC) - TP/(TP+FN)

	- F1 - 2*(Precision*Recall)/(Precision+Recall)

Output Example:

{1: {'172.31.65.123', '172.31.69.25', '172.31.67.46', '172.31.67.109'}}
=>day: 1 vitms: 1 FP: 3 => 25.0
-------------------------
prec: 25.0
Fscore: 40.0

################## RRCF.py ##################

Adaptation from https://klabum.github.io/rrcf/, used in prob_model.py



# C2BID
T. Fernandes, L. Dias, and M. Correia, “C2BID: Cluster Change-Based Intrusion Detection”, in Proceedings of the 19th IEEE International Conference on Trust, Security and Privacy in Computing and Communications (IEEE TrustCom 2020), 2020 

