'''
History Path building
Pmore inf: python3 history_path.py --help
'''

import warnings
from functools import reduce
from os import listdir
from sys import argv, exit, version_info

import pandas as pd
from halo import Halo  # Spinner: https://pypi.org/project/halo/
from joblib import load, Parallel, delayed  # ,dump
from numpy import nan
from pathlib import Path
from sklearn import preprocessing
from tqdm import tqdm
from collections  import Counter

warnings.filterwarnings("ignore") #remove warnings
def help_msg():
    '''
    :return: interface warning message 
    :rtype: str
    '''
    print('''
    execution with python3 
    python3 history_path.py [FILE PATH] [COMP] [TRESHOLD] 
    [FILE PATH] location of .csv files 
         The directory should contain:
            - .csv files with features, with the [WINDOW]_features_[DAY]_[TIME].csv format
            -directory named 'results', with two sub-directory: 'Ext' and 'Int'each with:
                -csv files with clusters per time window, with format [WINDOW]_Int_[DAY]_[END_TIME]_cluster.csv or [WINDOW]_Ext_[DAY]_[TIME]_cluster.csv
                -directory named 'classifier' with the classifiers, with format [WINDOW]_Int_[DAY]_[END_TIME]_classifier.joblib or [WINDOW]_Ext_[DAY]_[TIME]_classifier.joblib    
        [WINDOW]: min size of time window 
        [DAY]: DD day
        [TIME]: HH/MM/SS from start of time window
    [TRESHOLD] minimum overlap value, by default 0.5. NOTE: Treshold values below 0.5 can lead to rating problems
    [COMP] features comparison mode, R- compare time window based on relative values, A- compare based on absolute values, by default A 
        
    --help: to get help 
    --fast: fast mode, only produces csv files with cluster numbers and only analyzes Int (internal) clusters

    ''')
    exit()

if not version_info > (3, 0):  # python version
    help_msg()

try:
    if '--help' in argv[1]:  # help message 
        help_msg()
except IndexError:
    help_msg()

def test_path(path):
    '''
     confirms that the location provided meets all the requirements for the implementation of the programme.
    if any requirements are not met the programme ends execution
    :param path: path
    :type path: str
    :return: list with csv files of features + dic with list of Src and Dst+ day files and time window
    :rtype: list
    '''
    csv_file = [f for f in listdir(path) if '.csv' in str(f)]  #confirms existence of csv files
    if len(csv_file) <= 2:  #  considers days with more than 2 windows only
        print ('Ausencia de ficheiros .csv para analisar em: ' + path)
        help_msg()


    Time_window=str(csv_file[0]).split("_")[0]
    day=  str(csv_file[0]).split("_")[2]

    result_dir = [f for f in listdir(path) if 'results' in str(f)]  # confirms the existence of the directory of the directory
    try:
        result_dir.pop(0)
    except IndexError:
        print ('No directory result')
        help_msg()

    Src_Dst_dir = [f for f in listdir(path + '/results') if
                   'Int' in str(f) or 'Ext' in str(f)]  # confirms the existence of the directory of the directory
    if len(Src_Dst_dir) == 0:
        print ('no directory result\Ext e result\Int ')
        help_msg()

    result = {}
    for i in Src_Dst_dir:
        classifier_dir = [f for f in listdir(path + '/results/' + i) if
                          'classifier' in str(f)]  # confirms the existence of the directory of the sub-directory
        if len(classifier_dir) == 0:
            print ('no directory result\ ' + i + '\classifier')
            help_msg()

        csv_file_result = [f for f in listdir(path + '/results/' + i) if
                           '.csv' in str(f)]  # confirms existence of csv files
        if len(csv_file_result) == 0:
            print ('no .csv files in result\ ' + i)
            help_msg()

        joblib_file_result = [f for f in listdir(path + '/results/' + i + '/classifier') if
                              '.joblib' in str(f)]  #confirms existence of  joblib files
        if len(joblib_file_result) == 0:
            print ('no .joblib files in result\ ' + i + '\classifier')
            help_msg()

        result[i] = {'csv': sorted(csv_file_result), 'joblib': joblib_file_result}  #  reorganises results; sorts csv list

    return csv_file, result, day, Time_window

def overlap(r_curr, r_next, rl_prev, rl_curr):
    '''
     Comput the overlap value between two time window (t and t') and two clusters (C and C')

    :param r_curr: entities of t classified as C
    :type r_curr: list
    :param r_next: entities of t' classified as C
    :type r_next: list
    :param rl_prev: entities of t classified as C'
    :type rl_prev: list
    :param rl_curr: entities of t classified as C
    :type rl_curr: list
    :return: overlap value
    :rtype: float
    '''
    return len(set().union([value for value in r_curr if value in rl_prev],
                           [value for value in r_next if value in rl_curr])) / len(
        set().union(r_curr, r_next, rl_curr, rl_prev))  #overlap computation

def format_dataframe(classifier, featuresa, featuresb,compar_mode='A'):
    '''
    makes the classification and standardization of entities according to a classifier, Normalizes based on the reference sets,
    deals with the disagreement between features
    Note: dataframes are changeable targets, so if they are changed in a function this reflects in the main code (passed by reference), the copy avoids this problem

    :param classifier: classifier
    :type classifier: KMeans
    :param featuresa: data to classify
    :type featuresa: dataframe
    :param featuresb: reference data
    :type featuresb: dataframe
    :param compar_mode: comparation mode
    :type compar_mode: str
    :return: dataframe group by clusters
    :rtype: dataframe
    '''
    # dataframes are changeable objectives, so if they are changed in a function this is reflected in the main code (passed by reference), the copy avoids this problem
    features_a = featuresa.copy()
    features_b = featuresb.copy()
    features_b=features_b.loc[:, (features_b != 0).any(axis=0)]#remove zero column
    min_max_scaler = preprocessing.MinMaxScaler()  # normalization
    scaler_data={'R':features_a,'A':features_b}

    if Counter(features_b.columns) ==  Counter(features_a.columns): # check if features are the same 
        features_a = pd.DataFrame(min_max_scaler.fit(scaler_data[compar_mode]).transform(features_a[scaler_data[compar_mode].columns]),
                                  index=features_a.index,  # min_max_scaler.fit_transform
                                  columns=features_a.columns)

        R = pd.DataFrame({'clusters': classifier.predict(features_a)}, index=features_a.index).groupby(
            ['clusters'])  # group by clusters and classifies  t-1 data according to t

    else:
        for i in [i for i in features_b.columns if i not in features_a.columns]:
            features_a[i] = 0  # add missing column

        features_a.drop(labels=[i for i in features_a.columns if i not in features_b.columns], axis=1,
                        inplace=True)  # removed unused features

        features_a = pd.DataFrame(min_max_scaler.fit(scaler_data[compar_mode]).transform(features_a[scaler_data[compar_mode].columns]),
                                  index=features_a.index,  # min_max_scaler.fit_transform
                                  columns=features_a.columns)

        R = pd.DataFrame({'clusters': classifier.predict(features_a)}, index=features_a.index).groupby(
            ['clusters'])

    return R


def cluster_monitoring(clusters,files,time,dir):
    '''
    get ip that have not changed their cluster as well as a relationship between clusters

    :param clusters: data frame  with overlap value (only contemplates cases where the overlap is greater than treshold)
    :type clusters: dataframe
    :param path: file path
    :type path: str
    :return: ip list that keeps your cluster and dic with clusters that change id
    :rtype: (list,list(str,list))
    '''
    path = str(argv[1]) + '/results/' + dir + '/' + [file for file in files if
                                                     time in file].pop()  # file path
    df=pd.read_csv(path, sep=',',index_col=0)#get files from time window


    df=df[df['clusters'].isin(list(clusters.index))] #get ip

    changes=[]
    for  index, row in clusters.iterrows(): # obtains a relationship between the clusters that do not change {cluster current: new cluster} where the current is the resultant of the classification and the new is the value of the anteriror that is 'inherited

        changes.append((int(index),int(pd.to_numeric(row).idxmax(axis=0))))
    return ((time,list(df.index)),(time,changes))

def get_overlap(features_file, csv_file, classifier_file, tt,t, path,treshold=0.5,compar_mode='A'):
    '''
    load data and applies overlap expression

    :param features_file: file names and features
    :type features_file: list [str]
    :param csv_file:  file name, classified  (clustering result)
    :type csv_file: list [str]
    :param classifier_file:  classifiers name
    :type classifier_file: list [str]
    :param tt: previous time window
    :type tt: str
    :param t: actual time window
    :type t: str
    :param path: files path
    :type path: str
    :param treshold: overlap treshold
    :type treshold: float
    :param compar_mode: comparation mode
    :type compar_mode: str
    :return: time window and overlap values where the treshold is checked
    :rtype: list [str,dataframe]
    '''
    try:
        features_t = pd.read_csv(str(argv[1]) + '/' + [file for file in features_file if t in file].pop(), sep=',',
                                 index_col=0)  #get features from time window t
        features_tt = pd.read_csv(str(argv[1]) + '/' + [file for file in features_file if tt in file].pop(),
                                  sep=',',
                                  index_col=0)  # get features from time window t-1
        classifier_t = load(path + 'classifier/' + [file for file in classifier_file if
                                                    t in file].pop())  # get classifier  from time window t
        classifier_tt = load(path + 'classifier/' + [file for file in classifier_file if
                                                     tt in file].pop())  # get classifier  from time window t-1
        Rl_curr = pd.read_csv(path + [file for file in csv_file if t in file].pop(), sep=',',
                              index_col=0)  # get cluster  from time window t
        R_curr = pd.read_csv(path + [file for file in csv_file if tt in file].pop(), sep=',',
                             index_col=0)  # get cluster  from time window t-1
    except KeyError:
        print ('invalid file name ')
        help_msg()

    features_t = features_t[features_t.index.isin(Rl_curr.index)]  #  ip selection from t
    features_tt = features_tt[features_tt.index.isin(R_curr.index)]  # ip selection from t-1


    R_next = format_dataframe(classifier_tt, features_t, features_tt,compar_mode) #classifies data of t according to tt
    Rl_prev = format_dataframe(classifier_t, features_tt, features_t,compar_mode) #classifies data of tt according to t
    Rl_curr = Rl_curr.groupby(['clusters'])  # group by cluster 
    R_curr = R_curr.groupby(['clusters'])  # group by cluster 

    match = pd.DataFrame(columns=R_curr.groups.keys(), index=Rl_curr.groups.keys()) #corresponds a cluster from t that had originatin in cluster i from tt
    for i in R_curr.groups.keys(): #analyses all clusters from two time windows
        r_curr = R_curr.get_group(i).index
        try:
            r_next = R_next.get_group(i).index
        except KeyError: #if there are no cases 
            r_next = []
        for il in Rl_curr.groups.keys():
            rl_curr = Rl_curr.get_group(il).index
            try:
                rl_prev = Rl_prev.get_group(il).index
            except KeyError:#if there are no cases
                rl_prev = []
            over= overlap(r_curr, r_next, rl_prev, rl_curr)
            if over>=treshold: match.iat[il, i] = over

    aux = match[(match >= treshold).any(axis=1)]
    if not aux.empty:
        match = match.dropna(axis=1, how='all') #removes columns only with Nan
        match = match.dropna(axis=0, how='all') #removes lines only with Nan
        #print (match)
        return t,match


    # dd=pd.concat([df,df_t],axis=1)

    
def main():

    if argv[1]=='--fast':
        argv.pop(1)
        slow_mode=False
    else:
        slow_mode = True

    try:
        features, result,day, Time_window = test_path(argv[1])
    except IndexError:
        help_msg()

    try:
        treshold= float(argv[2])
        try:
            compar_mode = str((argv[3]))
        except IndexError:
            compar_mode = 'A'
    except IndexError:
        treshold= 0.5
        compar_mode='A'
    except ValueError:
        compar_mode = str((argv[2]))
        try: treshold= float(argv[3])
        except IndexError:
            treshold = 0.5
        except ValueError:
            treshold = 0.5

    #print( treshold)

    if slow_mode:
        Result=result
    else:
        Result=['Int']

    for dir in Result:
        timestamp = sorted([str(file_name).split("_")[-2] for file_name in result[dir][
            'csv']])  # obtain timestanps from time windows and sort in order of seniority
        features_file = [file for file in features if
                         any(s in file for s in timestamp)]  # remove unused features file
        similarity=[i for i in Parallel(n_jobs=-1)(delayed(get_overlap)(features_file, result[dir]['csv'], result[dir]['joblib'], tt,t,
                      str(argv[1]) + '/results/' + dir + '/',treshold,compar_mode) for tt,t in tqdm(total=len(timestamp)-1,iterable=zip(timestamp[:-1],timestamp[1:]), desc='loading: '+dir))
            if i is not None]#get timstamp and cluster 

        spinner = Halo(text='clusters analyses '+dir+': ')
        spinner.start() # spiner
        spinner.text='clusters analyses '+dir+': file reading '
        path = str(argv[1]) + '/results/' + dir + '/' #files paths
        read_files =Parallel(n_jobs=-1)(
            delayed(pd.read_csv)(path + file, sep=',',index_col=0,skiprows=1, names=[str(file).split("_")[-2]]) for file in result[dir]['csv']) #file reading
        df_cluster=pd.concat(read_files,axis=1)#brings together all the ips analysed and their clusters
        
        if slow_mode:
            df_change=df_cluster.copy()



        IP, changes = zip(*Parallel(n_jobs=-1)(
            delayed(cluster_monitoring)(data, result[dir]['csv'], time, dir) for time, data in
            similarity)) #get 1: IP and time window in which it maintains the cluster; 2: old and new cluster relationship

        if slow_mode:
            spinner.text='clusters analyses '+dir+': making the file change_history_path'
            #produces the matrix with the changes from the previous time window (M-maintain cluster, O doesn't exist, E goes to a new cluster)
            for time, ip in IP:
                df_change.loc[ip, time] = 'M'
            df_change=df_change.fillna('O') #fills in blank spaces
            df_change =Parallel(n_jobs=-1)(
                delayed(df_change[i].apply)(lambda x: 'E' if x not in ['M','O'] else x) for i in
               list(df_change.columns))         #replaces value
            df_change = pd.concat(df_change,axis=1)

        spinner.text='clusters analyses '+dir+': making the file cluster_history_path'
        max=0
        for column in df_cluster: #gives a different number to each cluster
            max_l = (df_cluster[column].max())
            df_cluster[column]=df_cluster[column]+max
            max+=1+max_l
            
        df_cluster=df_cluster.fillna(-1)  #replace nan for -1
        df_cluster.loc[-1]=-1 #ensure that all time windows have cluster -1


        window,cluster_all=zip(*[[time,sorted(df_cluster[time].unique())] for time in list(df_cluster.columns) ])# get the clusters in each time window

        cluster_all_old=reduce(lambda a,b: a+b,cluster_all) #  copy list with old clusters, put everything together in a flat list

        for time,clusters in changes:
            index=window.index(time)
            for  old_cluster, new_cluster in  clusters:
                try:
                    cluster_all[index][old_cluster+1] = (cluster_all[index-1][new_cluster+1]) #replaces remaining  cluster (keeps smaller number)
                except IndexError:
                    print ('erro in cluster id changing')
                    help_msg()

        df_cluster.drop([-1], inplace=True) # remove line

        cluster_all=reduce(lambda a,b: a+b,cluster_all) #plane list

        for old,new in zip(cluster_all_old,cluster_all): #remove unchange items
            if old==new:
                cluster_all_old.remove(old)
                cluster_all.remove(new)


        df_cluster =Parallel(n_jobs=-1)(
            delayed(df_cluster[i].replace)(cluster_all_old,cluster_all) for i in
            list(df_cluster.columns))
        df_cluster = pd.concat(df_cluster,axis=1) #change value

        mode={'R':'rel','A':'abs'}
        spinner.text='Clusters analyses '+dir+': saving results'
        #save results
        name =str(argv[1]) + '/results/' + dir+ '/history_path_'+mode[compar_mode]+'/'
        Path(name).mkdir(parents=True, exist_ok=True)
        if slow_mode:
            df_change.to_csv(name+dir+'_'+day+'_'+Time_window+'_'+str(treshold)+'_'+'change_history_path.csv')
        df_cluster.to_csv(name+dir+'_'+day+'_'+Time_window+'_'+str(treshold)+'_'+'cluster_history_path.csv')
        spinner.succeed('file saved')


if __name__ == '__main__':
    main()
