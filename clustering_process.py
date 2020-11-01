'''
Script clustering aplication (k-means)
more inf: python3 clustering_process --help
'''

import os
import warnings
from os import listdir
from sys import argv, exit, version_info

import matplotlib.pyplot as plt
import pandas as pd
import plotly
import plotly.graph_objs as go
from joblib import Parallel, delayed, dump  # , load
from numpy import sqrt, linspace, asarray
from pathlib import Path
from pathlib import Path
from progress.bar import Bar
from sklearn import preprocessing
from sklearn.cluster import KMeans
from kneed import KneeLocator
from tqdm import tqdm
import numpy as np

warnings.filterwarnings("ignore") #remove warnings

def help_msg():
    '''
    :return: interface help mensage
    :rtype: str
    '''
    print('''
    execution with python3 
    python3 clustering_process.py [FILE PATH] 
    [FILE PATH]: The path to the file. The name of the file must respect the format: [WINDOW]_features_[DAY]_[TIME].csv
        Fast mode: directory for files to analyze 
        [WINDOW]: size of time window in minutes
        [DAY]:  Day
        [TIME]: HH/MM/SS from start of time window
    
   
    --help: to get help 
    --fast: fast mode, only produces the csv files with cluster by entity, you can read more than one file 


    
    ''')
    exit()

if not version_info > (3, 0): #check Python version
    help_msg()

try:
    if '--help' in argv[1]:  #  --help
        help_msg()
except IndexError:
    help_msg()

def read_file_name(file):
    '''
    read the file name for information about the day, time and time window

    :param file:file name 
    :type file: str
    :return: day
    :rtype: int
    :return: time window
    :rtype: int
    :rturn: hour
    :rtype: str
    '''
    file_name=str(file).split("/")[-1]# removes directory from the name, gets only the name of the file
    day=str(file_name).split("_")[2]# get day
    time_window=str(file_name).split("_")[0]# get time window
    time = str(file_name).split("_")[-1].split('.')[0]# get hour
    return day, time_window, time

def model_train(data, k):
    '''
    creates and trains a Kmeans clustering model

    :param data: int training data
    :type data: dataframe
    :param k : number of clusters
    :type k: int
    :return: model and value of inertia
    :rtype: float
    '''
    warnings.filterwarnings("ignore")  # remove warnings
    kmeanModel = KMeans(n_clusters=k,max_iter=1000,random_state=0,  tol=1e-5).fit(data) #42
    return kmeanModel,kmeanModel.inertia_

def elbow_method(data,k_max=20,k_min=3):
    '''
   Apply the elbow_method to calculate the optimal cluster number to be used in Kmeans
    As a cost measure it uses the sum of the square distances of the samples to the nearest cluster centre.
    Produces a graph of the method

    :param data : data
    :type data: dataframe
    :param k_max : max number of clusters
    :type k_max: int
    :param k_min: min number of clusters
    :type K_min: int
    :retun: optimal model
    :rtype: KMeans
    '''
    K = range(k_min, k_max)
    K_means,distortions=zip(*Parallel(n_jobs=-1)(delayed(model_train) (data,k) for k in Bar('elbow method', suffix='%(percent)d%%').iter(K)))  # realiza em parealelo o treino dos varios modelos
    kn = KneeLocator(K,distortions, curve='convex', direction='decreasing')

    best_k=kn.knee  #obtains a value of k for which less distortion is obtained
    kmeanModel=K_means[best_k-k_min] #obtains the classification that gives rise to the least distortion
    plt.figure(figsize=(16, 8))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow method method for best k ('+str(best_k)+')')
    #plt.show() 
    return kmeanModel, plt

def optimal_number_of_clusters(wcss):
    x1,y1 = 2, wcss[0]
    x2, y2 = 30, wcss[len(wcss)-1]
    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = np.sqrt((y2-y1)**2 + (x2-x1)**2)
        distances.append(numerator/denominator)
    return distances.index(max(distances)) + 2

def elbow_method_fast(data,k_max=30,k_min=2):
    '''
    [fast version]
    Apply the elbow_method to calculate the optimal cluster number to be used in Kmeans
    As a cost measure it uses the sum of the square distances of the samples to the nearest cluster centre.

:param data : data
    :type data: dataframe
    :param k_max : max number of clusters
    :type k_max: int
    :param k_min: min number of clusters
    :type K_min: int
    :retun: optimal model
    :rtype: KMeans
    '''
    K = range(k_min, k_max)
    K_means,distortions=zip(*[model_train(data,x) for x in K])
    kn = KneeLocator(K, distortions, curve='convex', direction='decreasing')
    best_k = kn.knee  # obtains a value of k for which less distortion is obtained
   
    kmeanModel=K_means[best_k-k_min] #obtains the classification that gives rise to the least distortion
    return kmeanModel


def result_analysis(view, plt, clusters, file, data):
    '''
  Funcao, for analysis of results, saves results
    Produces and stores:
        -elbow method graph
        -heatmap of clusters
        -csv with entity vs cluster relationship assigned
        -save classifier with joblib

    :param view: identificator
    :type view: str
    :param plt:graph elbow method (math.plot)
    :type plt: ploty
    :param clusters: classifier
    :type clusters: KMeans
    :param file: file name
    :type file: str
    :param data: data
    :type data: dataframe
    '''

    data['clusters']=clusters.predict(data) #get clusters

    day, time_window, hour = read_file_name(file)
    file_name= str(time_window)+'_'+str(view)+'_'+str(day)+'_'+str(hour)+'_'
    file_prefix = 'results/'+ str(view)+'/'
    dir_path = os.path.dirname(os.path.realpath(file))+'/'

    # elbow method graphical storage
    Path(dir_path + file_prefix+'elbow/').mkdir(parents=True,exist_ok=True)
    plt.savefig(dir_path + file_prefix+'elbow/'+file_name+'elbow.pdf')
   

    #save Classification of entities
    Path(dir_path + file_prefix).mkdir(parents=True, exist_ok=True)
    data=data.sort_values('clusters')
    data['clusters'].to_csv(dir_path+file_prefix+'/'+file_name+'clusters.csv')

    #heatmap
    toz = data.groupby('clusters').mean()
    toz = toz.loc[:, (toz > 0.2).any(axis=0)]
    data1 = [go.Heatmap(z=toz.values.tolist(),
                        y=list(toz.index.values),
                        x=list(toz),
                        colorscale='Viridis')]
    #save heatmap
    Path(dir_path + file_prefix + 'heat/').mkdir(parents=True, exist_ok=True)
    plotly.offline.plot(data1, filename=dir_path + file_prefix+'heat/'+file_name+ 'heatmap.html',auto_open=False)

    #save Kmeans 
    Path(dir_path + file_prefix + 'classifier/').mkdir(parents=True, exist_ok=True)
    name=dir_path + file_prefix+'classifier/'+file_name+'classifier.joblib'
    dump(clusters, name)

def result_analysis_fast(view, clusters, file, data):
    '''
[fast version]
    Funcao, for analysis of results, saves results
    Produces and stores:
        -csv with entity vs cluster relationship assigned
        -save classifier with joblib
    :param plt: graph elbow method (math.plot)
    :param clusters: classifier
    :type clusters: KMeans
    :param file: file name
    :type file: str
    :param data: data
    :type data: dataframe
    '''

    data['clusters']=clusters.predict(data) #get clusters

    day, time_window, hour = read_file_name(file)
    file_name = str(time_window) + '_' + str(view) + '_' + str(day) + '_' + str(hour) + '_'
    file_prefix = 'results/' + str(view) + '/'
    dir_path = os.path.dirname(os.path.realpath(file)) + '/'

    #Classification of entities
    Path(dir_path + file_prefix).mkdir(parents=True, exist_ok=True)
    data=data.sort_values('clusters')
    data['clusters'].to_csv(dir_path+file_prefix+'/'+file_name+'clusters.csv')

    #Save Kmeans (trained classifier)
    Path(dir_path + file_prefix + 'classifier/').mkdir(parents=True, exist_ok=True)
    name=dir_path + file_prefix+'classifier/'+file_name+'classifier.joblib'
    dump(clusters, name)

def test_path(path):
    '''
      confirms that the location provided meets all the requirements for the implementation of the programme.
    if any requirements are not met the programme ends execution
    :param path: loc (file path)
    :type path: str
    :return:  csv files list ( features)
    :rtype: list
    '''
    csv_file = [f for f in listdir(path) if '.csv' in str(f)]  # confirms existence of csv files
    if len(csv_file) <= 2: #considers days with more than 2 windows only
        print ('file not found:: '+path)
        help_msg()

    return csv_file

def fast(file):
    '''
  [fast version]
    Processes a file/time window in order to obtain the cluster and the classifier
    :param file: file name 
    :type file: str

    '''
    try:
        file = str(argv[2]) + '/'+str(file)
    except IndexError:
        help_msg()

    try:
        data = pd.read_csv(file, sep=',', index_col=0)
    except FileNotFoundError:
        print ('File Not Found')
        help_msg()

    int_df = data[data.index.str.match('172.31.')] # selct internal IP
    ext_df = data[~data.index.str.match('172.31.')] # select external ip
    int_df = int_df.loc[:, (int_df != 0).any(axis=0)]  # delete zero columns
    ext_df = ext_df.loc[:, (ext_df != 0).any(axis=0)]  # delete zero columns

    if len(ext_df.index) >= 30:  # consider cases with more than 20 elements
        min_max_scaler = preprocessing.MinMaxScaler()  # normalization
        ext_df = pd.DataFrame(min_max_scaler.fit_transform(ext_df), index=ext_df.index, columns=ext_df.columns)
        clusters_ext = elbow_method_fast(ext_df)
        result_analysis_fast(view='Ext', clusters=clusters_ext, file=file, data=ext_df)

    else:
        pass

    if len(int_df.index) >= 30:  # consider cases with more than 20 elements
        min_max_scaler = preprocessing.MinMaxScaler()  # normalization
        int_df = pd.DataFrame(min_max_scaler.fit_transform(int_df), index=int_df.index, columns=int_df.columns)
        clusters_int = elbow_method_fast(int_df)
        result_analysis_fast(view='Int', clusters=clusters_int, file=file, data=int_df)
    else:
        pass

def main():
    '''
    analyses only one file, produces several results
    '''

    try:
        file = str(argv[1])
    except IndexError:
        help_msg()


    try:
        data=pd.read_csv(file, sep=',',index_col=0)
    except FileNotFoundError:
        print ('File Not Found'+file)
        help_msg()



    int_df = data[data.index.str.match('172.31.')] # select int ip
    ext_df = data[~data.index.str.match('172.31.')] # select ext ip
    int_df = int_df.loc[:, (int_df != 0).any(axis=0)]  # delete zero columns
    ext_df = ext_df.loc[:, (ext_df != 0).any(axis=0)]  #  delete zero columns

    if len(ext_df.index) >= 20: #consider cases with more than 20 elements
        print ('Ext Ip analysis')
        min_max_scaler = preprocessing.MinMaxScaler()  # normalization
        ext_df = pd.DataFrame(min_max_scaler.fit_transform(ext_df), index=ext_df.index, columns=ext_df.columns)
        clusters_ext, plt_ext = elbow_method(ext_df)
        result_analysis(view='Ext', plt=plt_ext, clusters=clusters_ext, file=file, data=ext_df)

    else:
        pass

    if len(int_df.index) >= 20: # consider cases with more than 20 elements
        print ('Int IP analysis')
        min_max_scaler = preprocessing.MinMaxScaler()  # normalization
        int_df = pd.DataFrame(min_max_scaler.fit_transform(int_df), index=int_df.index, columns=int_df.columns)
        clusters_int, plt_int = elbow_method(int_df)
        result_analysis(view='Int', plt=plt_int, clusters=clusters_int, file=file, data=int_df)
    else:
        pass

def main_fast():
    '''
    fast mode, produces only the list of clusters and classifiers
    '''
    files= test_path(str(argv[2]))
    Parallel(n_jobs=-1)(delayed(fast)(file) for file in tqdm(iterable=files, desc='loading clustering'))




if __name__ == '__main__' and '--fast' in argv[1]:
    main_fast()
elif __name__ == '__main__':
    main()
