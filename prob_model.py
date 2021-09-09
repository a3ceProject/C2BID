'''
Script to analyze probabilities and detect attacks
Analyzes the various HPs in order to detect Outliers
more inf run: python3 prob_model.py --help
'''

from collections import Counter, ChainMap
from datetime import datetime, timedelta
from functools import reduce
from sys import argv, exit, version_info

from halo import Halo  # Spinner: https://pypi.org/project/halo/
from joblib import Parallel, delayed
from matplotlib.pyplot import plot, scatter, show, figure, axes
from numpy import log, quantile, asarray, all, vectorize
from pandas import DataFrame, qcut, read_csv, concat
from sklearn.preprocessing import MinMaxScaler

from RRCF import Rrcf

IP_V={
    1:['172.31.69.25'],
    }
white_list=["172.31.0.2"]
class Marker():
    '''
 Classifier according to defined values, updated values (increase or decrease) according to defined rules
    '''
    def __init__(self,q_09=0,q_08=0,q_03=0,q_04=0):
        '''
        object initiation
        :param q_09:higgest Quartil 
        :type q_09: float
        :param q_08:second higgest Quartil
        :type q_08: float
        :param q_03: second smaller Quartil 
        :type q_03: float
        :param q_04:smallest Quartil
        :type q_04: float
        '''
        self.q_09=q_09
        self.q_08 =q_08
        self.q_03 =q_03
        self.q_04 =q_04
    def Mark_prob(self,x):
        '''
        Checks if x is between any defined interval and updates according to the interval
        :param x: value
        :type x: float
        :return: new value
        :rtype: float
        '''
        if x > self.q_09:
            return x * 1.15
        elif x > self.q_08:
            return x* 1.05
        elif x < self.q_03:
            return x* .85
        elif x < self.q_04:
            return x* 0.95
        return x


def help_msg():
    '''
    :return: hel mensage
    :rtype: str

    '''
    print('''
    execution with python3 
    python3 prob_model.py [FILE PATH] [Graph] [Filter Sensitivity] [Cluster Sensitivity] [TimeWindows] 
    [FILE PATH] The file name 
    [Graph] Show graphs (T -yes, F -no, default: F)
    [Filter Sensitivity] Filter sensitivity to be applied in the detection of outliers, 
                            the smaller the value, the less value should be a positive value
                            (defaul: 1)
     [Cluster Sensitivity] Algorithm sensitivity to be applied in the cluster algorithm, 
                            the higher the value, the lower the positive value, the lower the positive value, the lower the value of 100
                            (defaul: 99.7)
    [TimeWindows] time windows (min) (default 10,30,120 ), for example: 10,20,30
    

    Note: Parameters may be omitted, provided that the following are also omitted

    --help: to get help 

    ''')
    exit()

try:
    if '--help' in argv[1]:  # help mensage
        help_msg()
except IndexError:
    help_msg()

if not version_info > (3, 0): #python version
    help_msg()

def Generate_transition_matrix(data):
    '''
      Counting changes of status in a determiand list

    :param data: hp list for one ip
    :type data: list
    :return: counting
    :rtype: dict
    '''
    c = (Counter(list(zip(data, data[1:]))))
    return c

def get_prob(path, transition,Dv):
    '''
     Computes the probability of a path based on a given probability matrix.
    Values not in the matrix are replaced by the quartile value (or equivalent) of the median 

    :param path: path: History path
    :type path: list[str]
    :param transition: likelihood matrix
    :type transition: dataframe
    :param Dv: quartil value 
    :type Dv: dict
    :return: likelihood
    :rtype: float
    '''
    changes = list(zip(path, path[1:]))
    prob =[]
    zero_value=[]
    for i, j in changes:
        try:
            prob.append( transition.at[i, j]) #get likelihood
        except KeyError:
            zero_value.append(0) # counter for not existing cases
    if zero_value:
        decil=[]
        if prob:
            for p in prob:
                for d in sorted(list(Dv.keys()),reverse=True):
                    if p>= Dv[d]:
                        decil.append(d) # converts the probabilities to decile
                        break
        try:
            m= quantile(decil,0.5) #get  median
        except  IndexError:
            m=0
        for i in zero_value:
            prob.append(Dv[int(m)]) # add missing values 
    return sum(prob)

def  fill_Nan(df):
    '''
    Fill in the dataframe based on the decile to which the Nan line values belong
    :param df: dataframe
    :type df: DataFrame
    :return: Dataframe with no NaN
    :rtype: DataFrame
    '''
    df_cut=df.copy()
    for i in df.columns:
        df_cut[i]= qcut(df_cut[i], 10, labels=False,duplicates='drop') #marks the decil of each one according to column
    df_cut=DataFrame(df_cut.mean(axis=1),index=df.index) #calculate average per line
    df_q=DataFrame(index=df.index, columns=df.columns)
    for i in df_q.columns:
        df_q[i] = df_cut + 1#set values and create auxiliary Df
    deciles={}
    for i in list(df.columns) : #get decile values per column
        deciles[i]=[]
        for n  in range(1,11):
            deciles[i].append(float(df[i].quantile(n/10)))

    for i in df.columns:
        df[i]=df[i].fillna(df_q.applymap(lambda x:  deciles[i][int(x-1)])[i]) #replaces the Nan
    return df

def graph(f,p,dia,window):
    '''
    Graphs 2d or 3d according to cluster in f and p-values
    :param f: dataframe  clusters
    :type f: DataFrame
    :param p: Data frame values
    :type p: DataFrame
    :param dia: day
    :type dia: int
    :param window: time window
    :type window: list[str]
    '''
    f['cluster'] = f['cluster'].astype('int') #clusters convertions
    colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'goldenrod',
             'lightcyan', 'navy', 'olive'] #coulors
    vectorizer = vectorize(lambda x: colors[x % len(colors)])
    if len(p.columns) == 2: #graph 2D
        #for i in list(IP_V[dia]): #plot vitms
            #plot(p[p.columns[0]].loc[i], p[p.columns[1]].loc[i], 'r^')
        scatter(p[p.columns[0]], p[p.columns[1]], c=vectorizer(f['cluster'].tolist()))
        fig.suptitle(str('day ' + str(dia) + '  time:' + str(i)), fontsize=12)
        show()
    elif len(p.columns) == 3: #graph 3D
        from mpl_toolkits.mplot3d import Axes3D
        fig = figure()

        ax = axes(projection='3d')
        #for w in list(IP_V[dia]): #plot vitms
            #ax.scatter3D(p[p.columns[0]].loc[w], p[p.columns[1]].loc[w], p[p.columns[2]].loc[w], c='r', marker='^')
        ax.scatter3D(p[p.columns[0]], p[p.columns[1]], p[p.columns[2]], c=vectorizer(f['cluster'].tolist()));
        ax.set_xlabel(str(p.columns[0])) #axis title
        ax.set_ylabel(str(p.columns[1]))#axis title
        ax.set_zlabel(str(p.columns[2]))#axis title
        fig.suptitle(str('day ' + str(dia) + '  time:' + str(window)), fontsize=12)#graph title
        show()

def f1_score(TP,FP,TC=19):
    '''
   compute Fscore (F1)
    F1 = 2 * (precision * recall) / (precision + recall)
    :param TP: true positives
    :type TP: int
    :param FP:  false positives
    :type FP: int
    :param TC: TP+FN
    :type TC: int
    :return: f1 score
    :rtype: float
    '''
    try:
        return (2* ( (TP/(TP+FP)) * (TP/TC) ) / ( (TP/(TP+FP)) + (TP/TC) ) )*100
    except ZeroDivisionError:
        return 0


def day_analyse(dia,time_window,Path,filt_sens=1,cluster_sens=99.7, Graph=False,kind='Int_'):
    '''
    analyzes the likelihood of a day. can make a graph (visual aid) of the process

    :param dia: day
    :type dia: int
    :param time_window: analised time windows
    :type time_window: list[int]
    :param Path: file path
    :type Path: str
    :param filt_sens: filter treshold
    :type filt_sens: int
    :param cluster_sens: rrcf treshold
    :type cluster_sens: float
    :param Graph: graph mode
    :type Graph: bool
    :param kind: Internal network or external network ('Int_'/'Ext_')
    :type kind: str
    :return: list of outliers and ip not analysed with respective time window
    :rtype: list, dict
    '''
    IP_outlier= set()
    Data = {}
    not_detct={}
    for TW in time_window:
        path=Path

        loc = path.find('day') + 3
        loc1= path.find('/') 
        path = path[:loc] + str(dia) + path[loc1:] #day defenition in file path

        loc = path.find('/')+1
        loc2 =loc+ str(path)[loc:].find('.')
        path = path[:loc] + str(int(TW)) + path[loc2:] #time-window defenition on file


        loc = str(path).find(kind) + 6
        loc2=loc+str(path)[loc:].find('.')
        path = path[:loc] + str(int(TW)) + path[loc2:]# day defeniton on file 

        loc = path.find(kind) + 4

        loc1=str(path[loc:]).find('.') 


        path = path[:loc] + str(dia)+'_'+ str(int(TW))+ path[loc:][loc1:] #time window defenition on file path
  


        try:
            hp = read_csv(str(path), sep=',', index_col=0)  #csv loading
        except FileNotFoundError:
            print('File not found: ' + path,dia,TW)
            help_msg()
        else:
            Data[TW] = hp #old read file
    Max=max(list(Data.keys())) #get higger time-window
    TW_max=list(Data[Max].columns)# divid time windows
    indices = [i for i, elem in enumerate(TW_max) if '12:00'<= elem]
    try:
        Time=[TW_max[0:indices[0]+1],
               TW_max[indices[0]:]] 
    except IndexError: # if the perid is less than 12 h divides in half
        Time= [TW_max[0:int(len(TW_max)/2)+1],
               TW_max[int(len(TW_max)/2):]] 


    for window in Time:
        data = []
        t_ini = window[0] #get firts time window to analyze
        t_end = window[-1] #get last time window to analyze
        for TW in time_window:
            hp= Data[TW]
            list_col = [i for i in list(hp.columns) if
                        datetime.strptime(i, '%H:%M:%S') + timedelta(minutes=TW) >= datetime.strptime(t_ini, '%H:%M:%S')
                        and datetime.strptime(i, '%H:%M:%S') < datetime.strptime(t_end, '%H:%M:%S') + timedelta(
                            minutes=Max)] #seleciona janelas de tempo dentro do periodo em analise
            hp = hp[list_col]

            for index, row in hp.iterrows():  # removes ip with more than 60% of windows disabled
                check = dict(Counter(list(row)))
                if -1 in check.keys():
                    if check[-1] / float(hp.shape[1]) > 0.6:
                        if check[-1] / float(hp.shape[1]) < 1 and index not in white_list:
                            indices = [i for i, elem in enumerate(row) if -1 != elem]
                            while indices:# adds active time windows to a list
                                try:
                                    not_detct[(index,TW)].append(str(hp.columns[indices.pop()]))
                                except KeyError:
                                    not_detct[(index,TW)]=[str(hp.columns[indices.pop()])]


                        hp = hp.drop(index, axis=0)
            cntr =  [Generate_transition_matrix(l) for l in hp.values.tolist()]  #  counts the changes per line
            #cntr =  Parallel(n_jobs=-1)(delayed(Generate_transition_matrix)(l) for l in hp.values.tolist())  # conta as trazicoes por linha


            cntr = reduce(lambda a, b: a + b, cntr)  # add all values
            try:
                row, columns = zip(*cntr.keys()) #get all clusters 
                stages = list(set().union(row, columns))
            except ValueError:
                print('erro na contagem ',cntr.keys(), TW, list_col)
                exit()
            transition = DataFrame(index=sorted(stages), columns=sorted(stages))
            for i, j in cntr.keys():  # matrix creation
                transition.at[i, j] = cntr[(i, j)]
            try:
                transition = transition.drop(-1, axis=0) #remove column -1
            except KeyError:
                pass
            try:
                transition = transition.drop(-1, axis=1) #remove line -1
            except KeyError:
                pass
            transition = transition.dropna(how='all', axis=1)#remove columns with no vlaues
            transition = transition.dropna(how='all', axis=0)#remove lines with no vlaues

            transition = transition / transition.sum().sum()  #get likelihood
            transition = transition.applymap(log) * -1  # apply log


            V = transition.values.tolist()
            v = list(reduce(lambda a, b: a + b, V))
            v = [incom for incom in v if str(incom) != 'nan']
            v = list(set(v))

            q_09 = quantile(v, 0.9) # percentil 0.9
            q_08 = quantile(v, 0.8) # percentil 0.8
            q_04 = quantile(v, 0.4) # percentil 0.4
            q_03 = quantile(v, 0.3) #percentil 0.3


            mark= Marker(q_09,q_08,q_03,q_04)

            transition=transition.applymap(mark.Mark_prob)

            Df = {0: 0}
            for q in range(1, 101):#percentil computation
                Df[q] = quantile(v, q / 100)

            p = [get_prob(row.tolist(), transition, Df) for index, row in hp.iterrows()] # compute likelihood for each path
            #p = Parallel(n_jobs=-1)(delayed(get_prob)(row.tolist(), transition, Df) for index, row in hp.iterrows()) 
            p = DataFrame(p, columns=['likehood' + str(TW)], index=hp.index)
            data.append(p)

        p = concat(data, axis=1) #joins the different time windows
        p = fill_Nan(p)# fill Nan

        min_max_scaler = MinMaxScaler()  # normalization
        p = DataFrame(min_max_scaler.fit_transform(p), index=p.index, columns=p.columns)
        n = len(list(p.index))

        perc = ((n - filt_sens) / n) #filter defenition
        filtro = p.quantile([perc], axis=0).values 
        clustering = Rrcf(n_jobs=1, perc=cluster_sens).fit_predict(p)
        f = DataFrame(clustering, columns=['cluster'], index=p.index)

        all_larger = lambda a: all(asarray(filtro) > asarray(a)) #apply filter 
        tp_fp = list(f[f['cluster'] == -1].index) #select TP and FP
        out_list = []
        for prob, out in zip(p.loc[tp_fp].values, tp_fp):
            if all_larger(prob):
                f.loc[out] = -2
                out_list.append(out)
                #tp_fp.remove(out)
        tp_fp = [i for i in tp_fp if i not in out_list]
        white_list_2=[i for i in tp_fp if i in white_list] #apply whit list
        #for ip in white_list_2:  f.loc[ip]=-3 #
        f.loc[white_list_2] = -3

        if Graph: graph(f,p,dia,window) #Graph
        IP_outlier.update(f[f['cluster'] == -1].index)
    return IP_outlier, {dia:not_detct}#return outliers

def Cluster_analy(dia, janelas,PATH,kind='Int_'):
    '''
    Analisa os dados DYN3_100 em busca de entidades em clusters isolados

    :param dia: day
    :type dia: int
    :param janelas: IP set, time window size and window to analyze
    :type janelas: list[dict]
    :param PATH: HP path
    :type PATH: str
    :param kind: Internal network or external network ('Int_'/'Ext_')
    :type kind: str
    :return: IP marked
    :rtype: (int,list)
    '''
    result=set()
    for window in janelas:
        path=PATH
        IP, TW = window

        loc = str(path).rfind('/')
        path = path[:loc]
        loc = str(path).rfind('/')
        path = path[:loc + 1]  # cluster directory
        loc = str(path).find('day') + 3 #day
        path = path[:loc] + str(dia) + path[loc + 1:]
        loc = path.find('day') + 5
        loc2 = loc + str(path)[loc:].find('.')
        path = path[:loc] + str(TW) + path[loc2:]  # time-window defenition

        for tw in janelas[(IP,TW)]:
            Path = path+str(TW)+'.0_'+kind+str(dia)+'_'+str(tw)+'_clusters.csv' # cluster file
            try:
                data_cluster =read_csv(Path, sep=',', index_col=0) #reading file
            except FileNotFoundError:
                print('Ficheiro nao existente: ' + Path, dia, TW)
                help_msg()
            count = data_cluster['clusters'].value_counts() #computing elements per clustering
            for a in count.index:
                if count[a] == 1:#one elemente cluster
                    if IP in list(data_cluster[data_cluster['clusters'] == a].index): 
                        result.add(IP)
    if result:
        return (dia,list(result) )

def main():
    spinner = Halo(text=' Time-Windows analyse  ')
    spinner.start()  # spiner
    try:
        path = str(argv[1])
    except IndexError:
        spinner.fail('error path reading')
        help_msg()
    try:
        Graph = str(argv[2])
    except IndexError:
        Graph=False
    else:
        if Graph=='T':
            Graph = True
        elif Graph=='F':
            Graph = False
        else:
            spinner.fail('invalid graph option')
            help_msg()

    try:
        filt_sens = float(argv[3])
    except ValueError:
        spinner.fail('filter value invalid')
        help_msg()
    except IndexError:
        filt_sens=1
    else:
        if filt_sens<0:
            spinner.fail('filter value invalid, it must be >0')
            help_msg()

    try:
        cluster_sens = float(argv[4])
    except ValueError:
        spinner.fail('rrfc value invalid ')
        help_msg() 
    except IndexError:
        cluster_sens=99.6
    else:
        if cluster_sens<=0 or cluster_sens>=100:
            spinner.fail('rrfc value invalid, it mus be ]0,100[')
            help_msg()

    try:
        TimeWindons = [float(i) for i in str(argv[5]).split(",")]
    except ValueError:
        spinner.fail('invalid time window')
        help_msg()
    except IndexError:
        TimeWindons = [10,30,120]

    #count,not_detc=zip(*Parallel(n_jobs=-1)(delayed(day_analyse)(dia,TimeWindons,path,filt_sens,cluster_sens,Graph) for dia in [1,10])) #use to check multiple day (files)
    count,not_detc=day_analyse(10,TimeWindons,path,filt_sens,cluster_sens,Graph)
    print(count)
    count={dia:c  for dia,c  in zip([1],count)} 
    # if not_detc:

    #     not_detc=dict(ChainMap(*not_detc))#merge everything
    #     c=Parallel(n_jobs=-1)(delayed(Cluster_analy)(dia,not_detc[dia],path) for dia in not_detc.keys())
    #     c=[i for i in c if i != None]
    #     c={dia:ip for dia,ip in c}
    # spinner.succeed('Outliers detetados')

    # TP=0
    # FP=0
    # print(count)
    # for i in count:
    #     vit = [x for x in count[i] if x in IP_V[i]]
    #     TP += len(vit)
    #     FP += (len(count[i]) - len(vit))
    #     try:
    #         print('=>day: ' + str(i) + ' vitms: ' + str(len(vit)) + ' FP: ' + str(
    #             len(count[i]) - len(vit)) + ' => ' + str(100 * len(vit) / len(count[i])))
    #     except ZeroDivisionError:
    #         print('=>day: ' + str(i) + ' Vitms: ' + str(len(vit)) + ' FP: ' + str(
    #             len(count[i]) - len(vit)) + ' => 0/0')
    # print('-------------------------')
    # print('prec: ' + str(TP / (TP + FP) * 100))
    # print('Fscore: ' + str(f1_score(TP, FP,1)))  



if __name__ == '__main__':
    main()
