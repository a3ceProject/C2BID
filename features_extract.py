'''
Script for  features extraction  (based on the OutGene method)
For more information run: python3 features_extract.py --help
'''

import os
import warnings
from datetime import datetime
from sys import argv, exit, version_info  # le linha de comando

import numpy
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from progress.bar import IncrementalBar
from tqdm import tqdm

warnings.filterwarnings("ignore") #remove warnings
dates={ #relationship between file name, the days and dates to be considered and most used port (src and dst)
    'new_wednesday-14-02-18.csv':[1,datetime(2018,2,14,4,0,0),datetime(2018,2,15,4,0,0),[[22],[21,22]]],
    'new_thursday-15-02-18.csv':[2,datetime(2018,2,15,4,0,0),datetime(2018,2,16,4,0,0),[[],[80]]],
    'new_friday-16-02-18.csv':[3,datetime(2018,2,16,4,0,0),datetime(2018,2,17,4,0,0),[[80],[21,80]]],
    'new_tuesday-20-02-18.csv':[4,datetime(2018,2,20,4,0,0),datetime(2018,2,21,4,0,0),[[80],[80]]],
    'new_wednesday-21-02-18.csv':[5,datetime(2018,2,21,4,0,0),datetime(2018,2,22,4,0,0),[[80],[80]]],
    'new_thursday-22-02-18.csv':[6,datetime(2018,2,22,4,0,0),datetime(2018,2,23,4,0,0),[[80],[80]]],
    'new_friday-23-02-18.csv':[7,datetime(2018,2,23,4,0,0),datetime(2018,2,24,4,0,0),[[80],[80]]],
    'new_wednesday-28-02-18.csv':[8,datetime(2018,2,28,4,0,0),datetime(2018,3,1,4,0,0),[[51603,54751],[31337]]],
    'new_thursday-01-03-18.csv':[9,datetime(2018,3,1,4,0,0),datetime(2018,3,2,4,0,0),[[51040,31337,53445],[51040,31337]]],
    'new_friday-02-03-18.csv':[10,datetime(2018,3,2,4,0,0),datetime(2018,3,3,4,0,0),[[8080,0],[8080,0]]]}

def help_msg():
    '''
    :return: help mensage for intaerface
    :rtype: str

    '''
    print('''
    execution with python3 
    python3 features_extract.py [FILE PATH] [TimeWindows] [method] [X]
    [FILE PATH] The file name 
    [TimeWindows] time windows (default 10), for example: 10,20,30
    [method] 0 OutGene (defaul); 1  the day  ports; 2 Junction of ports 0 and 1 ;3 DN3_x, 4 DN3_x_Outgene, 5 FlowHacker
    [X] Port number to be used in DN3_X method, defaul: 100
    
    Note: If you specify the [method] you must also specify [TimeWindows].
    
    --help: to get help 
    --file: get possible files 


    ''')
    exit()
def files_name():

    '''
    :return: supported files 
    :rtype: str
    '''
    print ('FILES:')
    for file in dates.keys():
        print('     '+str(file))
    exit()

if not version_info > (3, 0): #verifica versao do Python
    help_msg()

try:
    if '--help' in argv[1]:  # Instrucoes --help
        help_msg()
except IndexError:
    help_msg()

if '--files' in argv[1]:#Instrucoes --files
    files_name()


def get_ports(method=0,X=None):
    '''
      Ports extraction

    :param method: service name
    :type: int
    :param X: method value (if applied)
    :return: extration port
    :rtype: list
    '''
    methods={0:get_port_outgene,1:get_port_day,2:get_port_mist,3:get_port_DYN3_x,4:get_port_get_port_DYN3_x_outgene,5:get_port_FlowHacker}

    try:
        return methods[method]()
    except TypeError:
        return methods[method](X)

def get_port_outgene():
    '''
     OutGene ports

    :return: ports 
    :rtype: list
    '''
    ports=[80,194,25,22] #colocar port 21
    return ports, ports

def get_port_FlowHacker():
    '''
    FlowHacker ports

    :return: ports 
    :rtype: list
    '''
    ports=[80,194,25,22,6667]
    return ports, ports


def get_port_DYN3_x(df):
    '''
    get according to df provided, the most used, the least used and the most unusual ports (most used for less than 10 IP)
    Ports analysed are between 0-49151, and for the less used and up to 1024

    :param df: dataframe with ports
    :type df: DataFrame
    :return:  ports list
    :rtype: list[int]
    '''
    try:
        x=int(argv[4])
    except IndexError:
        x=100
    except ValueError:
        help_msg()
    aux = df[df['Dst Port'] < 49151].groupby(['Dst Port']).nunique()  # dst port counting
    uniqueports = aux.sort_values(by=['Src IP'],
                                  ascending=False)  # orders from the most contacted to the least contacted ports
    uniqueports = uniqueports[
        uniqueports['Src IP'] < 10].index.tolist()  # list of Ports contacted by less than 10 IP
    Dst_port = df[df['Dst Port'] < 49151]
    Src_port = df[df['Src Port'] < 49151]
    most_used_ports_dst = Dst_port['Dst Port'].value_counts()  # counting the use of the ports of destination
    most_used_ports_src = Src_port['Src Port'].value_counts()  # counting the use of the ports of origin
    most_used_ports = most_used_ports_dst.append(most_used_ports_src)  # add counting 
    most_used_ports = most_used_ports.sort_values(ascending=False)  # descending  orders 
   
    port = list(set().union(
        list(most_used_ports.head(int(x / 3)).index),  # must used ports
        list(most_used_ports[most_used_ports.index.isin(uniqueports)].head(int(x/3)).index),
        # less common and more used ports
        list(most_used_ports[most_used_ports.keys() < 1024].tail(
            int(x / 3)).index)))  # less used ports and below 1024
    return port, port

def get_port_get_port_DYN3_x_outgene(df):
    '''
     Returns ports according to metogo DYN3_x and outgene, i.e. a union between the two
    :param df: ports dataframe 
    :type df: DataFrame
    :return: ports list 
    :rtype: list[int]
    '''
    port1a,port1b=get_port_DYN3_x(df)
    port2a, port2b= get_port_outgene()
    porta= list(set(port1a+port2a))
    portb = list(set(port1b+ port2b))
    return porta,portb


def get_port_day():
    '''
    returns ports involved in attacks on each of the days (point of view of origin and destination)
    :return: ports (org and dst)
    :rtype: list
    '''
    try:
        file=str(argv[1].split("/")[-1])
    except IndexError:
        help_msg()

    try:
        return dates[file][3][0],dates[file][3][1]
    except KeyError:
        print ('fle not found')
        help_msg()

def get_port_mist():
    '''
    returns the ports involved in attacks on each of the days (point of view of origin and destination)
    :return: ports (org and dst)
    :rtype: list
    '''
    try:
        file=str(argv[1].split("/")[-1])
    except IndexError:
        'wrong file name'
        help_msg()

    try:
         port_src=list(set().union(dates[file][3][0],[80,194,25,22]))
         port_dst=list(set().union(dates[file][3][1],[80,194,25,22]))
    except KeyError:
        print ('file not found')
        help_msg()
    return port_src,port_dst

def get_dst_pkts(df, list_ports):
    '''
   function to extract the values of each feature related to ports from the Destination point of view.
    Calculates the number of packages sent/received

    :param df: analysed dataframe 
    :type: dataframe
    :param list_ports: port list 
    :type: list
    :return: packege counting 
    :rtype: list
    '''
    #corrects Bwd and Fwd values
    df['Tot Bwd Pkts'] -= 1
    df['Tot Fwd Pkts'] += 1
    # remove unused ports
    df_DstTo = df[df['Dst Port'].isin(list_ports)]
    df_DstFrom = df[df['Src Port'].isin(list_ports)]
    df_DstTo = df_DstTo[['Dst IP', 'Dst Port','Tot Bwd Pkts','Tot Fwd Pkts']]
    df_DstFrom = df_DstFrom[['Dst IP', 'Src Port','Tot Bwd Pkts','Tot Fwd Pkts']]
    df_DstTo = df_DstTo.groupby(['Dst IP', 'Dst Port']).sum().reset_index()   # counts the number of occurrences according to an IP+ destination port
    df_DstTo = df_DstTo.pivot(index='Dst IP', columns='Dst Port', values='Tot Bwd Pkts')  # data cast
    df_DstTo = df_DstTo.rename(columns=lambda x: str(x) + 'DstTo')  # change name
    df_DstFrom = df_DstFrom.groupby(['Dst IP', 'Src Port']).sum().reset_index()  # ccounts the number of occurrences according to an IP+ origin port
    df_DstFrom = df_DstFrom.pivot(index='Dst IP', columns='Src Port', values='Tot Fwd Pkts')
    df_DstFrom = df_DstFrom.rename(columns=lambda x: str(x) + 'DstFrom')  # data cast
    result = pd.concat([df_DstFrom, df_DstTo], axis=1, sort=False)  #  dataframes union
    return result

def get_src_pkts(df, list_ports):
    '''
    function to extract the values of each feature related to ports from the point of view of Source
    Calculates the number of packages sent/received

    :param df: analysed dataframe 
    :type: dataframe
    :param list_ports: port list 
    :type: list
    :return: packege counting 
    :rtype: list
    '''
    #corrects Bwd and Fwd values
    df['Tot Bwd Pkts'] -= 1
    df['Tot Fwd Pkts'] += 1
    # remove unused ports
    df_SrcFrom=df[df['Dst Port'].isin(list_ports)]
    df_SrcTo=df[df['Src Port'].isin(list_ports)]
    df_SrcFrom = df_SrcFrom[['Src IP', 'Dst Port','Tot Bwd Pkts','Tot Fwd Pkts']]
    df_SrcTo = df_SrcTo[['Src IP', 'Src Port','Tot Bwd Pkts','Tot Fwd Pkts']]
    df_SrcFrom = df_SrcFrom.groupby(['Src IP', 'Dst Port']).sum().reset_index() #  counts the number of occurrences according to an IP+ destination port
    df_SrcFrom = df_SrcFrom.pivot(index='Src IP', columns='Dst Port', values='Tot Bwd Pkts')  # # data cast
    df_SrcFrom = df_SrcFrom.rename(columns=lambda x: str(x) + 'SrcFrom')  #  change name
    df_SrcTo = df_SrcTo.groupby(['Src IP', 'Src Port']).sum().reset_index()   # ccounts the number of occurrences according to an IP+ origin port
    df_SrcTo = df_SrcTo.pivot(index='Src IP', columns='Src Port', values='Tot Fwd Pkts')
    df_SrcTo = df_SrcTo.rename(columns=lambda x: str(x) + 'SrcTo')  # # data cast
    result = pd.concat([df_SrcTo, df_SrcFrom], axis=1, sort=False)  # dataframes union
    return result

def window_features(i, dataframe, day, time,port_list_src, port_list_dst):
    '''
      extracts the features for a window

    :param day: day
    :type day: str
    :param time: window size
    :type time: int
    :param i: timestamp 
    :type i: datetime
    :param dataframe: data
    :type dataframe: panda
    :param port_list_dst: dst ports
    :type port_list_dst: list[int]
    :param port_list_src: org ports
    :type port_list_src: list[int]
    '''
    warnings.filterwarnings("ignore")  # remove  warnings
    dataframe = dataframe.drop(columns='Timestamp')  # remove Timesatmp
    # select only those flows that have internal ip or as origin, or as destination
    dataframe = dataframe[dataframe['Src IP'].str.match('172.31.') | dataframe['Dst IP'].str.match('172.31.')]

    if (len(list(dataframe.index)) >= 10): #minimum of 10 events
        # Src
        # Extracting separate ports used to make communications
        SrcPortUsed = dataframe[['Src Port', 'Src IP']].groupby('Src IP', axis=0, as_index=True).nunique()
        SrcPortUsed = SrcPortUsed['Src Port']
        # Extract different ports contacted
        SrcPortContacted = dataframe[['Dst Port', 'Src IP']].groupby('Src IP', axis=0, as_index=True).nunique()
        SrcPortContacted = SrcPortContacted['Dst Port']
        # Extrair diferentes IPs de destino contactados
        SrcIPContacted = dataframe[['Dst IP', 'Src IP']].groupby('Src IP', axis=0, as_index=True).nunique()
        SrcIPContacted = SrcIPContacted['Dst IP']
        # Extract different destination IPs contacted
        SrcTotLenSent = dataframe[['TotLen Fwd Pkts', 'Src IP']].groupby('Src IP', axis=0, as_index=True).sum()
        # Extract total number of received package sizes
        SrcTotLenRcv = dataframe[['TotLen Bwd Pkts', 'Src IP']].groupby('Src IP', axis=0, as_index=True).sum()
        # Extrair numero total de sessoes estabelecidas
        SrcTotConn = dataframe[['Dst IP', 'Src IP']].groupby('Src IP', axis=0, as_index=True).count()

        SrcTotalNumPkts = dataframe[['Tot Bwd Pkts', 'Tot Fwd Pkts', 'Src IP']].groupby('Src IP', axis=0,
                                                                                        as_index=True).sum()
        SrcTotalNumPkts['Tot Pckts'] = SrcTotalNumPkts['Tot Bwd Pkts'] + SrcTotalNumPkts['Tot Fwd Pkts']
        SrcTotalNumPkts = SrcTotalNumPkts['Tot Pckts']  # feature Total number of packets exchanged

        SrcTotalNumBytes = dataframe[['TotLen Bwd Pkts', 'TotLen Fwd Pkts', 'Src IP']].groupby('Src IP', axis=0,
                                                                                               as_index=True).sum()
        SrcTotalNumBytes['TotLen Pckts'] = SrcTotalNumBytes['TotLen Fwd Pkts'] + SrcTotalNumBytes['TotLen Bwd Pkts']
        SrcTotalNumBytes = SrcTotalNumBytes['TotLen Pckts']  # feature Overall sum of bytes

        SrcPktRate = dataframe[['Flow Duration', 'Src IP']].groupby('Src IP', axis=0, as_index=True).sum()
        SrcPktRate = SrcPktRate.replace(0, 0.1)  # Avoids that when FlowDuration=0 becomes SrcPktRate=Infinity
        SrcPktRate['SrcPcktRate'] = SrcTotalNumPkts / SrcPktRate['Flow Duration'] # feature Ratio of the number of packets sent and its duration
        SrcPktRate = SrcPktRate['SrcPcktRate']

        SrcAvgPktSize = SrcTotalNumBytes / SrcTotalNumPkts  # feature Average packet size



        # Dst
        #  Extracting separate ports used to make communications
        DstPortUsed = dataframe[['Dst Port', 'Dst IP']].groupby('Dst IP', axis=0, as_index=True).nunique()
        DstPortUsed = DstPortUsed['Dst Port']
        # Extract different ports contacted
        DstPortContacted = dataframe[['Src Port', 'Dst IP']].groupby('Dst IP', axis=0, as_index=True).nunique()
        DstPortContacted = DstPortContacted['Src Port']
        #   Extrair diferentes IPs de destino contactados
        DstIPContacted = dataframe[['Src IP', 'Dst IP']].groupby('Dst IP', axis=0, as_index=True).nunique()
        DstIPContacted = DstIPContacted['Src IP']
        # Extract different destination IPs contacted
        DstTotLenSent = dataframe[['TotLen Bwd Pkts', 'Dst IP']].groupby('Dst IP', axis=0, as_index=True).sum()
        #  Extract total number of received package sizes
        DstTotLenRcv = dataframe[['TotLen Fwd Pkts', 'Dst IP']].groupby('Dst IP', axis=0, as_index=True).sum()
        # Extrair numero total de sessoes estabelecidas
        DstTotConn = dataframe[['Src IP', 'Dst IP']].groupby('Dst IP', axis=0, as_index=True).count()

        DstTotalNumPkts = dataframe[['Tot Bwd Pkts', 'Tot Fwd Pkts', 'Dst IP']].groupby('Dst IP', axis=0,
                                                                                        as_index=True).sum()
        DstTotalNumPkts['Tot Pckts'] = DstTotalNumPkts['Tot Bwd Pkts'] + DstTotalNumPkts['Tot Fwd Pkts']
        DstTotalNumPkts = DstTotalNumPkts['Tot Pckts']  # feature Total number of packets exchanged

        DstTotalNumBytes = dataframe[['TotLen Bwd Pkts', 'TotLen Fwd Pkts', 'Dst IP']].groupby('Dst IP', axis=0,
                                                                                               as_index=True).sum()
        DstTotalNumBytes['TotLen Pckts'] = DstTotalNumBytes['TotLen Fwd Pkts'] + DstTotalNumBytes['TotLen Bwd Pkts']
        DstTotalNumBytes = DstTotalNumBytes['TotLen Pckts']  # feature Overall sum of bytes

        DstPktRate = dataframe[['Flow Duration', 'Dst IP']].groupby('Dst IP', axis=0, as_index=True).sum()
        DstPktRate = DstPktRate.replace(0, 0.1)  # Avoids that when FlowDuration=0 becomes SrcPktRate=Infinity
        DstPktRate['DstPcktRate'] = DstTotalNumPkts / DstPktRate['Flow Duration']
        DstPktRate = DstPktRate['DstPcktRate']  # feature Ratio of the number of packets sent and its duration

        DstAvgPktSize = DstTotalNumBytes / DstTotalNumPkts  # feature Average packet size





        # Extract the packages sent and received at each port from the point of view of origin and destination
        SrcPkt = get_src_pkts(dataframe[['Src IP', 'Dst Port', 'Src Port', 'Tot Bwd Pkts', 'Tot Fwd Pkts']],
                              port_list_src)

        DstPkt = get_dst_pkts(dataframe[['Dst IP', 'Dst Port', 'Src Port', 'Tot Bwd Pkts', 'Tot Fwd Pkts']],
                              port_list_dst)




        #  Concatenation of all features
        Tot = pd.concat(
            [SrcIPContacted, SrcPortUsed, SrcPortContacted, SrcTotLenRcv, SrcTotLenSent, SrcTotConn,SrcPktRate, SrcAvgPktSize, SrcPkt,
             DstIPContacted, DstPortUsed, DstPortContacted, DstTotLenRcv, DstTotLenSent, DstTotConn,DstPktRate,DstAvgPktSize, DstPkt],
            axis=1, sort=False)
        Tot.fillna(value=0, inplace=True)  # change values with Nan
        Tot.columns = ['SrcIPContacted', 'SrcPortUsed', 'SrcPortContacted', 'SrcTotLenRcv', 'SrcTotLenSent','SrcConn','SrcPktRate','SrcAvgPktSize'] \
                      + list(SrcPkt.columns) + \
                      ['DstIPContacted', 'DstPortUsed', 'DstPortContacted','DstTotLenRcv', 'DstTotLenSent','DstTotConn','DstPktRate','DstAvgPktSize'] + list(DstPkt.columns)

        dir_path = os.path.dirname(os.path.realpath(str(argv[1])))
        i = datetime.strptime(str(i), "%Y-%m-%d %H:%M:%S")
        i = i.timestamp() - (4 * 60 * 60)  #  setting the time (lag)
        i = datetime.fromtimestamp(i)
        #  Save the extracted features
        Path(dir_path + '/day' + str(day) + '/' + str(time) + 'min/').mkdir(parents=True,
                                                                            exist_ok=True)
        Tot.to_csv(dir_path+'/day' + str(day) + '/' + str(time) + 'min/' + str(time) + '_features_' + str(
            day) + '_' + str(i).split(' ')[-1] + '.csv')  #save files with time in the name
    else:  # ignore time windows with less than 10 elements
        pass

def main():
    try:
        file=str(argv[1])
    except IndexError:
        help_msg()
    try:
        TimeWindons= [float(i) for i in str(argv[2]).split(",")]
    except ValueError:
        print ('invalid time window')
        help_msg()
    except IndexError:
        TimeWindons=[10]

    try:
        method=int(argv[3])
    except IndexError:
        method=0
    bar= IncrementalBar('loading file',max=13,suffix='%(percent)d%%')
    bar.next()

    try:
        data=pd.read_csv(file, sep=',')
    except FileNotFoundError:
        print ('''
        File Not Found''')
        files_name()
    bar.next()
    data=data.fillna(0) #fill in fields with NaN
    bar.next()
    data = data[data['Src Port'].map(lambda x: str(x) != "Src Port")]# remove the duplicate header lines
    bar.next()
    data = data[data['Timestamp'].map(lambda x: str(x) != '0')]    #remove timestamp-free lines
    bar.next()
    #convert data types
    data['Src Port'] = data['Src Port'].astype('int')
    bar.next()
    data['Dst Port'] = data['Dst Port'].astype('int')
    bar.next()
    data['Tot Fwd Pkts'] = data['Tot Fwd Pkts'].astype('int')
    bar.next()
    data['Tot Bwd Pkts'] = data['Tot Bwd Pkts'].astype('int')
    bar.next()
    data['TotLen Fwd Pkts'] = data['TotLen Fwd Pkts'].astype('float')
    bar.next()
    data['TotLen Bwd Pkts'] = data['TotLen Bwd Pkts'].astype('float')
    bar.next()
    data['Timestamp']= pd.to_datetime(data['Timestamp'].tolist(), format="%d/%m/%Y %I:%M:%S %p")
    bar.next()

    try:
        day,start_date, end_date = dates[file.split("/")[-1]][0:3]
    except KeyError:
        help_msg()
    bar.next()

      #remove values outside the desired dates
    data=data[data['Timestamp']>=start_date ]
    data=data[ data['Timestamp']<=end_date]
    bar.next()
    port_list_src, port_list_dst = get_ports(method, data)
    bar.finish()


    for time in tqdm(iterable=TimeWindons, desc='time window-day'+str(day)):
        grouped = data.groupby(by=pd.Grouper(key='Timestamp', freq=str(int(time * 60)) + 's')) #make groups according to the time window
        Parallel(n_jobs=-1)(delayed(window_features)(i,grouped.get_group(i),day,time,port_list_src, port_list_dst)
                            for i in tqdm(desc='selected time window ' + str(time) + ' min', iterable=grouped.indices.keys()))#performs in parallel the analysis of the various time windows


if __name__ == '__main__':
    main()
