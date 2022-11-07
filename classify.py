import pandas as pd
import numpy as np
import os
import cv2
from numba import jit
from numba import cuda
from numba.pycc import CC
import traceback
from tqdm import tqdm
aoiloc='conts/'
dataloc='all_data.csv'
AOI={'stim':[],'rect':[],'area':[]}
ao=os.listdir(aoiloc)
for a in ao:
   re=np.load(aoiloc+a)
   AOI['stim'].append(a.split('-')[0])
   AOI['area'].append(a.split('-')[1].split('.')[0])
   AOI['rect'].append(re)
   
df=pd.read_csv(dataloc,sep='\t')

x=df['GazeX'].values
y=df['GazeY'].values

cc = CC('nbspatial')
@cc.export('ray_tracing',  'b1(f8, f8, f8[:,:])')
@cuda.jit(nopython=True)
def ray_tracing(x,y,poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y
    return inside


##########
stim=df['StimulusName'].values
ineye=[]
inareas={}
for i in np.unique(AOI['area']):#create lists for AOIs
    inareas[i]=[]
pbar = tqdm(total=len(x))#create progress bar
for i,x1,y1 in zip(stim,x,y):#iterate over all gaze coordinates and check if the are within areas 
      try:
          se=np.where(np.array(AOI['stim'])==i)
          recs=[AOI['rect'][o] for o in se[0]]
          if len(recs)==0:
              1/0
          areas=np.array(AOI['area'])[np.where(np.array(AOI['stim'])==i)]
      #    print(arr)
          for rec,ar in zip(recs,areas):
              try:
                  cn= ray_tracing(x1,y1,rec)
                  if np.sum(np.isnan(np.array([x1,y1])))!=0 or min(np.array([x1,y1]))==-1:
                      cn=-1
                      raise ValueError
              except Exception:
                  cn=-1
              if cn==True:
                  inareas[ar].append(1)#if witin area
              if cn==False:
                  inareas[ar].append(0)#if outside area
              if cn==-1:
                  inareas[ar].append(-1)#if error, not on screen or nans 
      except Exception:
           for i in np.unique(AOI['area']):
               inareas[i].append(-1)     
      pbar.update(1)#progress bar

df=df[['Name', 'Age', 'Gender', 'StimulusName',
       'Condition', 'SlideType', 'EventSource', 'Timestamp', 'MediaTime',
       'TimeSignal']]
##calculate percent per AOI
for i in np.unique(AOI['area']):
    df[i]=inareas[i]
    
df2={'id':[],'stim':[],'percent_on_screen':[]}
for i in np.unique(AOI['area']):
    df2['percent_'+i]=[]
    
print('starts percent estimation')
pbar = tqdm(total=len(AOI['stim']))#create progress bar
for s in AOI['stim']:
    for ig in np.unique(AOI['area']):# iterate over AOIs
        g=df.loc[df['StimulusName']==s][ig].values
        i=df.loc[df['StimulusName']==s]['Name'].values##id
        for n in np.unique(i):## per individual
            g2=g[np.where(i==n)] #get all datapoints where the stimilus is equal to n
            pr=(np.sum(g2==1)/len(g2))*100 ##calculate percent on the area
            df2['percent_'+ig].append(pr) 
            df2['id'].append(n)
            df2['stim'].append(s)
            pr2=abs(1-(np.sum(g2==-1)/len(g2)))*100##percent on screen
            df2['percent_on_screen'].append(pr2)
    pbar.update(1)#update progress bar
    
df2=pd.DataFrame(df2)        
df2.to_csv('percent_stim_lookarea.csv')##save file for all percentages per area
