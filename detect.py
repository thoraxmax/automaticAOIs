##Copyright Max Thorsson 2022
import os
import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
import scipy.spatial.distance as dist

path='bilder/'##images loc
path2='bilder2/'#save images loc
path3='conts/'#save contours of AOIs loc
ww,hh=1024,768##target screen resolution pixel size

detector= dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

data=[]

files=os.listdir(path)

#AOIs based on landmarks
cnts={}
cnts['leftface']=np.array([0,1,2,3,4,5,6,7,8,30,29,27])#27,28,29,30,17,18,21])
cnts['rightface']=np.array([27,28,29,30,8,9,10,11,12,13,14,15,16])
cnts['eyes']=np.array([0,1,15,16,26,25,24,23,22,21,20,19,18,17])
##text stuff
font = cv2.FONT_HERSHEY_DUPLEX
th3=5
org = (50, 50)
fontScale = 1.5#
thickness = 2
#files=[f for f in files if '01F' in f]
for file in files[:]:
    print('starting with image: '+file)
    frame=cv2.imread(path+file)
    w2=int((hh/frame.shape[1])*frame.shape[1])
    frame=cv2.resize(frame,(w2,hh))
    im=np.zeros((hh,ww,3))+frame[:100,:100].mean() #color of outer border for fill
    c=int((ww-frame.shape[1])/2)
    im[:,c:c+frame.shape[1]]=frame
    frame=im.astype('uint8')
    face_Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boundary = detector(face_Gray, 1)
    for (index, rectangle) in enumerate([boundary[0].rect]):
        shape = predictor(face_Gray, rectangle)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rectangle)
        xmin,xmax=min(shape[48:].T[0]),max(shape[48:].T[0])
        ymin,ymax=min(shape[48:].T[1]),max(shape[48:].T[1])
        tol=4
        rect=[int(xmin)-tol,int(ymin)-tol,int(xmax)+tol,int(ymax)+tol]
        pr=tuple(np.mean(shape[36:42],0).round().astype(int))
        pl=tuple(np.mean(shape[42:48],0).round().astype(int))
        cl=tuple(np.random.randint(0,255,3))
        back=frame.copy()
        alpha = 0.25
        cols=[(119,158,27),
        (179,112,117),
        (2,95,217)]
        for c,cl in zip(cnts.keys(),cols):
            cnt=shape[cnts[c]].copy()
            if c=='lefteye':
                ce=np.mean(cnt,0)
                cnt=((cnt-ce)*2+ce).astype(int)
            if c=='righteye':
                ce=np.mean(cnt,0)
                cnt=((cnt-ce)*2+ce).astype(int)
            if c=='leftface':##create ellipse for left AOI
                v=((shape[30]-shape[27])*2).astype(int)
                cnt[-1]-=v
                r=dist.euclidean(cnt[0],cnt[-1])
                rx=cnt[0][0]-cnt[-1][0]
                ry=cnt[0][1]-cnt[-1][1]
                sts=cnt[-1][:]
                f=np.pi
                for i in np.flip(np.linspace(0+f,np.pi/2+f,5)):
                    cnt=np.append(cnt,(np.array([[-np.cos(i)*rx,np.sin(i)*ry]])+sts+np.array([[0,ry]])).astype('uint32'),0)
            if c=='rightface':##create ellipse for right AOI
               v=((shape[30]-shape[27])*2).astype(int)
               cnt[0]-=v
               r=dist.euclidean(cnt[0],cnt[-1])
               rx=cnt[0][0]-cnt[-1][0]
               ry=cnt[0][1]-cnt[-1][1]
               sts=cnt[0][:]
               f=np.pi
               for i in (np.linspace(0+f,np.pi/2+f,5)):
                   cnt=np.append(cnt,(np.array([[np.cos(i)*rx,-np.sin(i)*ry]])+sts+np.array([[0,-ry]])).astype('uint32'),0)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect) 
            box = np.int0(box)
            np.save(path3+file.split('.')[0]+'-'+c,cnt)
            back=frame.copy()
            cv2.drawContours(back, [cnt], 0, (int(cl[0]),int(cl[1]),int(cl[2])), -1)
            alpha = 0.45
            frame = cv2.addWeighted(frame, 1-alpha, back, alpha, 0)
            cv2.drawContours(frame,[cnt],0,(int(cl[0]),int(cl[1]),int(cl[2])),th3)
        nr=0
        for x,y in shape[:]:
             if nr in [0,1,15,16,27,30]:
                 textsize = cv2.getTextSize(str(nr+1), font, fontScale, thickness)[0]
                 textX = int((x - textsize[0]/2) )
                 textY = int((y + textsize[1]/2) )
                 ad2=int(textsize[1])
                 ad=int(textsize[0]/2)
                 cl=(255,255,255)
                 if False:
                     if nr<2:
                         frame = cv2.putText(frame, str(nr+1), (textX-ad,textY), font, 
                                            fontScale, cl, thickness, cv2.LINE_AA)
                     if nr==15 or nr==16:
                        frame = cv2.putText(frame, str(nr+1), (textX+ad,textY), font, 
                                           fontScale, cl, thickness, cv2.LINE_AA)
                     if nr==27 or nr==30:
                        frame = cv2.putText(frame, str(nr+1), (textX+3,textY-ad2), font, 
                                           fontScale, cl, thickness, cv2.LINE_AA)
                     cv2.circle(frame, (x,y), 7, (255,255,255),-1,cv2.LINE_AA)
             cv2.circle(frame, (x,y), 5, (0,0,255),-1,cv2.LINE_AA)
             nr+=1
        cv2.imshow("detected face", frame)
        cv2.imwrite(path2+file.split('.')[0]+'_edit.png', frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('finished with image: '+file)    
cv2.destroyAllWindows()
