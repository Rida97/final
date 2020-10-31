import cv2
import numpy as np
from flask import Flask,render_template

this is a change made by RIDA in feature/version1 branch

app=Flask(__name__)

@app.route('/')
def home_page():
    return "This is Home Page"

@app.route('/task')
def task_page():
    cap = cv2.VideoCapture('aerial.mp4')

    def frames(count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        frameTime = 600
        foog =  cv2.bgsegm.createBackgroundSubtractorMOG()

    count = 5   
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
    sigma = 0.33
    while(1):
        ret, frame_old = cap.read()  
   # if count%(10*fps) == 0 :
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_frame = int(current_frame)
    #count+=1
        frames(count)
        count += 12
    
        frame = frame_old[160:630, 50:1200]   # roi defined for aerial video  
        ht = frame.shape[0] 
        wd = frame.shape[1]
        frame_pixels = ht*wd
        
        if not ret:
            break
        fgmask = foog.apply(frame) 
        ret, fgmask = cv2.threshold(fgmask, 252, 255, cv2.THRESH_BINARY)
        fgmask = cv2.GaussianBlur(fgmask,(5,5), cv2.BORDER_DEFAULT)
        v = np.median(fgmask)
        lower = int(max(0, (1.0-sigma)*v))
        upper = int(min(255, (1.0+sigma)*v))
 #   fgmask = cv2.Canny(fgmask,lower,upper)
        kernel = np.ones((5, 5))
        fgmask = cv2.dilate(fgmask, kernel, iterations =  2)
        fgmask = cv2.erode(fgmask, kernel, iterations =  2)
        contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        list_of_object_area = []
    
        for cnt in contours:
            object_area = cv2.contourArea(cnt)
            if object_area >= 700:
                
                # list_of_object_area = list(object_area)
                #  sum_of_object_area = sum_of_object_area + object_area
            
                x,y,w,h = cv2.boundingRect(cnt)
                   
                cv2.rectangle(frame,(x ,y),(x+w,y+h),(255,0,255),2)
         
                list_of_object_area.append(object_area)
        
                cv2.putText(frame,'OBJECT AREA ' + str(int(object_area)), (x,y+20), 
                            cv2.FONT_HERSHEY_SIMPLEX,0.5, (0,0,255), 2 ,cv2.LINE_AA)
                text = "Frame Number: " + str(current_frame)
                cv2.putText(frame,"{}".format(text), (70, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0 ,255), 2)
            
       #     pixels = 115200 # 100 ft
            
            #n_black_pix = np.sum(fgmask == 0)
            n_white_pix = cv2.countNonZero(fgmask)
            #    n_white_pix = pixels - round(int(n_black_pix))
    
        print(" ")
        print(" -----------Frame # " + str(current_frame)+ "------------")
        print('Sum of all the White Pixels:    ' + str(round((n_white_pix))))
        total_object_area = round((sum(list_of_object_area)))
        print("Sum of all the Detected Pixels: " + str(total_object_area))
    
        ratio = round((total_object_area/frame_pixels)*100)
        print("Ratio is : " + str(ratio)+ "%")
    
        cv2.putText(frame,'  RATIO: '+ str(int(ratio))+'%',(45,45), cv2.FONT_HERSHEY_SIMPLEX,  
                    1, (255, 230, 150), 2, cv2.LINE_AA)
    
        fgmask_3 = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
    
        stacked = np.hstack((fgmask_3,frame))
        cv2.imshow('All ',cv2.resize(stacked,None, fx=0.55,fy=0.55))
        k = cv2.waitKey(frameTime) & 0xff
        if k == ord('q'):
            break
        cap.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    app.run(debug=True)#,host="192.168.43.161")
