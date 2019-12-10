#Made by Daniel Jambor, 12.03.2019
#
#Sources: pythonprogramming.net by harrison@pythonprogramming.net
#Pytorch.org
#stackoverflow.com

import cv2
import numpy as np
import torch
import sys
import webbrowser

from Net import *

#set file name as argument from batch script
filename = sys.argv[1]

#set path to the trained model
MPATH = './bin/model.pth'

#creating net variable
net = Net()

#loading trained model
net.load_state_dict(torch.load(MPATH))
#setting eval mode
net.eval()

#setting path to image
IPATH = './bin/userdata/' + filename

#reading image in grayscale mode
img = cv2.imread(IPATH, cv2.IMREAD_GRAYSCALE)
#resizing image
img = cv2.resize(img, (64, 64))
#adding image to np array
img = np.array(img)

#transfer image to tensor and reshaping it
img = torch.Tensor(img).view(-1, 64, 64)
#divide by the number of pixels
img = img/255.0

#get the unconverted output
netOut = net(img.view(-1, 1, 64, 64))

#get predicted answer
predicted = int(torch.argmax(netOut))

#checking the answer and saving it; setting the name of description file 
if predicted == 0:
    monkeyName = "Mantled Howler (n0)"
    monkeyFileName = "n0.txt"
elif predicted == 1:
    monkeyName = "Patas Monkey (n1)"
    monkeyFileName = "n1.txt"
elif predicted == 2:
    monkeyName = "Bald Uakari (n2)"
    monkeyFileName = "n2.txt"
elif predicted == 3:
    monkeyName = "Japanese Macaque (n3)"
    monkeyFileName = "n3.txt"
elif predicted == 4:
    monkeyName = "Pygmy Marmoset (n4)"
    monkeyFileName = "n4.txt"
elif predicted == 5:
    monkeyName = "White Headed Capuchin (n5)"
    monkeyFileName = "n5.txt"
elif predicted == 6:
    monkeyName = "Silvery Marmoset (n6)"
    monkeyFileName = "n6.txt"
elif predicted == 7:
    monkeyName = "Common Squirrel Monkey (n7)"
    monkeyFileName = "n7.txt"
elif predicted == 8:
    monkeyName = "Black Headed Night Monkey (n8)"
    monkeyFileName = "n8.txt"
elif predicted == 9:
    monkeyName = "Nilgiri Langur (n9)"
    monkeyFileName = "n9.txt"

#open the html file
f = open('result.html','w')

#open the description file
monf = open('./bin/data/description/' + monkeyFileName, 'r')
#save description into variable
description = monf.read()
#close description file
monf.close()

#save the content of html file
content = f"""<html>
    <head>
        <body>
            <center>
                    <img src="bin/userdata/{filename}" alt="Monkey" align="middle" border="5" style="width:600px;height:600px;">
                    <p> <b> <font size="6"> This is {monkeyName}! </font> </b> </p>
                    <p> <i> <font size="4"> {description} </font> </i> </p>
            </center>
    </head>
</html>"""

#save html file
f.write(content)
#close html file
f.close()

#open saved html as result
webbrowser.open_new_tab('result.html')