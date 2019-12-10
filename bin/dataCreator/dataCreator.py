#Made by Daniel Jambor, 12.03.2019
#
#Sources: pythonprogramming.net by harrison@pythonprogramming.net
#Pytorch.org
#stackoverflow.com

import os
import cv2
import numpy as np

class Monkeys():

    #setting the file path of each monkey
    n0 = 'C:/Users/danie/Documents/Studies/Term III/Scripting Languages/Project/bin/data/monkeys/n0'
    n1 = 'C:/Users/danie/Documents/Studies/Term III/Scripting Languages/Project/bin/data/monkeys/n1'
    n2 = 'C:/Users/danie/Documents/Studies/Term III/Scripting Languages/Project/bin/data/monkeys/n2'
    n3 = 'C:/Users/danie/Documents/Studies/Term III/Scripting Languages/Project/bin/data/monkeys/n3'
    n4 = 'C:/Users/danie/Documents/Studies/Term III/Scripting Languages/Project/bin/data/monkeys/n4'
    n5 = 'C:/Users/danie/Documents/Studies/Term III/Scripting Languages/Project/bin/data/monkeys/n5'
    n6 = 'C:/Users/danie/Documents/Studies/Term III/Scripting Languages/Project/bin/data/monkeys/n6'
    n7 = 'C:/Users/danie/Documents/Studies/Term III/Scripting Languages/Project/bin/data/monkeys/n7'
    n8 = 'C:/Users/danie/Documents/Studies/Term III/Scripting Languages/Project/bin/data/monkeys/n8'
    n9 = 'C:/Users/danie/Documents/Studies/Term III/Scripting Languages/Project/bin/data/monkeys/n9'

    #setting labels of monkeys
    LABELS = {n0: 0, n1: 1, n2: 2, n3: 3, n4: 4, n5: 5, n6: 6, n7: 7, n8: 8, n9: 9}

    #our data list
    data = []
    
    #print the message
    print("Creating the dataset...")

    def makingData(self):
        #for every label in label list
        for label in self.LABELS:
            #print the message
            print(f"Making data from {self.LABELS[label] + 1} label...")
            #for every file (image) in folder
            for f in os.listdir(label):
                try:
                    #setting path of image
                    path = os.path.join(label, f)
                    #lodaing the image in grayscale mode
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    #resizing the image
                    img = cv2.resize(img, (64, 64))
                    #adding to data list our image in array version and adding the array with ones on the diagonal and zeros elsewhere (size of the numbers of monkeys)
                    self.data.append([np.array(img), np.eye(10)[self.LABELS[label]]])
                except Exception as e:
                    #if error, print
                    print(e)

        #shuffle the data list randomly
        np.random.shuffle(self.data)
        #save our database as trainingData.npy
        np.save("C:/Users/danie/Documents/Studies/Term III/Scripting Languages/Project/bin/trainingData.npy", self.data)
        #print the message
        print("\nCreating dataset completed!\n")

monkeys = Monkeys()
monkeys.makingData()