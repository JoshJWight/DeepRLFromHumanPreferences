import pickle
import random
import time
import picker
from comparisonviewer import ComparisonViewer
import collections
from threading import Thread, Lock

class FeedbackManager:
    def __init__(self, savefile, show_picker=True):
        self.mutex = Lock()

        #I don't think the footprint of this many clips is that significant, but a human would take a while
        #to work through this many.
        self.clipLimit = 100
        self.clipStorage = collections.deque(maxlen=self.clipLimit)

        self.savefile = savefile
        try:
            self.comparisons = pickle.load(open(self.savefile, "rb"))
            print("Feedback loaded from " + savefile)
        except:
            print("Could not load feedback from " + savefile + ". Starting from scratch.")
            self.comparisons = []

        if show_picker:
            print("Show picker")
            self.pickerWindow = picker.PickerWindow()
            self.pickerWindow.gtkMain() 
            self.pickerThread = Thread(target=self.watchPicker)
            self.pickerThread.start()
        else:
            print("No show picker")
            
    
    def queueClips(self, clip1, clip2, obs1, obs2):
        self.mutex.acquire()
        obj = {
            "clips": [clip1, clip2],
            "observations": [obs1, obs2],
            "values": [0, 0]
        }
        self.clipStorage.append(obj)
        self.mutex.release()

    def addComparison(self, comparison):
        self.mutex.acquire()
        self.comparisons.append(comparison)
        self.mutex.release()

    def watchPicker(self):
        while True:
            self.updatePicker()
            time.sleep(1)
        

    def updatePicker(self):
        if not self.pickerWindow.judgingClip:
            self.mutex.acquire()
            if self.pickerWindow.hasResult:
                comparison, result = self.pickerWindow.getResult()
                if result is not picker.PickerResult.DISCARD:
                    if result is picker.PickerResult.LEFT:
                        comparison["values"] = [1, 0]
                    elif result is picker.PickerResult.RIGHT:
                        comparison["values"] = [0, 1]
                    elif result is picker.PickerResult.SAME:
                        comparison["values"] = [0.5, 0.5]
                    self.comparisons.append(comparison)
            if len(self.clipStorage) > 0:
                #Use the most recently provided clip.
                #Older clips are just kept in case the user picks before another clip is provided.
                self.pickerWindow.setClips(self.clipStorage.pop())
            self.mutex.release()

    def save(self):
        pickle.dump(self.comparisons, open(self.savefile, "wb"))

    def viewComparisons(self):
        self.comparisonViewer = ComparisonViewer(self.comparisons)

    def randomBatch(self, n):
        self.mutex.acquire()
        result = random.sample(self.comparisons, n)
        self.mutex.release()
        return result
