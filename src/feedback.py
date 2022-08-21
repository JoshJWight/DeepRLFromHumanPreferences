import pickle
import random
import picker
from comparisonviewer import ComparisonViewer
import collections

class FeedbackManager:
    def __init__(self, savefile):
        self.pickerWindow = picker.PickerWindow()

        #TODO might end up being situations where we're not using this to solicit feedback right now
        #And don't need to show the picker
        #self.pickerWindow.show_all()
        self.pickerWindow.gtkMain() 

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
            
    
    def queueClips(self, clip1, clip2, obs1, obs2):
        obj = {
            "clips": [clip1, clip2],
            "observations": [obs1, obs2],
            "values": [0, 0]
        }
        self.clipStorage.append(obj)

    def updatePicker(self):
        if not self.pickerWindow.judgingClip:
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

    def save(self):
        pickle.dump(self.comparisons, open(self.savefile, "wb"))

    def viewComparisons(self):
        self.comparisonViewer = ComparisonViewer(self.comparisons)

    def randomBatch(self, n):
        return random.sample(self.comparisons, n)
