import threading
from enum import Enum
import gi
import numpy as np

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import GdkPixbuf
from gi.repository import GLib
from gi.repository import GObject

from animatedimage import AnimatedImage


class PickerResult(Enum):
    LEFT=1
    RIGHT=2
    SAME=3
    DISCARD=4

class PickerWindow(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="Which Clip is Better?")

        #Constants
        self.timeout = 100
        #End Constants
        
        self.judgingClip = False
        self.hasResult = False

        topBox = Gtk.Box(spacing=6)
        bottomBox = Gtk.Box(spacing=6)

        leftImage = AnimatedImage(336)
        rightImage = AnimatedImage(336)
        
        topBox.pack_start(leftImage, True, True, 0)
        topBox.pack_start(rightImage, True, True, 0)

        self.imageViews = [leftImage, rightImage]

        leftButton = Gtk.Button(label="Left")
        leftButton.connect("clicked", self.onLeftClicked)
        bottomBox.pack_start(leftButton, True, True, 0)

        sameButton = Gtk.Button(label="Same")
        sameButton.connect("clicked", self.onSameClicked)
        bottomBox.pack_start(sameButton, True, True, 0)

        discardButton = Gtk.Button(label="Skip")
        discardButton.connect("clicked", self.onDiscardClicked)
        bottomBox.pack_start(discardButton, True, True, 0)

        rightButton = Gtk.Button(label="Right")
        rightButton.connect("clicked", self.onRightClicked)
        bottomBox.pack_start(rightButton, True, True, 0)

        mainBox = Gtk.VBox(spacing=6)

        mainBox.pack_start(topBox, True, True, 0)
        mainBox.pack_start(bottomBox, True, True, 0)

        self.add(mainBox)

        GObject.timeout_add(self.timeout, self.animStep)
        self.show_all()

    def gtkMain(self):
        self.gtkThread = threading.Thread(target=Gtk.main)
        self.gtkThread.start()

    def getResult(self):
        self.hasResult = False
        return (self.comparison, self.result)
        
    def setClips(self, comparison):
        self.comparison = comparison
        for i in range(len(self.imageViews)):
            self.imageViews[i].setClip(comparison["clips"][i])

        self.judgingClip = True
        self.hasResult = False
        

    def animStep(self):
        if self.judgingClip:
            for imageView in self.imageViews:
                imageView.update()
        GObject.timeout_add(self.timeout, self.animStep)

    def pickResult(self, result):
        if self.judgingClip:
            self.result = result
            self.judgingClip = False
            self.hasResult = True
            print(f"Result: {self.result}")
        else:
            print("No clip to pick result for")

    def onLeftClicked(self, widget):
        self.pickResult(PickerResult.LEFT)
        print("Left")

    def onSameClicked(self, widget):
        self.pickResult(PickerResult.SAME)
        print("Same")

    def onDiscardClicked(self, widget):
        self.pickResult(PickerResult.DISCARD)
        print("Discard")

    def onRightClicked(self, widget):
        self.pickResult(PickerResult.RIGHT)
        print("Right")


#win = PickerWindow()
#win.connect("destroy", Gtk.main_quit)
#win.show_all()
#Gtk.main()
