
import gi
import numpy as np

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import GdkPixbuf
from gi.repository import GLib
from gi.repository import GObject

from animatedimage import AnimatedImage

class ComparisonViewer(Gtk.Window):
    def __init__(self, comparisons):
        Gtk.Window.__init__(self, title=f"Viewing {len(comparisons)} Comparisons")
        self.comparisons = comparisons


        #Constants
        self.timeout = 100

        self.images = []
        GObject.timeout_add(0, self.initInGtkThread)

    def initInGtkThread(self):
        comparisons = self.comparisons
        mainBox = Gtk.VBox(spacing=6)

        print(f"{len(comparisons)} comparisons total")
        for comparison in comparisons:
            #Only show comparisons that still have clips to show
            if comparison["clips"] == None:
                continue
            print("comparison step")
            box = Gtk.Box(spacing=6)
            for i in range(2):
                image = AnimatedImage(200)
                self.images.append(image) 
                
                value = comparison["values"][i]
                color = []
                if value == 1:
                    color = [50, 200, 50] #green
                elif value == 0:
                    color = [200, 50, 50] #red
                else:
                    color = [150, 150, 50] #yellow
                image.setClipBordered(comparison["clips"][i], color=color)

                box.pack_start(image, True, True, 0)

            mainBox.pack_start(box, True, True, 0)

        self.set_size_request(450, 1000)
        scrolledWindow = Gtk.ScrolledWindow()
        scrolledWindow.add_with_viewport(mainBox)
        self.add(scrolledWindow)
        GObject.timeout_add(self.timeout, self.animStep)
        self.show_all()
        print("End constructor")
        
    
    def animStep(self):
        for imageView in self.images:
            imageView.update()
        GObject.timeout_add(self.timeout, self.animStep)
