import threading
from enum import Enum
import gi
import numpy as np

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import GdkPixbuf
from gi.repository import GLib
from gi.repository import GObject

def pixbuf_from_array(z, size):
    z=z.astype('uint8')
    h,w,c = z.shape
    if not(c==3 or c==4):
        print(f"Bad shape for pixbuf: {z.shape}")
        assert(False)
    Z = GLib.Bytes.new(z.tobytes())
    useAlpha = (c==4)
    pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(Z, GdkPixbuf.Colorspace.RGB, useAlpha, 8, w, h, w*c)
    pixbuf = pixbuf.scale_simple(size, size, GdkPixbuf.InterpType.NEAREST)
    return pixbuf

class AnimatedImage(Gtk.Image):
    def __init__(self, size): 
        Gtk.Window.__init__(self)
        self.size = size
        self.currentFrame = 0
        self.postblackout = 2
        self.clip=[]
         
    def setClip(self, clip):
        self.currentFrame = 0

        self.clip = []
        for frame in clip:
            self.clip.append(pixbuf_from_array(frame, self.size))
        for i in range(self.postblackout):
            #Size of the zero array isn't important since it will be scaled up anyway
            self.clip.append(pixbuf_from_array(np.zeros((100, 100, 3), dtype=np.int8), self.size))

    def setClipBordered(self, clip, color, bordersize=5):
        borderedclip = []
        for frame in clip:
            frameshape = frame.shape
            expandedsize = (frameshape[0] + bordersize*2, frameshape[1] + bordersize*2, 3)
            borderedframe = np.full(expandedsize, color)
            borderedframe[bordersize:bordersize+frameshape[0], bordersize:bordersize+frameshape[1], :] = frame
            borderedclip.append(borderedframe)
        self.setClip(borderedclip)

    def update(self):
        if len(self.clip) == 0:
            return 
        self.set_from_pixbuf(self.clip[self.currentFrame])
        self.currentFrame = (self.currentFrame + 1) % len(self.clip)
