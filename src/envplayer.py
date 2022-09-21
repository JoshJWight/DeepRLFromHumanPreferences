import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import GdkPixbuf
from gi.repository import GLib
from gi.repository import GObject
import numpy as np

class EnvPlayer(Gtk.Window):
    def __init__(self, env, agent, rewardmodel):
        Gtk.Window.__init__(self, title=f"Environment Playback Control")
       
        self.env = env
        self.agent = agent
        self.rewardmodel = rewardmodel

        self.env_state = np.array(self.env.reset())

        self.timeout_ms = 100
        self.paused = False

        GObject.timeout_add(0, self.initInGtkThread)

    def initInGtkThread(self):
        self.set_default_size(1000, 50)

        self.actionlabel = Gtk.Label("Action: _")
        self.rewardlabel = Gtk.Label("Reward: _")
        self.rewardbar = Gtk.ProgressBar()
        self.rewardbar.set_text("Reward: _")


        box = Gtk.Box(spacing=6)
        box.pack_start(self.actionlabel, True, True, 0)
        box.pack_start(self.rewardlabel, True, True, 0)
        box.pack_start(self.rewardbar, True, True, 0)
        
        self.pause_button = Gtk.Button(label="Pause")
        self.pause_button.connect("clicked", self.togglePause)
        box.pack_start(self.pause_button, True, True, 0)

        step_button = Gtk.Button(label="Step")
        step_button.connect("clicked", self.update)
        box.pack_start(step_button, True, True, 0)

        self.playspeed_adjustment = Gtk.Adjustment(50, 0, 100, 1, 10, 0)
        playspeed_scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=self.playspeed_adjustment)
        playspeed_scale.set_vexpand(True)
        playspeed_scale.set_hexpand(True)
        box.pack_start(playspeed_scale, True, True, 0)

        self.add(box)
        self.show_all()
        if not self.paused:
            GObject.timeout_add(self.timeout_ms, self.update)
        
    def update(self, event=None):
        playspeed = self.playspeed_adjustment.get_value()
        self.timeout_ms = ((100 - playspeed) ** 3) / 1000
        #print(f"timeout : {self.timeout_ms}")
    
        #Update internals
        self.env.render()
        
        reward = self.rewardmodel.evaluate(self.env_state)
        action, _ = self.agent.predict(self.env_state)
        next_state, _, done, _ = self.env.step(action)
        
        self.env_state = np.array(next_state)
        if done:
            self.env_state = np.array(self.env.reset())

        #Update UI
        self.actionlabel.set_text(f"Action: {action}")
        self.rewardbar.set_fraction((reward + 1.0) / 2.0)
        self.rewardlabel.set_text(f"Reward: {reward:.3f}")
        
        #TODO save rewards to make a graph of reward over time?
        if not self.paused:
            GObject.timeout_add(self.timeout_ms, self.update)

    def togglePause(self, event):
        self.paused = not self.paused
        self.pause_button.set_label("Resume" if self.paused else "Pause")
        self.update()
