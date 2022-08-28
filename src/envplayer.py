import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import GdkPixbuf
from gi.repository import GLib
from gi.repository import GObject

class EnvPlayer(Gtk.Window):
    def __init__(self, env, agent, rewardmodel):
        Gtk.Window.__init__(self, title=f"Environment Playback Control")
       
        self.env = env
        self.agent = agent
        self.rewardmodel = rewardmodel

        self.env_state = self.env.reset()

        self.timeout_ms = 100
        self.paused = False

        GObject.timeout_add(0, self.initInGtkThread)

    def initInGtkThread(self):
        self.actionlabel = Gtk.Label("Action: _")
        self.rewardbar = Gtk.ProgressBar()
        self.rewardbar.set_text("Reward: _")


        box = Gtk.Box(spacing=6)
        box.pack_start(self.actionlabel, True, True, 0)
        box.pack_start(self.rewardbar, True, True, 0)
        
        self.pause_button = Gtk.Button(label="Pause")
        self.pause_button.connect("clicked", self.togglePause)
        box.pack_start(self.pause_button, True, True, 0)

        step_button = Gtk.Button(label="Step")
        step_button.connect("clicked", self.update)
        box.pack_start(step_button, True, True, 0)

        self.add(box)
        self.show_all()
        if not self.paused:
            GObject.timeout_add(self.timeout_ms, self.update)
        
    def update(self, event=None):
        #Update internals
        self.env.render()
        
        reward = self.rewardmodel.evaluate(self.env_state)
        action, _ = self.agent.predict(self.env_state)
        next_state, _, done, _ = self.env.step(action)
        
        self.env_state = next_state
        if done:
            self.env_state = self.env.reset()

        #Update UI
        self.actionlabel.set_text(f"Action: {action}")
        self.rewardbar.set_fraction((reward + 1.0) / 2.0)
        self.rewardbar.set_text(f"Reward: {reward}")
        
        #TODO save rewards to make a graph of reward over time?
        if not self.paused:
            GObject.timeout_add(self.timeout_ms, self.update)

    def togglePause(self, event):
        self.paused = not self.paused
        self.pause_button.set_label("Resume" if self.paused else "Pause")
        self.update()
