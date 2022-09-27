import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import GdkPixbuf
from gi.repository import GLib
from gi.repository import GObject
import numpy as np

from training_backend import TrainingBackend

class TrainingWindow(Gtk.Window):
    def __init__(self, env_id, feedback_file, save_dir, conv=False, preprocess=False):
        Gtk.Window.__init__(self, title=f"RLHF Training")

        self.backend = TrainingBackend(env_id, feedback_file, save_dir, conv, preprocess)

        GObject.timeout_add(0, self.initInGtkThread)

    def initInGtkThread(self):
        self.grid = Gtk.Grid()
 
        self.reward_button = Gtk.Button(label="Train Reward")
        self.reward_button.connect("clicked", self.toggleTrainReward)
        self.grid.attach(self.reward_button, 0, 0, 1, 1)

        self.agent_button = Gtk.Button(label="Train Agent")
        self.agent_button.connect("clicked", self.toggleTrainAgent)
        self.grid.attach(self.agent_button, 0, 1, 1, 1)

        self.random_button = Gtk.Button(label="Harvest Random")
        self.random_button.connect("clicked", self.toggleHarvestRandom)
        self.grid.attach(self.random_button, 1, 0, 1, 1)

        self.ensemble_button = Gtk.Button(label="Harvest Ensemble")
        self.ensemble_button.connect("clicked", self.toggleHarvestEnsemble)
        self.grid.attach(self.ensemble_button, 1, 1, 1, 1)

        self.save_button = Gtk.Button(label="Save")
        self.save_button.connect("clicked", self.save)
        self.grid.attach(self.save_button, 0, 2, 1, 1)

        self.add(self.grid)

        self.updateButtons()

        self.show_all()

    def updateButtons(self):
        check = "\u2705"
        x = "\u274C"

        def update(button, text, flag):
            status = check if flag else x
            button.set_label(f"{text} {status}")

        update(self.reward_button, "Train Reward", self.backend.trainReward)
        update(self.agent_button, "Train Agent", self.backend.trainAgent)
        update(self.random_button, "Harvest Random", self.backend.harvestRandom)
        update(self.ensemble_button, "Harvest Ensemble", self.backend.harvestEnsemble)

    def toggleTrainReward(self, event):
        self.backend.trainReward = not self.backend.trainReward
        self.updateButtons()

    def toggleTrainAgent(self, event):
        self.backend.trainAgent = not self.backend.trainAgent
        self.updateButtons()

    def toggleHarvestRandom(self, event):
        self.backend.harvestRandom = not self.backend.harvestRandom
        self.updateButtons()

    def toggleHarvestEnsemble(self, event):
        self.backend.harvestEnsemble = not self.backend.harvestEnsemble        
        self.updateButtons()

    def save(self, event):
        self.backend.shouldSave = True
