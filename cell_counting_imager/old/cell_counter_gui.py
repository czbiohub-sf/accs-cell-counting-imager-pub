import kivy
kivy.require('1.11.1')
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.popup import Popup
from cell_counting_imager import CellCounterCore
import json
# import numpy as np

class CCIGui(BoxLayout):

    '''
    The CCIGui class is the root kivy widget which interacts with CCI Core.

        Properties:
            self._CCI_CORE: The CellCounterCore instance which the GUI interacts with
            self._STATE_MACHINE: Nested Dictionaries, state machine information loaded from the same JSON file which we pass to the CellCounterCore object
            self._STATE_PERMISSIONS: Nested Dictionaries, the permission matrix extracted from the _STATE_MACHINE information

        Methods:

            self.traverseAndToggle(self, root, to_disable): Recursively traverses widget tree starting from root, enabling all buttons by default in disabling those in "to_disable"
            self.getCurrentWellButton(self): Returns the selected WellButton out of all of the WellButtons, which are part of a ToggleButton set.
            self.buttonToIndex(self, button): Returns the key associated with the given WellButton in the position dictionary.
            self.updateGUI(self): This method should be bound to the CellCounterCore's change state method. It checks the CellCounterCore's new state, and updates the GUI visually to match that state.

            Callbacks:
                self.upCB
                self.downCB
                self.initializeCB
                self.stepSizeCB
                self.previewCB
                self.stopPreviewCB
                self.setCB
                self.triggeredScanCB
                self.directScanCB
                self.stopScanCB
                self.statusCB


    '''
    
    _TIC_COM_MODE = 'serial'
    _TIC_COM = 'COM3'
    _TIC_BAUDRATE = 9600
    _TIC_DEVICE_NUMBER = 14
    _DEFAULT_NUM_IDX_POS = 8
    _SM_FILEPATH = "stateMachine.json"



    #Constructor
    def __init__(self):

        super(CCIGui, self).__init__()

        #Instantiating this GUI's CellCounterCore object
        myStage = TicStage(_TIC_COM_MODE, [_TIC_COM, _TIC_BAUDRATE], _TIC_DEVICE_NUMBER, input_dist_per_rev=300, micro_step_factor=8)
        myStage.enable = True
        my_camera = PyFLIRCamera()
        self._CCI_CORE = CellCounterCore(my_stage, my_camera, "stateMachine.json")

        #updateGUI will be called when _CCI_CORE changes state
        self._CCI_CORE.bindTo(updateGUI)

        #loading stateMachine information from JSON file
        with open("stateMachine.json") as json_file:
            self._STATE_MACHINE = json.load(json_file)
            self._STATE_PERMISSIONS = self._STATE_MACHINE.get("_STATE_PERMISSIONS")
        print(self._STATE_PERMISSIONS)

        # TODO:
            # CellCounterCore:
            #     Needs a method which will return the position dictionary which contains indices as keys and positions as values.
            # Gui:
            #     Implement a for loop which creates a well for every key in the position dictionary, with a label matching the key value. Shown below.
            #
            #     for key in position_dictionary.keys():
            #         button = WellButton(text=key)
            #         button.bind(on_press=self.wellCB)
            #         self.children[1].children.append(button)
            #

#---------------------------------------------------------------Helper Functions----------------------------------------------------------------------

    def traverseAndToggle(self, root, to_disable):


        """
        Purpose: Traverses a widget tree and enables all the buttons in it, unless the button appears in the list "to_disable"

        Inputs:

            root: A Widget object. Each call to this function traverses the widget tree which originates at root.

            to_disable: A list of strings. Each string matches the text field of a button to be disabled.

        Outputs: None

        """

        for x in root.children:
            if len(x.children) > 0:
                self.traverseAndToggle(x, to_disable)
            else:
                disable = False
                for i in to_disable:
                    if i == x.text:
                        disable = True
                x.disabled = disable
        return

    # TODO: Decide if you want to leave getCurrentWellButton and buttonToIndex as separate functions. They were left separate to give the option of getting button object or getting
    #   button index, but so far in the GUI code they have only been used together in order to get the button index.



    def getCurrentWellButton(self):

        """
        Purpose: Returns the WellButton object which is currently in the "down" state out of the ToggleButton set.

        Inputs: None

        Outputs: WellButton object. The WellButton which is currently in the "down" state.

        """

        current = [t for t in WellButton.get_widgets('wells') if t.state == 'down']

        try:
            current = current[0]
        except IndexError:
            popup_button = Button(text="Close")
            popup = Popup(title = "Please Select A Well Before Pressing Set", content=popup_button)
            popup_button.bind(on_press= popup.dismiss)
            popup.open()
            return

        return current

    def buttonToIndex(self, button):

        # TODO: Decide if you want to keep this method at all. It exists because the position of the buttons in the widget tree does not match the text of each button. However,
        #   it is now being used to match the button text to the value in the position dictionary which corresponds to the button's position.

        """
        Purpose: Converts button object to its index.

        Inputs:

            button: A WellButton object.

        Outputs:
            index: An integer. Represents the position of the WellButton object in the list which stores the positions corresponding to each WellButton index.

        """

        index = int(button.text)

        return index
#---------------------------------------------------------Update Gui--------------------------------------------------------------------------------------------------


    def updateGUI(self):
        """
        Purpose: Updates appearance of this CCIGui object to match the current state of its CellCounterCore object.

        Inputs: None

        Outputs: None

        """

        state = self._CCI_CORE.state()
        state_dictionary = self._STATE_PERMISSIONS.get(state)
        disable_list = []

        #Checking each permission and changing buttons accordingly
        if not state_dictionary.get("Enter Preview"):
            disable_list.append("Preview")
            if state != "PREVIEW":
                disable_list.append("Stop Preview")
            else:
                pass

        if not state_dictionary.get("Start Scan"):
            disable_list.append("Direct Scan")
            disable_list.append("Triggered Scan")
            if state != "TRIGGERED SCAN" and state != "DIRECT SCAN":
                disable_list.append("Stop Scan")

        if not state_dictionary.get("Set Positions"):
            disable_list.append("Set")

        if not state_dictionary.get("Move Stage"):
            disable_list.append("Up")
            disable_list.append("Down")
            disable_list.append("Step Size")
            disable_list.append("Initialize")

        if not state_dictionary.get("Create Server"):
            disable_list.append("Create Server")
        print(disable_list)

        self.traverseAndToggle(self, disable_list)

        return


#---------------------------------------------------------------------------Button Callbacks---------------------------------------------------------------------------------------

    # TODO: Decide if you want to make upCB and downCB one function with a direction argument, "up" or "down", then binding appropriate call in the .kv file
    def upCB(self):

        """
        Purpose: Callback for Up button.
        """

        # TODO:
            # CellCounterCore: Implement function which requests 1 unit of movement in a certain direction from the stage object.
            #
            # Gui: Call this method with the correct direction argument.

        return

    def downCB(self):
        """
        Purpose: Callback for Down button.
        """

        # TODO:
            # CellCounterCore: Implement function which requests 1 unit of movement in a certain direction from the stage object.
            #
            # Gui: Call this method with the correct direction argument.

        return

    def initializeCB(self):
        """
        Purpose: Callback for Initialize button.
        """

        self._CCI_CORE.stageDiscoverRange()

        return


    def stepSizeCB(self, text):
        """
        Purpose: Callback for Step Size spinner.

        Inputs:
            text: A string. Text corresponding to the step size which we would like the stage to be using.
        """

        # TODO:
            # CellCounterCore: Implement function which tells stage object to change its step size
            #
            # Gui: Call this method with the correct argument based on selected text

        return

    def previewCB(self):
        """
        Purpose: Callback for Preview button.
        """
        # TODO: Decide if we are implementing the preview in the GUI window or as a matplotlib popup.
            # if matplotlib popup:
            #     Remove Stop Preview method and button.
            # if GUI window:
            #     implement a previewCB which is continually updating the CameraPreview label. You may want to switch to a kivy image which can be altered
            #     pixelwise using the texture property.

        self._CCI_CORE.livePreview()
        return

    def stopPreview(self):
        """
        Purpose: Callback for Stop Preview button.
        """

        self._CCI_CORE.stopPreview()

        #TODO: Possibly add code which re-sets the label or image appearance to a black screen when the preview ends.
        return

    def wellCB(self):
        """
        Purpose: Callback for well buttons.
        """
        button = getCurrentWellButton()
        index = buttonToIndex()
        self._CCI_CORE.moveToIndexedPosition(index)

        return


    def setCB(self):
        """
        Purpose: Callback for Set button.
        """


        button = self.getCurrentWellButton()
        index = self.buttonToIndex(button)
        self._CCI_CORE.setCurrentPositionAsIndex(index)

        return

    def triggeredScanCB(self):
        """
        Purpose: Callback for Triggered Scan button.
        """
        #TODO: Confirm this is the correct way to wait for OT2 permission.
        self._CCI_CORE.handleHTTPRequest()
        self._CCI_CORE.runScan()
        return

    def directScanCB(self):
        """
        Purpose: Callback for Direct Scan button.
        """

        self._CCI_CORE.runScan()
        return

    def stopScanCB(self):

        #TODO:
            # CellCounterCore: Implement function that will interrupt either a triggered scan or a direct scan if needed
            #
            # GUI: Call this function.
        pass

    def statusCB(self):

        #TODO: Implement the mechanism for the status spinner changing colors and displaying various information. This may involve changes to updateGUI.

        pass



class WellButton(ToggleButton):

    pass




class GUIApp(App):
    def build(self):
        return CCIGui()


if __name__ == '__main__':
    GUIApp().run()
