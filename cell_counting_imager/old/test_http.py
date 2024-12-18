# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 08:22:07 2019

@author: paul.lebel
"""

from cell_counting_imager import CellCounterCore 
import os
from time import sleep

_OT2IP = '192.168.101.114'
_SM_FILEPATH = os.path.join(os.path.dirname(__file__), 'stateMachine.json').replace('/','\\')
_CONFIG_FILEPATH = os.path.join(os.path.dirname(__file__), 'instrumentConfig.json').replace('/','\\')

def deleteAll(instList):
    print('Deleting all instruments')
    for inst in instList:
        try:
            print('Deleting ' + str(inst))
            del(inst)
        except Exception:
            print('Deletion failed')

def main():

    myCore = CellCounterCore(_SM_FILEPATH, _CONFIG_FILEPATH)
    myCore.createWebserver()

    text = ''
    while text != 'stop':
        text = input('Type something \n')
        sleep(.1)

    return

if __name__ == "__main__":
    main()
    
   
   