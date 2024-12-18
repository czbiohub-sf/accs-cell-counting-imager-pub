# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 08:22:07 2019

@author: paul.lebel
"""

from py_cameras import PyFLIRCamera
import numpy as np
import os
from time import sleep
import asyncio

_EXPOSURE_LIST = [10,50,100]

def deleteAll(instList):
    print('Deleting all instruments')
    for inst in instList:
        try:
            print('Deleting ' + str(inst))
            del(inst)
        except Exception:
            print('Deletion failed')

def mainPreview(camera):
    camera.preview()
    return

def main():
    
    # Only proceed if True
    proceed = True
    instList = []
  

    if proceed:
        try:
            myCamera = PyFLIRCamera()
            print('Created pyFLIRCamera object')
            instList.append(myCamera)
        except Exception as e:
            print("Could not create pyFLIRCamera object")
            print(e)
            proceed = False

    if proceed:
        try:
            proceed = myCamera.activateCamera()
        except Exception as e: 
            print("Could not activate camera")
            print(e)
            proceed = False
   
    for exp in _EXPOSURE_LIST:
        if proceed:
            try:
                print('Changing exposure time to ' + str(exp) + ' ms')
                myCamera.exposureTime_ms = exp
            except Exception as e:
                print("Could not change exposure time")
                print(e)
                proceed = False

        if proceed:
           try:
                print('\nTesting camera preview. Please close window to proceed...')
                myCamera.preview()

           except Exception as e:
               print('Could not start camera preview')
               print(e)
               proceed = False
    

    # Clean up
    deleteAll(instList)
    
    return proceed

if __name__ == "__main__":
    proceed = main()
    
    if not proceed:
        print('\nExiting with error(s)')
    else:
        print('\nExited with no errors')
   