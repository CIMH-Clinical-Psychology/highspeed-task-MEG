#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.0),
    on January 12, 2024, at 10:57
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'pygame'
prefs.hardware['audioLatencyMode'] = '0'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# Store info about the experiment session
psychopyVersion = '2023.2.0'
expName = 'untitled'  # from the Builder filename that created this script
expInfo = {
    'participant': '0',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}

# Run 'Before Experiment' code from startup
import meg_triggers
from meg_triggers import send_trigger
meg_triggers.set_default_duration(0.005)

# Run 'Before Experiment' code from functions
import os.path as osp
def get_image_name(filename):
    return osp.splitext(osp.basename(filename))[0]

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # temporarily remove keys which the dialog doesn't need to show
    poppedKeys = {
        'date': expInfo.pop('date', data.getDateStr()),
        'expName': expInfo.pop('expName', expName),
        'psychopyVersion': expInfo.pop('psychopyVersion', psychopyVersion),
    }
    # show participant info dialog
    dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # restore hidden keys
    expInfo.update(poppedKeys)
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\ElektaStimPC\\Desktop\\MEG-highspeed-task-main\\fast_replay_main_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file
    # return log file
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=[1400,1050], fullscr=False, screen=1,
            winType='pyglet', allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units=None
        )
        if expInfo is not None:
            # store frame rate of monitor if we can measure it
            expInfo['frameRate'] = win.getActualFrameRate()
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = None
    win.mouseVisible = True
    win.hideMessage()
    return win


def setupInputs(expInfo, thisExp, win):
    """
    Setup whatever inputs are available (mouse, keyboard, eyetracker, etc.)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    dict
        Dictionary of input devices by name.
    """
    # --- Setup input devices ---
    inputs = {}
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    ioSession = '1'
    if 'session' in expInfo:
        ioSession = str(expInfo['session'])
    ioServer = io.launchHubServer(window=win, **ioConfig)
    eyetracker = None
    
    # create a default keyboard (e.g. to check for escape)
    defaultKeyboard = keyboard.Keyboard(backend='iohub')
    # return inputs dict
    return {
        'ioServer': ioServer,
        'defaultKeyboard': defaultKeyboard,
        'eyetracker': eyetracker,
    }

def pauseExperiment(thisExp, inputs=None, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # prevent components from auto-drawing
    win.stashAutoDraw()
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # make sure we have a keyboard
        if inputs is None:
            inputs = {
                'defaultKeyboard': keyboard.Keyboard(backend='ioHub')
            }
        # check for quit (typically the Esc key)
        if inputs['defaultKeyboard'].getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win, inputs=inputs)
        # flip the screen
        win.flip()
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, inputs=inputs, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # restore auto-drawn components
    win.retrieveAutoDraw()
    # reset any timers
    for timer in timers:
        timer.reset()


def run(expInfo, thisExp, win, inputs, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    inputs : dict
        Dictionary of input devices by name.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = inputs['ioServer']
    defaultKeyboard = inputs['defaultKeyboard']
    eyetracker = inputs['eyetracker']
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "startup_code" ---
    # Run 'Begin Experiment' code from settings
    t_img_sequence = 0.1
    t_img_localizer = 0.5
    t_isi_localizer = 2.0
    
    t_buffer = 8
    
     # show break after these blocks
    breaks_after_block = [2, 4, 6] 
    
    reward_amount = 0.03  # +-3 ct per trial
    
    n_localizer_trials = 5
    n_sequence_trials = 3
    
    dot_color = (-0.5, -0.5, -0.5)
    
    trigger_img = {}
    trigger_img['Gesicht'] = 1
    trigger_img['Haus']   = 2
    trigger_img['Katze']   = 3
    trigger_img['Schuh']   = 4
    trigger_img['Stuhl']   = 5
    
    trigger_break_start = 91
    trigger_break_stop = 92
    trigger_buf_start = 61
    trigger_buf_stop = 62
    trigger_sequence_sound0 = 70
    trigger_sequence_sound1 = 71
    trigger_localizer_sound0 = 30
    trigger_localizer_sound1 = 31
    trigger_fixation_pre1 = 81
    trigger_fixation_pre2 = 82
    
    # these get added depending on the trial
    # ie. Gesicht in localizer = 1, as cue = 11, as sequence=21
    trigger_base_val_localizer = 0
    trigger_base_val_cue = 10
    trigger_base_val_sequence = 20
    
    # Run 'Begin Experiment' code from startup
    import pandas as pd
    
    subj_id = expInfo['participant']
    
    df_localizer = pd.read_csv(f'./sequences/localizer_{subj_id}.csv')
    df_sequences = pd.read_csv(f'./sequences/sequences_{subj_id}.csv')
    
    # set variables we will access later
    i_localizer = 0
    i_sequence = 0
    i_block = 0
    
    letter_height=0.04
    
    # store reward
    reward_count = 0
    false_alarms = 0
    misses = 0
    wrong_answers = 0
    
    # set the number of repetitions we have
    n_blocks = max(df_localizer.block)
    n_localizer_trials = 0#max(df_localizer.trial)
    n_sequence_trials = max(df_sequences.trial)
    sound_wait = sound.Sound('sounds/soundWait.wav', secs=-1, stereo=True, hamming=True,
        name='sound_wait')
    sound_wait.setVolume(1.0)
    
    # --- Initialize components for Routine "language_selection_screen" ---
    english_polygon = visual.Rect(
        win=win, name='english_polygon',
        width=(0.3, 0.3)[0], height=(0.3, 0.3)[1],
        ori=0.0, pos=(0.5, 0), anchor='center',
        lineWidth=1.5,     colorSpace='rgb',  lineColor='white', fillColor=None,
        opacity=None, depth=0.0, interpolate=True)
    german_polygon = visual.Rect(
        win=win, name='german_polygon',
        width=(0.3, 0.3)[0], height=(0.3, 0.3)[1],
        ori=0.0, pos=(-0.5, 0), anchor='center',
        lineWidth=1.5,     colorSpace='rgb',  lineColor='white', fillColor=None,
        opacity=None, depth=-1.0, interpolate=True)
    german_flag = visual.ImageStim(
        win=win,
        name='german_flag', 
        image='stimuli/german_flag.png', mask=None, anchor='center',
        ori=0.0, pos=(-0.5, 0), size=(0.3, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=True, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    english_flag = visual.ImageStim(
        win=win,
        name='english_flag', 
        image='stimuli/english_flag.png', mask=None, anchor='center',
        ori=0.0, pos=(0.5, 0), size=(0.3, 0.3),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    german_text = visual.TextStim(win=win, name='german_text',
        text='Drücken Sie die linke (grün) Taste für Deutsch',
        font='Open Sans',
        pos=(-0.5, -0.25), height=0.04, wrapWidth=None, ori=0.0, 
        color=[-1.0000, 1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    english_text = visual.TextStim(win=win, name='english_text',
        text='Press the right (blue) button for English',
        font='Open Sans',
        pos=(0.5, -0.25), height=0.04, wrapWidth=None, ori=0.0, 
        color=[0.0588, 0.6157, 0.9608], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    choice_key = keyboard.Keyboard()
    
    # --- Initialize components for Routine "instruct_pre1" ---
    key_resp_3 = keyboard.Keyboard()
    text_2 = visual.TextStim(win=win, name='text_2',
        text='\nWillkommen beim Hauptteil des Experiments.\n\nIm Folgenden werden sich die beiden Aufgaben aus der Übung immer abwechseln, d.h. erst wird ein paar Minuten lang Aufgabe 1 erscheinen, dann ein paar Minuten Aufgabe 2. Diese Abfolge wird 8x wiederholt. Alle paar Blöcke wird es eine kurze Verschnaufpause geben.\n\nDrücken Sie eine Taste um fortzufahren.',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "instruct_pre2" ---
    key_resp_2 = keyboard.Keyboard()
    text_4 = visual.TextStim(win=win, name='text_4',
        text='Für eine gute Datenqualität ist es wichtig, dass sie während des Experimentes so ruhig wie möglich sitzen bleiben. Versuchen Sie insbesondere Kopfbewegungen zu minimieren und wenn möglich mit Bewegungen bis zu den Pausen zu warten.\n\nZur Erinnerung: Für jede richtige Antwort erhalten Sie eine Belohnung von X Cent. Wenn Sie eine falsche Antwort geben werden Ihnen X Cent abgezogen. Bitte versuchen Sie so schnell wie möglich zu antworten.\n\nDrücken Sie eine Taste um mit dem Experiment und Aufgabe 1 zu starten.',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "instruct_pre3" ---
    text_7 = visual.TextStim(win=win, name='text_7',
        text='es geht gleich los...',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "fixation_dot_pre1" ---
    text_6 = visual.TextStim(win=win, name='text_6',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "fixation_dot" ---
    localizer_fixation = visual.TextStim(win=win, name='localizer_fixation',
        text='•',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=dot_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "localizer" ---
    localizer_img = visual.ImageStim(
        win=win,
        name='localizer_img', 
        image='stimuli/Gesicht.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=[0.25],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    localizer_isi = visual.TextStim(win=win, name='localizer_isi',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_localizer = keyboard.Keyboard()
    sound_correct = sound.Sound('sounds/soundCoin.wav', secs=0.5, stereo=True, hamming=True,
        name='sound_correct')
    sound_correct.setVolume(1.0)
    sound_wrong = sound.Sound('sounds/soundError.wav', secs=0.5, stereo=True, hamming=True,
        name='sound_wrong')
    sound_wrong.setVolume(1.0)
    
    # --- Initialize components for Routine "fixation_dot_pre2" ---
    text_9 = visual.TextStim(win=win, name='text_9',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    # Run 'Begin Experiment' code from code_dot2
    i_localizer=0
    i_sequence=0
    
    # --- Initialize components for Routine "cue" ---
    cue_text = visual.TextStim(win=win, name='cue_text',
        text='dummy',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "blank1500" ---
    text = visual.TextStim(win=win, name='text',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "fixation_dot" ---
    localizer_fixation = visual.TextStim(win=win, name='localizer_fixation',
        text='•',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=dot_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "sequence" ---
    sequence_img_1 = visual.ImageStim(
        win=win,
        name='sequence_img_1', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    sequence_isi_1 = visual.TextStim(win=win, name='sequence_isi_1',
        text='•',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=dot_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    sequence_img_2 = visual.ImageStim(
        win=win,
        name='sequence_img_2', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    sequence_isi_2 = visual.TextStim(win=win, name='sequence_isi_2',
        text='•',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=dot_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    sequence_img_3 = visual.ImageStim(
        win=win,
        name='sequence_img_3', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    sequence_isi_3 = visual.TextStim(win=win, name='sequence_isi_3',
        text='•',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=dot_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    sequence_img_4 = visual.ImageStim(
        win=win,
        name='sequence_img_4', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    sequence_isi_4 = visual.TextStim(win=win, name='sequence_isi_4',
        text='•',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=dot_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    sequence_img_5 = visual.ImageStim(
        win=win,
        name='sequence_img_5', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    sequence_isi_5 = visual.TextStim(win=win, name='sequence_isi_5',
        text='•',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=dot_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    
    # --- Initialize components for Routine "buffer" ---
    buffer_fixation = visual.TextStim(win=win, name='buffer_fixation',
        text='•',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color=dot_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "question" ---
    question_key_resp = keyboard.Keyboard()
    question_text = visual.TextStim(win=win, name='question_text',
        text='Did X come before Y?',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "feedback" ---
    text_feedback__answer = visual.TextStim(win=win, name='text_feedback__answer',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "break_2" ---
    text_8 = visual.TextStim(win=win, name='text_8',
        text='<Kurze Pause>\n\nBitte nehmen Sie eine kurze Verschnaufpause.\n\nDrücken Sie eine beliebige Taste, wenn Sie mit dem nächsten Block fortfahren wollen.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_7 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "instruct_end" ---
    text_5 = visual.TextStim(win=win, name='text_5',
        text='Die Übung ist nun beendet. \n\nBitte melden Sie sich beim Experimentleiter.\n\n--------------------\n\nThe experiment has ended.\n\nPlease let the experimentator know that you are finished.',
        font='Open Sans',
        pos=(0, 0), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    if globalClock is None:
        globalClock = core.Clock()  # to track the time since experiment started
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6)
    
    # --- Prepare to start Routine "startup_code" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('startup_code.started', globalClock.getTime())
    sound_wait.setSound('sounds/soundWait.wav', hamming=True)
    sound_wait.setVolume(1.0, log=False)
    # keep track of which components have finished
    startup_codeComponents = [sound_wait]
    for thisComponent in startup_codeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "startup_code" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from startup
        sound_wait.status=FINISHED
        # update sound_wait status according to whether it's playing
        if sound_wait.isPlaying:
            sound_wait.status = STARTED
        elif sound_wait.isFinished:
            sound_wait.status = FINISHED
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in startup_codeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "startup_code" ---
    for thisComponent in startup_codeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('startup_code.stopped', globalClock.getTime())
    sound_wait.stop()  # ensure sound has stopped at end of Routine
    # the Routine "startup_code" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "language_selection_screen" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('language_selection_screen.started', globalClock.getTime())
    choice_key.keys = []
    choice_key.rt = []
    _choice_key_allKeys = []
    # Run 'Begin Routine' code from choice_code
    win.mouseVisible = False
    # keep track of which components have finished
    language_selection_screenComponents = [english_polygon, german_polygon, german_flag, english_flag, german_text, english_text, choice_key]
    for thisComponent in language_selection_screenComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "language_selection_screen" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *english_polygon* updates
        
        # if english_polygon is starting this frame...
        if english_polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            english_polygon.frameNStart = frameN  # exact frame index
            english_polygon.tStart = t  # local t and not account for scr refresh
            english_polygon.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(english_polygon, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'english_polygon.started')
            # update status
            english_polygon.status = STARTED
            english_polygon.setAutoDraw(True)
        
        # if english_polygon is active this frame...
        if english_polygon.status == STARTED:
            # update params
            pass
        
        # *german_polygon* updates
        
        # if german_polygon is starting this frame...
        if german_polygon.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            german_polygon.frameNStart = frameN  # exact frame index
            german_polygon.tStart = t  # local t and not account for scr refresh
            german_polygon.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(german_polygon, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'german_polygon.started')
            # update status
            german_polygon.status = STARTED
            german_polygon.setAutoDraw(True)
        
        # if german_polygon is active this frame...
        if german_polygon.status == STARTED:
            # update params
            pass
        
        # *german_flag* updates
        
        # if german_flag is starting this frame...
        if german_flag.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            german_flag.frameNStart = frameN  # exact frame index
            german_flag.tStart = t  # local t and not account for scr refresh
            german_flag.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(german_flag, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'german_flag.started')
            # update status
            german_flag.status = STARTED
            german_flag.setAutoDraw(True)
        
        # if german_flag is active this frame...
        if german_flag.status == STARTED:
            # update params
            pass
        
        # *english_flag* updates
        
        # if english_flag is starting this frame...
        if english_flag.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            english_flag.frameNStart = frameN  # exact frame index
            english_flag.tStart = t  # local t and not account for scr refresh
            english_flag.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(english_flag, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'english_flag.started')
            # update status
            english_flag.status = STARTED
            english_flag.setAutoDraw(True)
        
        # if english_flag is active this frame...
        if english_flag.status == STARTED:
            # update params
            pass
        
        # *german_text* updates
        
        # if german_text is starting this frame...
        if german_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            german_text.frameNStart = frameN  # exact frame index
            german_text.tStart = t  # local t and not account for scr refresh
            german_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(german_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'german_text.started')
            # update status
            german_text.status = STARTED
            german_text.setAutoDraw(True)
        
        # if german_text is active this frame...
        if german_text.status == STARTED:
            # update params
            pass
        
        # *english_text* updates
        
        # if english_text is starting this frame...
        if english_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            english_text.frameNStart = frameN  # exact frame index
            english_text.tStart = t  # local t and not account for scr refresh
            english_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(english_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'english_text.started')
            # update status
            english_text.status = STARTED
            english_text.setAutoDraw(True)
        
        # if english_text is active this frame...
        if english_text.status == STARTED:
            # update params
            pass
        
        # *choice_key* updates
        waitOnFlip = False
        
        # if choice_key is starting this frame...
        if choice_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            choice_key.frameNStart = frameN  # exact frame index
            choice_key.tStart = t  # local t and not account for scr refresh
            choice_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(choice_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'choice_key.started')
            # update status
            choice_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(choice_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(choice_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if choice_key.status == STARTED and not waitOnFlip:
            theseKeys = choice_key.getKeys(keyList=['g','b'], ignoreKeys=["escape"], waitRelease=False)
            _choice_key_allKeys.extend(theseKeys)
            if len(_choice_key_allKeys):
                choice_key.keys = _choice_key_allKeys[-1].name  # just the last key pressed
                choice_key.rt = _choice_key_allKeys[-1].rt
                choice_key.duration = _choice_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in language_selection_screenComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "language_selection_screen" ---
    for thisComponent in language_selection_screenComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('language_selection_screen.stopped', globalClock.getTime())
    # check responses
    if choice_key.keys in ['', [], None]:  # No response was made
        choice_key.keys = None
    thisExp.addData('choice_key.keys',choice_key.keys)
    if choice_key.keys != None:  # we had a response
        thisExp.addData('choice_key.rt', choice_key.rt)
        thisExp.addData('choice_key.duration', choice_key.duration)
    thisExp.nextEntry()
    # Run 'End Routine' code from choice_code
    
    if choice_key.keys == "g":
        german_polygon.borderColor = [1, 0.2941, -1]
        language = "german"
    elif choice_key.keys == "b":
        english_polygon.borderColor = [1, 0.2941, -1]
        language = "english"
    # the Routine "language_selection_screen" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruct_pre1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruct_pre1.started', globalClock.getTime())
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # Run 'Begin Routine' code from code_3
    if language=='english':
        text_2.text = """Welcome to the main part of the experiment.
    
    In the following, the two tasks from the exercise will always alternate, i.e. task 1 will appear first for a few minutes, then task 2 for a few minutes. This sequence will be repeated 8 times. There will be a short breather every few blocks.
    
    Press any button to continue."""
    # keep track of which components have finished
    instruct_pre1Components = [key_resp_3, text_2]
    for thisComponent in instruct_pre1Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_pre1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['y','b','r', 'g'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_pre1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_pre1" ---
    for thisComponent in instruct_pre1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruct_pre1.stopped', globalClock.getTime())
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "instruct_pre1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruct_pre2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruct_pre2.started', globalClock.getTime())
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # Run 'Begin Routine' code from code_4
    if language=='english':
        text_4.text="""To ensure good data quality, it is important that you remain as still as possible during the experiment. In particular, try to minimize head movements and, if possible, wait until the breaks before making any movements.
    
    Reminder: You will receive a reward of X cents for every correct answer. If you give an incorrect answer, X cents will be deducted. Please try to answer as quickly as possible.
    
    Press any button to start the experiment and task 1."""
    # keep track of which components have finished
    instruct_pre2Components = [key_resp_2, text_4]
    for thisComponent in instruct_pre2Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_pre2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['y','b','r', 'g'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *text_4* updates
        
        # if text_4 is starting this frame...
        if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_4.frameNStart = frameN  # exact frame index
            text_4.tStart = t  # local t and not account for scr refresh
            text_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_4.started')
            # update status
            text_4.status = STARTED
            text_4.setAutoDraw(True)
        
        # if text_4 is active this frame...
        if text_4.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_pre2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_pre2" ---
    for thisComponent in instruct_pre2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruct_pre2.stopped', globalClock.getTime())
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "instruct_pre2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruct_pre3" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruct_pre3.started', globalClock.getTime())
    # Run 'Begin Routine' code from code_5
    if language=='english':
        text_7.text = 'the experiment is starting...'
    # keep track of which components have finished
    instruct_pre3Components = [text_7]
    for thisComponent in instruct_pre3Components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_pre3" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_7* updates
        
        # if text_7 is starting this frame...
        if text_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_7.frameNStart = frameN  # exact frame index
            text_7.tStart = t  # local t and not account for scr refresh
            text_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_7.started')
            # update status
            text_7.status = STARTED
            text_7.setAutoDraw(True)
        
        # if text_7 is active this frame...
        if text_7.status == STARTED:
            # update params
            pass
        
        # if text_7 is stopping this frame...
        if text_7.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_7.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_7.tStop = t  # not accounting for scr refresh
                text_7.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_7.stopped')
                # update status
                text_7.status = FINISHED
                text_7.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_pre3Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_pre3" ---
    for thisComponent in instruct_pre3Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruct_pre3.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    
    # set up handler to look after randomisation of conditions etc
    blocks = data.TrialHandler(nReps=n_blocks, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='blocks')
    thisExp.addLoop(blocks)  # add the loop to the experiment
    thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    
    for thisBlock in blocks:
        currentLoop = blocks
        thisExp.timestampOnFlip(win, 'thisRow.t')
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                inputs=inputs, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
        )
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # --- Prepare to start Routine "fixation_dot_pre1" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('fixation_dot_pre1.started', globalClock.getTime())
        # Run 'Begin Routine' code from code_dot1
        i_localizer=0
        i_sequence=0
        # keep track of which components have finished
        fixation_dot_pre1Components = [text_6]
        for thisComponent in fixation_dot_pre1Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_dot_pre1" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_6* updates
            
            # if text_6 is starting this frame...
            if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_6.frameNStart = frameN  # exact frame index
                text_6.tStart = t  # local t and not account for scr refresh
                text_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_6.started')
                # update status
                text_6.status = STARTED
                text_6.setAutoDraw(True)
            
            # if text_6 is active this frame...
            if text_6.status == STARTED:
                # update params
                pass
            
            # if text_6 is stopping this frame...
            if text_6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_6.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_6.tStop = t  # not accounting for scr refresh
                    text_6.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_6.stopped')
                    # update status
                    text_6.status = FINISHED
                    text_6.setAutoDraw(False)
            # Run 'Each Frame' code from code_dot1
            if frameN==0:
                send_trigger(trigger_fixation_pre1)
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation_dot_pre1Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_dot_pre1" ---
        for thisComponent in fixation_dot_pre1Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('fixation_dot_pre1.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        
        # set up handler to look after randomisation of conditions etc
        localizer_trials = data.TrialHandler(nReps=n_localizer_trials, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='localizer_trials')
        thisExp.addLoop(localizer_trials)  # add the loop to the experiment
        thisLocalizer_trial = localizer_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLocalizer_trial.rgb)
        if thisLocalizer_trial != None:
            for paramName in thisLocalizer_trial:
                globals()[paramName] = thisLocalizer_trial[paramName]
        
        for thisLocalizer_trial in localizer_trials:
            currentLoop = localizer_trials
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisLocalizer_trial.rgb)
            if thisLocalizer_trial != None:
                for paramName in thisLocalizer_trial:
                    globals()[paramName] = thisLocalizer_trial[paramName]
            
            # --- Prepare to start Routine "fixation_dot" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('fixation_dot.started', globalClock.getTime())
            # keep track of which components have finished
            fixation_dotComponents = [localizer_fixation]
            for thisComponent in fixation_dotComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation_dot" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *localizer_fixation* updates
                
                # if localizer_fixation is starting this frame...
                if localizer_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    localizer_fixation.frameNStart = frameN  # exact frame index
                    localizer_fixation.tStart = t  # local t and not account for scr refresh
                    localizer_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(localizer_fixation, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'localizer_fixation.started')
                    # update status
                    localizer_fixation.status = STARTED
                    localizer_fixation.setAutoDraw(True)
                
                # if localizer_fixation is active this frame...
                if localizer_fixation.status == STARTED:
                    # update params
                    pass
                
                # if localizer_fixation is stopping this frame...
                if localizer_fixation.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > localizer_fixation.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        localizer_fixation.tStop = t  # not accounting for scr refresh
                        localizer_fixation.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'localizer_fixation.stopped')
                        # update status
                        localizer_fixation.status = FINISHED
                        localizer_fixation.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixation_dotComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation_dot" ---
            for thisComponent in fixation_dotComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('fixation_dot.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            
            # --- Prepare to start Routine "localizer" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('localizer.started', globalClock.getTime())
            key_resp_localizer.keys = []
            key_resp_localizer.rt = []
            _key_resp_localizer_allKeys = []
            sound_correct.setSound('sounds/soundCoin.wav', secs=0.5, hamming=True)
            sound_correct.setVolume(1.0, log=False)
            sound_wrong.setSound('sounds/soundError.wav', secs=0.5, hamming=True)
            sound_wrong.setVolume(1.0, log=False)
            # Run 'Begin Routine' code from localizer_code
            df_block = df_localizer[df_localizer['block']==i_block].reset_index()
            df_trial = df_block.iloc[i_localizer]
            
            localizer_img.image = df_trial.img
            t_isi_localizer = df_trial.isi
            is_distractor = df_trial.distractor
            
            # prevent inactive sound from blocking trial finish
            sound_correct.status = FINISHED
            sound_wrong.status = FINISHED
            
            played = False
            
            if is_distractor:
                localizer_img.ori  = 180
            else:
                localizer_img.ori  = 0
            
            msg = f'localizer {i_localizer}/{len(df_block)}'
            msg += f' block {i_block}/{max(df_localizer["block"])} {"[FLIPPED]" if is_distractor else ""} isi={t_isi_localizer:.2f}s'
            print(msg)
            # keep track of which components have finished
            localizerComponents = [localizer_img, localizer_isi, key_resp_localizer, sound_correct, sound_wrong]
            for thisComponent in localizerComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "localizer" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *localizer_img* updates
                
                # if localizer_img is starting this frame...
                if localizer_img.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    localizer_img.frameNStart = frameN  # exact frame index
                    localizer_img.tStart = t  # local t and not account for scr refresh
                    localizer_img.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(localizer_img, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'localizer_img.started')
                    # update status
                    localizer_img.status = STARTED
                    localizer_img.setAutoDraw(True)
                
                # if localizer_img is active this frame...
                if localizer_img.status == STARTED:
                    # update params
                    pass
                
                # if localizer_img is stopping this frame...
                if localizer_img.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > localizer_img.tStartRefresh + t_img_localizer-frameTolerance:
                        # keep track of stop time/frame for later
                        localizer_img.tStop = t  # not accounting for scr refresh
                        localizer_img.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'localizer_img.stopped')
                        # update status
                        localizer_img.status = FINISHED
                        localizer_img.setAutoDraw(False)
                
                # *localizer_isi* updates
                
                # if localizer_isi is starting this frame...
                if localizer_isi.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    localizer_isi.frameNStart = frameN  # exact frame index
                    localizer_isi.tStart = t  # local t and not account for scr refresh
                    localizer_isi.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(localizer_isi, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'localizer_isi.started')
                    # update status
                    localizer_isi.status = STARTED
                    localizer_isi.setAutoDraw(True)
                
                # if localizer_isi is active this frame...
                if localizer_isi.status == STARTED:
                    # update params
                    pass
                
                # if localizer_isi is stopping this frame...
                if localizer_isi.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > localizer_isi.tStartRefresh + t_isi_localizer-frameTolerance:
                        # keep track of stop time/frame for later
                        localizer_isi.tStop = t  # not accounting for scr refresh
                        localizer_isi.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'localizer_isi.stopped')
                        # update status
                        localizer_isi.status = FINISHED
                        localizer_isi.setAutoDraw(False)
                
                # *key_resp_localizer* updates
                waitOnFlip = False
                
                # if key_resp_localizer is starting this frame...
                if key_resp_localizer.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_localizer.frameNStart = frameN  # exact frame index
                    key_resp_localizer.tStart = t  # local t and not account for scr refresh
                    key_resp_localizer.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_localizer, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_localizer.started')
                    # update status
                    key_resp_localizer.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_localizer.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_localizer.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if key_resp_localizer is stopping this frame...
                if key_resp_localizer.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > key_resp_localizer.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        key_resp_localizer.tStop = t  # not accounting for scr refresh
                        key_resp_localizer.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'key_resp_localizer.stopped')
                        # update status
                        key_resp_localizer.status = FINISHED
                        key_resp_localizer.status = FINISHED
                if key_resp_localizer.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_localizer.getKeys(keyList=['y', 'g', 'b', 'r'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_localizer_allKeys.extend(theseKeys)
                    if len(_key_resp_localizer_allKeys):
                        key_resp_localizer.keys = _key_resp_localizer_allKeys[-1].name  # just the last key pressed
                        key_resp_localizer.rt = _key_resp_localizer_allKeys[-1].rt
                        key_resp_localizer.duration = _key_resp_localizer_allKeys[-1].duration
                
                # if sound_correct is starting this frame...
                if sound_correct.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
                    # keep track of start time/frame for later
                    sound_correct.frameNStart = frameN  # exact frame index
                    sound_correct.tStart = t  # local t and not account for scr refresh
                    sound_correct.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('sound_correct.started', tThisFlipGlobal)
                    # update status
                    sound_correct.status = STARTED
                    sound_correct.play(when=win)  # sync with win flip
                
                # if sound_correct is stopping this frame...
                if sound_correct.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sound_correct.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        sound_correct.tStop = t  # not accounting for scr refresh
                        sound_correct.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sound_correct.stopped')
                        # update status
                        sound_correct.status = FINISHED
                        sound_correct.stop()
                # update sound_correct status according to whether it's playing
                if sound_correct.isPlaying:
                    sound_correct.status = STARTED
                elif sound_correct.isFinished:
                    sound_correct.status = FINISHED
                
                # if sound_wrong is starting this frame...
                if sound_wrong.status == NOT_STARTED and tThisFlip >= 5-frameTolerance:
                    # keep track of start time/frame for later
                    sound_wrong.frameNStart = frameN  # exact frame index
                    sound_wrong.tStart = t  # local t and not account for scr refresh
                    sound_wrong.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('sound_wrong.started', tThisFlipGlobal)
                    # update status
                    sound_wrong.status = STARTED
                    sound_wrong.play(when=win)  # sync with win flip
                
                # if sound_wrong is stopping this frame...
                if sound_wrong.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sound_wrong.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        sound_wrong.tStop = t  # not accounting for scr refresh
                        sound_wrong.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sound_wrong.stopped')
                        # update status
                        sound_wrong.status = FINISHED
                        sound_wrong.stop()
                # update sound_wrong status according to whether it's playing
                if sound_wrong.isPlaying:
                    sound_wrong.status = STARTED
                elif sound_wrong.isFinished:
                    sound_wrong.status = FINISHED
                # Run 'Each Frame' code from localizer_code
                if frameN == 0:
                    img_idx = trigger_img[get_image_name(df_trial.img)]
                    send_trigger(trigger_base_val_localizer + img_idx)
                
                # play for wrong/correct button presses
                if len(key_resp_localizer.keys) and not played:
                    played = True
                    if is_distractor:
                        sound_correct.play()
                        reward_count += reward_amount 
                        print('Correct press')
                        send_trigger(trigger_localizer_sound1)
                    else:
                        sound_wrong.play()
                        reward_count -= reward_amount 
                        false_alarms += 1
                        print('False alarm')
                        send_trigger(trigger_localizer_sound0)
                
                # play error in case of missed button press
                if is_distractor and key_resp_localizer.status==FINISHED and not played:
                    played = True
                    sound_wrong.play()
                    misses += 1
                    print('Miss')
                    send_trigger(trigger_localizer_sound0)
                    reward_count -= reward_amount 
                
                sound_correct.status = FINISHED
                sound_wrong.status = FINISHED
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in localizerComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "localizer" ---
            for thisComponent in localizerComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('localizer.stopped', globalClock.getTime())
            # check responses
            if key_resp_localizer.keys in ['', [], None]:  # No response was made
                key_resp_localizer.keys = None
            localizer_trials.addData('key_resp_localizer.keys',key_resp_localizer.keys)
            if key_resp_localizer.keys != None:  # we had a response
                localizer_trials.addData('key_resp_localizer.rt', key_resp_localizer.rt)
                localizer_trials.addData('key_resp_localizer.duration', key_resp_localizer.duration)
            sound_correct.stop()  # ensure sound has stopped at end of Routine
            sound_wrong.stop()  # ensure sound has stopped at end of Routine
            # Run 'End Routine' code from localizer_code
            i_localizer+=1
            # the Routine "localizer" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed n_localizer_trials repeats of 'localizer_trials'
        
        
        # --- Prepare to start Routine "fixation_dot_pre2" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('fixation_dot_pre2.started', globalClock.getTime())
        # keep track of which components have finished
        fixation_dot_pre2Components = [text_9]
        for thisComponent in fixation_dot_pre2Components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_dot_pre2" ---
        routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_9* updates
            
            # if text_9 is starting this frame...
            if text_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_9.frameNStart = frameN  # exact frame index
                text_9.tStart = t  # local t and not account for scr refresh
                text_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_9, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_9.started')
                # update status
                text_9.status = STARTED
                text_9.setAutoDraw(True)
            
            # if text_9 is active this frame...
            if text_9.status == STARTED:
                # update params
                pass
            
            # if text_9 is stopping this frame...
            if text_9.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_9.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_9.tStop = t  # not accounting for scr refresh
                    text_9.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_9.stopped')
                    # update status
                    text_9.status = FINISHED
                    text_9.setAutoDraw(False)
            # Run 'Each Frame' code from code_dot2
            if frameN==0:
                send_trigger(trigger_fixation_pre2)
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, inputs=inputs, win=win)
                return
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation_dot_pre2Components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_dot_pre2" ---
        for thisComponent in fixation_dot_pre2Components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('fixation_dot_pre2.stopped', globalClock.getTime())
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if routineForceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        
        # set up handler to look after randomisation of conditions etc
        sequence_trials = data.TrialHandler(nReps=n_sequence_trials, method='sequential', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='sequence_trials')
        thisExp.addLoop(sequence_trials)  # add the loop to the experiment
        thisSequence_trial = sequence_trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisSequence_trial.rgb)
        if thisSequence_trial != None:
            for paramName in thisSequence_trial:
                globals()[paramName] = thisSequence_trial[paramName]
        
        for thisSequence_trial in sequence_trials:
            currentLoop = sequence_trials
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisSequence_trial.rgb)
            if thisSequence_trial != None:
                for paramName in thisSequence_trial:
                    globals()[paramName] = thisSequence_trial[paramName]
            
            # --- Prepare to start Routine "cue" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('cue.started', globalClock.getTime())
            # Run 'Begin Routine' code from cue_code
            df_block = df_sequences[df_sequences['block']==i_block].reset_index()
            df_trial = df_block.iloc[i_sequence]
            
            translation =  {'Haus': 'house', 
                            'Gesicht': 'face', 
                            'Katze': 'cat', 
                            'Schuh': 'shoe', 
                            'Stuhl': 'chair'}
            if language=='english':
                cue_text.text = translation[df_trial.cue]
            else:
                cue_text.text = df_trial.cue
                
            t_isi = df_trial.isi/1000
            t_thisbuffer = t_buffer - (5*0.1 + 5*t_isi)
            msg = f'sequence {i_sequence}/{len(df_block)} '
            msg += f' block {i_block}/{max(df_localizer["block"])}  isi={t_isi} ms, buffer={t_thisbuffer:1f} s'
            msg += f'Expecting choice {df_trial.correct_pos} ({df_trial.target_idx+1})'
            print(msg, flush=True)
            # keep track of which components have finished
            cueComponents = [cue_text]
            for thisComponent in cueComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "cue" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *cue_text* updates
                
                # if cue_text is starting this frame...
                if cue_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    cue_text.frameNStart = frameN  # exact frame index
                    cue_text.tStart = t  # local t and not account for scr refresh
                    cue_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(cue_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_text.started')
                    # update status
                    cue_text.status = STARTED
                    cue_text.setAutoDraw(True)
                
                # if cue_text is active this frame...
                if cue_text.status == STARTED:
                    # update params
                    pass
                
                # if cue_text is stopping this frame...
                if cue_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > cue_text.tStartRefresh + 1.0-frameTolerance:
                        # keep track of stop time/frame for later
                        cue_text.tStop = t  # not accounting for scr refresh
                        cue_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'cue_text.stopped')
                        # update status
                        cue_text.status = FINISHED
                        cue_text.setAutoDraw(False)
                # Run 'Each Frame' code from cue_code
                if frameN == 0:
                    img_idx = trigger_img[df_trial.cue]
                    send_trigger(trigger_base_val_cue + img_idx)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in cueComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "cue" ---
            for thisComponent in cueComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('cue.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.000000)
            
            # --- Prepare to start Routine "blank1500" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('blank1500.started', globalClock.getTime())
            # keep track of which components have finished
            blank1500Components = [text]
            for thisComponent in blank1500Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "blank1500" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text* updates
                
                # if text is starting this frame...
                if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text.frameNStart = frameN  # exact frame index
                    text.tStart = t  # local t and not account for scr refresh
                    text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.started')
                    # update status
                    text.status = STARTED
                    text.setAutoDraw(True)
                
                # if text is active this frame...
                if text.status == STARTED:
                    # update params
                    pass
                
                # if text is stopping this frame...
                if text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        text.tStop = t  # not accounting for scr refresh
                        text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text.stopped')
                        # update status
                        text.status = FINISHED
                        text.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in blank1500Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "blank1500" ---
            for thisComponent in blank1500Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('blank1500.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.500000)
            
            # --- Prepare to start Routine "fixation_dot" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('fixation_dot.started', globalClock.getTime())
            # keep track of which components have finished
            fixation_dotComponents = [localizer_fixation]
            for thisComponent in fixation_dotComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "fixation_dot" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *localizer_fixation* updates
                
                # if localizer_fixation is starting this frame...
                if localizer_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    localizer_fixation.frameNStart = frameN  # exact frame index
                    localizer_fixation.tStart = t  # local t and not account for scr refresh
                    localizer_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(localizer_fixation, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'localizer_fixation.started')
                    # update status
                    localizer_fixation.status = STARTED
                    localizer_fixation.setAutoDraw(True)
                
                # if localizer_fixation is active this frame...
                if localizer_fixation.status == STARTED:
                    # update params
                    pass
                
                # if localizer_fixation is stopping this frame...
                if localizer_fixation.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > localizer_fixation.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        localizer_fixation.tStop = t  # not accounting for scr refresh
                        localizer_fixation.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'localizer_fixation.stopped')
                        # update status
                        localizer_fixation.status = FINISHED
                        localizer_fixation.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fixation_dotComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fixation_dot" ---
            for thisComponent in fixation_dotComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('fixation_dot.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            
            # --- Prepare to start Routine "sequence" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('sequence.started', globalClock.getTime())
            # Run 'Begin Routine' code from sequence_code
            for img_x in range(5):
                component_img = locals()[f'sequence_img_{img_x+1}']
                component_img.image = df_trial[f'img{img_x}']
            
            img_started = [0, 0, 0, 0, 0]
            # keep track of which components have finished
            sequenceComponents = [sequence_img_1, sequence_isi_1, sequence_img_2, sequence_isi_2, sequence_img_3, sequence_isi_3, sequence_img_4, sequence_isi_4, sequence_img_5, sequence_isi_5]
            for thisComponent in sequenceComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "sequence" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *sequence_img_1* updates
                
                # if sequence_img_1 is starting this frame...
                if sequence_img_1.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                    # keep track of start time/frame for later
                    sequence_img_1.frameNStart = frameN  # exact frame index
                    sequence_img_1.tStart = t  # local t and not account for scr refresh
                    sequence_img_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sequence_img_1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sequence_img_1.started')
                    # update status
                    sequence_img_1.status = STARTED
                    sequence_img_1.setAutoDraw(True)
                
                # if sequence_img_1 is active this frame...
                if sequence_img_1.status == STARTED:
                    # update params
                    pass
                
                # if sequence_img_1 is stopping this frame...
                if sequence_img_1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sequence_img_1.tStartRefresh + t_img_sequence-frameTolerance:
                        # keep track of stop time/frame for later
                        sequence_img_1.tStop = t  # not accounting for scr refresh
                        sequence_img_1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sequence_img_1.stopped')
                        # update status
                        sequence_img_1.status = FINISHED
                        sequence_img_1.setAutoDraw(False)
                
                # *sequence_isi_1* updates
                
                # if sequence_isi_1 is starting this frame...
                if sequence_isi_1.status == NOT_STARTED and sequence_img_1.status==FINISHED:
                    # keep track of start time/frame for later
                    sequence_isi_1.frameNStart = frameN  # exact frame index
                    sequence_isi_1.tStart = t  # local t and not account for scr refresh
                    sequence_isi_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sequence_isi_1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sequence_isi_1.started')
                    # update status
                    sequence_isi_1.status = STARTED
                    sequence_isi_1.setAutoDraw(True)
                
                # if sequence_isi_1 is active this frame...
                if sequence_isi_1.status == STARTED:
                    # update params
                    pass
                
                # if sequence_isi_1 is stopping this frame...
                if sequence_isi_1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sequence_isi_1.tStartRefresh + t_isi-frameTolerance:
                        # keep track of stop time/frame for later
                        sequence_isi_1.tStop = t  # not accounting for scr refresh
                        sequence_isi_1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sequence_isi_1.stopped')
                        # update status
                        sequence_isi_1.status = FINISHED
                        sequence_isi_1.setAutoDraw(False)
                
                # *sequence_img_2* updates
                
                # if sequence_img_2 is starting this frame...
                if sequence_img_2.status == NOT_STARTED and sequence_isi_1.status==FINISHED:
                    # keep track of start time/frame for later
                    sequence_img_2.frameNStart = frameN  # exact frame index
                    sequence_img_2.tStart = t  # local t and not account for scr refresh
                    sequence_img_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sequence_img_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sequence_img_2.started')
                    # update status
                    sequence_img_2.status = STARTED
                    sequence_img_2.setAutoDraw(True)
                
                # if sequence_img_2 is active this frame...
                if sequence_img_2.status == STARTED:
                    # update params
                    pass
                
                # if sequence_img_2 is stopping this frame...
                if sequence_img_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sequence_img_2.tStartRefresh + t_img_sequence-frameTolerance:
                        # keep track of stop time/frame for later
                        sequence_img_2.tStop = t  # not accounting for scr refresh
                        sequence_img_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sequence_img_2.stopped')
                        # update status
                        sequence_img_2.status = FINISHED
                        sequence_img_2.setAutoDraw(False)
                
                # *sequence_isi_2* updates
                
                # if sequence_isi_2 is starting this frame...
                if sequence_isi_2.status == NOT_STARTED and sequence_img_2.status==FINISHED:
                    # keep track of start time/frame for later
                    sequence_isi_2.frameNStart = frameN  # exact frame index
                    sequence_isi_2.tStart = t  # local t and not account for scr refresh
                    sequence_isi_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sequence_isi_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sequence_isi_2.started')
                    # update status
                    sequence_isi_2.status = STARTED
                    sequence_isi_2.setAutoDraw(True)
                
                # if sequence_isi_2 is active this frame...
                if sequence_isi_2.status == STARTED:
                    # update params
                    pass
                
                # if sequence_isi_2 is stopping this frame...
                if sequence_isi_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sequence_isi_2.tStartRefresh + t_isi-frameTolerance:
                        # keep track of stop time/frame for later
                        sequence_isi_2.tStop = t  # not accounting for scr refresh
                        sequence_isi_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sequence_isi_2.stopped')
                        # update status
                        sequence_isi_2.status = FINISHED
                        sequence_isi_2.setAutoDraw(False)
                
                # *sequence_img_3* updates
                
                # if sequence_img_3 is starting this frame...
                if sequence_img_3.status == NOT_STARTED and sequence_isi_2.status==FINISHED:
                    # keep track of start time/frame for later
                    sequence_img_3.frameNStart = frameN  # exact frame index
                    sequence_img_3.tStart = t  # local t and not account for scr refresh
                    sequence_img_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sequence_img_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sequence_img_3.started')
                    # update status
                    sequence_img_3.status = STARTED
                    sequence_img_3.setAutoDraw(True)
                
                # if sequence_img_3 is active this frame...
                if sequence_img_3.status == STARTED:
                    # update params
                    pass
                
                # if sequence_img_3 is stopping this frame...
                if sequence_img_3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sequence_img_3.tStartRefresh + t_img_sequence-frameTolerance:
                        # keep track of stop time/frame for later
                        sequence_img_3.tStop = t  # not accounting for scr refresh
                        sequence_img_3.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sequence_img_3.stopped')
                        # update status
                        sequence_img_3.status = FINISHED
                        sequence_img_3.setAutoDraw(False)
                
                # *sequence_isi_3* updates
                
                # if sequence_isi_3 is starting this frame...
                if sequence_isi_3.status == NOT_STARTED and sequence_img_3.status==FINISHED:
                    # keep track of start time/frame for later
                    sequence_isi_3.frameNStart = frameN  # exact frame index
                    sequence_isi_3.tStart = t  # local t and not account for scr refresh
                    sequence_isi_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sequence_isi_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sequence_isi_3.started')
                    # update status
                    sequence_isi_3.status = STARTED
                    sequence_isi_3.setAutoDraw(True)
                
                # if sequence_isi_3 is active this frame...
                if sequence_isi_3.status == STARTED:
                    # update params
                    pass
                
                # if sequence_isi_3 is stopping this frame...
                if sequence_isi_3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sequence_isi_3.tStartRefresh + t_isi-frameTolerance:
                        # keep track of stop time/frame for later
                        sequence_isi_3.tStop = t  # not accounting for scr refresh
                        sequence_isi_3.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sequence_isi_3.stopped')
                        # update status
                        sequence_isi_3.status = FINISHED
                        sequence_isi_3.setAutoDraw(False)
                
                # *sequence_img_4* updates
                
                # if sequence_img_4 is starting this frame...
                if sequence_img_4.status == NOT_STARTED and sequence_isi_3.status==FINISHED:
                    # keep track of start time/frame for later
                    sequence_img_4.frameNStart = frameN  # exact frame index
                    sequence_img_4.tStart = t  # local t and not account for scr refresh
                    sequence_img_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sequence_img_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sequence_img_4.started')
                    # update status
                    sequence_img_4.status = STARTED
                    sequence_img_4.setAutoDraw(True)
                
                # if sequence_img_4 is active this frame...
                if sequence_img_4.status == STARTED:
                    # update params
                    pass
                
                # if sequence_img_4 is stopping this frame...
                if sequence_img_4.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sequence_img_4.tStartRefresh + t_img_sequence-frameTolerance:
                        # keep track of stop time/frame for later
                        sequence_img_4.tStop = t  # not accounting for scr refresh
                        sequence_img_4.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sequence_img_4.stopped')
                        # update status
                        sequence_img_4.status = FINISHED
                        sequence_img_4.setAutoDraw(False)
                
                # *sequence_isi_4* updates
                
                # if sequence_isi_4 is starting this frame...
                if sequence_isi_4.status == NOT_STARTED and sequence_img_4.status==FINISHED:
                    # keep track of start time/frame for later
                    sequence_isi_4.frameNStart = frameN  # exact frame index
                    sequence_isi_4.tStart = t  # local t and not account for scr refresh
                    sequence_isi_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sequence_isi_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sequence_isi_4.started')
                    # update status
                    sequence_isi_4.status = STARTED
                    sequence_isi_4.setAutoDraw(True)
                
                # if sequence_isi_4 is active this frame...
                if sequence_isi_4.status == STARTED:
                    # update params
                    pass
                
                # if sequence_isi_4 is stopping this frame...
                if sequence_isi_4.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sequence_isi_4.tStartRefresh + t_isi-frameTolerance:
                        # keep track of stop time/frame for later
                        sequence_isi_4.tStop = t  # not accounting for scr refresh
                        sequence_isi_4.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sequence_isi_4.stopped')
                        # update status
                        sequence_isi_4.status = FINISHED
                        sequence_isi_4.setAutoDraw(False)
                
                # *sequence_img_5* updates
                
                # if sequence_img_5 is starting this frame...
                if sequence_img_5.status == NOT_STARTED and sequence_isi_4.status==FINISHED:
                    # keep track of start time/frame for later
                    sequence_img_5.frameNStart = frameN  # exact frame index
                    sequence_img_5.tStart = t  # local t and not account for scr refresh
                    sequence_img_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sequence_img_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sequence_img_5.started')
                    # update status
                    sequence_img_5.status = STARTED
                    sequence_img_5.setAutoDraw(True)
                
                # if sequence_img_5 is active this frame...
                if sequence_img_5.status == STARTED:
                    # update params
                    pass
                
                # if sequence_img_5 is stopping this frame...
                if sequence_img_5.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sequence_img_5.tStartRefresh + t_img_sequence-frameTolerance:
                        # keep track of stop time/frame for later
                        sequence_img_5.tStop = t  # not accounting for scr refresh
                        sequence_img_5.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sequence_img_5.stopped')
                        # update status
                        sequence_img_5.status = FINISHED
                        sequence_img_5.setAutoDraw(False)
                
                # *sequence_isi_5* updates
                
                # if sequence_isi_5 is starting this frame...
                if sequence_isi_5.status == NOT_STARTED and sequence_img_5.status==FINISHED:
                    # keep track of start time/frame for later
                    sequence_isi_5.frameNStart = frameN  # exact frame index
                    sequence_isi_5.tStart = t  # local t and not account for scr refresh
                    sequence_isi_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(sequence_isi_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'sequence_isi_5.started')
                    # update status
                    sequence_isi_5.status = STARTED
                    sequence_isi_5.setAutoDraw(True)
                
                # if sequence_isi_5 is active this frame...
                if sequence_isi_5.status == STARTED:
                    # update params
                    pass
                
                # if sequence_isi_5 is stopping this frame...
                if sequence_isi_5.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sequence_isi_5.tStartRefresh + t_isi-frameTolerance:
                        # keep track of stop time/frame for later
                        sequence_isi_5.tStop = t  # not accounting for scr refresh
                        sequence_isi_5.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sequence_isi_5.stopped')
                        # update status
                        sequence_isi_5.status = FINISHED
                        sequence_isi_5.setAutoDraw(False)
                # Run 'Each Frame' code from sequence_code
                for img_x, state in enumerate(img_started):
                    if state: continue
                    component_img = locals()[f'sequence_img_{img_x+1}']
                    if component_img.status==STARTED:
                        img_started[img_x] = 1
                        img_id = trigger_img[get_image_name(component_img.image)]
                        send_trigger(trigger_base_val_sequence + img_id)
                
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in sequenceComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "sequence" ---
            for thisComponent in sequenceComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('sequence.stopped', globalClock.getTime())
            # Run 'End Routine' code from sequence_code
            i_sequence += 1
            # the Routine "sequence" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "buffer" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('buffer.started', globalClock.getTime())
            # Run 'Begin Routine' code from buffer_code
            sound_stopped = False
            # keep track of which components have finished
            bufferComponents = [buffer_fixation]
            for thisComponent in bufferComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "buffer" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *buffer_fixation* updates
                
                # if buffer_fixation is starting this frame...
                if buffer_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    buffer_fixation.frameNStart = frameN  # exact frame index
                    buffer_fixation.tStart = t  # local t and not account for scr refresh
                    buffer_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(buffer_fixation, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'buffer_fixation.started')
                    # update status
                    buffer_fixation.status = STARTED
                    buffer_fixation.setAutoDraw(True)
                
                # if buffer_fixation is active this frame...
                if buffer_fixation.status == STARTED:
                    # update params
                    pass
                
                # if buffer_fixation is stopping this frame...
                if buffer_fixation.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > buffer_fixation.tStartRefresh + t_thisbuffer-frameTolerance:
                        # keep track of stop time/frame for later
                        buffer_fixation.tStop = t  # not accounting for scr refresh
                        buffer_fixation.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'buffer_fixation.stopped')
                        # update status
                        buffer_fixation.status = FINISHED
                        buffer_fixation.setAutoDraw(False)
                # Run 'Each Frame' code from buffer_code
                if frameN==0:
                    send_trigger(trigger_buf_start)
                    sound_wait.play()
                
                if buffer_fixation.status==FINISHED and not sound_stopped:
                    send_trigger(trigger_buf_stop)
                    sound_wait.pause()
                    sound_stopped = True
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in bufferComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "buffer" ---
            for thisComponent in bufferComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('buffer.stopped', globalClock.getTime())
            # the Routine "buffer" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "question" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('question.started', globalClock.getTime())
            question_key_resp.keys = []
            question_key_resp.rt = []
            _question_key_resp_allKeys = []
            # Run 'Begin Routine' code from question_code
            
            choices = [df_trial.target_idx, df_trial.lure_idx]
            if df_trial.correct_pos=='r':
                choices = choices[::-1]
                
            question_text.text = f'Position\n\n\n\n\n{df_trial.cue}\n\n\n\n\n'
            question_text.text += '             ?             '.join([str(c+1) for c in choices])
            # keep track of which components have finished
            questionComponents = [question_key_resp, question_text]
            for thisComponent in questionComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "question" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *question_key_resp* updates
                waitOnFlip = False
                
                # if question_key_resp is starting this frame...
                if question_key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    question_key_resp.frameNStart = frameN  # exact frame index
                    question_key_resp.tStart = t  # local t and not account for scr refresh
                    question_key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(question_key_resp, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'question_key_resp.started')
                    # update status
                    question_key_resp.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(question_key_resp.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(question_key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if question_key_resp is stopping this frame...
                if question_key_resp.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > question_key_resp.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        question_key_resp.tStop = t  # not accounting for scr refresh
                        question_key_resp.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'question_key_resp.stopped')
                        # update status
                        question_key_resp.status = FINISHED
                        question_key_resp.status = FINISHED
                if question_key_resp.status == STARTED and not waitOnFlip:
                    theseKeys = question_key_resp.getKeys(keyList=['g', 'b'], ignoreKeys=["escape"], waitRelease=False)
                    _question_key_resp_allKeys.extend(theseKeys)
                    if len(_question_key_resp_allKeys):
                        question_key_resp.keys = _question_key_resp_allKeys[-1].name  # just the last key pressed
                        question_key_resp.rt = _question_key_resp_allKeys[-1].rt
                        question_key_resp.duration = _question_key_resp_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *question_text* updates
                
                # if question_text is starting this frame...
                if question_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    question_text.frameNStart = frameN  # exact frame index
                    question_text.tStart = t  # local t and not account for scr refresh
                    question_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(question_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'question_text.started')
                    # update status
                    question_text.status = STARTED
                    question_text.setAutoDraw(True)
                
                # if question_text is active this frame...
                if question_text.status == STARTED:
                    # update params
                    pass
                
                # if question_text is stopping this frame...
                if question_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > question_text.tStartRefresh + 2.5-frameTolerance:
                        # keep track of stop time/frame for later
                        question_text.tStop = t  # not accounting for scr refresh
                        question_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'question_text.stopped')
                        # update status
                        question_text.status = FINISHED
                        question_text.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in questionComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "question" ---
            for thisComponent in questionComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('question.stopped', globalClock.getTime())
            # check responses
            if question_key_resp.keys in ['', [], None]:  # No response was made
                question_key_resp.keys = None
            sequence_trials.addData('question_key_resp.keys',question_key_resp.keys)
            if question_key_resp.keys != None:  # we had a response
                sequence_trials.addData('question_key_resp.rt', question_key_resp.rt)
                sequence_trials.addData('question_key_resp.duration', question_key_resp.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.500000)
            
            # --- Prepare to start Routine "feedback" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('feedback.started', globalClock.getTime())
            # Run 'Begin Routine' code from code_feedbackseq
            # play for wrong/correct button presses
            # mapping from keys to positions
            
            mapping = {'r': 'u', 'y': 'd', 'b': 'r', 'g': 'l'}
            if question_key_resp.keys and len(question_key_resp.keys):
                key_pos = mapping[question_key_resp.keys[-1]]
                if key_pos==df_trial.correct_pos:
                    sound_correct.stop()
                    sound_correct.play()
                    send_trigger(trigger_sequence_sound1)
                    print('Correct answer')
                else:
                    sound_wrong.stop()
                    sound_wrong.play()
                    wrong_answers += 1
                    send_trigger(trigger_sequence_sound0)
                    print('Wrong answer')
            else:
                # play error in case of missed button press
                sound_wrong.stop()
                sound_wrong.play()
                misses += 1
                send_trigger(trigger_sequence_sound0)
                print('Miss')
                reward_count -= reward_amount 
            
            # set the break conditional to true to display a break
            break_conditional_nReps = int(i_block in breaks_after_block)
            
            # keep track of which components have finished
            feedbackComponents = [text_feedback__answer]
            for thisComponent in feedbackComponents:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "feedback" ---
            routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_feedback__answer* updates
                
                # if text_feedback__answer is starting this frame...
                if text_feedback__answer.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_feedback__answer.frameNStart = frameN  # exact frame index
                    text_feedback__answer.tStart = t  # local t and not account for scr refresh
                    text_feedback__answer.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_feedback__answer, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_feedback__answer.started')
                    # update status
                    text_feedback__answer.status = STARTED
                    text_feedback__answer.setAutoDraw(True)
                
                # if text_feedback__answer is active this frame...
                if text_feedback__answer.status == STARTED:
                    # update params
                    pass
                
                # if text_feedback__answer is stopping this frame...
                if text_feedback__answer.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_feedback__answer.tStartRefresh + 0.5-frameTolerance:
                        # keep track of stop time/frame for later
                        text_feedback__answer.tStop = t  # not accounting for scr refresh
                        text_feedback__answer.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_feedback__answer.stopped')
                        # update status
                        text_feedback__answer.status = FINISHED
                        text_feedback__answer.setAutoDraw(False)
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in feedbackComponents:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "feedback" ---
            for thisComponent in feedbackComponents:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('feedback.stopped', globalClock.getTime())
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routineForceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.500000)
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed n_sequence_trials repeats of 'sequence_trials'
        
        
        # set up handler to look after randomisation of conditions etc
        break_conditional = data.TrialHandler(nReps=break_conditional_nReps, method='random', 
            extraInfo=expInfo, originPath=-1,
            trialList=[None],
            seed=None, name='break_conditional')
        thisExp.addLoop(break_conditional)  # add the loop to the experiment
        thisBreak_conditional = break_conditional.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisBreak_conditional.rgb)
        if thisBreak_conditional != None:
            for paramName in thisBreak_conditional:
                globals()[paramName] = thisBreak_conditional[paramName]
        
        for thisBreak_conditional in break_conditional:
            currentLoop = break_conditional
            thisExp.timestampOnFlip(win, 'thisRow.t')
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    inputs=inputs, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
            )
            # abbreviate parameter names if possible (e.g. rgb = thisBreak_conditional.rgb)
            if thisBreak_conditional != None:
                for paramName in thisBreak_conditional:
                    globals()[paramName] = thisBreak_conditional[paramName]
            
            # --- Prepare to start Routine "break_2" ---
            continueRoutine = True
            # update component parameters for each repeat
            thisExp.addData('break_2.started', globalClock.getTime())
            key_resp_7.keys = []
            key_resp_7.rt = []
            _key_resp_7_allKeys = []
            # Run 'Begin Routine' code from code_break
            if frameN==0:
                send_trigger(trigger_break_start)
                
            if language=='english':
                test_8.text = """<short break>
            
                Please take a short breather.
            
                Press any button if you want to continue with the next block."""
            # keep track of which components have finished
            break_2Components = [text_8, key_resp_7]
            for thisComponent in break_2Components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "break_2" ---
            routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *text_8* updates
                
                # if text_8 is starting this frame...
                if text_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_8.frameNStart = frameN  # exact frame index
                    text_8.tStart = t  # local t and not account for scr refresh
                    text_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_8, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_8.started')
                    # update status
                    text_8.status = STARTED
                    text_8.setAutoDraw(True)
                
                # if text_8 is active this frame...
                if text_8.status == STARTED:
                    # update params
                    pass
                
                # *key_resp_7* updates
                waitOnFlip = False
                
                # if key_resp_7 is starting this frame...
                if key_resp_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    key_resp_7.frameNStart = frameN  # exact frame index
                    key_resp_7.tStart = t  # local t and not account for scr refresh
                    key_resp_7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(key_resp_7, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp_7.started')
                    # update status
                    key_resp_7.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(key_resp_7.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(key_resp_7.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if key_resp_7.status == STARTED and not waitOnFlip:
                    theseKeys = key_resp_7.getKeys(keyList=['y', 'r', 'g', 'b'], ignoreKeys=["escape"], waitRelease=False)
                    _key_resp_7_allKeys.extend(theseKeys)
                    if len(_key_resp_7_allKeys):
                        key_resp_7.keys = _key_resp_7_allKeys[-1].name  # just the last key pressed
                        key_resp_7.rt = _key_resp_7_allKeys[-1].rt
                        key_resp_7.duration = _key_resp_7_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, inputs=inputs, win=win)
                    return
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in break_2Components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "break_2" ---
            for thisComponent in break_2Components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            thisExp.addData('break_2.stopped', globalClock.getTime())
            # check responses
            if key_resp_7.keys in ['', [], None]:  # No response was made
                key_resp_7.keys = None
            break_conditional.addData('key_resp_7.keys',key_resp_7.keys)
            if key_resp_7.keys != None:  # we had a response
                break_conditional.addData('key_resp_7.rt', key_resp_7.rt)
                break_conditional.addData('key_resp_7.duration', key_resp_7.duration)
            # Run 'End Routine' code from code_break
            if frameN==0:
                send_trigger(trigger_break_stop)
            # the Routine "break_2" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed break_conditional_nReps repeats of 'break_conditional'
        
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed n_blocks repeats of 'blocks'
    
    
    # --- Prepare to start Routine "instruct_end" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruct_end.started', globalClock.getTime())
    # keep track of which components have finished
    instruct_endComponents = [text_5]
    for thisComponent in instruct_endComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruct_end" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_5* updates
        
        # if text_5 is starting this frame...
        if text_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_5.frameNStart = frameN  # exact frame index
            text_5.tStart = t  # local t and not account for scr refresh
            text_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_5.started')
            # update status
            text_5.status = STARTED
            text_5.setAutoDraw(True)
        
        # if text_5 is active this frame...
        if text_5.status == STARTED:
            # update params
            pass
        
        # if text_5 is stopping this frame...
        if text_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_5.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_5.tStop = t  # not accounting for scr refresh
                text_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_5.stopped')
                # update status
                text_5.status = FINISHED
                text_5.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, inputs=inputs, win=win)
            return
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instruct_endComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_end" ---
    for thisComponent in instruct_endComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruct_end.stopped', globalClock.getTime())
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    
    # mark experiment as finished
    endExperiment(thisExp, win=win, inputs=inputs)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, inputs=None, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    inputs : dict
        Dictionary of input devices by name.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # shut down eyetracker, if there is one
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()


def quit(thisExp, win=None, inputs=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    inputs : dict
        Dictionary of input devices by name.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    if inputs is not None:
        if 'eyetracker' in inputs and inputs['eyetracker'] is not None:
            eyetracker.setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    inputs = setupInputs(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win, 
        inputs=inputs
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win, inputs=inputs)
