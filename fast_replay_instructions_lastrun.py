#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2023.2.2),
    on Januar 12, 2024, at 11:43
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
prefs.hardware['audioLib'] = 'sounddevice'
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
psychopyVersion = '2023.2.2'
expName = 'untitled'  # from the Builder filename that created this script
expInfo = {
    'participant': '0',
    'date': data.getDateStr(),  # add a simple timestamp
    'expName': expName,
    'psychopyVersion': psychopyVersion,
}


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
        originPath='C:\\Users\\simon.kern\\Nextcloud\\ZI\\2023.09_MEG_Fast_Replay\\MEG-highspeed-task\\fast_replay_instructions_lastrun.py',
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
    # this outputs to the screen, not a file
    logging.console.setLevel(logging.EXP)
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log', level=logging.EXP)
    
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
            units='norm'
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
        win.units = 'norm'
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
    
    reward_amount = 0.03  # +-3 ct per trial
    
    n_localizer_trials = 5
    n_sequence_trials = 3
    
    dot_color = (-0.5, -0.5, -0.5)
    
    image_size = 0.25
    
    letter_height=0.06
    # Run 'Begin Experiment' code from startup
    import pandas as pd
    
    subj_id = 0
    
    df_localizer = pd.read_csv(f'./sequences/localizer_{subj_id}.csv')
    df_sequences = pd.read_csv(f'./sequences/sequences_{subj_id}.csv')
    
    print('DUMMY TRIALS: subj_id is set to 0')
    
    # set variables we will access later
    i_localizer = 0
    i_sequence = 0
    i_block = 0
    
    # store reward
    reward_count = 0
    false_alarms = 0
    misses = 0
    wrong_answers = 0
    
    # set the number of repetitions we have
    n_blocks = 1
    n_localizer_trials = 6
    n_sequence_trials = 3
    
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
    
    # --- Initialize components for Routine "instruct_welcome" ---
    key_resp = keyboard.Keyboard()
    image = visual.ImageStim(
        win=win,
        name='image', 
        image='stimuli/Gesicht.jpg', mask=None, anchor='center',
        ori=0.0, pos=(2, 2), size=[0.25],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', 
        image='stimuli/Haus.jpg', mask=None, anchor='center',
        ori=0.0, pos=(2, 0), size=[0.25],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    image_3 = visual.ImageStim(
        win=win,
        name='image_3', 
        image='stimuli/Katze.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=[0.25],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    image_4 = visual.ImageStim(
        win=win,
        name='image_4', 
        image='stimuli/Schuh.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=[0.25],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    image_5 = visual.ImageStim(
        win=win,
        name='image_5', 
        image='stimuli/Stuhl.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=[0.25],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-5.0)
    textbox_3 = visual.TextBox2(
         win, text='Willkommen zur Studie \n<b>“Visuelle Objekte Erkennen”</b>\n\nDas Experiment “Visuelle Objekte Erkennen” besteht aus zwei Aufgaben bei denen wir Ihnen Bilder von alltäglichen Objekten zeigen. Dabei sollen Sie sich immer jedes Bild genau anschauen. Während des Experiments werden Sie abwechselnd zwei verschiedene Aufgaben bearbeiten, die wir Ihnen im Folgenden genau erklären.\n\nDrücken Sie eine Taste um fortzufahren.', placeholder='Type here...', font='Arial',
         pos=(0, 0),     letterHeight=letter_height,
         size=(1, 1), borderWidth=0.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='top-center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textbox_3',
         depth=-6, autoLog=True,
    )
    
    # --- Initialize components for Routine "instruct_task1" ---
    image_6 = visual.ImageStim(
        win=win,
        name='image_6', 
        image='stimuli/Stuhl.jpg', mask=None, anchor='center',
        ori=180.0, pos=(0, 0), size=[0.25],
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_resp_3 = keyboard.Keyboard()
    text_2 = visual.TextStim(win=win, name='text_2',
        text='\nIn der ersten Aufgabe sollen sie immer eine Taste drücken, wenn ein Bild falsch herum (auf den Kopf gedreht) gezeigt wird. Beispiel:\n\n\n\n\n\n\n\n\n\nBitte antworten Sie jeweils so schnell und genau wie möglich. Sie haben bei jedem Bild 1,5 Sekunden Zeit zu antworten.\n\nDrücken Sie eine Taste um fortzufahren.',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "instruct_task1_2" ---
    key_resp_2 = keyboard.Keyboard()
    text_4 = visual.TextStim(win=win, name='text_4',
        text='Immer wenn Sie eine richtige Antwort gegeben haben, hören Sie kurz das Geräusch einer Münze über die Kopfhörer. Bitte sagen Sie Bescheid, falls Sie kein Geräusch hören sollten.\n\nSie sollen nicht reagieren, wenn ein Bild richtig herum gezeigt wird.\n\nFür jedes umgedrehte Bild, das Sie richtig erkannt haben, erhalten Sie eine Belohnung von 3 Cent. Wenn Sie eine falsche Antwort geben (also eine Taste drücken, obwohl das Bild richtig herum dargestellt wurde), werden Ihnen 3 Cent abgezogen.\n\nSie sehen nun eine Übung der Aufgabe. \nDrücken Sie eine Taste um fortzufahren.',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "fixation_dot" ---
    localizer_fixation = visual.TextStim(win=win, name='localizer_fixation',
        text='•',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color=dot_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "localizer" ---
    localizer_img = visual.ImageStim(
        win=win,
        name='localizer_img', units='height', 
        image='stimuli/Gesicht.jpg', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=image_size,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    localizer_isi = visual.TextStim(win=win, name='localizer_isi',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
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
    
    # --- Initialize components for Routine "localizer_feedback" ---
    key_resp_4 = keyboard.Keyboard()
    text_feedback = visual.TextStim(win=win, name='text_feedback',
        text='... dummy ... something went wrong',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "instruct_exp2" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text='\n\nIn der zweiten Aufgabe wir Ihnen erst ein Wort gezeigt, das eines der fünf Bilder beschreibt.\n\nKurz danach werden ihnen alle fünf Bilder in sehr schneller, zufälliger Reihenfolge abgespielt. Ihre Aufgabe ist es, sich zu merken an welcher Stelle der Gegenstand des vorher gezeigten Wortes ist.\nDie Geschwindigkeit mit der die Bilder gezeigt werden, ist in jedem Durchgang unterschiedlich und wird in einigen Fällen sehr schnell sein.\n\nDeshalb ist es wichtig, dass Sie sich die Abfolge der Bilder aufmerksamm anschauen und sich die Position merken, an der das vorgegebene Objekt das erste Mal erschienen ist.\n\nBitte drücken Sie eine Taste um Fortzufahren.\n',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_5 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "instructexp2_dot" ---
    text_6 = visual.TextStim(win=win, name='text_6',
        text='\nFür jede richtige Antwort erhalten sie erneut 3ct Belohnung.\n\nZwischen den Bildern wird, wie in der vorherigen Aufgabe, ein Punkt auf dem Bildschirm erscheinen ( • ). Bitte schauen Sie wenn möglich immer den Punkt an, wenn dieser auf dem Bildschirm erscheint.\n\nBitte drücken Sie eine Taste um mit einer Übung zu starten.\n\n',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_7 = keyboard.Keyboard()
    
    # --- Initialize components for Routine "cue" ---
    cue_text = visual.TextStim(win=win, name='cue_text',
        text='dummy',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "blank1500" ---
    text = visual.TextStim(win=win, name='text',
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
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color=dot_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "sequence" ---
    sequence_img_1 = visual.ImageStim(
        win=win,
        name='sequence_img_1', units='height', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=image_size,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    sequence_isi_1 = visual.TextStim(win=win, name='sequence_isi_1',
        text='•',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color=dot_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    sequence_img_2 = visual.ImageStim(
        win=win,
        name='sequence_img_2', units='height', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=image_size,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    sequence_isi_2 = visual.TextStim(win=win, name='sequence_isi_2',
        text='•',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color=dot_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    sequence_img_3 = visual.ImageStim(
        win=win,
        name='sequence_img_3', units='height', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=image_size,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-4.0)
    sequence_isi_3 = visual.TextStim(win=win, name='sequence_isi_3',
        text='•',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color=dot_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    sequence_img_4 = visual.ImageStim(
        win=win,
        name='sequence_img_4', units='height', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=image_size,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-6.0)
    sequence_isi_4 = visual.TextStim(win=win, name='sequence_isi_4',
        text='•',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color=dot_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    sequence_img_5 = visual.ImageStim(
        win=win,
        name='sequence_img_5', units='height', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), size=image_size,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-8.0)
    sequence_isi_5 = visual.TextStim(win=win, name='sequence_isi_5',
        text='•',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color=dot_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    
    # --- Initialize components for Routine "buffer" ---
    buffer_fixation = visual.TextStim(win=win, name='buffer_fixation',
        text='•',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color=dot_color, colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    sound_birds = sound.Sound('sounds/soundWait.wav', secs=-1, stereo=True, hamming=True,
        name='sound_birds')
    sound_birds.setVolume(1.0)
    
    # --- Initialize components for Routine "question" ---
    question_key_resp = keyboard.Keyboard()
    question_text = visual.TextStim(win=win, name='question_text',
        text='Did X come before Y?',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "feedback" ---
    text_feedback__answer = visual.TextStim(win=win, name='text_feedback__answer',
        text=None,
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "sequence_feedback" ---
    key_resp_6 = keyboard.Keyboard()
    text_feedback_seq = visual.TextStim(win=win, name='text_feedback_seq',
        text='dummy... something went wrong',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "instruct_end" ---
    text_5 = visual.TextStim(win=win, name='text_5',
        text='Die Übung ist nun beendet. \n\nBitte melden Sie sich beim Experimentleider um mit dem eigentlichen Experiment zu beginnen..\n\n-------------------------\n\nThe experiment has ended. Please let the experimentator know that you are finished, so that you can start the main experiment.',
        font='Open Sans',
        pos=(0, 0), height=letter_height, wrapWidth=None, ori=0.0, 
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
    # keep track of which components have finished
    startup_codeComponents = []
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
    
    # --- Prepare to start Routine "instruct_welcome" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruct_welcome.started', globalClock.getTime())
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    textbox_3.reset()
    # Run 'Begin Routine' code from code
    for i, img in enumerate([image, image_2, image_3, image_4, image_5]):
        img.pos = [(i-2)/3, -0.6]
        #img.size = 0.2
        
    if language=='english':
        print('selected english')
        textbox_3.setText("""\n\nWelcome to the study
    <b>“Recognizing Visual Objects”</b>
    
    The experiment “Recognizing Visual Objects” consists of two tasks in which we show you pictures of everyday objects. You should always look closely at each picture. During the experiment, you will alternate between two different tasks, which we will now explain in detail.
    
    Press any button to continue.""")
        textbox_3.draw()
    # keep track of which components have finished
    instruct_welcomeComponents = [key_resp, image, image_2, image_3, image_4, image_5, textbox_3]
    for thisComponent in instruct_welcomeComponents:
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
    
    # --- Run Routine "instruct_welcome" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['r','g', 'b', 'y'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # *image* updates
        
        # if image is starting this frame...
        if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image.frameNStart = frameN  # exact frame index
            image.tStart = t  # local t and not account for scr refresh
            image.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image.started')
            # update status
            image.status = STARTED
            image.setAutoDraw(True)
        
        # if image is active this frame...
        if image.status == STARTED:
            # update params
            pass
        
        # *image_2* updates
        
        # if image_2 is starting this frame...
        if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_2.frameNStart = frameN  # exact frame index
            image_2.tStart = t  # local t and not account for scr refresh
            image_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_2.started')
            # update status
            image_2.status = STARTED
            image_2.setAutoDraw(True)
        
        # if image_2 is active this frame...
        if image_2.status == STARTED:
            # update params
            pass
        
        # *image_3* updates
        
        # if image_3 is starting this frame...
        if image_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_3.frameNStart = frameN  # exact frame index
            image_3.tStart = t  # local t and not account for scr refresh
            image_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_3.started')
            # update status
            image_3.status = STARTED
            image_3.setAutoDraw(True)
        
        # if image_3 is active this frame...
        if image_3.status == STARTED:
            # update params
            pass
        
        # *image_4* updates
        
        # if image_4 is starting this frame...
        if image_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_4.frameNStart = frameN  # exact frame index
            image_4.tStart = t  # local t and not account for scr refresh
            image_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_4.started')
            # update status
            image_4.status = STARTED
            image_4.setAutoDraw(True)
        
        # if image_4 is active this frame...
        if image_4.status == STARTED:
            # update params
            pass
        
        # *image_5* updates
        
        # if image_5 is starting this frame...
        if image_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_5.frameNStart = frameN  # exact frame index
            image_5.tStart = t  # local t and not account for scr refresh
            image_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_5.started')
            # update status
            image_5.status = STARTED
            image_5.setAutoDraw(True)
        
        # if image_5 is active this frame...
        if image_5.status == STARTED:
            # update params
            pass
        
        # *textbox_3* updates
        
        # if textbox_3 is starting this frame...
        if textbox_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            textbox_3.frameNStart = frameN  # exact frame index
            textbox_3.tStart = t  # local t and not account for scr refresh
            textbox_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(textbox_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'textbox_3.started')
            # update status
            textbox_3.status = STARTED
            textbox_3.setAutoDraw(True)
        
        # if textbox_3 is active this frame...
        if textbox_3.status == STARTED:
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
        for thisComponent in instruct_welcomeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_welcome" ---
    for thisComponent in instruct_welcomeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruct_welcome.stopped', globalClock.getTime())
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instruct_welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruct_task1" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruct_task1.started', globalClock.getTime())
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # Run 'Begin Routine' code from code_instruct1
    if language=='english':
        text_2.text = """In the first task, you should always press a button when a picture is shown the wrong way round (upside down). Example:
    
    
    
    
    
    
    
    
    
    Please answer as quickly and accurately as possible. You have 1.5 seconds to answer each picture.
    
    Press any button to continue."""
    # keep track of which components have finished
    instruct_task1Components = [image_6, key_resp_3, text_2]
    for thisComponent in instruct_task1Components:
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
    
    # --- Run Routine "instruct_task1" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *image_6* updates
        
        # if image_6 is starting this frame...
        if image_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_6.frameNStart = frameN  # exact frame index
            image_6.tStart = t  # local t and not account for scr refresh
            image_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_6.started')
            # update status
            image_6.status = STARTED
            image_6.setAutoDraw(True)
        
        # if image_6 is active this frame...
        if image_6.status == STARTED:
            # update params
            pass
        
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
        for thisComponent in instruct_task1Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_task1" ---
    for thisComponent in instruct_task1Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruct_task1.stopped', globalClock.getTime())
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "instruct_task1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instruct_task1_2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruct_task1_2.started', globalClock.getTime())
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # Run 'Begin Routine' code from code_instruct2
    if language=='english':
        text_4.text = """Whenever you have given a correct answer, you will briefly hear the sound of a coin through the headphones. Please let us know if you do not hear a sound.
    
    You should not react when a picture is shown the right way round.
    
    You will receive a reward of 3 cents for each upside-down picture that you have correctly recognized. If you give an incorrect answer (i.e. press a button even though the picture was shown the right way round), you will be deducted 3 cents.
    
    You will now see an exercise of the task. 
    Press any button to continue."""
    # keep track of which components have finished
    instruct_task1_2Components = [key_resp_2, text_4]
    for thisComponent in instruct_task1_2Components:
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
    
    # --- Run Routine "instruct_task1_2" ---
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
        for thisComponent in instruct_task1_2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_task1_2" ---
    for thisComponent in instruct_task1_2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruct_task1_2.stopped', globalClock.getTime())
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "instruct_task1_2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    localizer_repetition = data.TrialHandler(nReps=99.0, method='random', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='localizer_repetition')
    thisExp.addLoop(localizer_repetition)  # add the loop to the experiment
    thisLocalizer_repetition = localizer_repetition.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisLocalizer_repetition.rgb)
    if thisLocalizer_repetition != None:
        for paramName in thisLocalizer_repetition:
            globals()[paramName] = thisLocalizer_repetition[paramName]
    
    for thisLocalizer_repetition in localizer_repetition:
        currentLoop = localizer_repetition
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
        # abbreviate parameter names if possible (e.g. rgb = thisLocalizer_repetition.rgb)
        if thisLocalizer_repetition != None:
            for paramName in thisLocalizer_repetition:
                globals()[paramName] = thisLocalizer_repetition[paramName]
        
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
            sound_correct.seek(0)
            sound_wrong.setSound('sounds/soundError.wav', secs=0.5, hamming=True)
            sound_wrong.setVolume(1.0, log=False)
            sound_wrong.seek(0)
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
            msg += f' block {i_block}/{max(df_localizer["block"])} {"[FLIPPED]" if is_distractor else ""} isi={t_isi_localizer}'
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
                # play for wrong/correct button presses
                if len(key_resp_localizer.keys) and not played:
                    played = True
                    if is_distractor:
                        sound_correct.play()
                        reward_count += reward_amount 
                        print('Correct press')
                    else:
                        sound_wrong.play()
                        reward_count -= reward_amount 
                        false_alarms += 1
                        print('False alarm')
                
                # play error in case of missed button press
                if is_distractor and key_resp_localizer.status==FINISHED and not played:
                    played = True
                    sound_wrong.play()
                    misses += 1
                    print('Miss')
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
            sound_correct.pause()  # ensure sound has stopped at end of Routine
            sound_wrong.pause()  # ensure sound has stopped at end of Routine
            # Run 'End Routine' code from localizer_code
            i_localizer+=1
            # the Routine "localizer" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
        # completed n_localizer_trials repeats of 'localizer_trials'
        
        
        # --- Prepare to start Routine "localizer_feedback" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('localizer_feedback.started', globalClock.getTime())
        key_resp_4.keys = []
        key_resp_4.rt = []
        _key_resp_4_allKeys = []
        # Run 'Begin Routine' code from code_2
        if false_alarms==0 and misses==0:
            localizer_repetition.finished=True
            if language=='german':
                feedback_string = 'Gratuliere, Sie haben alles richtig gemacht!\n\n'
                feedback_string += 'Es geht nun weiter mit einer Übung der zweiten Aufgabe.\n\n'
                feedback_string += 'Bitte drücken Sie eine Taste um fortzufahren.'
            elif language=='english':
                feedback_string = 'Congratulations, you did everything correctly!\n\n'
                feedback_string += 'Now we continue with the second task.\n\n'
                feedback_string += 'Please press any button to continue.'
            text_feedback.text = feedback_string
        else:
            feedback_string = ''
            if misses:
                if language=='german':
                    feedback_string += f'Sie haben {misses}x verpasst eine Taste bei einem auf dem Kopf stehenden Bild zu drücken.\n'
                elif language=='english':
                    feedback_string += f'You missed {misses}x to press a button when the picture was upside down.\n'
        
            if false_alarms:
                if language=='german':
                    feedback_string += f'Sie haben {false_alarms}x an der falschen  Stelle gedrückt.\n'  
                elif language=='english':
                    feedback_string += f'You have pressed at the wrong time {false_alarms}x.\n'  
            if language=='german':
        
                feedback_string += '\n\nDrücken Sie nur eine Taste, wenn ein Bild falsch herum ist. '
                feedback_string += 'Falls Sie Fragen haben, stellen Sie diese gerne dem Experimentleiter.\n'
                feedback_string += '\nDrücken Sie eine beliebige Taste um die Übung zu wiederholen.'
            elif language=='english':
                feedback_string += '\nPlease only press a button when a picture is upside-down. '
                feedback_string += 'If you have any questions, please ask the experimentator now.\n'
                feedback_string += '\nPlease press any button to repeat the task.'
            text_feedback.text = feedback_string
        
            # reset counters
            i_localizer = 0
            false_alarms = 0
            misses = 0
        # keep track of which components have finished
        localizer_feedbackComponents = [key_resp_4, text_feedback]
        for thisComponent in localizer_feedbackComponents:
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
        
        # --- Run Routine "localizer_feedback" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *key_resp_4* updates
            waitOnFlip = False
            
            # if key_resp_4 is starting this frame...
            if key_resp_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_4.frameNStart = frameN  # exact frame index
                key_resp_4.tStart = t  # local t and not account for scr refresh
                key_resp_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_4, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_4.started')
                # update status
                key_resp_4.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_4.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_4.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_4.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_4.getKeys(keyList=['y', 'g', 'b', 'r'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_4_allKeys.extend(theseKeys)
                if len(_key_resp_4_allKeys):
                    key_resp_4.keys = _key_resp_4_allKeys[-1].name  # just the last key pressed
                    key_resp_4.rt = _key_resp_4_allKeys[-1].rt
                    key_resp_4.duration = _key_resp_4_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_feedback* updates
            
            # if text_feedback is starting this frame...
            if text_feedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_feedback.frameNStart = frameN  # exact frame index
                text_feedback.tStart = t  # local t and not account for scr refresh
                text_feedback.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_feedback, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_feedback.started')
                # update status
                text_feedback.status = STARTED
                text_feedback.setAutoDraw(True)
            
            # if text_feedback is active this frame...
            if text_feedback.status == STARTED:
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
            for thisComponent in localizer_feedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "localizer_feedback" ---
        for thisComponent in localizer_feedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('localizer_feedback.stopped', globalClock.getTime())
        # check responses
        if key_resp_4.keys in ['', [], None]:  # No response was made
            key_resp_4.keys = None
        localizer_repetition.addData('key_resp_4.keys',key_resp_4.keys)
        if key_resp_4.keys != None:  # we had a response
            localizer_repetition.addData('key_resp_4.rt', key_resp_4.rt)
            localizer_repetition.addData('key_resp_4.duration', key_resp_4.duration)
        # the Routine "localizer_feedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 99.0 repeats of 'localizer_repetition'
    
    
    # --- Prepare to start Routine "instruct_exp2" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instruct_exp2.started', globalClock.getTime())
    key_resp_5.keys = []
    key_resp_5.rt = []
    _key_resp_5_allKeys = []
    # Run 'Begin Routine' code from code_instruct3
    if language=='english':
        text_3.text = """
    
    In the second task, you are first shown a word that describes one of the five pictures.
    
    Shortly afterwards, all five pictures are played back to you in a very fast, random order. Your task is to remember at which position the object of the previously shown word is.
    The speed at which the pictures are shown is different in each round and will be very fast in some cases.
    
    It is therefore important that you watch the sequence of pictures carefully and remember the position at which the given object first appeared.
    
    Please press any button to continue."""
    # keep track of which components have finished
    instruct_exp2Components = [text_3, key_resp_5]
    for thisComponent in instruct_exp2Components:
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
    
    # --- Run Routine "instruct_exp2" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_3* updates
        
        # if text_3 is starting this frame...
        if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_3.frameNStart = frameN  # exact frame index
            text_3.tStart = t  # local t and not account for scr refresh
            text_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_3.started')
            # update status
            text_3.status = STARTED
            text_3.setAutoDraw(True)
        
        # if text_3 is active this frame...
        if text_3.status == STARTED:
            # update params
            pass
        
        # *key_resp_5* updates
        waitOnFlip = False
        
        # if key_resp_5 is starting this frame...
        if key_resp_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_5.frameNStart = frameN  # exact frame index
            key_resp_5.tStart = t  # local t and not account for scr refresh
            key_resp_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_5.started')
            # update status
            key_resp_5.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_5.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_5.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_5.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_5.getKeys(keyList=['r', 'g', 'b', 'y'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_5_allKeys.extend(theseKeys)
            if len(_key_resp_5_allKeys):
                key_resp_5.keys = _key_resp_5_allKeys[-1].name  # just the last key pressed
                key_resp_5.rt = _key_resp_5_allKeys[-1].rt
                key_resp_5.duration = _key_resp_5_allKeys[-1].duration
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
        for thisComponent in instruct_exp2Components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruct_exp2" ---
    for thisComponent in instruct_exp2Components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instruct_exp2.stopped', globalClock.getTime())
    # check responses
    if key_resp_5.keys in ['', [], None]:  # No response was made
        key_resp_5.keys = None
    thisExp.addData('key_resp_5.keys',key_resp_5.keys)
    if key_resp_5.keys != None:  # we had a response
        thisExp.addData('key_resp_5.rt', key_resp_5.rt)
        thisExp.addData('key_resp_5.duration', key_resp_5.duration)
    thisExp.nextEntry()
    # the Routine "instruct_exp2" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "instructexp2_dot" ---
    continueRoutine = True
    # update component parameters for each repeat
    thisExp.addData('instructexp2_dot.started', globalClock.getTime())
    key_resp_7.keys = []
    key_resp_7.rt = []
    _key_resp_7_allKeys = []
    # Run 'Begin Routine' code from code_instruct4
    if language=='english':
        text_6.text = """For each correct answer you will receive a reward of 3ct.
    
    Between the pictures, as in the previous task, a dot will appear on the screen ( • ). If possible, please always look at the dot when it appears on the screen.
    
    Please press a button to start the task."""
    # keep track of which components have finished
    instructexp2_dotComponents = [text_6, key_resp_7]
    for thisComponent in instructexp2_dotComponents:
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
    
    # --- Run Routine "instructexp2_dot" ---
    routineForceEnded = not continueRoutine
    while continueRoutine:
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
            theseKeys = key_resp_7.getKeys(keyList=['r', 'g', 'b', 'y'], ignoreKeys=["escape"], waitRelease=False)
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
        for thisComponent in instructexp2_dotComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructexp2_dot" ---
    for thisComponent in instructexp2_dotComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    thisExp.addData('instructexp2_dot.stopped', globalClock.getTime())
    # check responses
    if key_resp_7.keys in ['', [], None]:  # No response was made
        key_resp_7.keys = None
    thisExp.addData('key_resp_7.keys',key_resp_7.keys)
    if key_resp_7.keys != None:  # we had a response
        thisExp.addData('key_resp_7.rt', key_resp_7.rt)
        thisExp.addData('key_resp_7.duration', key_resp_7.duration)
    thisExp.nextEntry()
    # the Routine "instructexp2_dot" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    sequence_repetition = data.TrialHandler(nReps=99.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=[None],
        seed=None, name='sequence_repetition')
    thisExp.addLoop(sequence_repetition)  # add the loop to the experiment
    thisSequence_repetition = sequence_repetition.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisSequence_repetition.rgb)
    if thisSequence_repetition != None:
        for paramName in thisSequence_repetition:
            globals()[paramName] = thisSequence_repetition[paramName]
    
    for thisSequence_repetition in sequence_repetition:
        currentLoop = sequence_repetition
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
        # abbreviate parameter names if possible (e.g. rgb = thisSequence_repetition.rgb)
        if thisSequence_repetition != None:
            for paramName in thisSequence_repetition:
                globals()[paramName] = thisSequence_repetition[paramName]
        
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
                component_seq = locals()[f'sequence_img_{img_x+1}']
                component_seq.image = df_trial[f'img{img_x}']
            
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
            sound_birds.setSound('sounds/soundWait.wav', secs=t_buffer - (5*t_img_sequence + 5*t_isi), hamming=True)
            sound_birds.setVolume(1.0, log=False)
            sound_birds.seek(0)
            # keep track of which components have finished
            bufferComponents = [buffer_fixation, sound_birds]
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
                
                # if sound_birds is starting this frame...
                if sound_birds.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    sound_birds.frameNStart = frameN  # exact frame index
                    sound_birds.tStart = t  # local t and not account for scr refresh
                    sound_birds.tStartRefresh = tThisFlipGlobal  # on global time
                    # add timestamp to datafile
                    thisExp.addData('sound_birds.started', tThisFlipGlobal)
                    # update status
                    sound_birds.status = STARTED
                    sound_birds.play(when=win)  # sync with win flip
                
                # if sound_birds is stopping this frame...
                if sound_birds.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > sound_birds.tStartRefresh + t_buffer - (5*t_img_sequence + 5*t_isi)-frameTolerance:
                        # keep track of stop time/frame for later
                        sound_birds.tStop = t  # not accounting for scr refresh
                        sound_birds.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'sound_birds.stopped')
                        # update status
                        sound_birds.status = FINISHED
                        sound_birds.stop()
                # update sound_birds status according to whether it's playing
                if sound_birds.isPlaying:
                    sound_birds.status = STARTED
                elif sound_birds.isFinished:
                    sound_birds.status = FINISHED
                # Run 'Each Frame' code from buffer_code
                if buffer_fixation.status==FINISHED:
                    sound_birds.stop()
                
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
            sound_birds.pause()  # ensure sound has stopped at end of Routine
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
                    print('Correct answer')
                else:
                    sound_wrong.stop()
                    sound_wrong.play()
                    wrong_answers += 1
                    print('Wrong answer')
            else:
                # play error in case of missed button press
                sound_wrong.stop()
                sound_wrong.play()
                misses += 1
                print('Miss')
            
                reward_count -= reward_amount 
            
            
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
        
        
        # --- Prepare to start Routine "sequence_feedback" ---
        continueRoutine = True
        # update component parameters for each repeat
        thisExp.addData('sequence_feedback.started', globalClock.getTime())
        key_resp_6.keys = []
        key_resp_6.rt = []
        _key_resp_6_allKeys = []
        # Run 'Begin Routine' code from code_seqfeedback
        if wrong_answers==0 and misses==0:
            sequence_repetition.finished=True
            if language=='german':
                feedback_string = 'Sie haben bei allen drei Sequenzen die richtige Position bestimmt. Gratulation!\n\n'
                feedback_string += 'Die Übung ist nun beendet. Bitte melden Sie sich beim Experimentleiter.\n\n'
                feedback_string += 'Falls Sie noch Fragen haben, stellen Sie diese gerne.'
            elif language=='english':
                feedback_string = 'You have detected the right order in all three examples. Congratulation!\n\n'
                feedback_string += 'The exercise is now complete. Please let the experimentator know you are finished.\n\n'
                feedback_string += 'If you have any remaining questions, please let us know.'
        
            text_feedback_seq.text = feedback_string
        
        else:
            feedback_string = ''
            if wrong_answers:
                if language=='german':
                    feedback_string += f'Sie haben {wrong_answers}x falsch geantwortet. '
                elif language=='english':
                    feedback_string += f'You gave a wrong answere {wrong_answers} time(s). '
        
            if misses:
                if language=='german':
                    feedback_string += f'Sie haben {misses}x verpasst zu antworten. Bitte antworten Sie schnell genug! '
                elif language=='english':
                    feedback_string += f'You have missed to answer {misses} time(s). Please always respond fast enough! '
            if language=='german':
                feedback_string += '\n\nFalls Sie Fragen haben, stellen Sie diese gerne dem Experimentleiter.\n'
                feedback_string += '\nDrücken Sie eine beliebige Taste um die Übung zu wiederholen.'
            elif language=='english':
                feedback_string += '\nIf you have any questions, please ask the experimentator now.\n'
                feedback_string += '\nPlease press any button to repeat the task.'
            text_feedback_seq.text = feedback_string
            
            # reset counters
            wrong_answers = 0
            misses = 0
            i_sequence = 0
        
        # keep track of which components have finished
        sequence_feedbackComponents = [key_resp_6, text_feedback_seq]
        for thisComponent in sequence_feedbackComponents:
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
        
        # --- Run Routine "sequence_feedback" ---
        routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *key_resp_6* updates
            waitOnFlip = False
            
            # if key_resp_6 is starting this frame...
            if key_resp_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_6.frameNStart = frameN  # exact frame index
                key_resp_6.tStart = t  # local t and not account for scr refresh
                key_resp_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_6, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_6.started')
                # update status
                key_resp_6.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_6.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_6.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_6.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_6.getKeys(keyList=['y', 'g', 'b', 'r'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_6_allKeys.extend(theseKeys)
                if len(_key_resp_6_allKeys):
                    key_resp_6.keys = _key_resp_6_allKeys[-1].name  # just the last key pressed
                    key_resp_6.rt = _key_resp_6_allKeys[-1].rt
                    key_resp_6.duration = _key_resp_6_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *text_feedback_seq* updates
            
            # if text_feedback_seq is starting this frame...
            if text_feedback_seq.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_feedback_seq.frameNStart = frameN  # exact frame index
                text_feedback_seq.tStart = t  # local t and not account for scr refresh
                text_feedback_seq.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_feedback_seq, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_feedback_seq.started')
                # update status
                text_feedback_seq.status = STARTED
                text_feedback_seq.setAutoDraw(True)
            
            # if text_feedback_seq is active this frame...
            if text_feedback_seq.status == STARTED:
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
            for thisComponent in sequence_feedbackComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "sequence_feedback" ---
        for thisComponent in sequence_feedbackComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.addData('sequence_feedback.stopped', globalClock.getTime())
        # check responses
        if key_resp_6.keys in ['', [], None]:  # No response was made
            key_resp_6.keys = None
        sequence_repetition.addData('key_resp_6.keys',key_resp_6.keys)
        if key_resp_6.keys != None:  # we had a response
            sequence_repetition.addData('key_resp_6.rt', key_resp_6.rt)
            sequence_repetition.addData('key_resp_6.duration', key_resp_6.duration)
        # the Routine "sequence_feedback" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
    # completed 99.0 repeats of 'sequence_repetition'
    
    
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
            inputs['eyetracker'].setConnectionState(False)
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
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
