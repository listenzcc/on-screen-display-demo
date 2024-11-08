"""
File: path-animation.py
Author: Chuncheng Zhang
Date: 2024-10-23
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    On-screen display of the path animation.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-10-23 ------------------------
# Requirements and constants
import sys
import time
import noise
import itertools
import contextlib
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageQt import ImageQt
from threading import Thread, RLock

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QMainWindow, QApplication, QLabel

from pathlib import Path
from omegaconf import OmegaConf
from rich import print
from loguru import logger

# %% ---- 2024-10-23 ------------------------
# Function and class


def prepare():
    # Handle logger
    logger.add('log/path-animation.log', rotation='5 MB')

    # Handle error messages
    p = Path('error.log')
    with contextlib.suppress(Exception):
        p.unlink(missing_ok=True)
    logger.add('error.log', level='ERROR')

    # Load the configuration file
    try:
        config = OmegaConf.load('path-animation.yaml')
    except Exception as err:
        logger.error(f'Unable to read configuration file, {err}')

    return config, logger


CONFIG, logger = prepare()
logger.info(f'Using config: {CONFIG}')

app = QApplication(sys.argv)
logger.debug(f'App: {app}')

screen = app.screens()[CONFIG.screen.id]
logger.debug(f'Screen: {CONFIG.screen.id}: {screen}, {screen.size()}')


class MyFont:
    small_font = ImageFont.truetype(
        'font/MesloLGLDZNerdFont-Bold.ttf', CONFIG.unit.em)
    large_font = ImageFont.truetype(
        'font/MesloLGLDZNerdFont-Bold.ttf', CONFIG.unit.em*2)


class MyColor:
    debug = CONFIG.color.debug
    text = CONFIG.color.text
    scatter = CONFIG.color.scatter


logger.debug(f'Loaded fonts: {MyFont.small_font}, {MyFont.large_font}')

# %%


class DynamicScatter(object):
    count = CONFIG.scatter.count
    affect_range = CONFIG.scatter.affectRange
    duration = CONFIG.scatter.duration
    t_start = CONFIG.checkpoints[0].seconds
    t_stop = CONFIG.checkpoints[-1].seconds
    min_r = CONFIG.scatter.minR
    max_r = CONFIG.scatter.maxR
    start_alpha = CONFIG.scatter.startAlpha
    stop_alpha = CONFIG.scatter.stopAlpha
    epochs = 0

    def __init__(self):
        pass

    def generate(self):
        nodes = []
        times = sorted(np.random.uniform(
            self.t_start, self.t_stop-self.duration, self.count))
        for t in times:
            radians = np.random.uniform(0, np.pi*2)
            r = np.random.uniform(0, self.affect_range)
            dx = r * np.cos(radians)
            dy = r * np.sin(radians)
            nodes.append(dict(dx=float(dx), dy=float(
                dy), t0=float(t), t1=float(t+self.duration)))
        self.nodes = nodes
        logger.debug(f'Generated nodes: {len(nodes)}')
        return self.nodes

    def remove_exceeded_nodes(self, t: float):
        '''
        Removes all the exceeded nodes.

        Args:
            t (float): Current time in seconds
        '''
        nodes = [e for e in self.nodes if e['t1'] > t]
        if len(self.nodes) > len(nodes):
            logger.debug(
                f'Removed exceeded nodes, {t:.2f} | {len(self.nodes)} -> {len(nodes)}')
        self.nodes = nodes
        return self.nodes

    def place_nodes(self, t: float, x: float, y: float):
        '''
        Place the node with the given coordinate, if and only if the coordinate is not already set.

        Args:
            t (float): Current time in seconds
            x (float): X coordinate
            y (float): Y coordinate

        Returns:
            self.nodes
        '''
        for n in self.nodes:
            if all((n['t1'] > t, n['t0'] < t, n.get('x') is None)):
                n['x'] = x
                n['y'] = y
                logger.debug(f'Placed node: {n}')
        return self.nodes


ds = DynamicScatter()
ds.generate()


def generate_img(passed: float, key_pressed: str = ''):
    # Size and img
    width = CONFIG.window.width
    height = CONFIG.window.height
    img = Image.fromarray(np.zeros((height, width, 4)), mode='RGBA')
    drawer = ImageDraw.Draw(img)

    overlay_img = img.copy()
    overlay_drawer = ImageDraw.Draw(overlay_img)

    # Time in seconds
    t_start = CONFIG.checkpoints[0].seconds
    t_stop = CONFIG.checkpoints[-1].seconds
    t_current = passed % t_stop

    # Shuffle the dynamic scatters
    current_epochs = passed // t_stop
    if current_epochs > ds.epochs:
        ds.epochs = current_epochs
        ds.generate()
    else:
        ds.remove_exceeded_nodes(t_current)

    # Draw progressBar
    if CONFIG.toggle.progressBar:
        drawer.rectangle(
            (0, height-CONFIG.unit.em-4, width*t_current/t_stop, height), outline=MyColor.debug, width=2)
        if t_current > t_start:
            drawer.rectangle(
                (width*t_start/t_stop, height-CONFIG.unit.em-4, width*t_current/t_stop, height), fill=MyColor.debug)
        text = f'>> {t_current:.2f} | {t_start:.2f} | {t_stop:.2f}'
        drawer.text((0, height), text, font=MyFont.small_font,
                    anchor='lb', fill=MyColor.text)

    # Draw windowFrame
    if CONFIG.toggle.windowFrame:
        drawer.rectangle(
            (0, 0, width, height), outline=MyColor.debug, width=2)

    # Draw pathCurve
    if CONFIG.toggle.pathCurve:
        for a, b in zip(CONFIG.checkpoints[:-1], CONFIG.checkpoints[1:]):
            drawer.line((a['x'], a['y'], b['x'], b['y']),
                        fill=MyColor.debug, width=3)

    # Draw checkpointNode
    if CONFIG.toggle.checkpointNode:
        r = CONFIG.unit.em/2
        for i, n in enumerate(CONFIG.checkpoints):
            x = n['x']
            y = n['y']
            s = n['seconds']
            drawer.ellipse((x-r, y-r, x+r, y+r), outline=MyColor.debug)
            drawer.text((x, y), f'.{i}({s:.2f})', anchor='lb',
                        font=MyFont.small_font, fill=MyColor.text)

    # ----------------------------------------
    # ---- Draw scatters ----

    n = None

    for a, b in zip(CONFIG.checkpoints[:-1], CONFIG.checkpoints[1:]):
        if all([a['seconds'] <= t_current, b['seconds'] >= t_current]):
            r = (t_current - a['seconds']) / (b['seconds'] - a['seconds'])
            n = dict(
                seconds=t_current, x=b['x']*r + a['x']*(1-r), y=b['y']*r + a['y']*(1-r))
            break

    if n:
        affect_range = CONFIG.scatter.affectRange
        x = n['x']
        y = n['y']
        ds.place_nodes(t_current, x, y)

        # Draw currentNode
        if CONFIG.toggle.currentNode:
            r = CONFIG.unit.em/2
            drawer.ellipse(
                (x-r, y-r, x+r, y+r), fill=MyColor.debug)
            overlay_drawer.ellipse(
                (x-affect_range, y-affect_range, x+affect_range, y+affect_range), fill=(255, 0, 0, 30))

        # Only draw the placed nodes
        for node in [e for e in ds.nodes if e.get('x')]:
            x = node['x'] + node['dx']
            y = node['y'] + node['dy']
            k = (t_current - node['t0'])/(node['t1'] - node['t0'])
            r = ds.max_r * k + ds.min_r*(1-k)
            alpha = ds.stop_alpha * k + ds.start_alpha * (1-k)
            hex_alpha = hex(int(alpha)).replace('x', '')[-2:]
            overlay_drawer.ellipse(
                (x-r, y-r, x+r, y+r), fill=MyColor.scatter+hex_alpha)

    if CONFIG.toggle.keyPressedAnnotation:
        drawer.text((width, height), key_pressed, font=MyFont.small_font,
                    anchor='rb', fill=MyColor.text)

    # Composite the overlay image
    img = Image.composite(img, overlay_img, img)

    return img


class OnScreenPainter(object):
    app = app
    screen = screen

    window = QMainWindow()
    pixmap_container = QLabel(window)
    pixmap = None

    width = CONFIG.window.width
    height = CONFIG.window.height

    _rlock = RLock()
    running = False
    key_pressed = ''

    def __init__(self):
        self._prepare_window()
        logger.info('Initialized {}, {}'.format(
            self, {k: self.__getattribute__(k) for k in dir(self) if not k.startswith('_')}))

    def _prepare_window(self):
        # Translucent image by its RGBA A channel
        self.window.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Disable frame and keep the window on the top layer.
        # It is necessary to set the FramelessWindowHint for the WA_TranslucentBackground works.
        # The WindowTransparentForInput option disables interaction.
        self.window.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowTransparentForInput
        )

        # Set the window size
        self.window.resize(self.width, self.height)

        # Put the window to the NW corner
        rect = self.screen.geometry()
        self.window.move(rect.x(), rect.y())

        # Set the pixmap_container accordingly,
        # and it is within the window bounds
        self.pixmap_container.setGeometry(0, 0, self.width, self.height)

    @contextlib.contextmanager
    def acquire_lock(self):
        self._rlock.acquire()
        try:
            yield
        finally:
            self._rlock.release()

    def repaint(self):
        # pos = mouse_position()
        # img = generate_snoise_img(self.width, self.height, pos)
        # pixmap = QPixmap.fromImage(ImageQt(img))
        # self.pixmap_container.setPixmap(pixmap)
        with self.acquire_lock():
            if pixmap := self.pixmap:
                self.pixmap_container.setPixmap(pixmap)
        return

    def main_loop(self):
        Thread(target=self._main_loop, daemon=True).start()

    def _main_loop(self):
        self.running = True
        tic = time.time()
        while self.running:
            img = generate_img(
                passed=time.time()-tic, key_pressed=self.key_pressed)
            with self.acquire_lock():
                self.pixmap = QPixmap.fromImage(ImageQt(img))
            time.sleep(0.01)


# %% ---- 2024-10-23 ------------------------
# Play ground
if __name__ == "__main__":
    osp = OnScreenPainter()

    osp.window.show()
    # osp.main_loop()

    def _on_timeout():
        osp.repaint()

    def _on_key_pressed(event):
        '''
        Handle the key pressed event.

        Args:
            - event: The pressed event.
        '''

        try:
            key = event.key()
            enum = Qt.Key(key)
            logger.debug(f'Key pressed: {key}, {enum.name}')

            osp.key_pressed = str(enum.name)

            # If esc is pressed, quit the app
            if enum.name == 'Key_Escape':
                osp.app.quit()

            if enum.name == 'Key_S' and not osp.running:
                osp.main_loop()

        except Exception as err:
            logger.error(f'Key pressed but I got an error: {err}')

    def _about_to_quit():
        '''
        Safely quit
        '''
        logger.debug('Safely quit the application')
        return

    # Bind the _about_to_quit and _on_key_pressed methods
    osp.app.aboutToQuit.connect(_about_to_quit)
    osp.window.keyPressEvent = _on_key_pressed

    # Bind to the timer and run
    timer = QTimer()
    timer.timeout.connect(_on_timeout)
    timer.start()

    # Proper exit
    sys.exit(osp.app.exec())
# %% ---- 2024-10-23 ------------------------
# Pending


# %% ---- 2024-10-23 ------------------------
# Pending
