"""
File: app.py
Author: Chuncheng Zhang
Date: 2024-10-22
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    On-screen display application.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2024-10-22 ------------------------
# Requirements and constants
import sys
import time
import noise
import pyautogui
import itertools
import contextlib
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageQt import ImageQt
from threading import Thread, RLock

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QMainWindow, QApplication, QLabel

from rich import print
from loguru import logger

# %% ---- 2024-10-22 ------------------------
# Function and class

app = QApplication(sys.argv)

snoise = noise.snoise3


def mouse_position():
    size = pyautogui.size()
    pos = pyautogui.position()
    x = pos.x / size.width
    y = pos.y / size.height
    return x, y


def generate_snoise_img(width: int, height: int, pos: tuple = None):
    # Rulers
    if width > height:
        _width = 100
        _w_value = 2
        _height = int(_width * height / width)
        _h_value = _w_value * height / width
    else:
        _height = 100
        _h_value = 2
        _width = int(_height * width / height)
        _w_value = _h_value * width / height

    # print(_width, _height, _w_value, _h_value)

    # Generate simplex noise image
    mat = np.zeros((_width, _height, 4))
    if False:
        z = time.time() % 3600
        for i, j in itertools.product(range(_width), range(_height)):
            x = i / _width * _w_value
            y = j / _height * _h_value
            mat[i, j] = snoise(x, y, z)
        mat += 1
        mat *= 128
    mat = mat.astype(np.uint8)

    img = Image.fromarray(
        mat.transpose((1, 0, 2)), mode='RGBA').resize((width, height))

    # Put mouse cursor
    if pos is not None:
        x, y = pos
        x *= width
        y *= height

        # res: the temporary resolution
        # r: the max radius
        # T: period in seconds
        R = width / 20
        T = 2
        # t is the value of (0, 1)
        t = (time.time() % T) / T
        r = t * R
        alpha = np.exp(-t)

        drawer = ImageDraw.Draw(img)
        drawer.ellipse(
            (x-r, y-r, x+r, y+r), fill=(233, 163, 104, int(alpha*255)))

    return img


class OnScreenPainter(object):
    app = app
    screen = app.primaryScreen()

    window = QMainWindow()
    pixmap_container = QLabel(window)
    pixmap = None

    width = screen.size().width() // 1  # pixels
    height = screen.size().height() // 1  # pixels

    rlock = RLock()

    def __init__(self):
        self.prepare_window()

    def prepare_window(self):
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

        # Put the window to the right
        self.window.move(self.screen.size().width()-self.width, 0)

        # Set the pixmap_container accordingly,
        # and it is within the window bounds
        self.pixmap_container.setGeometry(0, 0, self.width, self.height)

    @contextlib.contextmanager
    def acquire_lock(self):
        self.rlock.acquire()
        try:
            yield
        finally:
            self.rlock.release()

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
        while self.running:
            pos = mouse_position()
            img = generate_snoise_img(self.width, self.height, pos)
            with self.acquire_lock():
                self.pixmap = QPixmap.fromImage(ImageQt(img))
            time.sleep(0.01)


# %% ---- 2024-10-22 ------------------------
# Play ground
if __name__ == "__main__":
    osp = OnScreenPainter()
    osp.window.show()
    osp.main_loop()

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

            # If esc is pressed, quit the app
            if enum.name == 'Key_Escape':
                osp.app.quit()

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

# %% ---- 2024-10-22 ------------------------
# Pending


# %% ---- 2024-10-22 ------------------------
# Pending
