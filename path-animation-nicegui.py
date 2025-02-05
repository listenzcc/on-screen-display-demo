"""
File: path-animation-nicegui.py
Author: Chuncheng Zhang
Date: 2025-01-23
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-01-23 ------------------------
# Requirements and constants
import sys
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMainWindow, QApplication, QLabel
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QTimer
from omegaconf import OmegaConf
import time
import bezier
import contextlib
import socket
import threading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from loguru import logger
from threading import Thread, RLock
from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt
from rich import print
from tqdm.auto import tqdm
from IPython.display import display
from nicegui import ui

# %%
CONFIG = OmegaConf.load('path-animation-nicegui.yaml')

app = QApplication(sys.argv)
logger.debug(f'App: {app}')

screen = app.screens()[CONFIG.screen.id]
logger.debug(f'Screen: {CONFIG.screen.id}: {screen}, {screen.size()}')

# %% ---- 2025-01-23 ------------------------
# Function and class


def random_check_points(n: int):
    '''
    Create the random check points.

    :param n: the number of check points.
    :return: the check points.
    '''
    check_points = np.random.random((n, 2)) * 0.8 + 0.1
    return check_points


def mk_curve(check_points: np.ndarray):
    '''
    Create the bezier curve from a list of points.

    :param check_points: list of points.
    :return: the bezier curves.
    '''
    n = len(check_points)
    control_points = {}
    control_points[0] = dict(
        left=check_points[0], right=check_points[0]*0.5 + check_points[1]*0.5)
    control_points[n-1] = dict(
        left=check_points[-1]*0.5 + check_points[-2]*0.5, right=check_points[-1])
    for i in range(1, n-1):
        a = check_points[i-1]
        b = check_points[i]
        c = check_points[i+1]
        control_points[i] = dict(
            left=b+(a-c)*0.5,
            right=b-(a-c)*0.5
        )
    curves = []
    for i in range(1, n):
        a = check_points[i-1]
        b = check_points[i]
        c = control_points[i-1]['right']
        d = control_points[i]['left']
        cp = np.array((a, c, d, b))
        curves.append(bezier.Curve.from_nodes(cp.T))
    return curves


class RoadBook:
    schedule_time_cost: float = 10  # 30 * 60  # Seconds
    fps_in_schedule: float = 100  # fps
    resolution: int = int(schedule_time_cost * fps_in_schedule)  # Points

    def generate_data(self, n: int = 5):
        # Generate check_points and curves.
        check_points = random_check_points(n)
        curves = mk_curve(check_points)

        # Calculate the time cost and length of each curve.
        # Generate segment_table: (idx, start, end, curve_length, curve_obj)
        segment_table = []
        for i, curve in enumerate(curves):
            p1 = check_points[i]
            p2 = check_points[i+1]
            length = curve.length
            segment_table.append((i, p1, p2, length, curve))
        segment_table = pd.DataFrame(
            segment_table, columns=['idx', 'start', 'end', 'length', 'curve'])

        # Create the large_table
        total_curve_length = segment_table['length'].sum()
        speed_unit = total_curve_length / self.schedule_time_cost
        segment_table['n'] = (segment_table['length'] / total_curve_length *
                              self.resolution).map(int)
        segment_table['distance_offset'] = np.cumsum(
            segment_table['length']) - segment_table['length']

        large_table = []
        for i, slice in segment_table.iterrows():
            s = np.linspace(0, 1, slice['n'])
            distance = np.linspace(slice['distance_offset'],
                                   slice['distance_offset'] + slice['length'], slice['n'])
            df = pd.DataFrame(s, columns=['_s'])
            df['segment'] = i
            df['distance'] = distance
            df['pos'] = df['_s'].map(
                lambda s: slice['curve'].evaluate(s).squeeze())
            large_table.append(df)
        large_table = pd.concat(large_table)
        n = len(large_table)
        large_table['progress'] = np.linspace(0, 1, n)
        large_table['schedule_time'] = np.linspace(
            0, self.schedule_time_cost, n)
        large_table.index = range(n)

        # Instance data
        self.speed_unit = speed_unit
        self.total_curve_length = total_curve_length
        self.large_table = large_table
        self.check_points = check_points
        self.curves = curves
        self.data = segment_table
        return segment_table, large_table

    def plot_with_matplotlib(self):
        '''
        the trace is linear interpolated between the check points.
        plot the trace and mark the time between checkpoints.
        '''
        if self.check_points is None:
            print("No check points to plot.")
            return

        fig, ax = plt.subplots()

        # Plot check points
        ax.scatter(self.check_points[:, 0],
                   self.check_points[:, 1], c='r', marker='o')

        for curve in self.curves:
            curve.plot(num_pts=1000, ax=ax)

        # Plot linear interpolated trace with arrows and color based on time cost
        for i in range(len(self.check_points) - 1):
            p1 = self.check_points[i]
            p2 = self.check_points[i + 1]
            ax.annotate('', xy=(p2[0], p2[1]), xytext=(p1[0], p1[1]),
                        arrowprops=dict(arrowstyle="->", color='gray'))
            ax.text(x=p1[0], y=p1[1], s=f'{i}')
        ax.text(x=p2[0], y=p2[1], s=f'{i+1}')

        return fig

    def generate_road_map_image(self, width=CONFIG.window.width, height=CONFIG.window.height, padding=0):
        # 1. Generate the PIL.Image as the size of (width + 2*padding, height + 2*padding), the mode is RGBA.
        image = Image.new('RGBA', (width + 2 * padding,
                          height + 2 * padding), (255, 255, 255, 50))
        draw = ImageDraw.Draw(image)

        # 2. Draw the curves from the large_table's pos using a colormap
        # colormap = cm.get_cmap('viridis', len(self.curves))
        colormap = plt.colormaps.get_cmap('viridis')
        for i, curve in enumerate(self.curves):
            color = tuple(int(c * 255)
                          for c in colormap(i/len(self.curves))[:3]) + (255,)
            points = curve.evaluate_multi(np.linspace(0, 1, 1000)).T
            scaled_points = [(x * width + padding, y * height + padding)
                             for x, y in points]
            draw.line(scaled_points, fill=color, width=2)

        # Draw the linear interpolation between check points
        for i in range(len(self.check_points) - 1):
            p1 = self.check_points[i]
            p2 = self.check_points[i + 1]
            x1, y1 = p1[0] * width + padding, p1[1] * height + padding
            x2, y2 = p2[0] * width + padding, p2[1] * height + padding
            draw.line([(x1, y1), (x2, y2)], fill=(128, 128, 128, 255), width=1)

        # 3. Return the image
        self.road_map_image = image
        self.padding = padding
        self.width = width
        self.height = height
        return image

    def draw_node_at_distance(self, distance: float, radius: int = 3, color=(255, 0, 0, 255)):
        image = self.road_map_image.copy()
        draw = ImageDraw.Draw(image)

        try:
            slice = self.large_table[
                self.large_table['distance'].lt(distance)].iloc[-1]
        except:
            slice = self.large_table.iloc[0]

        x = slice['pos'][0] * self.width + self.padding
        y = slice['pos'][1] * self.height + self.padding
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius), fill=color)
        return image


class MovingNode:
    speed = 1
    speed_limit = (0.1, 10)
    _rlock = RLock()

    radius = 3
    color = (255, 0, 0, 255)
    name = 'Node1'

    def __init__(self, speed_unit: float, total_length: float):
        self.speed_unit = speed_unit
        self.total_length = total_length

    def go(self):
        Thread(target=self._moving_loop, daemon=True).start()

    @contextlib.contextmanager
    def lock(self):
        self._rlock.acquire()
        try:
            yield
        finally:
            self._rlock.release()

    def _moving_loop(self):
        self.t0 = time.time()
        self.t = self.t0
        self.distance = 0.0
        self.running = True
        logger.info(f'Node({self.name}) starts moving.')
        while self.running:
            time.sleep(0.01)
            with self.lock():
                dt = time.time() - self.t
                self.t = time.time()
                step = self.speed * self.speed_unit * dt
                self.distance += step

                if self.distance > self.total_length:
                    self.distance = 0.0
                    self.t0 = time.time()
        logger.info(f'Node({self.name}) stops moving.')

    def set_speed(self, speed: float):
        with self.lock():
            self.speed = max(self.speed_limit[0], min(
                self.speed_limit[1], speed))
        logger.info(f'Node({self.name}) speed set to {self.speed}')

    def set_radius(self, radius: int):
        with self.lock():
            self.radius = radius
        logger.info(f'Node({self.name}) radius set to {self.radius}')

    def set_color(self, color: tuple):
        with self.lock():
            self.color = color
        logger.info(f'Node({self.name}) color set to {self.color}')

    def start_socket_server(self, host='localhost', port=65432):
        def handle_client(client_socket):
            with client_socket:
                while True:
                    data = client_socket.recv(1024)
                    if not data:
                        break
                    try:
                        command, value = data.decode('utf-8').split(':')
                        if command == 'speed':
                            self.set_speed(float(value))
                        elif command == 'radius':
                            self.set_radius(int(value))
                        elif command == 'color':
                            self.set_color(tuple(map(int, value.split(','))))
                        client_socket.sendall(b'OK')
                    except ValueError:
                        client_socket.sendall(b'ERROR')

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((host, port))
        server.listen()

        def server_thread():
            while True:
                client_socket, addr = server.accept()
                threading.Thread(target=handle_client,
                                 args=(client_socket,)).start()

        threading.Thread(target=server_thread, daemon=True).start()
        logger.info(f'Socket server started on {host}:{port}')


# %%
rb = RoadBook()
segment_table, large_table = rb.generate_data(5)
display(segment_table)
display(large_table)
print(rb.speed_unit)

# Draw in matplotlib
fig = rb.plot_with_matplotlib()
# plt.show()

road_map_image = rb.generate_road_map_image()
display(road_map_image)

img = rb.draw_node_at_distance(0.0)
display(img)

# %%
mn1 = MovingNode(rb.speed_unit, rb.total_curve_length)
mn1.go()
mn1.start_socket_server()


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
        if not self.running:
            Thread(target=self._main_loop, daemon=True).start()
        else:
            logger.error(
                'Failed to start main_loop, since it is already running.')

    def _main_loop(self):
        self.running = True
        while self.running:
            # img = rb.road_map_image
            img = rb.draw_node_at_distance(
                mn1.distance, mn1.radius, mn1.color)
            with self.acquire_lock():
                self.pixmap = QPixmap.fromImage(ImageQt(img))
            time.sleep(0.01)


class AppThread(QThread):
    finished = pyqtSignal()

    def __init__(self, app):
        super().__init__()
        self.app = app

    def run(self):
        self.app.exec()
        self.finished.emit()


# %% ---- 2025-01-23 ------------------------
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

            if enum.name == 'Key_S':
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

# %% ---- 2025-01-23 ------------------------
# Pending

# %% ---- 2025-01-23 ------------------------
# Pending

# %%

# %%
