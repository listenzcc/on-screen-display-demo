from nicegui import ui
import socket


def send_command(command, value):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(('localhost', 65432))
        s.sendall(f'{command}:{value}'.encode('utf-8'))
        response = s.recv(1024)
        print('Received', repr(response))


def hex2rgb(hex):
    hex = hex.lstrip('#')
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4)) + (255,)


ui.label('Speed Control')
ui.slider(min=0.1, max=10, value=1, step=0.1).props('label-always').on_value_change(
    lambda e: send_command('speed', e.value))
ui.label('Radius Control')
ui.slider(min=1, max=20, value=3, step=1).props('label-always').on_value_change(
    lambda e: send_command('radius', e.value))
ui.label('Color Control')
ui.color_input(label='Color', value='#ff0000').on_value_change(
    lambda e: send_command('color', ','.join(map(str, hex2rgb(e.value)))))
ui.run(reload=False)
