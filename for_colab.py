#! python3
# -*- coding: utf-8 -*-

# colabを継続して使う

import time
import pyautogui

print()
print('''10分ごとにマウスカーソルを左右に少し動かし、クリックします。
12時間後に終了します。途中で停止するときは、Ctrl + c を押してください。
''')

try:
    dir = -10
    counter = 0

    while counter < 72:
        time.sleep(600)
        pyautogui.moveRel(dir, 0)
        dir = - dir

        pyautogui.click()
        counter += 1
        # print('カウンター：', counter)  # クリックしたときに出力したい場合はコメントアウト

    print('停止：１２時間経過')


except KeyboardInterrupt:
    print('停止：Ctrl + c による終了')