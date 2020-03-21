from tkinter import *

root = Tk()
canvas = Canvas(root,width=784,height=784)
canvas.pack()

img = []
for i in range(784):
    img.append([0])

def mouse_press(event):
    global img

    x = event.x//28
    y = event.y//28

    img[x+y*28] = 1

root.bind('<Button-1>',mouse_press)

while True:
    canvas.delete(ALL)
    for i in range(784):
        x = i % 28
        y = i // 28
        x *= 28
        y *= 28
        if img[i] == 1:
            #print(x,y)
            canvas.create_rectangle(x,y,x+28,y+28,fill='black')

    root.update()
