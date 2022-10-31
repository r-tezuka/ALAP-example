import tkinter as tk

def btn_clicked():
	print("Button Clicked")

# ウィンドウ作成
root = tk.Tk()

# ボタンの作成と配置
button1 = tk.Button(root, text="Button", command=btn_clicked)
button1.place(x=10, y=20, width=100, height=50)

# メインループ
root.mainloop()