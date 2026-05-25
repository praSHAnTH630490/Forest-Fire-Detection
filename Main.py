from tkinter import *
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import datetime


main = Tk()
main.title("Forest Fire Detection System")
main.geometry("1300x850")
main.config(bg="#F7C74F")

filename = None
processing = False
video = None
frame_count = 0
current_frame_index = 0
previous_frame = None



def ColorFeaturesDetectFire(frame):
    msg = "No Fire Detected"
    blur = cv2.GaussianBlur(frame, (21, 21), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 120, 200], dtype=np.uint8)
    upper = np.array([35, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    no_red = cv2.countNonZero(mask)
    if int(no_red) > 4000:
        msg = "Fire Detected"
    return msg, mask

def upload():
    global filename, video, frame_count, current_frame_index, previous_frame
    filename = filedialog.askopenfilename(initialdir="UAV_Videos")
    if filename:
        log_text.config(state=NORMAL)
        log_text.delete('1.0', END)
        log_text.insert(END, f"Loaded Video: {filename}\n")
        log_text.config(state=DISABLED)
        progress_bar['value'] = 0

        video = cv2.VideoCapture(filename)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame_index = 0
        previous_frame = None

def show_frames(color_frame, mask_frame, fire_status):
    color_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    img_color = Image.fromarray(color_rgb)
    img_color = img_color.resize((color_canvas.winfo_width(), color_canvas.winfo_height()))
    imgtk_color = ImageTk.PhotoImage(image=img_color)
    color_canvas.imgtk = imgtk_color
    color_canvas.create_image(0, 0, anchor=NW, image=imgtk_color)

    mask_bgr = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)
    img_mask = Image.fromarray(mask_bgr)
    img_mask = img_mask.resize((mask_canvas.winfo_width(), mask_canvas.winfo_height()))
    imgtk_mask = ImageTk.PhotoImage(image=img_mask)
    mask_canvas.imgtk = imgtk_mask
    mask_canvas.create_image(0, 0, anchor=NW, image=imgtk_mask)
    
    if fire_status == "Fire Detected":
        fire_status_label.config(text=fire_status, fg="red")
    else:
        fire_status_label.config(text=fire_status, fg="lime green")



    

def process_next_frame():
    global current_frame_index, video, previous_frame, processing
    if not processing or video is None:
        return
    
    ret, frame = video.read()

    if not ret:
        messagebox.showinfo("Finished", "Video processing completed!")
        processing = False
        progress_bar['value'] = 0
        video.release()
        return
    
    msg, mask = ColorFeaturesDetectFire(frame)

   
    if msg == "Fire Detected":
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Get largest contour (main fire area)
            largest_contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest_contour) > 1000:  # strong filtering
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

                # Highlight region (semi-transparent red overlay)
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 0, 255), -1)
                alpha = 0.3  # transparency
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Label
                cv2.putText(frame, "FIRE DETECTED", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
  


    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    if previous_frame is None:
        previous_frame = gray_frame
    diff_frame = cv2.absdiff(previous_frame, gray_frame)
    previous_frame = gray_frame

    kernel = np.ones((5, 5))
    diff_frame = cv2.dilate(diff_frame, kernel, 1)
    thresh_frame = cv2.threshold(diff_frame, 20, 255, cv2.THRESH_BINARY)[1]

    cv2.putText(frame, msg, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    show_frames(frame, thresh_frame, msg)

    time_now = datetime.datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, time_now, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    log_text.config(state=NORMAL)
    log_text.insert(END, f"{msg}\n")
    log_text.see(END)
    log_text.config(state=DISABLED)

    current_frame_index += 1
    progress_bar['value'] = (current_frame_index / frame_count) * 100

    main.after(30, process_next_frame)

def start_processing():
    global processing
    if not filename:
        messagebox.showerror("Error", "Please upload a video first!")
        return
    processing = True
    process_next_frame()

def stop_processing():
    global processing
    processing = False

def exit_app():
    global processing
    processing = False
    main.destroy()



style = ttk.Style()
style.configure('TButton', font=('Helvetica', 12, 'bold'), padding=6)
style.configure('TLabel', font=('Helvetica', 14, 'bold'))

title_label = Label(main, text=" FOREST FIRE DETECTION SYSTEM ",
                    bg="#F9F180", fg="#DC0202")
title_label.config(font=('times', 20, 'bold'), height=2)
title_label.pack(fill=X, pady=5)

fire_status_label = Label(main, text="No Fire Detected",
                          font=('Helvetica', 18, 'bold'),
                          fg="lime green", bg="white",
                          width=30, height=2, relief=RIDGE, bd=3)
fire_status_label.pack(pady=5)

top_frame = Frame(main, bg="#F7C74F")
top_frame.pack(pady=10, fill=X)

ttk.Button(top_frame, text="Upload Video", command=upload).pack(side=LEFT, padx=15)
ttk.Button(top_frame, text="Start", command=start_processing).pack(side=LEFT, padx=15)
ttk.Button(top_frame, text="Stop", command=stop_processing).pack(side=LEFT, padx=15)
ttk.Button(top_frame, text="Exit", command=exit_app).pack(side=RIGHT, padx=15)

notebook = ttk.Notebook(main)
notebook.pack(expand=True, fill=BOTH, pady=5)

video_tab = Frame(notebook, bg="#F7C74F")
notebook.add(video_tab, text="Video Display")

video_frame = Frame(video_tab, bg="#F7C74F")
video_frame.pack(expand=True, fill=BOTH)

color_canvas = Canvas(video_frame, bg="#E0DFDD", width=640, height=400)
color_canvas.pack(side=LEFT, expand=True, fill=BOTH, padx=5, pady=5)

mask_canvas = Canvas(video_frame, bg="#E0DFDD", width=640, height=400)
mask_canvas.pack(side=LEFT, expand=True, fill=BOTH, padx=5, pady=5)

logs_tab = Frame(notebook, bg="#2E8B57")
notebook.add(logs_tab, text="Logs")

log_text = Text(logs_tab, font=('Helvetica', 12), state=DISABLED, wrap=WORD)
log_text.pack(side=LEFT, expand=True, fill=BOTH, padx=5, pady=5)

scrollbar = Scrollbar(logs_tab, command=log_text.yview)
scrollbar.pack(side=LEFT, fill=Y)
log_text.config(yscrollcommand=scrollbar.set)

progress_bar = ttk.Progressbar(main, orient=HORIZONTAL, length=1200, mode='determinate')
progress_bar.pack(pady=10)

main.mainloop()
