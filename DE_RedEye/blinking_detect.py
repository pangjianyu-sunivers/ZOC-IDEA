import sys
import os
import dlib
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


# Module-level resources reused by every worker process.
predictor_path = '/data/DE_system/shape_predictor_68_face_landmarks.dat'
faces_folder_path = "/data/DE_system/examples"
scaling = 0.3
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def process_video(args):
    """Run dlib on every frame of one video and return the per-frame opening / browline series."""
    folder_path, video_name = args
    video_path = folder_path + '/' + video_name
    # output_video_path = 'video_results_240925/' + video_name


    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(scaling * frame_width), int(scaling * frame_height)))

    blinking = []
    periorbital_motion = []
    ax_x = 0
    xx = []
    yy = []
    xx1 = []
    yy1 = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (0, 0), fx=scaling, fy=scaling)
        img = frame
        dets = detector(img, 1)

        for k, d in enumerate(dets):
            shape = predictor(img, d)
            right_eye_status = abs(shape.part(37).y - shape.part(41).y) + abs(shape.part(38).y - shape.part(40).y)
            left_eye_status = abs(shape.part(43).y - shape.part(47).y) + abs(shape.part(44).y - shape.part(48).y)
            both_eye_status = (right_eye_status + left_eye_status) / 2
            blinking.append(round(both_eye_status, 2))

            # Browline-to-eye distance (used as a QC signal: large values = brow raise = open eye)
            right_brow_status = (abs(shape.part(22).y - shape.part(42).y) + abs(shape.part(23).y - shape.part(42).y) + abs(shape.part(24).y - shape.part(42).y) + abs(shape.part(25).y - shape.part(42).y) + abs(shape.part(26).y - shape.part(42).y))/(5*(abs(shape.part(22).y - shape.part(42).y)))
            left_brow_status = (abs(shape.part(17).y - shape.part(39).y) + abs(shape.part(18).y - shape.part(39).y) + abs(shape.part(19).y - shape.part(39).y) + abs(shape.part(20).y - shape.part(39).y) + abs(shape.part(21).y - shape.part(39).y))/(5*(abs(shape.part(22).y - shape.part(42).y)))
            both_brow_status = (right_brow_status + left_brow_status) / 2
            # print('result:',round(both_eye_status,2))
            periorbital_motion.append(round(both_brow_status, 2))

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            # points1 = np.array([[shape.part(36).x, shape.part(36).y], [shape.part(37).x, shape.part(37).y], [shape.part(38).x, shape.part(38).y], [shape.part(39).x, shape.part(39).y], [shape.part(40).x, shape.part(40).y], [shape.part(41).x, shape.part(41).y]])
            # hull1 = cv2.convexHull(points1)
            # points2 = np.array([[shape.part(42).x, shape.part(42).y], [shape.part(43).x, shape.part(43).y], [shape.part(44).x, shape.part(44).y], [shape.part(45).x, shape.part(45).y], [shape.part(46).x, shape.part(46).y], [shape.part(47).x, shape.part(47).y]])
            # hull2 = cv2.convexHull(points2)
            # cv2.fillConvexPoly(mask, hull1, 255)
            # cv2.fillConvexPoly(mask, hull2, 255)
            # img = cv2.bitwise_and(img, img, mask=mask)
            # Left periocular polygon
            points1 = np.array([[shape.part(17).x,shape.part(17).y],[shape.part(18).x,shape.part(18).y],[shape.part(19).x,shape.part(19).y],[shape.part(20).x,shape.part(20).y],[shape.part(21).x,shape.part(21).y],[shape.part(27).x,shape.part(27).y],[shape.part(28).x,shape.part(28).y],[shape.part(21).x,shape.part(21).y+2*(shape.part(39).y-shape.part(21).y)],[shape.part(20).x,shape.part(20).y+2*(shape.part(39).y-shape.part(20).y)],[shape.part(19).x,shape.part(19).y+2*(shape.part(39).y-shape.part(19).y)],[shape.part(18).x,shape.part(18).y+2*(shape.part(39).y-shape.part(18).y)],[shape.part(17).x,shape.part(17).y+2*(shape.part(39).y-shape.part(17).y)]])
            hull1 = cv2.convexHull(points1)

            # Right periocular polygon
            points2 = np.array([[shape.part(27).x,shape.part(27).y],[shape.part(22).x,shape.part(22).y],[shape.part(23).x,shape.part(23).y],[shape.part(24).x,shape.part(24).y],[shape.part(25).x,shape.part(25).y],[shape.part(26).x,shape.part(26).y],[shape.part(26).x,shape.part(26).y+2*(shape.part(39).y-shape.part(26).y)],[shape.part(25).x,shape.part(25).y+2*(shape.part(39).y-shape.part(25).y)],[shape.part(24).x,shape.part(24).y+2*(shape.part(39).y-shape.part(24).y)],[shape.part(23).x,shape.part(23).y+2*(shape.part(39).y-shape.part(23).y)],[shape.part(22).x,shape.part(22).y+2*(shape.part(39).y-shape.part(22).y)],[shape.part(28).x,shape.part(28).y]])
            hull2 = cv2.convexHull(points2)

            # Fill the periocular masks
            cv2.fillConvexPoly(mask, hull1, 255)
            cv2.fillConvexPoly(mask, hull2, 255)

            # Keep only the masked region
            img = cv2.bitwise_and(img, img, mask=mask)

            for n in range(68):
                x = shape.part(n).x
                y = shape.part(n).y
                if n <= 41 and n >= 36:
                    cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
                    cv2.putText(img, f'{n}', (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 255, 0), 1)
                elif n <= 47 and n >= 42:
                    cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
                    cv2.putText(img, f'{n}', (x + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 255, 0), 1)
                else:
                    cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

            height, width = img.shape[:2]
            fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
            ax.set_position([0, 0, 1, 1])
            ax.set_xlim(0, width)
            ax.set_ylim(0, height)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ax.imshow(img, extent=[0, width, 0, height])


            # ax_y = both_eye_status * height / 10 + height * 1 / 2
            # ax_y = both_eye_status + height * 1 / 2
            # xx.append(ax_x)
            # yy.append(ax_y)
            # ax_x += 1
            # new_xx = [xxx * width / len(xx) for xxx in xx]
            # ax.plot(new_xx, yy, color='white', linewidth=1)


            # 5.1. Build the eye-opening trace data
            # ax_x += 1
            # ax_y = both_eye_status*height/10 + height*2/3
            ax_y = both_eye_status + height * 2 / 3

            xx.append(ax_x)
            yy.append(ax_y)

            # ax_x += 1

            new_xx = []
            for xxx in xx:
                new_xx.append(xxx * width / len(xx))

            # 5.2. Build the periorbital trace data
            # ax_x += 1
            ax_y1 = both_brow_status + height*2/3

            xx1.append(ax_x)
            yy1.append(ax_y1)

            # ax_x += 1

            new_xx1 = []
            for xxx1 in xx1:
                new_xx1.append(xxx1 * width / len(xx1))


            # Frame counter
            ax_x += 1


            # 6. Render both traces on top of the masked frame
            ax.plot(new_xx, yy, color='white', linewidth=1)
            ax.plot(new_xx1, yy1, color='blue', linewidth=1)


            plot_text = f'seconds:{round(ax_x / fps, 2)}\nblink parameters:{round(both_eye_status, 2)}'
            ax.text(0, 0, plot_text, fontsize=8, color='red')
            ax.axis('off')
            plt.savefig(f'/data/DE_system/middle_stage_fig/{video_name}.png', bbox_inches='tight', pad_inches=0)
            plt.close()

        image = cv2.imread(f'/data/DE_system/middle_stage_fig/{video_name}.png')
        # out.write(image)
    cap.release()
    # out.release()

    return video_name, blinking, periorbital_motion

def blinking_extract(folder_path):
    """Process every video in ``folder_path`` in parallel and dump the per-video time series."""
    # folder_path = '/data/videos/2025-02-12/21-10/'
    # folder_path = 'videos_test/'
    video_name_list = os.listdir(folder_path)
    df = pd.DataFrame()
    # video_tasks = [(folder, video_name) for folder in folder_list for video_name in os.listdir(folder_path + folder)]
    video_tasks = [(folder_path, video_name) for video_name in video_name_list]

    with Pool(30) as pool:
        results = pool.map(process_video, video_tasks)

    for i, (video_name, blinking, periorbital_motion) in enumerate(results):
        df.at[i, 'video_name'] = video_name
        df.at[i, 'blinking'] = str(blinking)
        df.at[i, 'periorbital_motion'] = str(periorbital_motion)

    df.to_excel(f'{folder_path}/blinking_series.xlsx', index=False)

if __name__ == '__main__':
    main()
