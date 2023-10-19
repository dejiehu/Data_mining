
import os
# import cv2



def read_video_names(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".mkv"):
            if filename.__contains__('@'):

                # print(filename)
                # 提取视频文件名（不包括扩展名）
                video_name = os.path.splitext(filename)[0]
                video_name = video_name.split('@')[1]
                # 生成新的文件名
                new_filename = video_name  + os.path.splitext(filename)[1]

                # 构建原文件路径和新文件路径
                old_path = os.path.join(folder_path, filename)
                new_path = os.path.join(folder_path, new_filename)

                # 重命名文件
                # print(old_path,new_path)
                os.rename(old_path, new_path)
                print(f"重命名文件：{filename} -> {new_filename}")

if __name__ == '__main__':
    folder_path = "I:/大电影/高质量WM"

    read_video_names(folder_path)
