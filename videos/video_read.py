
import os
# import cv2

def compare_video_sizes(file1, file2,filename1,filename2):
    video_path1 = os.path.join(filename1, file1)
    video_path2 = os.path.join(filename2, file2)
    size1 = os.path.getsize(video_path1)
    size2 = os.path.getsize(video_path2)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    if size1 == size2 and file1 != file2:
        print(f"{file1} 和 {file2} 的大小相同，均为 {size1} 字节")
    # cap.release()

def read_video_names(filename1,filename2):
    for file1 in os.listdir(filename1):
        if file1.endswith(".mp4") or file1.endswith(".avi") or file1.endswith(".mkv") or file1.endswith(".wmv"):
            for file2 in os.listdir(filename2):
                if file2.endswith(".mp4") or file2.endswith(".avi") or file2.endswith(".mkv") or file1.endswith(".wmv"):
                    compare_video_sizes(file1, file2,filename1,filename2)


if __name__ == '__main__':
    filename1 = "J:/联想/1/WMZE"
    filename2 = "J:/联想/1/无码看"
    filename3 = "J:/新"
    # filename4 = "J:/大电影"
    # filename5 = "D:/新建文件夹"
    filename4 =  "F:/新建文件夹"
    filename5 = "I:/新"
    # filename8 = "J:/按摩"
    # filename9 = "I:move"
    filename6 = "K:/按摩"
    filename7 = "K:/大电影"
    read_video_names(filename5,filename7)
