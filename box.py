'''
函数描述：对给定的图片和坐标信息在图片上标框，并在框的上方标注出该框的名称
函数参数：img_file_path=图片的绝对路径，new_img_file_path=保存后的绝对路径，points=[(str,[b0,b1,b2,b3])]
返回值：无返回值
注意事项：坐标[b0,b1,b2,b3]依次为左上角和右下角的坐标
'''

import cv2

def draw_rectangle_by_point(img_file_path, new_img_file_path, points):
    image = cv2.imread(img_file_path)
    for item in points:
        print("当前字符：",item)
        point=item[1]
        first_point=(int(point[0]),int(point[1]))
        last_point=(int(point[2]) + int(point[0]),int(point[3]) + int(point[1]))

        # first_point = (point[0] * 2, point[1] * 2)
        # last_point = (point[2]* 2, point[3] * 2)
        print("左上角：",first_point)
        print("右下角：",last_point)
        cv2.rectangle(image, first_point, last_point, (0, 255, 0), 15)   # 在图片上进行绘制框
        cv2.putText(image, item[0], first_point, cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(255,0,0), thickness=4)  # 在矩形框上方绘制该框的名称
        
    cv2.imwrite(new_img_file_path, image)


if __name__ == '__main__':
    #points=[('8F.', [66.72106971740723, 193.4539794921875, 80.52160511016845, 207.30389404296875]), ('Zhongshan', [241.5278513590495, 193.4539794921875, 282.9294575373332, 207.30389404296875])]
    points1=[('Atelectasis', [494.1016949, 577.3920981, 271.1864407, 154.0338983])]
    points2=[('Cardiomegaly', [412.8507937, 468.1142857, 548.3005291, 501.7058201])]
    points3=[('Effusion', [195.5271181, 407.6468229, 669.0133333, 64.85333333])]
    points4=[('Infiltrate', [182.6133333, 392.8557118, 202.5244444, 309.4755556])]
    points5=[('Mass', [361.2444444, 284.7668229, 106.9511111, 135.3955556])]
    points6=[('Nodule', [695.6698413, 319.6613757, 100.7746032, 88.85502646])]
    points7=[('Pneumonia', [469.1978836, 465.9470899, 374.9248677, 373.8412698])]
    points8=[('Pneumothorax', [403.0933333, 681.8512674, 171.68, 182.0444444])]
    points9=[('Pneumothorax', [560.2201058, 137.6169312, 320.7449735, 273.0666667])]
    
    draw_rectangle_by_point(r'../../../data/ChestX-ray14/images/00001373_009.png',
                                 "box/00001373_009.jpg", points2)

