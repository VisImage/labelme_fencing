import os
import cv2
import numpy as np
import json

class piste_base:
    def __init__(self, id):
        self.piste_id = id
        self.keypoints = {}

# th elabel used in labelme for annotation
label_list = ["piste_left_end_far","piste_left_end_near",
               "piste_left_warn_far","piste_left_warn_near",
               "piste_left_onguard_far","piste_left_onguard_near",
               "piste_center_far","piste_center_near",
                "piste_right_onguard_far","piste_right_onguard_near",
                "piste_right_warn_far","piste_right_warn_near",
                "piste_right_end_far","piste_right_end_near"]

piste_width = 150
piste_temp_1 = piste_base(0)
piste_temp_1.keypoints[label_list[0]] = [-700, piste_width]
piste_temp_1.keypoints[label_list[1]] = [-700, 0]
piste_temp_1.keypoints[label_list[2]] = [-500, piste_width]
piste_temp_1.keypoints[label_list[3]] = [-500, 0]
piste_temp_1.keypoints[label_list[4]] = [-200, piste_width]
piste_temp_1.keypoints[label_list[5]] = [-200, 0]
piste_temp_1.keypoints[label_list[6]] = [0, piste_width]
piste_temp_1.keypoints[label_list[7]] = [0, 0]
piste_temp_1.keypoints[label_list[8]] = [200, piste_width]
piste_temp_1.keypoints[label_list[9]] = [200, 0]
piste_temp_1.keypoints[label_list[10]] = [500, piste_width]
piste_temp_1.keypoints[label_list[11]] = [500, 0]
piste_temp_1.keypoints[label_list[12]] = [700, piste_width]
piste_temp_1.keypoints[label_list[13]] = [700, 0]

class Piste(piste_base):
    def __init__(self, piste_temp, id):
        super().__init__(id)
        self.piste_temp = piste_temp
        self.annotateKPs = {}
        self.homography = [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]

    def Calculate_homography(self):
        pts_src = []
        pts_dst =[]
        for key in self.annotateKPs.keys():
            pts_src.append(self.piste_temp.keypoints[key])
            pts_dst.append(self.annotateKPs[key])

        if len(pts_src) < 4:
            print("There is not enought points for calculating homography!")
        else:
            # Compute the homography matrix
            _src = np.array(pts_src, dtype=np.float32)
            _dst = np.array(pts_dst, dtype=np.float32)
            self.homography, _ = cv2.findHomography(_src,_dst,cv2.RANSAC,5.0)

    def Draw(self,img):
        # Load your image
        h, w = img.shape[:2]

        # Create a black canvas
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        #Draw annotated point if any
        for key in self.annotateKPs.keys():
            pt1, pt2 = self.annotateKPs[key]
            cv2.circle(canvas, (int(pt1),int(pt2)), radius=5, color=(0, 255,0), thickness=-1)

        #Draw point from detected
        temp_keypoints = []
        for key in self.piste_temp.keypoints.keys():
            temp_keypoints.append([self.piste_temp.keypoints[key]])

        src = np.array(temp_keypoints,dtype=np.float32)
        perspect_keypoints = cv2.perspectiveTransform(src, self.homography)

        #transformed_kp = perspect_keypoints.tolist()
        transformed_kp  = [item[0] for item in perspect_keypoints.tolist()]
        pt = []
        for index in range(14): #key in self.keypoints.keys():
            pt.append(np.array(transformed_kp[index], dtype=np.int32))
        piste_lines = [[pt[0],pt[12]],[pt[1],pt[13]],[pt[0],pt[1]],[pt[2],pt[3]],[pt[4],pt[5]],
                       [pt[6],pt[7]],[pt[8],pt[9]],[pt[10],pt[11]],[pt[12],pt[13]]]
        # Draw each line
        for pt1, pt2 in piste_lines:
            cv2.line(canvas, pt1, pt2, (0, 255, 255), 2)

        transparent_result = cv2.addWeighted(canvas, 0.5, img, 0.5, 0)

        return transparent_result
        
        # cv2.imshow("out.jpg",transparent_result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


labelme_path = "J:\\labelme_piste_data\\images"
files = [f for f in os.listdir(labelme_path) if f.endswith('.json')]

for f in files:
    file_path = os.path.join(labelme_path, f)
    # Open and load the JSON file
    with open(file_path, 'r') as file:
        labelme_data = json.load(file)

    piste_1 = Piste(piste_temp_1,0)

    for pt in labelme_data["shapes"]:
        piste_1.annotateKPs[pt["label"]] = pt["points"][0]

    piste_1.Calculate_homography()

    image_path = os.path.join(labelme_path, labelme_data["imagePath"])
    result_image_name = os.path.join(labelme_path, "PisteAdded",labelme_data["imagePath"])
    image = cv2.imread(image_path)
    print(image_path)
    result_img = piste_1.Draw(image)

    cv2.imwrite(result_image_name,result_img)



 