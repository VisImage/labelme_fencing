import os
import cv2
import numpy as np
import json

# this file is originated at F:\gitSources\labelme_fencing\fencing
# as a clone from https://github.com/VisImage/labelme_fencing
# please do not modify otherwise

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
            self.homography, _ = cv2.findHomography(_src,_dst,cv2.RANSAC,15.0)
    
    def extend_line(self,w,h,line):   
        def get_boundary_point(width,height,pt,pt_end):
            # Ray origin and direction
            # origin = np.array(pt_end, dtype=np.float32)
            # direction = np.array(pt - pt_end, dtype=np.float32)  # Example direction

            # # Normalize direction
            # direction /= np.linalg.norm(direction)
            origin = pt_end
            direction = pt - pt_end  # Example direction

            pt0 = pt
            count = 0
            while width > pt[0] >=0 and height > pt[1] >0:
                count = count + 1
                pt = pt0 + count*direction 
            return pt
        
        output_line = [[],[]]
        if len(line) != 2:
            print("please input a list with 2 points")
            return []
        (k0, v0),  = line[0].items()
        (k1, v1),  = line[1].items()

        if "_end_" in k0:
            output_line[0] = v0
        else:           
            pt0 = get_boundary_point (w,h,v0,v1)
            output_line[0] = pt0

        if "_end_" in k1:
            output_line[1] = v1
        else:           
            pt1 = get_boundary_point (w,h,v1,v0)
            output_line[1] = pt1

        return output_line
        
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
        
        transfored_dict = {}
        pts = []
        for index in range(14): #key in self.keypoints.keys():
            pt = np.array(transformed_kp[index], dtype=np.int32)
            pts.append(pt)
            transfored_dict[label_list[index]] = pt

        # piste_lines = [[pt[0],pt[12]],[pt[1],pt[13]],[pt[0],pt[1]],[pt[2],pt[3]],[pt[4],pt[5]],
        #                [pt[6],pt[7]],[pt[8],pt[9]],[pt[10],pt[11]],[pt[12],pt[13]]]
        piste_lines = [[pts[0],pts[1]],[pts[2],pts[3]],[pts[4],pts[5]],
                       [pts[6],pts[7]],[pts[8],pts[9]],[pts[10],pts[11]],[pts[12],pts[13]]]
        piste_side_lines = [[],[]]
        # Draw each line
        for pt1, pt2 in piste_lines:
            cv2.line(canvas, pt1, pt2, (0, 255, 255), 2)

        # two long lines are handled below, seperately to avoid sign flip due to error on w
        # as appeared in "20191129_IMG_2244_0.json"
        far_line = [{},{}]
        near_line = [{},{}]
        for key in transfored_dict.keys():
            pt = transfored_dict[key]
            if "end_far" in key and 0 <= pt[0] < w and 0 <= pt[1] < h:
                if len(far_line[0]) == 0:
                    far_line[0] = {key:pt} 
                else:
                    far_line[1] = {key:pt} 
            if "end_near" in key and 0 <= pt[0] < w and 0 <= pt[1] < h:
                if len(near_line[0]) == 0:
                    near_line[0] = {key:pt} 
                else:
                    near_line[1] = {key:pt} 

        for key in transfored_dict.keys():
            pt = transfored_dict[key]
            if "far" in key and "end" not in key and 0 <= pt[0] < w and 0 <= pt[1] < h:
                if len(far_line[0]) == 0:
                    far_line[0] = {key:pt} 
                elif len(far_line[1]) == 0:
                    far_line[1] = {key:pt} 
                else:
                    (k0, v0),  = far_line[0].items()
                    (k1, v1),  = far_line[1].items()
                    d1 = np.linalg.norm(v0-v1) 
                    d2 = np.linalg.norm(v0-pt) 
                    if d1 < d2:
                        far_line[1] = {key:pt} 
            if "near" in key and "end" not in key and 0 <= pt[0] < w and 0 <= pt[1] < h:
                if len(near_line[0]) == 0:
                    near_line[0] = {key:pt} 
                elif len(near_line[1]) == 0:
                    near_line[1] = {key:pt} 
                else:
                    (k0, v0),  = near_line[0].items()
                    (k1, v1),  = near_line[1].items()
                    d1 = np.linalg.norm(v0-v1) 
                    d2 = np.linalg.norm(v0-pt) 
                    if d1 < d2:
                        near_line[1] = {key:pt} 

        piste_side_lines[0] = self.extend_line (w,h,far_line)
        piste_side_lines[1] = self.extend_line (w,h,near_line)
        
        for pt1, pt2 in piste_side_lines:
            cv2.line(canvas, pt1,pt2, (0, 255, 255), 2)

        transparent_result = cv2.addWeighted(canvas, 0.5, img, 0.5, 0)

        return transparent_result
        
        # cv2.imshow("out.jpg",transparent_result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


labelme_path = "J:\\labelme_piste_data\\images"
files = [f for f in os.listdir(labelme_path) if f.endswith('.json')]

#for f in files:
if 2 > 1:
    f = "[Semi Final] Impeccable!! Vivian Kong v Alexanne Verret l Vancouver Epee Fencing WC 2022_0.json"
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



 